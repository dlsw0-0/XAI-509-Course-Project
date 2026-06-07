# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
from transformers import AutoModelForCTC, TrainingArguments, Trainer
from transformers import AutoProcessor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
import evaluate
import numpy as np

# Custom imports
import sample_util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Same dataset/processor setup as the MaxEnt and baseline scripts.
db_top_dir = os.path.join(SCRIPT_DIR, "dataset")
train_top_dir = os.path.join(db_top_dir, "train")
test_top_dir = os.path.join(db_top_dir, "test-clean")
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

sample_util.processor = processor

train_dataset = sample_util.make_dataset(train_top_dir)
test_dataset = sample_util.make_dataset(test_top_dir)


def compute_metrics(pred) -> Dict[str, float]:
    """Compute WER between predictions and labels (same as baseline/MaxEnt)."""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_metric = evaluate.load("wer")
    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_score}


@dataclass
class DataCollatorCTCWithPadding:
    """Same dynamic-padding collator used by the baseline/MaxEnt scripts."""

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


class LabelSmoothingCTCTrainer(Trainer):
    # Novelty: CTC + Label Smoothing.
    #
    # Standard label smoothing replaces the one-hot target with
    #   y_smooth = (1 - alpha) * one_hot + alpha * uniform
    # which is equivalent to adding a KL(uniform || p) term to the loss.
    # For CTC there is no per-frame hard target, but the same regularization
    # can be applied directly on the per-frame output distribution:
    #
    #   smooth_loss = -mean_over_(B,T,V) [ (1/V) * sum_v log p_v ]
    #               ~ KL(uniform || p) up to an additive constant
    #
    # Final loss:
    #   loss = (1 - alpha) * CTC_loss + alpha * smooth_loss
    #
    # Compared to MaxEntropyCTCTrainer (which uses H(p) = -sum p log p),
    # label smoothing weights every vocab entry equally instead of weighting
    # by the predicted probability.
    """Trainer that adds label smoothing on top of the CTC loss."""

    def __init__(self, *args, label_smoothing: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        outputs = model(**inputs)
        ctc_loss = outputs.loss

        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        # KL(uniform || p) per (batch, time) step, dropping the constant
        # -log(V) term. mean over vocab gives -(1/V) * sum_v log p_v.
        smooth_loss = -log_probs.mean(dim=-1).mean()

        alpha = self.label_smoothing
        loss = (1.0 - alpha) * ctc_loss + alpha * smooth_loss

        return (loss, outputs) if return_outputs else loss


data_collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding="longest"
)

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    use_safetensors=True,
)
model.freeze_feature_encoder()

max_steps = int(os.environ.get("MAX_STEPS", "2000"))
eval_steps = int(os.environ.get("EVAL_STEPS", "100"))
save_steps = int(os.environ.get("SAVE_STEPS", str(max_steps)))
label_smoothing = float(os.environ.get("LABEL_SMOOTHING", "0.1"))
smoothing_tag = str(label_smoothing).replace(".", "p")
default_output_dir = os.path.join(
    SCRIPT_DIR,
    "models",
    f"wav2vec2_label_smoothing_{smoothing_tag}"
)
output_dir = os.environ.get("OUTPUT_DIR", default_output_dir)
use_cuda = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=int(os.environ.get("TRAIN_BATCH_SIZE", "16")),
    gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM_STEPS", "2")),
    learning_rate=float(os.environ.get("LEARNING_RATE", "1e-4")),
    warmup_steps=500,
    max_steps=max_steps,
    gradient_checkpointing=True,
    fp16=use_cuda,
    eval_strategy="steps",
    per_device_eval_batch_size=int(os.environ.get("EVAL_BATCH_SIZE", "24")),
    save_steps=save_steps,
    eval_steps=eval_steps,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
    report_to="none",
)

trainer = LabelSmoothingCTCTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    label_smoothing=label_smoothing,
)

trainer.train()
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
