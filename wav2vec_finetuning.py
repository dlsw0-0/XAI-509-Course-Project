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

# ⭐️ Added to make all paths relative to this `run/` directory instead of a
# machine-specific absolute path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: Correct paths depending on your environment
# ⭐️ TODO completed: use the dataset folders already placed under `run/dataset`.
db_top_dir = os.path.join(SCRIPT_DIR, "dataset")
train_top_dir = os.path.join(db_top_dir, "train")
test_top_dir = os.path.join(db_top_dir, "test-clean")
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
# End of ToDO

# ⭐️ Ensure sample_util uses the same processor object as this training script.
sample_util.processor = processor

train_dataset = sample_util.make_dataset(train_top_dir)
test_dataset = sample_util.make_dataset(test_top_dir)


def compute_metrics(pred) -> Dict[str, float]:
    """Compute word error rate (WER) between predictions and labels.

    This function decodes the model's predicted token IDs and ground truth
    label IDs into strings, replacing ignored label tokens with the padding
    token ID. Then it computes WER using the `evaluate` library.

    Args:
        pred: A prediction object with attributes:
            - predictions: logits or probabilities of shape
                (batch_size, seq_len, vocab_size).
            - label_ids: ground truth token IDs with padding replaced by -100.

    Returns:
        Dict[str, float]: Dictionary with WER under the key 'wer'.
    """
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 in labels with tokenizer pad token ID to enable decoding
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_metric = evaluate.load("wer")
    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_score}


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that dynamically pads input values and labels for CTC training.

    This class pads the input audio features and the corresponding label sequences
    (token IDs) to the length of the longest element in the batch. It also replaces
    padding tokens in the labels with -100 to ensure they are ignored during the loss
    computation, as required by PyTorch's CTC loss implementation.

    Attributes:
        processor (AutoProcessor): The processor used for feature extraction and tokenization.
        padding (Union[bool, str]): Padding strategy. Defaults to "longest" to pad to the
            longest sequence in the batch.
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Pad inputs and labels in a batch for model training.

        Args:
            features: A list of feature dictionaries, each containing:
                - "input_values": the audio features (list or tensor).
                - "labels": the tokenized label sequence.

        Returns:
            A dictionary with padded input tensors and labels ready for the model:
            - "input_values": Padded input audio feature tensor.
            - "labels": Padded label tensor with padding tokens replaced by -100.
        """
        # Separate the input audio features and label sequences from the batch.
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Use the processor's pad method to pad input audio features to the same length.
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Pad the label sequences separately using the processor's pad method.
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Replace padding tokens in labels with -100 so that the loss function ignores them.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Add the processed labels to the batch dictionary.
        batch["labels"] = labels

        return batch


class MaxEntropyCTCTrainer(Trainer):
    # ⭐️ Novelty: Maximum Entropy Training.
    # This regularizes the CTC model by rewarding higher output entropy, which
    # can reduce over-confident predictions on the small 1-hour fine-tuning set.
    """Trainer that adds maximum entropy regularization to CTC training."""

    def __init__(self, *args, entropy_weight: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_weight = entropy_weight

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        outputs = model(**inputs)
        ctc_loss = outputs.loss

        # ⭐️ Add entropy regularization: final loss = CTC loss - lambda * entropy.
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = ctc_loss - self.entropy_weight * entropy

        return (loss, outputs) if return_outputs else loss


# Instantiate the data collator for CTC loss with padding support.
# It dynamically pads the inputs and labels in each batch to the longest
# sequence, enabling efficient batch processing without manual padding.
data_collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding="longest"
)

# Load the pretrained Wav2Vec2 model with CTC (Connectionist Temporal Classification)
# head for speech recognition.
# - ctc_loss_reduction="mean" averages the CTC loss over the batch.
# - pad_token_id is set to the tokenizer's pad token to ensure correct masking.
model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    # ⭐️ Force safetensors loading because newer Transformers blocks
    # torch.load on torch<2.6 for security reasons.
    use_safetensors=True,
)
# ⭐️ Freeze the feature encoder for more stable fine-tuning on the small dataset.
model.freeze_feature_encoder()

# ⭐️ Environment-variable overrides make quick smoke tests possible, e.g.
# MAX_STEPS=2 EVAL_STEPS=1 python wav2vec_finetuning.py
max_steps = int(os.environ.get("MAX_STEPS", "2000"))
eval_steps = int(os.environ.get("EVAL_STEPS", "100"))
save_steps = int(os.environ.get("SAVE_STEPS", str(max_steps)))
entropy_weight = float(os.environ.get("ENTROPY_WEIGHT", "0.01"))
entropy_tag = str(entropy_weight).replace(".", "p")
default_output_dir = os.path.join(
    SCRIPT_DIR,
    "models",
    f"wav2vec2_maxent_{entropy_tag}"
)
output_dir = os.environ.get("OUTPUT_DIR", default_output_dir)
use_cuda = torch.cuda.is_available()

# Define the training arguments for the Hugging Face Trainer.
# These control training hyperparameters and runtime behavior:
training_args = TrainingArguments(
    # Directory to save model checkpoints and outputs.
    # ⭐️ Save the fine-tuned maximum-entropy model inside this project.
    # The default path includes ENTROPY_WEIGHT, e.g. wav2vec2_maxent_0p03.
    output_dir=output_dir,

    # Batch size per device (GPU/CPU) for training.
    per_device_train_batch_size=int(os.environ.get("TRAIN_BATCH_SIZE", "16")),

    # Number of batches to accumulate gradients over before updating model weights.
    gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM_STEPS", "2")),

    # Initial learning rate for the optimizer.
    learning_rate=float(os.environ.get("LEARNING_RATE", "1e-4")),

    # Number of warmup steps to gradually increase learning rate at start.
    warmup_steps=500,

    # Total number of training steps.
    max_steps=max_steps,

    # Enable gradient checkpointing to reduce memory usage at the cost of extra compute.
    gradient_checkpointing=True,

    # Use mixed precision training (float16) to speed up training and reduce memory.
    # ⭐️ Only enable fp16 when CUDA is actually available.
    fp16=use_cuda,

    # Performs evaluation every N steps (eval_strategy="steps").
    eval_strategy="steps",

    # Batch size per device during evaluation.
    per_device_eval_batch_size=int(os.environ.get("EVAL_BATCH_SIZE", "24")),

    # Save model checkpoints every N steps.
    save_steps=save_steps,

    # Run evaluation every N steps during training.
    eval_steps=eval_steps,

    # Log training progress every N steps.
    logging_steps=25,

    # Load the best model (lowest WER) at the end of training automatically.
    load_best_model_at_end=True,

    # Metric to use for selecting the best model checkpoint.
    metric_for_best_model="wer",

    # Indicates that a lower metric score (WER) is better.
    greater_is_better=False,

    # Disable pushing model to the Hugging Face hub.
    push_to_hub=False,

    # Keep WebDataset sample fields until the data collator extracts them.
    # ⭐️ Required because our WebDataset samples include extra fields used later.
    remove_unused_columns=False,

    # Avoid optional online experiment trackers for this class project script.
    report_to="none",
)

# TODO
# Create the Trainer instance to handle training and evaluation.
# This ties together the model, datasets, tokenizer, data collator, and metrics.
# ⭐️ TODO completed: use the custom maximum-entropy Trainer instead of the
# plain Trainer so the novelty is included during fine-tuning.
trainer = MaxEntropyCTCTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    entropy_weight=entropy_weight,
)
# End of TODO

trainer.train()
# ⭐️ Save both model and processor so wav2vec_inference.py can load them later.
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
