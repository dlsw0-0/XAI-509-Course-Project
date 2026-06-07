# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
# ⭐️ Added to choose GPU device automatically when CUDA is available.
import torch
from transformers import AutoProcessor
from transformers import pipeline

# Custom imports
# ⭐️ Replaced the missing `sample_util_solution` import with the local
# `sample_util.py` that we completed.
import sample_util
from evaluate_wer import evaluate_file

# ⭐️ Use paths relative to this `run/` directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ⭐️ Use the provided dataset folders under `run/dataset`.
db_top_dir = os.path.join(SCRIPT_DIR, "dataset")

test_clean_top_dir = os.path.join(db_top_dir, "test-clean")
test_other_top_dir = os.path.join(db_top_dir, "test-other")
# ⭐️ Default to the model produced by wav2vec_finetuning.py. You can override
# this with MODEL_DIR=/path/to/checkpoint if needed.
model_dir = os.environ.get(
    "MODEL_DIR",
    os.path.join(SCRIPT_DIR, "models", "wav2vec2_maxent")
)

model_name = os.path.basename(os.path.normpath(model_dir))
if model_name.startswith("checkpoint-"):
    model_name = (
        f"{os.path.basename(os.path.dirname(os.path.normpath(model_dir)))}_"
        f"{model_name}"
    )

# ⭐️ Save inference outputs under run/results/<model_name>/ so baseline and
# maximum-entropy results do not overwrite each other.
result_dir = os.environ.get(
    "RESULT_DIR",
    os.path.join(SCRIPT_DIR, "results", model_name)
)
os.makedirs(result_dir, exist_ok=True)

# ⭐️ Use the same processor saved with the fine-tuned model for preprocessing.
sample_util.processor = AutoProcessor.from_pretrained(model_dir)

# TODO Complete the following parts:
# ⭐️ TODO completed: load both official evaluation sets.
test_clean_dataset = sample_util.make_dataset(test_clean_top_dir)
test_other_dataset = sample_util.make_dataset(test_other_top_dir)

# ⭐️ TODO completed: create an ASR pipeline from the fine-tuned checkpoint.
transcriber = pipeline(
    "automatic-speech-recognition",
    model=model_dir,
    tokenizer=model_dir,
    feature_extractor=model_dir,
    device=0 if torch.cuda.is_available() else -1
)
# End of TODO

# Function to write REF/HYP pairs to a file
def write_results(dataset, transcriber, output_file, split_name):
    print(f"Writing {split_name} predictions to: {output_file}")
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, data in enumerate(dataset, start=1):
            count = idx
            # ⭐️ REF should be the original transcript text, not token IDs.
            ref = data["text"]
            # TODO complete the following part
            # ⭐️ TODO completed: pass raw waveform and sampling rate to the ASR pipeline.
            hyp = transcriber({
                "array": data["audio"],
                "sampling_rate": data["sampling_rate"]
            })["text"]
            # End of TODO
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n\n")  # double newline for readability
            if idx % 100 == 0:
                print(f"  processed {idx} utterances for {split_name}")
    print(f"Finished {split_name}: {count} utterances")

# Write test_clean_dataset
print(f"Using model_dir: {model_dir}")
print(f"Using result_dir: {result_dir}")
test_clean_result_path = os.path.join(result_dir, "test_clean_result.txt")
write_results(
    test_clean_dataset,
    transcriber,
    test_clean_result_path,
    "test-clean"
)
evaluate_file(test_clean_result_path)

# Write test_other_dataset
test_other_result_path = os.path.join(result_dir, "test_other_result.txt")
write_results(
    test_other_dataset,
    transcriber,
    test_other_result_path,
    "test-other"
)
evaluate_file(test_other_result_path)
