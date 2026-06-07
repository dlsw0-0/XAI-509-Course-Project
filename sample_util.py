# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard library imports
import glob
import io
import os
from typing import Dict, Optional

# Third-party imports
# ⭐️ Added for decoding FLAC bytes from the WebDataset shards and keeping
# waveform arrays available for inference.
import numpy as np
import soundfile as sf
import torch
import torchaudio
import webdataset as wds
from transformers import AutoProcessor

# ⭐️ The training/inference scripts set this explicitly after loading the
# correct processor. Keep it lazy so importing this file does not contact HF Hub.
processor: Optional[AutoProcessor] = None


def get_processor() -> AutoProcessor:
    """Return the active Wav2Vec2 processor, loading the base one if needed."""
    global processor
    if processor is None:
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    return processor

def preprocess_sample(sample: Dict) -> Dict:
    """Preprocess a single raw sample from the WebDataset.

    This function loads the waveform from the raw bytes using soundfile,
    extracts features using the processor's feature extractor, and tokenizes
    the transcript text.

    Args:
        sample (Dict): A dictionary containing keys 'audio' (raw audio bytes)
            and 'text' (transcript bytes).

    Returns:
        Dict: A dictionary with keys:
            - 'input_values': processed audio feature tensor.
            - 'labels': list of token IDs corresponding to the transcript.
    """
    # ⭐️ TODO completed: read the `.audio` bytes and `.text` transcript from
    # the provided WebDataset format.
    audio_bytes = sample["audio"]
    transcript = sample["text"]

    if isinstance(transcript, bytes):
        transcript = transcript.decode("utf-8")
    transcript = transcript.strip().upper()

    # ⭐️ The shard stores FLAC bytes under `.audio`, so decode with soundfile
    # instead of relying on WebDataset's automatic audio decoder.
    waveform, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # ⭐️ Wav2Vec2 expects 16 kHz audio. Resample only when needed.
    active_processor = get_processor()
    target_sampling_rate = active_processor.feature_extractor.sampling_rate
    if sampling_rate != target_sampling_rate:
        waveform_tensor = torch.from_numpy(waveform)
        waveform = torchaudio.functional.resample(
            waveform_tensor,
            sampling_rate,
            target_sampling_rate
        ).numpy()

    # ⭐️ Create model inputs and CTC labels for fine-tuning.
    input_values = active_processor(
        waveform,
        sampling_rate=target_sampling_rate
    ).input_values[0]
    labels = active_processor.tokenizer(transcript).input_ids

    # ⭐️ Keep text/audio/sampling_rate too, because inference needs REF text
    # and raw waveform input for the ASR pipeline.
    return {
        "input_values": input_values,
        "labels": labels,
        "text": transcript,
        "audio": np.asarray(waveform, dtype=np.float32),
        "sampling_rate": target_sampling_rate,
        "meta": sample.get("meta"),
    }


def make_dataset(data_dir: str) -> wds.WebDataset:
    """Create a WebDataset pipeline that loads and preprocesses data shards.

    It reads all shards named 'shard-*.tar' in the given directory,
    extracts 'wav' and 'txt' entries as tuples, converts them into dictionaries,
    and applies the preprocessing function.

    Args:
        data_dir (str): Path to the directory containing dataset shards.

    Returns:
        wds.WebDataset: The prepared dataset pipeline with preprocessing.
    """
    # ⭐️ Make shard order deterministic and fail clearly if the dataset path
    # is wrong.
    shards = sorted(glob.glob(os.path.join(data_dir, "shard-*.tar")))
    if not shards:
        raise FileNotFoundError(f"No shard-*.tar files found in {data_dir}")

    dataset = (
        # ⭐️ Removed `.decode(wds.torch_audio)` because these files use the
        # `.audio` extension and are decoded explicitly in preprocess_sample.
        wds.WebDataset(shards, shardshuffle=False)
            .to_tuple("audio", "text", "meta")
            .map(lambda sample: {"audio": sample[0], "text": sample[1], "meta": sample[2]})
            .map(preprocess_sample)
    )
    return dataset
