import argparse
import os
from typing import Dict, Optional

from jiwer import process_words, wer


def _default_output_path(file_path: str) -> str:
    root, _ = os.path.splitext(file_path)
    return f"{root}_wer.txt"


def evaluate_file(file_path: str, output_path: Optional[str] = None) -> Dict[str, float]:
    """Compute WER from a REF/HYP result file and save a summary."""
    refs = []
    hyps = []

    print(f"Loading result file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) % 2 != 0:
        raise ValueError(
            "Invalid result file format: expected alternating REF/HYP lines."
        )

    for i in range(0, len(lines), 2):
        ref_line = lines[i]
        hyp_line = lines[i + 1]

        if not ref_line.startswith("REF:") or not hyp_line.startswith("HYP:"):
            raise ValueError(
                f"Invalid REF/HYP pair near non-empty line {i + 1}: "
                f"{ref_line!r} / {hyp_line!r}"
            )

        refs.append(ref_line[len("REF:"):].strip())
        hyps.append(hyp_line[len("HYP:"):].strip())

    test_wer = wer(refs, hyps)
    details = process_words(refs, hyps)

    if output_path is None:
        output_path = _default_output_path(file_path)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    summary = {
        "num_utterances": len(refs),
        "wer": test_wer,
        "wer_percent": test_wer * 100,
        "hits": details.hits,
        "substitutions": details.substitutions,
        "deletions": details.deletions,
        "insertions": details.insertions,
    }

    summary_lines = [
        f"file: {file_path}",
        f"num_utterances: {summary['num_utterances']}",
        f"wer: {summary['wer']:.6f}",
        f"wer_percent: {summary['wer_percent']:.2f}",
        f"hits: {summary['hits']}",
        f"substitutions: {summary['substitutions']}",
        f"deletions: {summary['deletions']}",
        f"insertions: {summary['insertions']}",
    ]

    print("WER evaluation summary")
    for line in summary_lines:
        print(f"  {line}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
        f.write("\n")

    print(f"Saved WER summary to: {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compute WER from a REF/HYP text file"
    )
    parser.add_argument("file_path", type=str, help="Path to the REF/HYP results file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the WER summary. Defaults to <input>_wer.txt",
    )
    args = parser.parse_args()
    evaluate_file(args.file_path, args.output)


if __name__ == "__main__":
    main()
