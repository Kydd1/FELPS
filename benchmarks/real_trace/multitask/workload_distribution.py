"""Plot per-model input/output token length distributions from a trace JSON.

This script expects a JSON file shaped like the provided real-trace datasets,
where each element is a dict containing a ``model`` field and a ``conversation``
list of chat turns. It tokenizes each message with a LLaMA tokenizer, sums the
token counts for non-assistant turns (inputs) and assistant turns (outputs),
and draws histograms for every model.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot input/output token length distributions per model",
	)
	parser.add_argument(
		"--json_path",
		type=Path,
		help="Path to the trace JSON file ",
	)
	parser.add_argument(
		"--tokenizer",
		default="huggyllama/llama-7b",
		help=(
			"Hugging Face model id or local path for the LLaMA tokenizer. "
			"Set to a local path if you do not have network access."
		),
	)
	parser.add_argument(
		"--bins",
		type=int,
		default=60,
		help="Number of histogram bins",
	)
	parser.add_argument(
		"--max-records",
		type=int,
		default=None,
		help="Optional cap on number of records to process (for quick previews)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("workload_distribution.png"),
		help="Output path for the saved figure",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Show the plot interactively as well as saving it",
	)
	return parser.parse_args()


def load_records(path: Path, max_records) -> Iterable[dict]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if max_records:
		data = data[:max_records]
	return data


def count_tokens(tokenizer, text: str) -> int:
	# Avoid adding BOS/EOS so we only measure content length.
	return len(tokenizer.encode(text, add_special_tokens=False))


def collect_lengths(records: Iterable[dict], tokenizer) -> Dict[str, Dict[str, List[int]]]:
	per_model: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"inputs": [], "outputs": []})

	for rec in records:
		model = rec.get("model", "unknown")
		conversation = rec.get("conversation", []) or []

		input_len = 0
		output_len = 0
		for msg in conversation:
			content = msg.get("content") or ""
			role = msg.get("role", "")
			token_count = count_tokens(tokenizer, str(content))
			if role == "assistant":
				output_len += token_count
			else:
				input_len += token_count

		per_model[model]["inputs"].append(input_len)
		per_model[model]["outputs"].append(output_len)

	return per_model


def plot_distributions(per_model: Dict[str, Dict[str, List[int]]], bins: int, output_path: Path, show: bool) -> None:
	models = sorted(per_model.keys())
	if not models:
		raise ValueError("No models found in the provided data")

	fig, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 8), sharex="col")
	if len(models) == 1:
		# Normalize axes indexing when only one model is present.
		axes = [[axes[0]], [axes[1]]]

	for idx, model in enumerate(models):
		inputs = per_model[model]["inputs"]
		outputs = per_model[model]["outputs"]
		print(f"Model {model}: avg input tokens {sum(inputs) / len(inputs):.1f}, "
              f"avg output tokens {sum(outputs) / len(outputs):.1f}")

		ax_in = axes[0][idx]
		ax_out = axes[1][idx]

		ax_in.hist(inputs, bins=bins, color="#3b82f6", alpha=0.8)
		ax_in.set_title(f"{model} input")
		ax_in.set_ylabel("Count")

		ax_out.hist(outputs, bins=bins, color="#f97316", alpha=0.8)
		ax_out.set_title(f"{model} output")
		ax_out.set_xlabel("Tokens")
		ax_out.set_ylabel("Count")

	fig.tight_layout()
	fig.savefig(output_path, dpi=200)
	if show:
		plt.show()
	plt.close(fig)


def main() -> None:
	args = parse_args()
	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
	records = load_records(args.json_path, args.max_records)
	per_model = collect_lengths(records, tokenizer)
	plot_distributions(per_model, args.bins, args.output, args.show)
	print(f"Saved plot to {args.output}")


if __name__ == "__main__":
	main()
