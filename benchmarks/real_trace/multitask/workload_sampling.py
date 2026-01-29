"""Sample workload conversations from multiple JSON files.

This script reads every ``*.json`` under a given directory, samples up to
``sample_size`` entries per file, and combines them into a single dataset with
the following shape per entry::

	{
		"question_id": idx or generated id,
		"model": "<source>-<n>" cycling over the configured suffix count,
		"conversation": [
			{"content": prompt, "role": "user"},
			{"content": response, "role": "assistant"}
		],
		"tstamp": poisson-distributed timestamp,
		"turn": 1
	}

Output is JSONL if the output path ends with ``.jsonl``; otherwise the script
writes a JSON array to the output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _load_records(path: Path) -> List[Dict[str, Any]]:
	"""Load records from a JSON or JSONL file."""

	raw_text = path.read_text(encoding="utf-8")

	try:
		data = json.loads(raw_text)
		if isinstance(data, list):
			return data
	except Exception:
		# Fall back to JSONL parsing below.
		pass

	records: List[Dict[str, Any]] = []
	for line in raw_text.splitlines():
		if not line.strip():
			continue
		try:
			records.append(json.loads(line))
		except json.JSONDecodeError as exc:  # pragma: no cover - debug helper
			raise ValueError(f"Failed to parse line in {path}: {line}") from exc

	return records


def _sample_records(records: List[Dict[str, Any]], sample_size: int, rng: random.Random) -> List[Dict[str, Any]]:
	if len(records) <= sample_size:
		return list(records)
	return rng.sample(records, sample_size)


def _question_id(entry: Dict[str, Any], rng: random.Random) -> Any:
	if "idx" in entry:
		return entry["idx"]
	# Generate a reproducible random 10-digit id.
	return rng.randint(1_000_000_000, 9_999_999_999)

def _lam_for_source(source_name: str, base_lam: float) -> float:
	"""Create a deterministic lam per source file so tasks vary in arrival rate."""

	scale = 0.5 + random.random()
	return base_lam * scale


def generate_random_timestamp(current_time: int, lam: float, rng: np.random.Generator) -> int:
	"""Generate a random timestamp within the last year using a Poisson distribution."""

	one_year_ago = current_time - 24 * 60 * 60
	interval = int(rng.poisson(lam=lam))
	return max(one_year_ago, current_time - interval)

def generate_random_uniform_timestamp(current_time: int) -> int:
	"""Generate a random timestamp within the last year using a Poisson distribution."""

	interval = int(random.uniform(0, 24*60*60))
	return  current_time - interval


def _build_items(
	samples: Iterable[Dict[str, Any]],
	model_prefix: str,
	base_time: int,
	lam: float,
	time_rng: np.random.Generator,
	id_rng: random.Random,
	num_suffixes: int,
) -> List[Dict[str, Any]]:
	items: List[Dict[str, Any]] = []

	for i, entry in enumerate(samples):
		prompt = entry.get("prompt")
		response = entry.get("response")
		prompt_len = entry.get("prompt_len")
		response_len = entry.get("response_len")
		if prompt is None or response is None:
			continue

		# current_ts = generate_random_timestamp(base_time, lam, time_rng)
		current_ts = generate_random_uniform_timestamp(base_time)

		suffix_idx = i % num_suffixes + 1
		model_suffix = f"-{suffix_idx}"

		items.append(
			{
				"question_id": _question_id(entry, id_rng),
				"model": f"{model_prefix}{model_suffix}",
				"conversation": [
					{"content": prompt, "role": "user"},
					{"content": response, "role": "assistant"},
				],
				"prompt_len": prompt_len,
				"response_len": response_len,
				"tstamp": current_ts,
				"turn": 1,
			}
		)

	return items


def _write_output(path: Path, items: List[Dict[str, Any]]) -> None:
	if path.suffix.lower() == ".jsonl":
		path.write_text("\n".join(json.dumps(obj, ensure_ascii=False) for obj in items) + "\n", encoding="utf-8")
	else:
		path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Sample workload conversations from JSON/JSONL files.")
	parser.add_argument("--input-dir", type=Path, help="Directory containing *.json files to sample.")
	parser.add_argument("--output", type=Path, help="Output path (.jsonl for JSONL, otherwise JSON array).")
	parser.add_argument("--sample-size", type=int, default=5000, help="Samples per source file (default: 2000).")
	parser.add_argument(
		"--model-count",
		type=int,
		default=2,
		help="Number of suffix variants to cycle per model (default: 2).",
	)
	parser.add_argument(
		"--lam",
		type=float,
		default=12*60*60,
		help="Base lambda (seconds) for Poisson timestamp increments; each source scales this deterministically (default: 43200).",
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")

	args = parser.parse_args(argv)

	if args.model_count < 1:
		raise SystemExit("--model-suffix-count must be at least 1")

	rng = random.Random(args.seed)
	time_rng = np.random.default_rng(args.seed)

	input_dir: Path = args.input_dir
	if not input_dir.is_dir():
		raise SystemExit(f"Input directory does not exist: {input_dir}")

	json_files = [p for p in sorted(input_dir.glob("*.json")) if "multitask" not in p.name]
	if not json_files:
		raise SystemExit(f"No *.json files found in {input_dir}")

	base_time = int(time.time())
	combined: List[Dict[str, Any]] = []

	for path in json_files:
		records = _load_records(path)
		sampled = _sample_records(records, args.sample_size, rng)
		model_prefix = path.stem
		lam_for_source = _lam_for_source(model_prefix, args.lam)
		combined.extend(
			_build_items(
				sampled,
				model_prefix,
				base_time,
				lam_for_source,
				time_rng,
				rng,
				args.model_count,
			)
		)

	_write_output(args.output, combined)
	print(f"Wrote {len(combined)} records to {args.output}")
	return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
	sys.exit(main())
