import argparse
from collections import OrderedDict
from pathlib import Path
import re

import torch


RENAME_RULES = (
    (re.compile(r"^module\.cost_agg\."), "module.geo_decoder."),
    (re.compile(r"^module\.cov_volume\."), "module.geo_encoder."),
    (re.compile(r"^cost_agg\."), "geo_decoder."),
    (re.compile(r"^cov_volume\."), "geo_encoder."),
)

LEGACY_BLOCK_SUFFIX_RULES = (
    ("pwconv.0.weight", "spital_agg.0.weight"),
    ("pwconv.1.weight", "spital_agg.1.weight"),
    ("pwconv.1.bias", "spital_agg.1.bias"),
    ("pwconv.1.running_mean", "spital_agg.1.running_mean"),
    ("pwconv.1.running_var", "spital_agg.1.running_var"),
    ("pwconv.1.num_batches_tracked", "spital_agg.1.num_batches_tracked"),
    ("dwconv.0.weight", "spital_agg.3.weight"),
    ("dwconv.1.weight", "spital_agg.4.weight"),
    ("dwconv.1.bias", "spital_agg.4.bias"),
    ("dwconv.1.running_mean", "spital_agg.4.running_mean"),
    ("dwconv.1.running_var", "spital_agg.4.running_var"),
    ("dwconv.1.num_batches_tracked", "spital_agg.4.num_batches_tracked"),
    ("pwliner.0.weight", "disparity_agg.0.weight"),
    ("pwliner.1.weight", "disparity_agg.1.weight"),
    ("pwliner.1.bias", "disparity_agg.1.bias"),
    ("pwliner.1.running_mean", "disparity_agg.1.running_mean"),
    ("pwliner.1.running_var", "disparity_agg.1.running_var"),
    ("pwliner.1.num_batches_tracked", "disparity_agg.1.num_batches_tracked"),
)


def rename_key(key):
    for pattern, replacement in RENAME_RULES:
        if pattern.match(key):
            key = pattern.sub(replacement, key, count=1)
            break

    for old_suffix, new_suffix in LEGACY_BLOCK_SUFFIX_RULES:
        if key.endswith(old_suffix):
            return key[:-len(old_suffix)] + new_suffix

    return key


def count_keys(keys, needle):
    return sum(1 for key in keys if needle in key)


def rename_model_state(model_state):
    renamed_state = OrderedDict()
    renamed_pairs = []

    for key, value in model_state.items():
        new_key = rename_key(key)
        if new_key in renamed_state:
            raise ValueError(f"Key collision after rename: {key} -> {new_key}")
        renamed_state[new_key] = value
        if new_key != key:
            renamed_pairs.append((key, new_key))

    original_keys = list(model_state.keys())
    renamed_keys = list(renamed_state.keys())

    stats = {
        "orig_cost_agg": count_keys(original_keys, "cost_agg"),
        "orig_cov_volume": count_keys(original_keys, "cov_volume"),
        "orig_geo_decoder": count_keys(original_keys, "geo_decoder"),
        "orig_geo_encoder": count_keys(original_keys, "geo_encoder"),
        "orig_pwconv": count_keys(original_keys, "pwconv"),
        "orig_dwconv": count_keys(original_keys, "dwconv"),
        "orig_pwliner": count_keys(original_keys, "pwliner"),
        "orig_spital_agg": count_keys(original_keys, "spital_agg"),
        "orig_disparity_agg": count_keys(original_keys, "disparity_agg"),
        "new_geo_decoder": count_keys(renamed_keys, "geo_decoder"),
        "new_geo_encoder": count_keys(renamed_keys, "geo_encoder"),
        "new_spital_agg": count_keys(renamed_keys, "spital_agg"),
        "new_disparity_agg": count_keys(renamed_keys, "disparity_agg"),
        "renamed_total": len(renamed_pairs),
    }

    if len(original_keys) != len(renamed_keys):
        raise ValueError("model_state key count changed after rename")
    if stats["orig_cost_agg"] + stats["orig_geo_decoder"] != stats["new_geo_decoder"]:
        raise ValueError("cost_agg -> geo_decoder rename count mismatch")
    if stats["orig_cov_volume"] + stats["orig_geo_encoder"] != stats["new_geo_encoder"]:
        raise ValueError("cov_volume -> geo_encoder rename count mismatch")
    if count_keys(renamed_keys, "cost_agg") != 0:
        raise ValueError("Found leftover cost_agg keys after rename")
    if count_keys(renamed_keys, "cov_volume") != 0:
        raise ValueError("Found leftover cov_volume keys after rename")
    if count_keys(renamed_keys, "pwconv") != 0:
        raise ValueError("Found leftover pwconv keys after rename")
    if count_keys(renamed_keys, "dwconv") != 0:
        raise ValueError("Found leftover dwconv keys after rename")
    if count_keys(renamed_keys, "pwliner") != 0:
        raise ValueError("Found leftover pwliner keys after rename")
    if stats["orig_pwconv"] + stats["orig_dwconv"] + stats["orig_spital_agg"] != stats["new_spital_agg"]:
        raise ValueError("pwconv/dwconv -> spital_agg rename count mismatch")
    if stats["orig_pwliner"] + stats["orig_disparity_agg"] != stats["new_disparity_agg"]:
        raise ValueError("pwliner -> disparity_agg rename count mismatch")

    return renamed_state, stats


def build_output_path(input_path, suffix):
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def process_checkpoint(input_path, suffix):
    checkpoint = torch.load(input_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"{input_path} does not contain a model_state checkpoint")

    renamed_model_state, stats = rename_model_state(checkpoint["model_state"])
    output_path = build_output_path(input_path, suffix)

    updated_checkpoint = dict(checkpoint)
    updated_checkpoint["model_state"] = renamed_model_state
    torch.save(updated_checkpoint, output_path)

    reloaded = torch.load(output_path, map_location="cpu")
    reloaded_keys = list(reloaded["model_state"].keys())
    if len(reloaded_keys) != len(renamed_model_state):
        raise ValueError(f"{output_path} failed post-save key count validation")
    for legacy_name in ("cost_agg", "cov_volume", "pwconv", "dwconv", "pwliner"):
        if count_keys(reloaded_keys, legacy_name) != 0:
            raise ValueError(f"{output_path} still contains old key prefix {legacy_name}")

    return output_path, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename DBStereo checkpoint keys from cost_agg/cov_volume to geo_decoder/geo_encoder."
    )
    parser.add_argument(
        "--input-dir",
        default="ckeckpoint",
        help="Directory containing checkpoint .pth files.",
    )
    parser.add_argument(
        "--suffix",
        default="_v2",
        help="Suffix appended to the output checkpoint filename stem.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pth",
        help="Glob pattern used to select input checkpoint files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_paths = sorted(
        path for path in input_dir.glob(args.pattern)
        if path.is_file() and not path.stem.endswith(args.suffix)
    )
    if not input_paths:
        raise FileNotFoundError(f"No checkpoint files matched {args.pattern} under {input_dir}")

    for input_path in input_paths:
        output_path, stats = process_checkpoint(input_path, args.suffix)
        print(
            f"{input_path.name} -> {output_path.name} | "
            f"renamed={stats['renamed_total']} "
            f"geo_encoder={stats['new_geo_encoder']} "
            f"geo_decoder={stats['new_geo_decoder']} "
            f"spital_agg={stats['new_spital_agg']} "
            f"disparity_agg={stats['new_disparity_agg']}"
        )


if __name__ == "__main__":
    main()
