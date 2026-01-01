import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline_step1_preprocess import preprocess_project
from pipeline_step2_analyze import analyze_project
from pipeline_step3_merge import merge_and_plot

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("/data/config.json")


def resolve_config_path(config_path: Path) -> Path:
    if config_path.is_dir():
        config_path = config_path / "config.json"
    if config_path.exists():
        return config_path
    if config_path != DEFAULT_CONFIG_PATH and DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    raise FileNotFoundError(
        "Config file was not found. "
        f"Tried: {config_path} and {DEFAULT_CONFIG_PATH}"
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    resolved_path = resolve_config_path(config_path)
    with resolved_path.open() as f:
        return json.load(f)


def ensure_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    preprocess = {
        "prj_dir": "/p-antipsychotics-sleep/raw_data/kaist",
        "result_dir_name": "result",
        "epoch_len_sec": 8,
        "sample_freq": 128,
        "overwrite": False,
        "offset_in_msec": 0,
        **config.get("preprocess", {}),
    }

    analysis = {
        "prj_dir": preprocess["prj_dir"],
        "output_dir_name": "analyzed",
        "faster_dir_list": None,
        "epoch_len_sec": preprocess["epoch_len_sec"],
        "result_dir_name": preprocess["result_dir_name"],
        **config.get("analysis", {}),
    }

    merge = {
        "analyzed_dir_list": [],
        "rename_dict": {},
        "exclude_mouse_list": [],
        "target_group": "WT",
        "comparison_mode": "drug",
        "comparison_drug": "vehicle",
        "mouse_groups_to_compare": [],
        "output_dir": "/p-antipsychotics-sleep/figure/output",
        "epoch_len_sec": preprocess["epoch_len_sec"],
        "sample_freq": preprocess["sample_freq"],
        "quant_time_windows": {},
        **config.get("merge", {}),
    }

    return {"preprocess": preprocess, "analysis": analysis, "merge": merge}


def run_pipeline(config_path: Path, executed_dir: Optional[Path] = None) -> None:
    config = ensure_defaults(load_config(config_path))
    LOGGER.info("Starting preprocessing step")
    preprocess_project(**config["preprocess"])

    LOGGER.info("Starting analysis step")
    analyze_project(**config["analysis"])

    LOGGER.info("Starting merge and plot step")
    merge_and_plot(**config["merge"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Run the EEG/EMG notebook pipeline in batch mode.")
    parser.add_argument(
        "--config",
        type=Path,

        default=DEFAULT_CONFIG_PATH,

        help=(
            "Path to JSON configuration file describing inputs and outputs "
            "(default: /data/config.json)."
        ),
    )
    parser.add_argument(
        "--executed-dir",
        type=Path,
        default=None,
        help="Where to store executed notebook copies (for debugging).",
    )
    args = parser.parse_args()
    run_pipeline(args.config, args.executed_dir)
