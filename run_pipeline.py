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
        "overwrite": preprocess["overwrite"],
        "injection_before_hours": 6,
        "injection_after_hours": 18,
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
        "drug_names": ["vehicle", "rapalog"],
        "output_dir": "/p-antipsychotics-sleep/figure/output",
        "epoch_len_sec": preprocess["epoch_len_sec"],
        "sample_freq": preprocess["sample_freq"],
        "quant_time_windows": {},
        "include_individual_plots": False,
        **config.get("merge", {}),
    }

    return {"preprocess": preprocess, "analysis": analysis, "merge": merge}


def resolve_analyzed_dir_list(config: Dict[str, Any]) -> Dict[str, Any]:
    merge_conf = config["merge"]
    if merge_conf.get("analyzed_dir_list"):
        return config

    analysis_conf = config["analysis"]
    prj_dir = Path(analysis_conf["prj_dir"])
    output_dir_name = analysis_conf["output_dir_name"]
    result_dir_name = analysis_conf["result_dir_name"]
    faster_dir_list = analysis_conf.get("faster_dir_list")

    if faster_dir_list is None:
        faster_dir_list = sorted(
            str(path)
            for path in prj_dir.rglob(result_dir_name)
            if path.is_dir()
        )

    def _output_root_for_faster_dir(faster_dir: str) -> Path:
        faster_path = Path(faster_dir)
        if faster_path.name == result_dir_name:
            faster_path = faster_path.parent
        if "raw_data" in faster_path.parts:
            raw_data_index = faster_path.parts.index("raw_data")
            base_dir = Path(*faster_path.parts[:raw_data_index]) or prj_dir.parent
            rel_parts = list(faster_path.parts[raw_data_index + 1 :])
            if rel_parts:
                last_part = rel_parts[-1]
                if last_part.startswith("raw_data"):
                    suffix = last_part[len("raw_data") :]
                    rel_parts[-1] = f"{output_dir_name}{suffix}"
            rel_path = Path(*rel_parts)
            return base_dir / output_dir_name / rel_path
        return prj_dir / output_dir_name / faster_path.name

    analyzed_dirs = sorted({str(_output_root_for_faster_dir(fd)) for fd in faster_dir_list})
    merge_conf["analyzed_dir_list"] = analyzed_dirs
    LOGGER.info(
        "merge.analyzed_dir_list was empty; auto-discovered %d analyzed directories from step2 outputs",
        len(analyzed_dirs),
    )
    return config


def run_pipeline(config_path: Path, executed_dir: Optional[Path] = None) -> None:
    resolved_path = resolve_config_path(config_path)
    config = ensure_defaults(load_config(resolved_path))
    config = resolve_analyzed_dir_list(config)
    config["merge"]["config_path"] = str(resolved_path)
    LOGGER.info("Starting preprocessing step")
    preprocess_project(**config["preprocess"])

    LOGGER.info("Starting analysis step")
    analyze_project(**config["analysis"])

    LOGGER.info("Starting merge and plot step")
    config = resolve_analyzed_dir_list(config)
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
