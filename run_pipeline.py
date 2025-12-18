import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import papermill as pm

LOGGER = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open() as f:
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
        "output_dir": "/p-antipsychotics-sleep/figure/output",
        "epoch_len_sec": preprocess["epoch_len_sec"],
        "sample_freq": preprocess["sample_freq"],
        **config.get("merge", {}),
    }

    return {"preprocess": preprocess, "analysis": analysis, "merge": merge}


def execute_notebook(notebook: Path, output: Path, parameters: Dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Running %s", notebook)
    pm.execute_notebook(
        str(notebook),
        str(output),
        parameters=parameters,
        progress_bar=False,
        report_mode=False,
    )


def run_pipeline(config_path: Path, executed_dir: Optional[Path] = None) -> None:
    config = ensure_defaults(load_config(config_path))
    executed_dir = executed_dir or Path("executed_notebooks")
    root = Path(__file__).parent

    execute_notebook(
        root / "1_preprocess_edf.ipynb",
        executed_dir / "1_preprocess_edf.executed.ipynb",
        config["preprocess"],
    )

    execute_notebook(
        root / "2_analyze_stage_n_PSD.ipynb",
        executed_dir / "2_analyze_stage_n_PSD.executed.ipynb",
        config["analysis"],
    )

    execute_notebook(
        root / "3_merge_and_plot_data.ipynb",
        executed_dir / "3_merge_and_plot_data.executed.ipynb",
        config["merge"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Run the EEG/EMG notebook pipeline in batch mode.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON configuration file describing inputs and outputs.",
    )
    parser.add_argument(
        "--executed-dir",
        type=Path,
        default=None,
        help="Where to store executed notebook copies (for debugging).",
    )
    args = parser.parse_args()
    run_pipeline(args.config, args.executed_dir)
