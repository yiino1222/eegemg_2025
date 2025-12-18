from pathlib import Path
import argparse
import json
import sys

# Ensure repository modules are importable when running in Docker
sys.path.append('/p-antipsychotics-sleep')

import analysis as ana


def merge_and_plot(analyzed_dir_list, rename_dict, exclude_mouse_list, target_group, output_dir, epoch_len_sec, sample_freq):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_result = ana.merge_n_plot(
        analyzed_dir_list,
        epoch_len_sec,
        sample_freq,
        exclude_mouse_list,
        target_group,
        str(output_dir),
        group_rename_dic=rename_dict,
    )

    stage_df = (output_dir / "meta_stage_n_bout_df_after.csv")
    psd_df = (output_dir / "merge_norm_psd_ts_df_after.csv")
    bout_df = (output_dir / "meta_stage_n_bout_df_after.csv")

    if stage_df.exists() and psd_df.exists() and bout_df.exists():
        import pandas as pd

        stage_df = pd.read_csv(stage_df)
        psd_df = pd.read_csv(psd_df)
        bout_df = pd.read_csv(bout_df)

        for stage in ("NREM", "Wake", "REM"):
            ana.wilcoxon_n_paried_t(stage_df, psd_df, bout_df, target_group, stage)

    return merge_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge analyzed directories and plot summary figures")
    parser.add_argument("--analyzed-dir-list", nargs="*", required=True, help="List of analyzed directories to merge")
    parser.add_argument("--rename-dict", type=str, default="{}", help="Optional group rename mapping as JSON string")
    parser.add_argument("--exclude-mouse-list", nargs="*", default=[], help="Mouse IDs to exclude from merging")
    parser.add_argument("--target-group", default="WT", help="Target group name used for statistical tests")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store merged outputs")
    parser.add_argument("--epoch-len-sec", type=int, default=8)
    parser.add_argument("--sample-freq", type=int, default=128)

    args = parser.parse_args()
    rename_dict = json.loads(args.rename_dict)

    merge_and_plot(
        args.analyzed_dir_list,
        rename_dict,
        args.exclude_mouse_list,
        args.target_group,
        args.output_dir,
        args.epoch_len_sec,
        args.sample_freq,
    )


if __name__ == "__main__":
    main()
