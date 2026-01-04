from pathlib import Path
import shutil
import argparse
import json

import analysis as ana


def merge_and_plot(
    analyzed_dir_list,
    rename_dict,
    exclude_mouse_list,
    target_group,
    output_dir,
    epoch_len_sec,
    sample_freq,
    comparison_mode="drug",
    comparison_drug="vehicle",
    mouse_groups_to_compare=None,
    quant_time_windows=None,
    config_path=None,
):
    output_dir = Path(output_dir)
    if comparison_mode == "mouse_group" and mouse_groups_to_compare:
        compare_label = "_vs_".join(mouse_groups_to_compare)
        output_dir = output_dir / compare_label
    else:
        output_dir = output_dir / target_group
    output_dir.mkdir(parents=True, exist_ok=True)
    if config_path:
        try:
            shutil.copy2(config_path, output_dir / "config.json")
        except FileNotFoundError:
            print(f"[WARN] Config file not found for copy: {config_path}")

    merge_result = ana.merge_n_plot(
        analyzed_dir_list,
        epoch_len_sec,
        sample_freq,
        exclude_mouse_list,
        target_group,
        str(output_dir),
        group_rename_dic=rename_dict,
        comparison_mode=comparison_mode,
        comparison_drug=comparison_drug,
        mouse_groups_to_compare=mouse_groups_to_compare,
        quant_time_windows=quant_time_windows,
    )

    stage_df = (output_dir / "meta_stage_n_bout_df_after.csv")
    psd_df = (output_dir / "merge_norm_psd_ts_df_after.csv")
    bout_df = (output_dir / "meta_stage_n_bout_df_after.csv")

    if comparison_mode == "drug" and stage_df.exists() and psd_df.exists() and bout_df.exists():
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
    parser.add_argument(
        "--comparison-mode",
        choices=["drug", "mouse_group"],
        default="drug",
        help="drug: compare drugs within a mouse group (default). mouse_group: compare mouse groups within a single drug.",
    )
    parser.add_argument(
        "--comparison-drug",
        default="vehicle",
        help="Drug name to use when comparison-mode is mouse_group",
    )
    parser.add_argument(
        "--mouse-groups-to-compare",
        nargs="*",
        default=None,
        help="Mouse groups to compare when comparison-mode is mouse_group (defaults to all groups found).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store merged outputs")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional config.json to copy into output dir")
    parser.add_argument("--epoch-len-sec", type=int, default=8)
    parser.add_argument("--sample-freq", type=int, default=128)
    parser.add_argument(
        "--quant-time-windows",
        type=str,
        default="{}",
        help="JSON mapping for quantification windows, e.g. "
        '{"stage_after":[6,7],"psd_after":[6,7],"psd_before":[5,5],"stage_before":[3,5]}',
    )

    args = parser.parse_args()
    rename_dict = json.loads(args.rename_dict)
    quant_time_windows = json.loads(args.quant_time_windows)

    merge_and_plot(
        args.analyzed_dir_list,
        rename_dict,
        args.exclude_mouse_list,
        args.target_group,
        args.output_dir,
        args.epoch_len_sec,
        args.sample_freq,
        args.comparison_mode,
        args.comparison_drug,
        args.mouse_groups_to_compare,
        quant_time_windows,
        args.config_path,
    )


if __name__ == "__main__":
    main()
