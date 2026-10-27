#!/usr/bin/env python3
"""
Summarize per-participant test metrics, with flexible plotting options to
color-code participants by any specified categorical variable from questionnaire data.
"""
import argparse
import glob
import os
import pickle
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None

# --- Configuration ---
FILE_PATTERN = r"results_p([A-Za-z0-9]+)\.pkl"
METRICS_TO_ANALYZE = ['test_accuracy', 'test_balanced_accuracy', 'test_macro_f1']
PRIMARY_METRIC = 'test_balanced_accuracy'

METRIC_DISPLAY_NAMES = {
    "test_accuracy": "Accuracy",
    "test_balanced_accuracy": "Balanced Accuracy",
    "test_macro_f1": "Macro F1-Score"
}

# Functions for loading and summarizing results
def load_results(results_dir: str) -> Dict[str, Dict[str, List[float]]]:
    
    per_participant: Dict[str, Dict[str, List[float]]] = {}
    for path in glob.glob(os.path.join(results_dir, "results_p*.pkl")):
        fname = os.path.basename(path)
        m = re.match(FILE_PATTERN, fname)
        if not m: continue
        pid = f"p{m.group(1)}"
        per_participant[pid] = {key: [] for key in METRICS_TO_ANALYZE}
        try:
            with open(path, "rb") as f: data = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            continue
        for k, result in data.items():
            if not isinstance(result, dict): continue
            for metric_key in METRICS_TO_ANALYZE:
                if metric_key in result:
                    per_participant[pid][metric_key].append(result[metric_key])
    return per_participant

def summarize(per_participant: Dict[str, Dict[str, List[float]]], chance: float, tol: float) -> pd.DataFrame:
    
    rows = []
    for pid, all_metric_data in per_participant.items():
        primary_metric_values = all_metric_data.get(PRIMARY_METRIC, [])
        n_runs = len(primary_metric_values)
        if n_runs == 0: continue
        row = {"participant": pid, "n_runs": n_runs}
        for metric_key, values in all_metric_data.items():
            if not values:
                row[f'mean_{metric_key}'] = np.nan
                row[f'std_{metric_key}'] = np.nan
                continue
            arr = np.array(values, dtype=float)
            row[f'mean_{metric_key}'] = arr.mean()
            row[f'std_{metric_key}'] = arr.std(ddof=1) if n_runs > 1 else 0.0
        mean_primary = row[f'mean_{PRIMARY_METRIC}']
        near = abs(mean_primary - chance) <= tol
        row["near_chance"] = bool(near)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty: return df
    return df.sort_values(["participant"])

def load_and_process_questionnaires(path: str) -> Optional[pd.DataFrame]:
   
    if not path or not os.path.exists(path):
        print(f"[WARN] Questionnaire file not found at: '{path}'. Skipping.")
        return None
    
    print(f"[INFO] Loading questionnaire data from: {path}")
    df = pd.read_excel(path)
    df.rename(columns={'ID': 'participant'}, inplace=True)

    # Create the 'Group' column for default coloring
    if all(col in df.columns for col in ['Level', 'EDEQ_Level']):
        df['IPAQ_binary'] = df['Level'].apply(lambda x: 1 if x == 'ALTO' else 0)
        conditions = [
            (df['EDEQ_Level'] == 0) & (df['IPAQ_binary'] == 0),
            (df['EDEQ_Level'] == 1) & (df['IPAQ_binary'] == 0),
            (df['EDEQ_Level'] == 0) & (df['IPAQ_binary'] == 1),
            (df['EDEQ_Level'] == 1) & (df['IPAQ_binary'] == 1)
        ]
        group_values = [1, 2, 3, 4]
        df['Group'] = np.select(conditions, group_values, default=0)
    
    return df

# Plotting function
def make_bar_figure(df: pd.DataFrame, out_path: str, chance: float, metric_key: str, task_name: str, color_by: str):
    if plt is None:
        print("[WARN] Matplotlib not available; skipping bar figure.")
        return

    mean_col, std_col = f'mean_{metric_key}', f'std_{metric_key}'
    df_plot = df.sort_values(mean_col, ascending=True).reset_index(drop=True)

    legend_patches = []
    title_suffix = ""

    # Generic coloring logic based on the 'color_by' argument ---
    can_color_by_var = color_by in df_plot.columns and df_plot[color_by].notna().any()
    
    if can_color_by_var:
        unique_values = sorted(df_plot[color_by].dropna().unique())
        # Use a standard color cycle
        color_cycle = plt.get_cmap('tab10').colors
        color_map = {val: color_cycle[i % len(color_cycle)] for i, val in enumerate(unique_values)}
        
        colors = df_plot[color_by].map(color_map).fillna('lightgray').to_list()
        for value, color in color_map.items():
            legend_patches.append(mpatches.Patch(color=color, label=f"{color_by}: {value}"))
        title_suffix = f"(colored by {color_by})"
    else:
        # Fallback to near-chance coloring
        is_near = abs(df_plot[mean_col] - chance) <= 0.02
        colors = ["tab:gray" if near else "tab:blue" for near in is_near]
        title_suffix = "(near-chance in gray)"

    fig_w, fig_h = max(5.0, 0.20 * len(df_plot)), 6.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.bar(x=np.arange(len(df_plot)), height=df_plot[mean_col].values, yerr=df_plot[std_col].values,
           align="center", ecolor="black", capsize=2, color=colors, alpha=0.9)
    ax.axhline(chance, linestyle="--", linewidth=1.5, color="black", alpha=0.7, label=f"chance={chance:.2f}")
    
    ax.set_xticks(np.arange(len(df_plot)))
    ax.set_xticklabels(df_plot["participant"].tolist(), rotation=90, fontsize=8)
    
    display_name = METRIC_DISPLAY_NAMES.get(metric_key, metric_key)
    ax.set_ylabel(f"{display_name} (proportion)")
    ax.set_ylim(0.0, 1.0)
    
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right", title=color_by.replace('_', ' ').title())
    else:
        ax.legend(loc="lower right")

    ax.set_title(f"Per-participant mean {display_name} for {task_name.capitalize()} {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved bar figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize per-participant metrics and generate plots.")
    
    parser.add_argument("--task", type=str, default="condition", 
                        choices=['happiness', 'weight', 'condition'], 
                        help="The task to analyze. Default: 'happiness'")
                        
    parser.add_argument("--questionnaires", type=str, 
                        default="../dataset/allLabels/demographics_v2.xlsx",
                        help="Path to Excel file with questionnaire data.")
    
    # Argument to control excel colum to color from
    parser.add_argument("--color-by", type=str, default="GenderIdentity_Ideal",
                        help="Column name from the questionnaire file to color bars by. Default: 'Group'")
                        
    parser.add_argument("--chance", type=float, default=1.0/3.0)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--no-plots", action="store_true")
    
    args = parser.parse_args()

    # Load and merge data ---
    results_dir = f"../results_{args.task}"
    per_participant_data = load_results(results_dir)
    df_summary = summarize(per_participant_data, chance=args.chance, tol=args.tol)
    if df_summary.empty: raise SystemExit("No valid participants to summarize.")

    if args.questionnaires:
        questionnaires_df = load_and_process_questionnaires(args.questionnaires)
        if questionnaires_df is not None:
            df_summary = pd.merge(df_summary, questionnaires_df, on="participant", how="left")
            print("[INFO] Successfully merged performance and questionnaire data.")

    # Save results and generate plots ---
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f"summary_all_metrics_by_{args.color_by}.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\n[INFO] Wrote consolidated summary CSV to {out_csv}")

    if not args.no_plots:
        for metric_key in METRICS_TO_ANALYZE:
            # Dynamic PDF filename ---
            out_name = f"summary_bar_{metric_key}_by_{args.color_by}.pdf"
            bar_fig_path = os.path.join(results_dir, out_name)
            make_bar_figure(df_summary, bar_fig_path, chance=args.chance, 
                            metric_key=metric_key, task_name=args.task, color_by=args.color_by)

if __name__ == "__main__":
    main()