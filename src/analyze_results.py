import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter

# This script assumes it is in the same directory as dataset.py to allow this import
from dataset import DatasetManager

# --- CONFIGURATION ---
TASKS = ['happiness', 'weight', 'condition']
DATASET_FOLDER = '../dataset/unified_db'
CHANCE_LEVEL = 1/3
PRIMARY_METRIC = 'balanced_accuracy' # Metric used for the 'Responders' calculation

def get_full_label_distribution(subject_id, task_label, features_idx):
    """
    Loads a subject's data to get the label distribution of their complete dataset for a given task.
    """
    dm = DatasetManager(DATASET_FOLDER, subject_id, featuresIdx=features_idx, label=task_label)
    patient = dm.patientObj
    all_labels = [walk.label for condition in patient.conditions for walk in condition.walks]
    counts = Counter(all_labels)
    total = len(all_labels)
    
    distribution = {
        0: counts.get(0, 0) / total * 100 if total > 0 else 0,
        1: counts.get(1, 0) / total * 100 if total > 0 else 0,
        2: counts.get(2, 0) / total * 100 if total > 0 else 0,
    }
    return distribution

def analyze_task(task_label, features_idx):
    """
    Analyzes all results for a single task and returns a summary DataFrame.
    """
    results_folder = f"../results_{task_label}"
    
    print(f"\n{'='*25} ANALYZING TASK: {task_label.upper()} {'='*25}")

    if not os.path.isdir(results_folder):
        print(f"Directory not found: {results_folder}. Skipping task.")
        return pd.DataFrame() # Return empty DataFrame if folder not found

    all_files = [f for f in os.listdir(results_folder) if f.startswith('results_') and f.endswith('.pkl')]
    subject_ids = sorted([f.replace('results_', '').replace('.pkl', '') for f in all_files])
    
    if not subject_ids:
        print(f"No result files found in {results_folder}.")
        return pd.DataFrame()

    print(f"Found {len(subject_ids)} result files.")
    
    subject_summary_list = []
    
    for subject_id in subject_ids:
        file_path = os.path.join(results_folder, f'results_{subject_id}.pkl')
        with open(file_path, 'rb') as f:
            results_data = pickle.load(f)

        first_seed_key = f"{subject_id}_seed1"
        if first_seed_key not in results_data:
            print(f"  - Warning: Data for seed 1 not found for subject {subject_id}. Skipping.")
            continue
            
        test_labels_sample = results_data[first_seed_key]['test_labels']
        if len(np.unique(test_labels_sample)) < 2:
            print(f"  - Subject {subject_id} excluded (non-classifiable, only 1 label class found).")
            continue

        subject_metrics = {
            'accuracy': [], 'balanced_accuracy': [],
            'macro_f1_score': [], 'majority_baseline': []
        }

        for seed in range(1, 11):
            key = f"{subject_id}_seed{seed}"
            if key in results_data:
                run_data = results_data[key]
                
                subject_metrics['accuracy'].append(run_data['test_accuracy'])
                subject_metrics['balanced_accuracy'].append(run_data['test_balanced_accuracy'])
                subject_metrics['macro_f1_score'].append(run_data['test_macro_f1'])
                
                labels = run_data['test_labels']
                if len(labels) > 0:
                    counts = Counter(labels)
                    majority_count = counts.most_common(1)[0][1]
                    baseline = majority_count / len(labels)
                    subject_metrics['majority_baseline'].append(baseline)

        avg_subject_results = {metric: np.mean(values) for metric, values in subject_metrics.items()}
        avg_subject_results['subject_id'] = subject_id
        
        class_dist = get_full_label_distribution(subject_id, task_label, features_idx)
        avg_subject_results['dist_0'] = class_dist[0]
        avg_subject_results['dist_1'] = class_dist[1]
        avg_subject_results['dist_2'] = class_dist[2]
        
        subject_summary_list.append(avg_subject_results)

    return pd.DataFrame(subject_summary_list)

def print_summary_table(df, task_label):
    """
    Calculates final statistics from the summary DataFrame and prints a formatted table.
    """
    if df.empty:
        print(f"No classifiable subjects found for task '{task_label}'. Cannot generate table.")
        return

    n_classifiable = len(df)
    
    # Infer total number of subjects from the folder, as df only contains classifiable ones.
    results_folder = f"../results_{task_label}"
    all_files = [f for f in os.listdir(results_folder) if f.startswith('results_') and f.endswith('.pkl')]
    n_total = len(all_files)
    
    dist_0_mean, dist_0_std = df['dist_0'].mean(), df['dist_0'].std()
    dist_1_mean, dist_1_std = df['dist_1'].mean(), df['dist_1'].std()
    dist_2_mean, dist_2_std = df['dist_2'].mean(), df['dist_2'].std()
    
    baseline_mean, baseline_std = df['majority_baseline'].mean() * 100, df['majority_baseline'].std() * 100
    acc_mean, acc_std = df['accuracy'].mean() * 100, df['accuracy'].std() * 100
    ba_mean, ba_std = df['balanced_accuracy'].mean() * 100, df['balanced_accuracy'].std() * 100
    f1_mean, f1_std = df['macro_f1_score'].mean() * 100, df['macro_f1_score'].std() * 100
    
    responders = (df[PRIMARY_METRIC] > CHANCE_LEVEL).sum()
    # BUG FIX: Added safeguard against division by zero
    responders_pct = (responders / n_classifiable) * 100 if n_classifiable > 0 else 0
    
    print("\n" + "="*85)
    # BUG FIX: Title is now dynamic
    print(f"FINAL RESULTS TABLE FOR TASK: '{task_label.upper()}'")
    print("="*85)
    print(f"{'Metric':<45} | {'Value'}")
    print("-"*85)
    print(f"{'Classifiable Participants (N / Total)':<45} | {n_classifiable} / {n_total}")
    print(f"{'Class Distribution [0/1/2] (%) (μ ± σ)':<45} | [{dist_0_mean:.1f}±{dist_0_std:.1f} / {dist_1_mean:.1f}±{dist_1_std:.1f} / {dist_2_mean:.1f}±{dist_2_std:.1f}]")
    print(f"{'(Context) Majority Class Baseline (Acc, %)':<45} | {baseline_mean:.1f} ± {baseline_std:.1f}")
    print(f"{'(Context) Model Accuracy (%)':<45} | {acc_mean:.1f} ± {acc_std:.1f}")
    print(f"{'(Main) Balanced Accuracy (%)':<45} | {ba_mean:.1f} ± {ba_std:.1f}")
    print(f"{'(Main) Macro F1-Score (%)':<45} | {f1_mean:.1f} ± {f1_std:.1f}")
    print(f"{'Responders > Chance (BA > 33.3%) (%)':<45} | {responders_pct:.1f}% ({responders}/{n_classifiable})")
    print("="*85)

def main():

    try:
        with open('accelerometer_columns.txt', 'r') as f:
            features_idx = [int(line.strip()) for line in f]
    except FileNotFoundError:
        print("[ERROR] 'accelerometer_columns.txt' not found. Please ensure this file is in the same directory.")
        return

    for task in TASKS:
        
        # 1. Analyze the task and get the summary DataFrame
        summary_df = analyze_task(task, features_idx)
        
        # 2. Print the final table using the DataFrame
        print_summary_table(summary_df, task)

if __name__ == "__main__":
    main()