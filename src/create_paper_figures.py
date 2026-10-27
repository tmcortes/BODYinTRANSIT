import os
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. DATA LOADER (Raw I/O)
# ==========================================
class DataLoader:
    """
    Loads raw data from disk into memory without processing.
    """
    def __init__(self, results_folders, demographics_path, labels_path):
        self.results_folders = results_folders
        self.demographics_path = demographics_path
        self.labels_path = labels_path
        
        # Data Containers
        self.demographics = None
        self.labels = None
        self.results = {} # Structure: { 'folder_name': { 'subject_seed': {data...} } }

        print("=== INITIALIZING DATA LOADER ===")
        self._load_metadata()
        self._load_raw_results()
        print("=== LOADING COMPLETE ===\n")

    def _load_metadata(self):
        try:
            if os.path.exists(self.demographics_path):
                self.demographics = pd.read_excel(self.demographics_path)
            if os.path.exists(self.labels_path):
                self.labels = pd.read_excel(self.labels_path)
        except Exception as e:
            print(f"[ERR] Metadata load failed: {e}")

    def _load_raw_results(self):
        for folder_path in self.results_folders:
            folder_name = os.path.basename(os.path.normpath(folder_path))
            self.results[folder_name] = {}
            
            pkl_files = glob.glob(os.path.join(folder_path, "results_*.pkl"))
            
            for pkl_file in pkl_files:
                try:
                    with open(pkl_file, "rb") as f:
                        self.results[folder_name].update(pickle.load(f))
                except Exception as e:
                    print(f"[ERR] Failed reading {pkl_file}: {e}")
            
            print(f"[IO] Loaded {len(self.results[folder_name])} runs from {folder_name}")


# ==========================================
# 2. BASE FIGURE (Style & Saving)
# ==========================================
class BaseFigure:
    def __init__(self, output_dir="../results/figures"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Publication Style
        sns.set_context("paper", font_scale=1.4)
        sns.set_style("white")
        plt.rcParams['pdf.fonttype'] = 42

    def save_plot(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f"[PLOT] Saved: {path}")
        plt.close(fig)


# ==========================================
# 3. CONFUSION MATRIX PLOTTER (Logic)
# ==========================================
class ConfusionMatrixPlotter(BaseFigure):
    """
    Encapsulates the configuration and logic to generate all confusion matrices.
    """
    def __init__(self, data_loader):
        super().__init__() 
        self.loader = data_loader
        
        self.experiments_map = [
            {
                'folder_key': 'results_condition',
                'title': 'Confusion Matrix: Auditory Condition',
                'filename': 'cm_condition.pdf',
                'classes': ['Low Freq', 'Control', 'High Freq']
            },
            {
                'folder_key': 'results_weight',
                'title': 'Confusion Matrix: Perceived Weight',
                'filename': 'cm_weight.pdf',
                'classes': ['Lighter', 'Same', 'Heavier']
            },
            {
                'folder_key': 'results_happiness',
                'title': 'Confusion Matrix: Emotional State',
                'filename': 'cm_happiness.pdf',
                'classes': ['Less Happy', 'Same', 'Happier']
            }
        ]

    def run(self):
        print("\n--- Generating Confusion Matrices ---")
        for exp in self.experiments_map:
            self._process_single_experiment(exp)

    def _process_single_experiment(self, exp_config):
        folder = exp_config['folder_key']
        
        if folder not in self.loader.results:
            print(f"[SKIP] Folder '{folder}' not found.")
            return

        y_true = []
        y_pred = []
        
        runs = self.loader.results[folder]
        skipped_subjects = 0
        included_subjects = 0

        # Iterate over each subject/seed individually
        for run_id, content in runs.items():
            if 'test_labels' in content and 'test_predictions' in content:
                labels = np.array(content['test_labels'])
                preds = np.array(content['test_predictions'])
                
                # --- FILTERING LOGIC ---
                # Check if the ground truth has at least 2 different classes.
                # If a subject only has 1 class (e.g., all 0s), they are excluded
                # to match the logic used in the results table.
                if len(np.unique(labels)) < 2:
                    skipped_subjects += 1
                    # Optional: Print detailed log if needed
                    # print(f"[FILTER] Skipping {run_id} in {folder}: Single class in ground truth.")
                    continue
                
                # If valid, add to the pool
                y_true.extend(labels)
                y_pred.extend(preds)
                included_subjects += 1
        
        print(f"   -> {exp_config['title']}: Included {included_subjects} runs, Skipped {skipped_subjects} (single-class).")

        if not y_true:
            print(f"[SKIP] No valid data entries for {folder} after filtering.")
            return

        self._plot(
            np.array(y_true), 
            np.array(y_pred), 
            exp_config['classes'], 
            exp_config['title'], 
            exp_config['filename']
        )

    def _plot(self, y_true, y_pred, classes, title, filename):
        # Compute & Normalize
        cm = confusion_matrix(y_true, y_pred)
        # Normalize by Row (True Label)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm_norm, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            vmin=0, vmax=1, square=True
        )
        
        plt.title(title, fontsize=14, pad=15)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        self.save_plot(fig, filename)


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Configuration Paths
    folders = ["../results_condition", "../results_weight", "../results_happiness"]
    demo_path = "../dataset/allLabels/demographics_v2.xlsx"
    label_path = "../dataset/allLabels/labels.xlsx"

    # 2. Load Data (One-time execution)
    loader = DataLoader(folders, demo_path, label_path)

    # 3. Generate Confusion Matrices
    cm_plotter = ConfusionMatrixPlotter(loader)
    cm_plotter.run()

    # (Future) 4. Generate Sensor Importance
    # sensor_plotter = SensorImportancePlotter(loader)
    # sensor_plotter.run()