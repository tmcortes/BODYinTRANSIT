import pickle
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- CONFIGURACIÓN ---
TARGET = 'condition' 

# Configuración automática de rutas y títulos
VAR_CONFIGS = {
    'weight': {
        'dir': '../results_weight',
        'title': 'Perceived Weight Prediction',
        'file': 'figure_sensor_importance_global_weight.pdf'
    },
    'happiness': {
        'dir': '../results_happiness',
        'title': 'Happiness Prediction',
        'file': 'figure_sensor_importance_global_happiness.pdf'
    },
    'condition': {
        'dir': '../results_condition',
        'title': 'Auditory Condition Prediction',
        'file': 'figure_sensor_importance_global_condition.pdf'
    }
}

# Selección automática
if TARGET not in VAR_CONFIGS:
    raise ValueError(f"Target '{TARGET}' no válido. Usa: {list(VAR_CONFIGS.keys())}")

RESULTS_DIR = VAR_CONFIGS[TARGET]['dir']
OUTPUT_FILE = VAR_CONFIGS[TARGET]['file']
PLOT_TITLE = f"Sensor Importance for {VAR_CONFIGS[TARGET]['title']}"

SENSOR_MAP = {
    0: "Pelvis",
    1: "L.UpLeg", 2: "L.LowLeg", 3: "L.Foot",
    4: "R.UpLeg", 5: "R.LowLeg", 6: "R.Foot",
    7: "Chest",
    8: "L.Shoulder", 9: "L.UpArm", 10: "L.ForeArm", 11: "L.Hand",
    12: "Head",
    13: "R.Shoulder", 14: "R.UpArm", 15: "R.ForeArm", 16: "R.Hand"
}

# Coordenadas (Igual que antes)
SKELETON_COORDS = {
    12: (0, 1.75), 7: (0, 1.25), 0: (0, 0.0),
    8: (-0.4, 1.25), 9: (-0.5, 0.8), 10: (-0.6, 0.35), 11: (-0.7, -0.1),
    13: (0.4, 1.25), 14: (0.5, 0.8), 15: (0.6, 0.35), 16: (0.7, -0.1),
    1: (-0.25, -0.7), 2: (-0.25, -1.4), 3: (-0.3, -1.8),
    4: (0.25, -0.7), 5: (0.25, -1.4), 6: (0.3, -1.8)
}

BONES = [
    (12, 7), (7, 0),
    (7, 8), (8, 9), (9, 10), (10, 11),
    (7, 13), (13, 14), (14, 15), (15, 16),
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6)
]

def load_data():
    pkl_files = glob.glob(os.path.join(RESULTS_DIR, "results_*.pkl"))
    all_means = []
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            participant_vecs = []
            for key, val in data.items():
                if 'sensor_importance' in val and val['sensor_importance'] is not None:
                    participant_vecs.append(val['sensor_importance'])
            if participant_vecs:
                all_means.append(np.mean(np.vstack(participant_vecs), axis=0))
        except: continue
            
    if not all_means: return None, None
    matrix = np.vstack(all_means)
    return np.mean(matrix, axis=0), np.std(matrix, axis=0)

def main():
    means, stds = load_data()
    if means is None: return

    indices = np.arange(17)
    labels = [SENSOR_MAP[i] for i in indices]
    
    # Ajustamos el tamaño de figura para que sea más compacta
    fig = plt.figure(figsize=(10, 6))
    
    # Título General más claro
    fig.suptitle(PLOT_TITLE, fontsize=16, fontweight='bold')
    
    # --- SUBPLOT 1: BARRAS ---
    ax_bar = fig.add_subplot(1, 2, 1)
    sorted_idx = np.argsort(means)
    norm = mcolors.Normalize(vmin=np.min(means), vmax=np.max(means))
    cmap = cm.cividis
    
    ax_bar.barh(
        [labels[i] for i in sorted_idx], 
        means[sorted_idx], 
        xerr=stds[sorted_idx], 
        color=cmap(norm(means[sorted_idx])),
        capsize=4, alpha=0.8, edgecolor='black'
    )
    
    ax_bar.set_xlabel('Mean Importance Weight')
    ax_bar.set_title('Global Ranking', fontsize=12) # Subtítulo más simple
    ax_bar.grid(axis='x', linestyle='--', alpha=0.5)
    ax_bar.axvline(1/17, color='gray', linestyle='--', label='Uniform Chance')
    ax_bar.legend(loc='lower right')

    # --- SUBPLOT 2: MANIQUÍ ---
    ax_man = fig.add_subplot(1, 2, 2)
    ax_man.set_aspect('equal')
    ax_man.axis('off')
    # ax_man.set_title('Spatial Distribution', fontsize=12) 
    
    # Dibujar Huesos
    for start, end in BONES:
        x_vals = [SKELETON_COORDS[start][0], SKELETON_COORDS[end][0]]
        y_vals = [SKELETON_COORDS[start][1], SKELETON_COORDS[end][1]]
        ax_man.plot(x_vals, y_vals, color='gray', linewidth=2, zorder=1)
    
    # Dibujar Sensores
    xs = [SKELETON_COORDS[i][0] for i in indices]
    ys = [SKELETON_COORDS[i][1] for i in indices]
    
    sc = ax_man.scatter(
        xs, ys, 
        c=means, 
        s=400, # Círculos un poco más grandes
        cmap='cividis', 
        norm=norm,
        edgecolor='black', 
        zorder=2
    )
    
    # --- CORRECCIONES VISUALES ---
    
    # 1. Ajustar límites para que no se corten las manos (Hands @ X=0.7, Y=-0.1)
    ax_man.set_xlim(-1.0, 1.0) # Damos margen lateral
    ax_man.set_ylim(-2.0, 2.0) # Damos margen vertical (pies y cabeza)
    
    # 2. Reducir espacio entre subplots
    plt.subplots_adjust(wspace=-0.4, top=0.88, bottom=0.1) # wspace pequeño junta los plots

    # Colorbar más pequeña
    cbar = plt.colorbar(sc, ax=ax_man, fraction=0.046, pad=0.04)
    cbar.set_label('Importance')

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"[OK] Figura corregida guardada: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()