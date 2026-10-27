import pickle
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
TARGET = 'happiness' 

# Configuración automática
VAR_CONFIGS = {
    'weight': {
        'dir': '../results_weight',
        'title': 'Perceived Weight Prediction',
        'file': 'figure_sensor_importance_individual_weight.pdf'
    },
    'happiness': {
        'dir': '../results_happiness',
        'title': 'Happiness Prediction',
        'file': 'figure_sensor_importance_individual_happiness.pdf'
    },
    'condition': {
        'dir': '../results_condition',
        'title': 'Auditory Condition Prediction',
        'file': 'figure_sensor_importance_individual_condition.pdf'
    }
}

if TARGET not in VAR_CONFIGS:
    raise ValueError(f"Target '{TARGET}' no válido. Usa: {list(VAR_CONFIGS.keys())}")

RESULTS_DIR = VAR_CONFIGS[TARGET]['dir']
OUTPUT_FILE = VAR_CONFIGS[TARGET]['file']
# Título adaptado para la matriz individual
PLOT_TITLE = f"Sensor Importance by Participant ({VAR_CONFIGS[TARGET]['title']})"

SENSOR_MAP = {
    0: "Pelvis", 1: "L.UpLeg", 2: "L.LowLeg", 3: "L.Foot",
    4: "R.UpLeg", 5: "R.LowLeg", 6: "R.Foot", 7: "Chest",
    8: "L.Shoulder", 9: "L.UpArm", 10: "L.ForeArm", 11: "L.Hand",
    12: "Head", 13: "R.Shoulder", 14: "R.UpArm", 15: "R.ForeArm", 16: "R.Hand"
}

def load_data_matrix():
    pkl_files = glob.glob(os.path.join(RESULTS_DIR, "results_*.pkl"))
    print(f"[INFO] Procesando {len(pkl_files)} participantes...")
    
    data_list = []
    
    for pkl_path in pkl_files:
        # Extraer ID del nombre del archivo (results_p011.pkl -> p011)
        filename = os.path.basename(pkl_path)
        pid_str = filename.replace("results_", "").replace(".pkl", "")
        
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            # Recolectar semillas
            participant_vecs = []
            for key, val in data.items():
                if 'sensor_importance' in val and val['sensor_importance'] is not None:
                    participant_vecs.append(val['sensor_importance'])
            
            if participant_vecs:
                # Media de las 10 semillas para este participante
                p_mean = np.mean(np.vstack(participant_vecs), axis=0)
                
                # Guardamos ID numérico para ordenar y el vector
                # Quitamos la 'p' para ordenar numéricamente (11 en vez de p011)
                pid_num = int(pid_str.replace('p', '').replace('b', '')) # Maneja el 069b quitando la b si quieres
                
                entry = {'PID_Str': pid_str, 'PID_Num': pid_num}
                for i, val in enumerate(p_mean):
                    entry[i] = val
                    
                data_list.append(entry)
                
        except Exception as e:
            print(f"[WARN] Error en {pkl_path}: {e}")
            continue
            
    # Crear DataFrame
    df = pd.DataFrame(data_list)
    # Ordenar por ID numérico
    df = df.sort_values('PID_Num')
    
    return df

def main():
    df = load_data_matrix()
    if df.empty:
        print("[ERROR] No hay datos.")
        return

    # Preparar datos para heatmap
    # Eje Y: IDs (Strings)
    y_labels = df['PID_Str'].values
    
    # Matriz de valores (Quitamos columnas de ID)
    heatmap_data = df.drop(columns=['PID_Str', 'PID_Num']).values
    
    # Eje X: Nombres de sensores
    x_labels = [SENSOR_MAP[i] for i in range(17)]
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 16)) # Figura alta porque hay 98 filas
    
    # Usamos Seaborn heatmap porque gestiona muy bien los labels
    ax = sns.heatmap(
        heatmap_data,
        yticklabels=y_labels,
        xticklabels=x_labels,
        cmap='jet',
        cbar_kws={'label': 'Importance Weight', 'shrink': 0.5},
        linewidths=0.0, # Sin líneas para que parezca una imagen continua
        rasterized=True # Importante para que el PDF no pese 100MB con tantos cuadraditos
    )
    
    plt.title(PLOT_TITLE, fontsize=16)
    plt.xlabel('Sensor Location', fontsize=12)
    plt.ylabel('Participant ID', fontsize=12)
    
    # Ajustar etiquetas del eje X para que se lean bien
    plt.xticks(rotation=45, ha='right')
    # Ajustar etiquetas eje Y (fuente pequeña porque son muchos)
    plt.yticks(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"[OK] Matriz guardada: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()