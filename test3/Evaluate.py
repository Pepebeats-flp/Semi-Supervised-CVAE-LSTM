import numpy as np
import pandas as pd

_lambda = 0.6
epsilon = 1e-6

csv_path = "extracted_feature_values.csv"
df = pd.read_csv(csv_path, index_col=0)  # <-- Aquí cargamos con índice

# Columnas vector p(c)
pitch_cols = [f'Pitch_Class_Histogram_{i}' for i in range(12)]
p_cols = pitch_cols + ['Major_or_Minor', 'Vertical_Dissonance_Ratio']

# Columnas para riqueza armónica
richness_cols = ['Dominant_Seventh_Chords', 'Seventh_Chords', 'Non-Standard_Chords', 'Complex_Chords']

def coherence(p_orig, p_gen):
    diff_norm = np.linalg.norm(p_gen - p_orig)
    orig_norm = np.linalg.norm(p_orig) + epsilon
    return 1 - diff_norm / orig_norm

def richness(row):
    scores = row[richness_cols].values
    return np.mean(scores)

results = []

for i in range(1, 5):
    orig_name = f"TEST{i}.mid"
    gen_name = f"GENERATED{i}.mid"
    
    # Buscar en índice
    orig_row = df.loc[[idx for idx in df.index if orig_name in idx]]
    gen_row = df.loc[[idx for idx in df.index if gen_name in idx]]
    
    if orig_row.empty or gen_row.empty:
        print(f"Archivo {orig_name} o {gen_name} no encontrado en CSV")
        continue
    
    p_orig = orig_row[p_cols].values.flatten().astype(float)
    p_gen = gen_row[p_cols].values.flatten().astype(float)
    
    coh = coherence(p_orig, p_gen)
    rich = richness(gen_row.iloc[0])
    score = _lambda * coh + (1 - _lambda) * rich
    
    results.append({
        'Original': orig_name,
        'Generated': gen_name,
        'Coherence': coh,
        'Richness': rich,
        'Score': score
    })

df_results = pd.DataFrame(results)
print(df_results)
