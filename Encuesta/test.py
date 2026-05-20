# ==========================================================
# FIGURE 3 — STACKED BAR: PHC polarity × MSI × BLOCK
# ==========================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# 1. Definir MSI (si no existe previamente)
# ----------------------------------------------------------
df_filtered["MSI"] = df_filtered["SCORE_NORM"]

df_filtered["MSI_bin"] = pd.cut(
    df_filtered["MSI"],
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# ----------------------------------------------------------
# 2. Expandir df_audio con MSI del participante
# ----------------------------------------------------------
df_audio = df_audio.merge(
    df_filtered[["MSI_bin"]],
    left_on="participant_id",
    right_index=True,
    how="left"
)

# ----------------------------------------------------------
# 3. Categoría de preferencia
# ----------------------------------------------------------
df_audio["pref_type"] = np.where(
    df_audio["score"] > 0, "Positive",
    np.where(df_audio["score"] < 0, "Negative", "Neutral")
)

# ----------------------------------------------------------
# 4. Agregación: conteos por BLOCK × MSI × PREF
# ----------------------------------------------------------
summary = (
    df_audio
    .groupby(["block", "MSI_bin", "pref_type"])
    .size()
    .reset_index(name="count")
)

# Pivot para stacked bars
pivot = summary.pivot_table(
    index=["block", "MSI_bin"],
    columns="pref_type",
    values="count",
    fill_value=0
).reset_index()

# ----------------------------------------------------------
# 5. Plot stacked bar
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8,5))

colors = {
    "Negative": "#d73027",
    "Neutral": "#fdae61",
    "Positive": "#1a9850"
}

blocks = pivot["block"].unique()

x_labels = []
x_pos = []
offset = 0

for block in blocks:

    sub = pivot[pivot["block"] == block]

    for _, row in sub.iterrows():

        bottom = 0

        x_labels.append(f"{block.split('—')[-1].strip()}\n{row['MSI_bin']}")
        x_pos.append(offset)

        for cat in ["Negative", "Neutral", "Positive"]:

            val = row.get(cat, 0)

            ax.bar(
                offset,
                val,
                bottom=bottom,
                color=colors[cat],
                edgecolor="black",
                linewidth=0.5
            )

            # annotation dentro del segmento
            if val > 0:
                ax.text(
                    offset,
                    bottom + val / 2,
                    str(int(val)),
                    ha="center",
                    va="center",
                    fontsize=8
                )

            bottom += val

        offset += 1

# ----------------------------------------------------------
# 6. Styling paper
# ----------------------------------------------------------
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=0)

ax.set_ylabel("Number of responses")

# sin título (formato paper)

plt.tight_layout()
plt.show()