"""
Paso 9: Casos de uso comparativos - los 3 algoritmos en 15 clientes
====================================================================
Compara las predicciones de Árbol, Logística y KNN sobre los 15
clientes seleccionados, contra el valor real (ground truth).
Genera:
  - results/casos_uso_3_modelos.csv     Tabla maestra con esperado vs obtenido
  - results/figures/18_casos_uso_comparativo.png    Heatmap de aciertos
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Cargar modelos
with open(os.path.join(MODEL_DIR, "arbol_decision.pkl"), "rb") as f:
    arbol = pickle.load(f)
with open(os.path.join(MODEL_DIR, "regresion_logistica.pkl"), "rb") as f:
    logreg = pickle.load(f)
with open(os.path.join(MODEL_DIR, "knn.pkl"), "rb") as f:
    knn = pickle.load(f)
with open(os.path.join(PROC_DIR, "scaler.pkl"), "rb") as f:
    meta = pickle.load(f)
scaler = meta["scaler"]
feature_names = meta["feature_names"]
numericas = meta["numericas"]
categoricas = meta["categoricas"]

# Cargar clientes
clientes = pd.read_csv(os.path.join(PROC_DIR, "clientes_demo_metadata.csv"))


def preparar_features(df_clientes):
    """Aplica el mismo preprocesamiento que usamos en train."""
    df = df_clientes.drop(columns=["cliente_id", "default_real"], errors="ignore").copy()
    df_enc = pd.get_dummies(df, columns=categoricas, drop_first=False, dtype=int)
    for col in feature_names:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[feature_names]
    # Versión escalada (para logística y KNN)
    df_scaled = df_enc.copy()
    df_scaled[numericas] = scaler.transform(df_enc[numericas])
    return df_enc, df_scaled


X_raw, X_scaled = preparar_features(clientes)

# Predicciones
pred_arbol = arbol.predict(X_raw)
proba_arbol = arbol.predict_proba(X_raw)[:, 1]

pred_log = logreg.predict(X_scaled)
proba_log = logreg.predict_proba(X_scaled)[:, 1]

pred_knn = knn.predict(X_scaled)
proba_knn = knn.predict_proba(X_scaled)[:, 1]

# Tabla maestra
def etiqueta(v):
    return "RECHAZADO" if v == 1 else "APROBADO"

tabla = pd.DataFrame({
    "cliente_id": clientes["cliente_id"],
    "edad": clientes["age"],
    "monto": clientes["amount"],
    "duracion_meses": clientes["months_loan_duration"],
    "esperado": clientes["default_real"].map({0: "APROBADO", 1: "RECHAZADO"}),
    "arbol_pred": [etiqueta(p) for p in pred_arbol],
    "arbol_prob": proba_arbol.round(3),
    "logistica_pred": [etiqueta(p) for p in pred_log],
    "logistica_prob": proba_log.round(3),
    "knn_pred": [etiqueta(p) for p in pred_knn],
    "knn_prob": proba_knn.round(3),
})
tabla["acierta_arbol"] = (pred_arbol == clientes["default_real"].values).astype(int)
tabla["acierta_logistica"] = (pred_log == clientes["default_real"].values).astype(int)
tabla["acierta_knn"] = (pred_knn == clientes["default_real"].values).astype(int)

print("=" * 110)
print("CASOS DE USO COMPARATIVOS - 15 CLIENTES")
print("=" * 110)
print(tabla.to_string(index=False))
print()

# Resumen de aciertos
n = len(tabla)
print(f"\nAciertos por modelo (de {n}):")
print(f"  Árbol de Decisión:   {tabla['acierta_arbol'].sum()}/{n}  ({tabla['acierta_arbol'].mean():.1%})")
print(f"  Regresión Logística: {tabla['acierta_logistica'].sum()}/{n}  ({tabla['acierta_logistica'].mean():.1%})")
print(f"  KNN:                 {tabla['acierta_knn'].sum()}/{n}  ({tabla['acierta_knn'].mean():.1%})")

# Guardar tabla
tabla.to_csv(os.path.join(RESULTS_DIR, "casos_uso_3_modelos.csv"), index=False)
print(f"\n[ok] Tabla guardada en {RESULTS_DIR}/casos_uso_3_modelos.csv")

# ----------------------------------------------------------------------
# Visualización: heatmap de aciertos
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5.5))

modelos = ["Árbol", "Logística", "KNN"]
matriz = np.array([
    tabla["acierta_arbol"].values,
    tabla["acierta_logistica"].values,
    tabla["acierta_knn"].values,
])

# Color: verde si acierta, rojo si falla
colors = np.where(matriz == 1, 0.7, 0.0)   # 0.7 = verde, 0.0 = rojo
im = ax.imshow(colors, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(n))
ax.set_xticklabels(tabla["cliente_id"], rotation=0, fontsize=9)
ax.set_yticks(range(3))
ax.set_yticklabels(modelos, fontsize=11)

# Anotar cada celda con la predicción y si acertó
for i, mod_pred_col in enumerate(["arbol_pred", "logistica_pred", "knn_pred"]):
    for j in range(n):
        pred = tabla.iloc[j][mod_pred_col]
        acierta = matriz[i, j] == 1
        symbol = "✓" if acierta else "✗"
        texto = f"{symbol}\n{'APR' if pred=='APROBADO' else 'RECH'}"
        ax.text(j, i, texto, ha="center", va="center",
                fontsize=8.5, fontweight="bold",
                color="black" if acierta else "white")

# Fila adicional indicando ground truth
ax.set_title("Casos de uso comparativos: 15 clientes reales del test set\n"
             "(Verde = predicción correcta, Rojo = predicción incorrecta)",
             fontsize=12, fontweight="bold", pad=15)

# Anotar el ground truth abajo
for j in range(n):
    real = tabla.iloc[j]["esperado"]
    color_real = "green" if real == "APROBADO" else "red"
    ax.text(j, 3.0, "REAL:\n" + ("APR" if real == "APROBADO" else "RECH"),
            ha="center", va="center", fontsize=8,
            color=color_real, fontweight="bold")
ax.set_ylim(3.5, -0.5)

# Leyenda
green_patch = mpatches.Patch(color="#2ECC71", label="Acierto")
red_patch = mpatches.Patch(color="#E74C3C", label="Error")
ax.legend(handles=[green_patch, red_patch], loc="upper right",
          bbox_to_anchor=(1.0, 1.18), ncol=2, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "18_casos_uso_comparativo.png"),
            dpi=130, bbox_inches="tight")
plt.close()
print(f"[ok] Heatmap guardado en {FIG_DIR}/18_casos_uso_comparativo.png")

# ----------------------------------------------------------------------
# Gráfica de barras: aciertos totales
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
aciertos = [tabla["acierta_arbol"].sum(),
            tabla["acierta_logistica"].sum(),
            tabla["acierta_knn"].sum()]
colors_b = ["#C0392B", "#2E7D8F", "#8E44AD"]
bars = ax.bar(modelos, aciertos, color=colors_b, edgecolor="black",
              linewidth=0.6, alpha=0.85)
ax.axhline(n, color="black", linestyle="--", linewidth=1, alpha=0.4,
           label=f"Máximo ({n})")
for bar, val in zip(bars, aciertos):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{val}/{n}\n({val/n:.0%})", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Aciertos (de 15 clientes)", fontsize=11)
ax.set_title("Comparativa de aciertos en la demo (15 clientes)", fontweight="bold")
ax.set_ylim(0, n + 2)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "19_aciertos_demo.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"[ok] Aciertos totales guardado en {FIG_DIR}/19_aciertos_demo.png")
