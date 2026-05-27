"""
Paso 6: Comparación de los tres modelos
========================================
- Tabla comparativa de métricas
- Gráfica de barras comparativa
- Curvas ROC superpuestas
- Análisis: cuál modelo gana y por qué
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Cargar datos
X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
X_test_raw = pd.read_csv(os.path.join(PROC_DIR, "X_test_raw.csv"))
y_test = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv")).iloc[:, 0]

# Cargar modelos
with open(os.path.join(MODEL_DIR, "arbol_decision.pkl"), "rb") as f:
    arbol = pickle.load(f)
with open(os.path.join(MODEL_DIR, "regresion_logistica.pkl"), "rb") as f:
    logreg = pickle.load(f)
with open(os.path.join(MODEL_DIR, "knn.pkl"), "rb") as f:
    knn = pickle.load(f)

# Cargar métricas
with open(os.path.join(MODEL_DIR, "metrics_arbol.json")) as f:
    m_arbol = json.load(f)
with open(os.path.join(MODEL_DIR, "metrics_logistica.json")) as f:
    m_log = json.load(f)
with open(os.path.join(MODEL_DIR, "metrics_knn.json")) as f:
    m_knn = json.load(f)

# Tabla comparativa
df_metrics = pd.DataFrame([
    {"Modelo": "Árbol de Decisión", **{k: m_arbol[k] for k in ["accuracy","precision","recall","f1","roc_auc"]}},
    {"Modelo": "Regresión Logística", **{k: m_log[k] for k in ["accuracy","precision","recall","f1","roc_auc"]}},
    {"Modelo": "KNN", **{k: m_knn[k] for k in ["accuracy","precision","recall","f1","roc_auc"]}},
])
df_metrics = df_metrics.round(4)
print("=" * 70)
print("TABLA COMPARATIVA")
print("=" * 70)
print(df_metrics.to_string(index=False))

df_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_comparacion.csv"), index=False)

# Determinar ganador (priorizamos F1 y ROC-AUC por desbalance)
df_metrics["score_combinado"] = df_metrics["f1"] * 0.5 + df_metrics["roc_auc"] * 0.5
mejor = df_metrics.loc[df_metrics["score_combinado"].idxmax(), "Modelo"]
print(f"\n>>> MEJOR MODELO (F1 + ROC-AUC): {mejor}")

# ----------------------------------------------------------------------
# Gráfica comparativa de barras
# ----------------------------------------------------------------------
COLORS = {"arbol": "#C0392B", "logistica": "#2E7D8F", "knn": "#8E44AD"}

metricas_a_graficar = ["accuracy", "precision", "recall", "f1", "roc_auc"]
metricas_labels = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
x = np.arange(len(metricas_a_graficar))
width = 0.27

fig, ax = plt.subplots(figsize=(11, 5.5))
v_arbol = [m_arbol[k] for k in metricas_a_graficar]
v_log = [m_log[k] for k in metricas_a_graficar]
v_knn = [m_knn[k] for k in metricas_a_graficar]
ax.bar(x - width, v_arbol, width, label="Árbol de Decisión",
       color=COLORS["arbol"], edgecolor="black", linewidth=0.5)
ax.bar(x, v_log, width, label="Regresión Logística",
       color=COLORS["logistica"], edgecolor="black", linewidth=0.5)
ax.bar(x + width, v_knn, width, label="KNN",
       color=COLORS["knn"], edgecolor="black", linewidth=0.5)
# valores encima de cada barra
for i, (a, l, k) in enumerate(zip(v_arbol, v_log, v_knn)):
    ax.text(i - width, a + 0.012, f"{a:.2f}", ha="center", fontsize=8)
    ax.text(i, l + 0.012, f"{l:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(i + width, k + 0.012, f"{k:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(metricas_labels)
ax.set_ylabel("Valor")
ax.set_ylim(0, 1.0)
ax.set_title("Comparación de métricas: Árbol vs Logística vs KNN (test, n=300)",
             fontweight="bold", fontsize=12)
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "15_comparacion_metricas.png"), dpi=130, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# Curvas ROC superpuestas
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for nombre, mod, X, color, m in [
    ("Árbol de Decisión", arbol, X_test_raw, COLORS["arbol"], m_arbol),
    ("Regresión Logística", logreg, X_test, COLORS["logistica"], m_log),
    ("KNN", knn, X_test, COLORS["knn"], m_knn),
]:
    proba = mod.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=color, lw=2.2,
            label=f"{nombre} (AUC = {m['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Azar")
ax.set_xlabel("Tasa de falsos positivos (FPR)")
ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
ax.set_title("Curvas ROC comparativas - Test set", fontweight="bold", fontsize=12)
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "16_roc_comparacion.png"), dpi=130, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# Matrices de confusión lado a lado
# ----------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
modelos = [
    ("Árbol de Decisión", arbol.predict(X_test_raw)),
    ("Regresión Logística", logreg.predict(X_test)),
    ("KNN", knn.predict(X_test)),
]
for ax, (nombre, y_pred) in zip(axes, modelos):
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Default", "Default"])
    ax.set_yticklabels(["No Default", "Default"])
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(nombre, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
plt.suptitle("Matrices de confusión comparativas", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "17_matrices_confusion_comparacion.png"), dpi=130, bbox_inches="tight")
plt.close()

print(f"\n[ok] Comparación guardada en {FIG_DIR}")
print(f"[ok] Tabla en {RESULTS_DIR}/metrics_comparacion.csv")
