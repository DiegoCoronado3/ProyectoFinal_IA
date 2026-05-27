"""
Paso 4: Regresión Logística
=============================
- GridSearchCV sobre C (regularización L2)
- class_weight='balanced'
- Métricas, matriz de confusión, curva ROC, top coeficientes
"""
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
SEED = 42

# Para regresión logística usamos los datos ESCALADOS (numéricas)
X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROC_DIR, "y_train.csv")).iloc[:, 0]
y_test = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv")).iloc[:, 0]

print("[paso 4] GridSearchCV Regresión Logística...")
param_grid = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid = GridSearchCV(
    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    param_grid=param_grid, cv=cv, scoring="f1", n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print(f"  Mejores parámetros: {grid.best_params_}")
print(f"  Mejor F1 (CV):      {grid.best_score_:.4f}")

# Evaluación
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]
metrics = {
    "modelo": "Regresion Logistica",
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "best_params": grid.best_params_,
}
print("\n[test] Métricas:")
for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    print(f"  {k:10s}: {metrics[k]:.4f}")
print("\n[test] Reporte:")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

# Gráficas
COLORS = {"no_default": "#2E7D8F", "default": "#C0392B"}
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm, cmap="Blues", aspect="auto")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["No Default", "Default"]); ax.set_yticklabels(["No Default", "Default"])
ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
ax.set_title("Matriz de Confusión - Regresión Logística", fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=15, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "09_cm_logistica.png"), dpi=130, bbox_inches="tight")
plt.close()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=COLORS["no_default"], lw=2.2,
        label=f"Logística (AUC = {metrics['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Azar")
ax.set_xlabel("Tasa de falsos positivos (FPR)")
ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
ax.set_title("Curva ROC - Regresión Logística", fontweight="bold")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "10_roc_logistica.png"), dpi=130, bbox_inches="tight")
plt.close()

# Top coeficientes (positivos y negativos)
coefs = pd.Series(best.coef_[0], index=X_train.columns).sort_values()
top_pos = coefs.tail(8)   # más asociadas a default
top_neg = coefs.head(8)   # más asociadas a NO default
combined = pd.concat([top_neg, top_pos])
colors_bar = ([COLORS["no_default"]] * len(top_neg)) + ([COLORS["default"]] * len(top_pos))
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.barh(range(len(combined)), combined.values, color=colors_bar,
        alpha=0.85, edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(combined)))
ax.set_yticklabels(combined.index, fontsize=9)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Coeficiente β")
ax.set_title("Coeficientes más relevantes - Regresión Logística\n"
             "(rojo: aumenta riesgo de default | azul: lo disminuye)",
             fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "11_coeficientes_logistica.png"), dpi=130, bbox_inches="tight")
plt.close()

with open(os.path.join(MODEL_DIR, "regresion_logistica.pkl"), "wb") as f:
    pickle.dump(best, f)
with open(os.path.join(MODEL_DIR, "metrics_logistica.json"), "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print(f"\n[ok] Modelo guardado.")
