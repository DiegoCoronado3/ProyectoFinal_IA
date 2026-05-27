"""
Paso 5: K-Nearest Neighbors (KNN)
==================================
- GridSearchCV sobre k, distancia, ponderación
- Datos ESCALADOS (KNN es sensible a escala)
- Métricas, matriz de confusión, curva ROC
- Curva de validación para k
"""
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
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

# Para KNN usamos los datos ESCALADOS (estandarizados)
X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROC_DIR, "y_train.csv")).iloc[:, 0]
y_test = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv")).iloc[:, 0]

print("[paso 5] GridSearchCV KNN...")
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 25, 31],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid = GridSearchCV(
    KNeighborsClassifier(),
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
    "modelo": "KNN",
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
COLORS = {"no_default": "#2E7D8F", "default": "#C0392B", "knn": "#8E44AD"}

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm, cmap="Blues", aspect="auto")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["No Default", "Default"]); ax.set_yticklabels(["No Default", "Default"])
ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
ax.set_title("Matriz de Confusión - KNN", fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=15, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "12_cm_knn.png"), dpi=130, bbox_inches="tight")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=COLORS["knn"], lw=2.2,
        label=f"KNN (AUC = {metrics['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Azar")
ax.set_xlabel("Tasa de falsos positivos (FPR)")
ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
ax.set_title("Curva ROC - KNN", fontweight="bold")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "13_roc_knn.png"), dpi=130, bbox_inches="tight")
plt.close()

# Curva de validación: F1 vs k
print("\n[extra] Curva de validación F1 vs k...")
ks = list(range(1, 41))
f1_train, f1_val = [], []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k,
                               weights=grid.best_params_["weights"],
                               metric=grid.best_params_["metric"])
    scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    f1_val.append(scores.mean())
    knn.fit(X_train, y_train)
    f1_train.append(f1_score(y_train, knn.predict(X_train)))

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(ks, f1_train, "o-", color=COLORS["no_default"], label="Train F1", lw=1.6, ms=4)
ax.plot(ks, f1_val, "s-", color=COLORS["default"], label="Validation F1 (5-fold CV)", lw=1.8, ms=5)
ax.axvline(grid.best_params_["n_neighbors"], color="black", linestyle="--",
           alpha=0.6, label=f"k óptimo = {grid.best_params_['n_neighbors']}")
ax.set_xlabel("k (número de vecinos)")
ax.set_ylabel("F1-score (clase default)")
ax.set_title("Curva de validación KNN", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "14_curva_validacion_knn.png"), dpi=130, bbox_inches="tight")
plt.close()

with open(os.path.join(MODEL_DIR, "knn.pkl"), "wb") as f:
    pickle.dump(best, f)
with open(os.path.join(MODEL_DIR, "metrics_knn.json"), "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print(f"\n[ok] Modelo guardado.")
