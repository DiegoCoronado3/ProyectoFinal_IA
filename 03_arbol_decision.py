"""
Paso 3: Árbol de Decisión
==========================
- GridSearchCV para hiperparámetros (max_depth, criterion, etc.)
- class_weight='balanced' por desbalance de clases
- Métricas en test, matriz de confusión, curva ROC, importancia de features
- Visualización del árbol (primeros niveles)
- Persistencia del modelo
"""
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
os.makedirs(FIG_DIR, exist_ok=True)
SEED = 42

# Para árbol usamos los datos SIN escalar (no le hace daño y conserva la
# interpretabilidad de las reglas en su escala original)
X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train_raw.csv"))
X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test_raw.csv"))
y_train = pd.read_csv(os.path.join(PROC_DIR, "y_train.csv")).iloc[:, 0]
y_test = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv")).iloc[:, 0]

# ----------------------------------------------------------------------
# 1) GridSearchCV
# ----------------------------------------------------------------------
print("[paso 3] Búsqueda de hiperparámetros (GridSearchCV)...")
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 5, 10, 20],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    param_grid=param_grid,
    cv=cv,
    scoring="f1",  # F1 sobre clase positiva (default), buena para desbalance
    n_jobs=-1,
    verbose=0,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print(f"  Mejores parámetros: {grid.best_params_}")
print(f"  Mejor F1 (CV):      {grid.best_score_:.4f}")

# ----------------------------------------------------------------------
# 2) Evaluación en test
# ----------------------------------------------------------------------
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]

metrics = {
    "modelo": "Arbol de Decision",
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
print("\n[test] Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

# ----------------------------------------------------------------------
# 3) Gráficas: matriz de confusión, curva ROC, importancia de features
# ----------------------------------------------------------------------
COLORS = {"no_default": "#2E7D8F", "default": "#C0392B", "neutral": "#7D7D7D"}

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm, cmap="Blues", aspect="auto")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["No Default", "Default"])
ax.set_yticklabels(["No Default", "Default"])
ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
ax.set_title("Matriz de Confusión - Árbol de Decisión", fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=15, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_cm_arbol.png"), dpi=130, bbox_inches="tight")
plt.close()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=COLORS["default"], lw=2.2,
        label=f"Árbol (AUC = {metrics['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Azar")
ax.set_xlabel("Tasa de falsos positivos (FPR)")
ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
ax.set_title("Curva ROC - Árbol de Decisión", fontweight="bold")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_roc_arbol.png"), dpi=130, bbox_inches="tight")
plt.close()

# Importancia de features (top 15)
importances = pd.Series(best.feature_importances_, index=X_train.columns)
top = importances.sort_values(ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(len(top)), top.values, color=COLORS["default"], alpha=0.8,
        edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.index, fontsize=9)
ax.set_xlabel("Importancia (Gini importance)")
ax.set_title("Top 15 features más importantes - Árbol", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_importancia_arbol.png"), dpi=130, bbox_inches="tight")
plt.close()

# Visualización del árbol (recortado para legibilidad)
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(best, feature_names=X_train.columns,
          class_names=["No Default", "Default"],
          filled=True, max_depth=3, fontsize=9, ax=ax, rounded=True)
ax.set_title("Árbol de Decisión (primeros 3 niveles)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "08_arbol_visualizado.png"), dpi=120, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 4) Persistencia
# ----------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "arbol_decision.pkl"), "wb") as f:
    pickle.dump(best, f)
with open(os.path.join(MODEL_DIR, "metrics_arbol.json"), "w") as f:
    json.dump({k: (v if not isinstance(v, dict) else v)
               for k, v in metrics.items()}, f, indent=2, default=str)

print(f"\n[ok] Modelo y métricas guardados en {MODEL_DIR}")
print(f"[ok] Profundidad del árbol final: {best.get_depth()}")
print(f"[ok] Número de hojas:             {best.get_n_leaves()}")
