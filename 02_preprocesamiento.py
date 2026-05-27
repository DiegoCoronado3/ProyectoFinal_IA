"""
Paso 2: Preprocesamiento del dataset German Credit
====================================================
- One-hot encoding de variables categóricas
- Estandarización de variables numéricas (StandardScaler)
- Train/test split 70/30 estratificado por la variable objetivo
- Persistencia de los conjuntos preprocesados para los pasos siguientes

Genera en ../data/processed/:
    X_train.csv, X_test.csv, y_train.csv, y_test.csv, scaler.pkl, feature_names.txt
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Rutas relativas
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROC_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROC_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "german_credit_raw.csv")
SEED = 42

# ----------------------------------------------------------------------
# 1) Carga y recodificación de target
# ----------------------------------------------------------------------
print("[paso 2] Cargando dataset...")
df = pd.read_csv(RAW_PATH)
df["default"] = (df["default"] == 2).astype(int)   # 1=good->0, 2=bad->1

y = df["default"].copy()
X = df.drop(columns=["default"])

numericas = X.select_dtypes(include=[np.number]).columns.tolist()
categoricas = X.select_dtypes(include=["object"]).columns.tolist()
print(f"  numéricas:   {len(numericas)} -> {numericas}")
print(f"  categóricas: {len(categoricas)}")

# ----------------------------------------------------------------------
# 2) One-hot encoding de categóricas (drop_first=True para evitar
#    colinealidad perfecta, importante para Regresión Logística)
# ----------------------------------------------------------------------
X_encoded = pd.get_dummies(X, columns=categoricas, drop_first=True, dtype=int)
print(f"  shape tras one-hot: {X_encoded.shape}")

# ----------------------------------------------------------------------
# 3) Train/Test Split (estratificado por desbalance 70/30)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.30,
    stratify=y,
    random_state=SEED,
)
print(f"\n[split] train: {X_train.shape}, test: {X_test.shape}")
print(f"  tasa default train: {y_train.mean():.2%}")
print(f"  tasa default test:  {y_test.mean():.2%}")

# ----------------------------------------------------------------------
# 4) Estandarización SOLO de columnas numéricas (no de one-hot)
#    El scaler se ajusta con TRAIN y se aplica a TEST
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numericas] = scaler.fit_transform(X_train[numericas])
X_test_scaled[numericas] = scaler.transform(X_test[numericas])

# ----------------------------------------------------------------------
# 5) Persistencia
# ----------------------------------------------------------------------
X_train_scaled.to_csv(os.path.join(PROC_DIR, "X_train.csv"), index=False)
X_test_scaled.to_csv(os.path.join(PROC_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(PROC_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROC_DIR, "y_test.csv"), index=False)

# También conservamos versión SIN escalar (el árbol no la necesita)
X_train.to_csv(os.path.join(PROC_DIR, "X_train_raw.csv"), index=False)
X_test.to_csv(os.path.join(PROC_DIR, "X_test_raw.csv"), index=False)

with open(os.path.join(PROC_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump({"scaler": scaler, "numericas": numericas,
                 "categoricas": categoricas,
                 "feature_names": list(X_encoded.columns)}, f)

with open(os.path.join(PROC_DIR, "feature_names.txt"), "w") as f:
    f.write("\n".join(X_encoded.columns))

print(f"\n[ok] Archivos guardados en {PROC_DIR}")
print(f"     Total de features tras preprocesamiento: {X_encoded.shape[1]}")
