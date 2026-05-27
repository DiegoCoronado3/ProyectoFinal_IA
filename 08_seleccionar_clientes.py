"""
Paso 8: Selección de 15 clientes del test set para la demo
=============================================================
Selecciona 15 solicitudes reales del conjunto de prueba que servirán
como casos de uso. Mezcla intencionalmente:
  - 8 clientes que NO cayeron en default (clase 0)
  - 7 clientes que SÍ cayeron en default (clase 1)
Esta proporción (~53/47) es más balanceada que el dataset original
(70/30) para que la demo sea didácticamente clara.

Salida:
    ../data/processed/clientes_demo.csv             (formato legible humano)
    ../data/processed/clientes_demo_metadata.csv    (con ground truth)
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

SEED = 42
np.random.seed(SEED)

# Cargamos test set crudo (sin encoding) para tener las features originales
df_raw = pd.read_csv(os.path.join(BASE_DIR, "data", "german_credit_raw.csv"))
df_raw["default_real"] = (df_raw["default"] == 2).astype(int)

# Necesitamos saber qué filas pertenecen al test set:
# replicamos exactamente el split (stratify + random_state=42)
from sklearn.model_selection import train_test_split
y = df_raw["default_real"]
X_full = df_raw.drop(columns=["default", "default_real"])
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.30, stratify=y, random_state=SEED
)

# Reunir test set con su target real
test_set = X_test.copy()
test_set["default_real"] = y_test
test_set = test_set.reset_index(drop=True)

# Muestrear 8 no-default + 7 default
no_default = test_set[test_set["default_real"] == 0].sample(n=8, random_state=SEED)
default = test_set[test_set["default_real"] == 1].sample(n=7, random_state=SEED)
clientes = pd.concat([no_default, default]).sample(frac=1, random_state=SEED).reset_index(drop=True)
clientes.insert(0, "cliente_id", [f"C{i+1:03d}" for i in range(len(clientes))])

print(f"[ok] {len(clientes)} clientes seleccionados")
print(f"     - No default reales: {(clientes['default_real']==0).sum()}")
print(f"     - Default reales:    {(clientes['default_real']==1).sum()}")
print()
print(clientes[["cliente_id", "checking_balance", "months_loan_duration",
                "amount", "savings_balance", "age", "default_real"]].to_string(index=False))

# Versión limpia (sin la columna target) que el sistema usará como entrada
clientes_input = clientes.drop(columns=["default_real"])
clientes_input.to_csv(os.path.join(PROC_DIR, "clientes_demo.csv"), index=False)

# Versión con metadata (para validar contra ground truth)
clientes.to_csv(os.path.join(PROC_DIR, "clientes_demo_metadata.csv"), index=False)

print(f"\n[ok] Guardado en:")
print(f"     {PROC_DIR}/clientes_demo.csv          (input del sistema)")
print(f"     {PROC_DIR}/clientes_demo_metadata.csv (con ground truth)")
