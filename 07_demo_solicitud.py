"""
Paso 7: Sistema explicable - Inferencia para una nueva solicitud
==================================================================
Implementa la funcionalidad clave de la propuesta original:
  "El sistema debe recibir como entrada los datos de una nueva
   solicitud de crédito y devolver como salida una de dos clases:
   'aprobado' o 'rechazado'. Adicionalmente, gracias a la naturaleza
   del algoritmo elegido, el sistema podrá entregar la trayectoria de
   decisiones que llevó a esa clasificación (es decir, las reglas
   activadas dentro del árbol)."

Usa el ÁRBOL DE DECISIÓN como modelo explicable (aunque la Regresión
Logística tiene mejores métricas globales, el árbol cumple la
exigencia de explicabilidad pedida en la propuesta).
"""
import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")

# ----------------------------------------------------------------------
# Cargar modelo y metadatos
# ----------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "arbol_decision.pkl"), "rb") as f:
    arbol = pickle.load(f)
with open(os.path.join(PROC_DIR, "scaler.pkl"), "rb") as f:
    meta = pickle.load(f)

feature_names = meta["feature_names"]   # 48 columnas tras one-hot
numericas = meta["numericas"]
categoricas = meta["categoricas"]


def preparar_solicitud(solicitud: dict) -> pd.DataFrame:
    """
    Recibe un dict con los 20 campos originales y devuelve un DataFrame
    de 1 fila con las 48 columnas que espera el modelo.

    IMPORTANTE: usamos drop_first=False y luego alineamos con
    feature_names. Si usáramos drop_first=True sobre una sola fila,
    pandas eliminaría la única columna que se generó por categoría,
    rompiendo el encoding.
    """
    df = pd.DataFrame([solicitud])
    df_enc = pd.get_dummies(df, columns=categoricas, drop_first=False, dtype=int)
    # Agregar columnas faltantes en 0 y descartar las que no estén en train
    for col in feature_names:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[feature_names]
    return df_enc


def predecir_con_trayectoria(solicitud: dict, modelo=arbol, umbral: float = 0.5):
    """
    Devuelve:
      - decision: 'APROBADO' o 'RECHAZADO'
      - prob_default: probabilidad de default
      - trayectoria: lista de reglas activadas (decisiones del árbol)
    """
    X = preparar_solicitud(solicitud)
    proba_default = float(modelo.predict_proba(X)[0, 1])
    decision = "RECHAZADO" if proba_default >= umbral else "APROBADO"

    # Reconstruir trayectoria nodo por nodo
    tree = modelo.tree_
    node_indicator = modelo.decision_path(X)
    leaf_id = modelo.apply(X)[0]
    sample_id = 0
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]
    ]
    trayectoria = []
    x_arr = X.values[0]
    for nid in node_index:
        if leaf_id == nid:   # hoja: no hay regla, solo predicción
            n_no_default = tree.value[nid, 0, 0]
            n_default = tree.value[nid, 0, 1]
            total = n_no_default + n_default
            trayectoria.append({
                "nodo": int(nid), "tipo": "HOJA",
                "regla": (f"Predicción final: {decision} "
                          f"(prob. default = {proba_default:.2%})"),
                "muestras_train_en_nodo": int(total),
                "no_default_en_nodo": int(n_no_default),
                "default_en_nodo": int(n_default),
            })
            continue
        feature = feature_names[tree.feature[nid]]
        threshold = tree.threshold[nid]
        value = x_arr[tree.feature[nid]]
        operador = "<=" if value <= threshold else ">"
        regla = f"{feature} = {value:.2f}  {operador}  {threshold:.2f}"
        trayectoria.append({
            "nodo": int(nid), "tipo": "DECISION",
            "regla": regla, "feature": feature,
            "valor_solicitud": float(value), "umbral": float(threshold),
            "decision_rama": operador,
        })
    return {
        "decision": decision,
        "prob_default": proba_default,
        "umbral_usado": umbral,
        "trayectoria": trayectoria,
    }


def imprimir_resultado(solicitud: dict, resultado: dict):
    print("\n" + "=" * 72)
    print("  SISTEMA AUTOMATIZADO DE EVALUACIÓN CREDITICIA")
    print("=" * 72)
    print("\n--- Datos de la solicitud ---")
    for k, v in solicitud.items():
        print(f"  {k:25s}: {v}")
    decision = resultado["decision"]
    color = "\033[92m" if decision == "APROBADO" else "\033[91m"
    reset = "\033[0m"
    print(f"\n{'-' * 72}")
    print(f"  DECISIÓN: {color}{decision}{reset}")
    print(f"  Probabilidad de default: {resultado['prob_default']:.2%}")
    print(f"  Umbral usado: {resultado['umbral_usado']:.2%}")
    print(f"{'-' * 72}")
    print("\n--- Trayectoria de decisiones (reglas activadas) ---")
    for i, paso in enumerate(resultado["trayectoria"], start=1):
        if paso["tipo"] == "DECISION":
            print(f"  Paso {i}: {paso['regla']}")
        else:
            print(f"  -> {paso['regla']}")
            print(f"     (muestras de entrenamiento en esta hoja: "
                  f"{paso['muestras_train_en_nodo']}, "
                  f"de las cuales {paso['default_en_nodo']} fueron default)")
    print("=" * 72)


# ----------------------------------------------------------------------
# EJEMPLOS DE PRUEBA
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo 1: solicitante con buen perfil (esperamos APROBADO)
    solicitud_1 = {
        "checking_balance": "> 200 DM",
        "months_loan_duration": 12,
        "credit_history": "critical",
        "purpose": "radio/tv",
        "amount": 1200,
        "savings_balance": "> 1000 DM",
        "employment_length": "> 7 yrs",
        "installment_rate": 2,
        "personal_status": "single male",
        "other_debtors": "none",
        "residence_history": 4,
        "property": "real estate",
        "age": 45,
        "installment_plan": "none",
        "housing": "own",
        "existing_credits": 1,
        "dependents": 1,
        "telephone": "yes",
        "foreign_worker": "no",
        "job": "skilled employee",
    }
    print("\n\n############### EJEMPLO 1: PERFIL FAVORABLE ###############")
    res1 = predecir_con_trayectoria(solicitud_1)
    imprimir_resultado(solicitud_1, res1)

    # Ejemplo 2: solicitante con perfil riesgoso (esperamos RECHAZADO)
    solicitud_2 = {
        "checking_balance": "< 0 DM",
        "months_loan_duration": 48,
        "credit_history": "fully repaid",
        "purpose": "car (new)",
        "amount": 12000,
        "savings_balance": "< 100 DM",
        "employment_length": "0 - 1 yrs",
        "installment_rate": 4,
        "personal_status": "single male",
        "other_debtors": "none",
        "residence_history": 1,
        "property": "unknown/none",
        "age": 23,
        "installment_plan": "bank",
        "housing": "rent",
        "existing_credits": 2,
        "dependents": 2,
        "telephone": "none",
        "foreign_worker": "yes",
        "job": "unskilled resident",
    }
    print("\n\n############### EJEMPLO 2: PERFIL RIESGOSO ###############")
    res2 = predecir_con_trayectoria(solicitud_2)
    imprimir_resultado(solicitud_2, res2)

    # Ejemplo 3: caso intermedio
    solicitud_3 = {
        "checking_balance": "1 - 200 DM",
        "months_loan_duration": 24,
        "credit_history": "repaid",
        "purpose": "furniture",
        "amount": 4500,
        "savings_balance": "101 - 500 DM",
        "employment_length": "1 - 4 yrs",
        "installment_rate": 3,
        "personal_status": "single male",
        "other_debtors": "none",
        "residence_history": 2,
        "property": "car or other",
        "age": 32,
        "installment_plan": "none",
        "housing": "own",
        "existing_credits": 1,
        "dependents": 1,
        "telephone": "yes",
        "foreign_worker": "yes",
        "job": "skilled employee",
    }
    print("\n\n############### EJEMPLO 3: PERFIL INTERMEDIO ###############")
    res3 = predecir_con_trayectoria(solicitud_3)
    imprimir_resultado(solicitud_3, res3)
