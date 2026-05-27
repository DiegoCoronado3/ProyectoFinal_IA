"""
Paso 10: Sistema Final - Logística (decide) + Árbol (explica)
================================================================
Implementación final del sistema productivo:
  - Recibe un CSV de solicitudes de crédito
  - La REGRESIÓN LOGÍSTICA decide aprobar/rechazar (mejor desempeño)
  - El ÁRBOL DE DECISIÓN provee la trayectoria de reglas como
    EXPLICACIÓN paralela del caso (cumple requisito regulatorio)
  - Adicionalmente, se reportan los TOP-3 factores de la Logística
    (coeficiente × valor) para complementar la justificación
  - Salida: CSV con decisiones + justificación completa por cliente

Diseño: surrogate explanation pattern
=====================================
La Logística es el modelo de decisión primario (mejor AUC = 0.81).
El Árbol funciona como "modelo explicativo paralelo": no justifica
internamente a la Logística, sino que genera una explicación
basada en reglas usando los MISMOS datos de entrada. Esto cumple
la exigencia regulatoria de "decisión justificable" aunque ambos
sistemas son paralelos.

Uso:
    python scripts/10_sistema_final.py
"""
import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ----------------------------------------------------------------------
# Cargar modelos y metadatos
# ----------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "arbol_decision.pkl"), "rb") as f:
    arbol = pickle.load(f)
with open(os.path.join(MODEL_DIR, "regresion_logistica.pkl"), "rb") as f:
    logreg = pickle.load(f)
with open(os.path.join(PROC_DIR, "scaler.pkl"), "rb") as f:
    meta = pickle.load(f)
scaler = meta["scaler"]
feature_names = meta["feature_names"]
numericas = meta["numericas"]
categoricas = meta["categoricas"]

UMBRAL = 0.5   # umbral de probabilidad para decidir aprobar/rechazar


def preparar_solicitud(solicitud_dict):
    """Aplica el mismo encoding usado en train."""
    df = pd.DataFrame([solicitud_dict])
    df_enc = pd.get_dummies(df, columns=categoricas, drop_first=False, dtype=int)
    for col in feature_names:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[feature_names]
    df_scaled = df_enc.copy()
    df_scaled[numericas] = scaler.transform(df_enc[numericas])
    return df_enc, df_scaled


def trayectoria_arbol(X_raw):
    """Reconstruye la trayectoria nodo por nodo del árbol."""
    tree = arbol.tree_
    node_indicator = arbol.decision_path(X_raw)
    leaf_id = arbol.apply(X_raw)[0]
    indices = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]]
    x_arr = X_raw.values[0]
    pasos = []
    for nid in indices:
        if leaf_id == nid:
            break
        feature = feature_names[tree.feature[nid]]
        threshold = tree.threshold[nid]
        value = x_arr[tree.feature[nid]]
        op = "<=" if value <= threshold else ">"
        pasos.append(f"{feature}={value:.2f}{op}{threshold:.2f}")
    return " ; ".join(pasos)


def top_factores_logistica(X_scaled, top_k=3):
    """Top-k features que más contribuyeron a la decisión (coef * valor)."""
    coefs = logreg.coef_[0]
    valores = X_scaled.values[0]
    contribuciones = coefs * valores
    idx_sorted = np.argsort(np.abs(contribuciones))[::-1][:top_k]
    factores = []
    for i in idx_sorted:
        feat = feature_names[i]
        direccion = "↑riesgo" if contribuciones[i] > 0 else "↓riesgo"
        factores.append(f"{feat}({direccion})")
    return " ; ".join(factores)


def procesar_solicitud(solicitud_dict, cliente_id=""):
    """Procesa una solicitud individual. Devuelve dict con la decisión completa."""
    X_raw, X_scaled = preparar_solicitud(solicitud_dict)

    # DECISIÓN PRIMARIA: Regresión Logística
    proba_log = float(logreg.predict_proba(X_scaled)[0, 1])
    decision = "RECHAZADO" if proba_log >= UMBRAL else "APROBADO"

    # EXPLICACIÓN COMPLEMENTARIA: Árbol (trayectoria) + Logística (top factores)
    explicacion_arbol = trayectoria_arbol(X_raw)
    factores_log = top_factores_logistica(X_scaled, top_k=3)

    # Proba paralela del árbol (informativa)
    proba_arbol = float(arbol.predict_proba(X_raw)[0, 1])

    return {
        "cliente_id": cliente_id,
        "decision": decision,
        "prob_default_logistica": round(proba_log, 4),
        "prob_default_arbol": round(proba_arbol, 4),
        "top3_factores_logistica": factores_log,
        "trayectoria_arbol": explicacion_arbol,
    }


def procesar_csv(input_csv, output_csv):
    """Procesa un CSV completo de solicitudes."""
    print(f"\n[sistema] Leyendo {input_csv}...")
    df_in = pd.read_csv(input_csv)
    print(f"          {len(df_in)} solicitudes a procesar.\n")

    resultados = []
    for _, row in df_in.iterrows():
        cliente_id = row.get("cliente_id", "")
        solicitud = row.drop("cliente_id", errors="ignore").to_dict()
        # Convertir tipos
        for col in numericas:
            if col in solicitud:
                solicitud[col] = float(solicitud[col])
        res = procesar_solicitud(solicitud, cliente_id=cliente_id)
        resultados.append(res)

    df_out = pd.DataFrame(resultados)

    # Merge con info original para mantener contexto
    df_final = df_in.merge(df_out, on="cliente_id")
    df_final.to_csv(output_csv, index=False)

    # Resumen consola
    n_aprob = (df_final["decision"] == "APROBADO").sum()
    n_rech = (df_final["decision"] == "RECHAZADO").sum()
    print("=" * 78)
    print("  RESUMEN")
    print("=" * 78)
    print(f"  Total procesados: {len(df_final)}")
    print(f"  APROBADOS:        {n_aprob}")
    print(f"  RECHAZADOS:       {n_rech}")
    print(f"\n  CSV de salida:    {output_csv}\n")

    print("=" * 78)
    print("  DECISIONES POR CLIENTE")
    print("=" * 78)
    for _, r in df_final.iterrows():
        emoji = "✅" if r["decision"] == "APROBADO" else "❌"
        print(f"\n{emoji}  {r['cliente_id']}  →  {r['decision']}  "
              f"(prob default: {r['prob_default_logistica']:.1%})")
        print(f"   Top factores (Logística): {r['top3_factores_logistica']}")
        print(f"   Trayectoria (Árbol):      {r['trayectoria_arbol'][:120]}...")

    return df_final


# ----------------------------------------------------------------------
# Main: procesar los 15 clientes de la demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    INPUT = os.path.join(PROC_DIR, "clientes_demo.csv")
    OUTPUT = os.path.join(RESULTS_DIR, "decisiones_clientes.csv")
    df_resultados = procesar_csv(INPUT, OUTPUT)

    # Validación contra ground truth (opcional, solo en demo)
    metadata = pd.read_csv(os.path.join(PROC_DIR, "clientes_demo_metadata.csv"))
    if "default_real" in metadata.columns:
        df_resultados = df_resultados.merge(
            metadata[["cliente_id", "default_real"]], on="cliente_id")
        df_resultados["real"] = df_resultados["default_real"].map(
            {0: "APROBADO", 1: "RECHAZADO"})
        df_resultados["acierta"] = (df_resultados["decision"] ==
                                     df_resultados["real"]).astype(int)
        print("\n" + "=" * 78)
        print(f"  VALIDACIÓN CONTRA GROUND TRUTH: "
              f"{df_resultados['acierta'].sum()}/{len(df_resultados)} aciertos "
              f"({df_resultados['acierta'].mean():.1%})")
        print("=" * 78)
