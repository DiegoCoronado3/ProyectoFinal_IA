"""
Paso 1: Descarga del dataset German Credit y Análisis Exploratorio (EDA)
=========================================================================
Proyecto: Sistema Automatizado de Evaluación Crediticia
Autor: Coronado Pérez Diego
Materia: Inteligencia Artificial

Uso:
    python 01_descarga_y_eda.py

Genera:
    ../data/german_credit_raw.csv          (dataset original)
    ../results/figures/01_*.png .. 04_*.png (gráficas)
"""
import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Rutas relativas (compatibles con cualquier Mac/Linux)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DATASET_URL = ("https://raw.githubusercontent.com/stedy/"
               "Machine-Learning-with-R-datasets/master/credit.csv")
DATASET_PATH = os.path.join(DATA_DIR, "german_credit_raw.csv")

# Paleta consistente
COLORS = {"no_default": "#2E7D8F", "default": "#C0392B"}


# ----------------------------------------------------------------------
# 1) Descarga
# ----------------------------------------------------------------------
def descargar_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"[skip] Dataset ya existe en {DATASET_PATH}")
        return
    print(f"[download] {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    print(f"[ok] Guardado en {DATASET_PATH}")


# ----------------------------------------------------------------------
# 2) Carga y exploración básica
# ----------------------------------------------------------------------
def cargar_dataset():
    df = pd.read_csv(DATASET_PATH)
    # Recodificación de target a convención de credit scoring:
    #   1 (good) -> 0 (no default)
    #   2 (bad)  -> 1 (default, clase positiva a detectar)
    df["default"] = (df["default"] == 2).astype(int)
    return df


def resumen(df):
    print("\n" + "=" * 60)
    print(f"Dimensiones del dataset: {df.shape}")
    print("=" * 60)
    print(f"\nNaNs totales: {df.isna().sum().sum()}")
    print(f"\nDistribución de la variable objetivo:")
    print(df["default"].value_counts().rename({0: "No default (good)", 1: "Default (bad)"}))
    print(f"Tasa de default: {df['default'].mean():.2%}")

    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    numericas.remove("default")
    categoricas = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nVariables numéricas ({len(numericas)}): {numericas}")
    print(f"\nVariables categóricas ({len(categoricas)}):")
    for c in categoricas:
        print(f"  - {c}: {df[c].nunique()} valores únicos")

    print("\n--- Estadísticas descriptivas (numéricas) ---")
    print(df[numericas].describe().round(2))
    return numericas, categoricas


# ----------------------------------------------------------------------
# 3) Gráficas
# ----------------------------------------------------------------------
def fig_distribucion_target(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df["default"].value_counts().sort_index()
    labels = ["No Default\n(Buen crédito)", "Default\n(Mal crédito)"]
    bars = ax.bar(labels, counts.values,
                  color=[COLORS["no_default"], COLORS["default"]],
                  edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val}\n({val/len(df)*100:.0f}%)", ha="center",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("Cantidad de solicitudes", fontsize=11)
    ax.set_title("Distribución de la variable objetivo (n=1000)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 850)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_distribucion_target.png"),
                dpi=130, bbox_inches="tight")
    plt.close()


def fig_distribuciones_numericas(df):
    numericas = ["months_loan_duration", "amount", "installment_rate",
                 "age", "residence_history", "existing_credits"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, numericas):
        ax.hist(df[df["default"] == 0][col], bins=25, alpha=0.65,
                label="No default", color=COLORS["no_default"],
                edgecolor="black", linewidth=0.3)
        ax.hist(df[df["default"] == 1][col], bins=25, alpha=0.65,
                label="Default", color=COLORS["default"],
                edgecolor="black", linewidth=0.3)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_ylabel("Frecuencia", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Distribución de variables numéricas por clase",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_distribuciones_numericas.png"),
                dpi=130, bbox_inches="tight")
    plt.close()


def fig_tasa_default_categoricas(df):
    cats = ["checking_balance", "credit_history", "employment_length", "savings_balance"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, col in zip(axes.flat, cats):
        rates = df.groupby(col)["default"].mean().sort_values(ascending=False)
        ax.bar(range(len(rates)), rates.values, color=COLORS["default"],
               alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.axhline(0.30, color="black", linestyle="--", linewidth=1,
                   label="Tasa global (30%)")
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([str(x) for x in rates.index],
                           rotation=25, ha="right", fontsize=8)
        ax.set_title(f"Tasa de default por {col}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Tasa de default", fontsize=9)
        ax.set_ylim(0, max(rates.values) * 1.15)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_tasa_default_categoricas.png"),
                dpi=130, bbox_inches="tight")
    plt.close()


def fig_correlacion(df):
    num_cols = ["months_loan_duration", "amount", "installment_rate",
                "residence_history", "age", "existing_credits",
                "dependents", "default"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                    fontsize=8.5)
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title("Matriz de correlación (variables numéricas)",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_correlacion.png"),
                dpi=130, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    descargar_dataset()
    df = cargar_dataset()
    numericas, categoricas = resumen(df)

    print("\nGenerando gráficas...")
    fig_distribucion_target(df)
    fig_distribuciones_numericas(df)
    fig_tasa_default_categoricas(df)
    fig_correlacion(df)
    print(f"[ok] Gráficas guardadas en {FIG_DIR}")
