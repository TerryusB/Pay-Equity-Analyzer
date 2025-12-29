import sys
import pandas as pd
import numpy as np
from tabulate import tabulate
import statsmodels.formula.api as smf
import os

# Configuración Global: Midpoints del Mercado (Basado en tu Diccionario)
MARKET_MIDPOINTS = {
    "N1": 20000, "N2": 32000, "N3": 50000, "N4": 75000, 
    "N5": 110000, "N6": 165000, "N7": 235000, "N8": 350000
}

def main():
    """
    Función principal que orquesta el Pay Equity Analysis completo.
    Solicita interactivamente el archivo y guarda resultados en la misma ubicación.
    """
    print("\n📊 --- Executive Search & Pay Equity Analyzer ---")
    
    # 1. Solicitar ruta del archivo CSV al usuario
    file_path = input("📂 Por favor, ingresa la ruta completa del archivo 'data.csv': ").strip()
    
    # Si el usuario presiona Enter sin escribir nada, intentamos buscar en local
    if not file_path:
        file_path = "data.csv"
        print(f"⚠️  No se ingresó ruta, buscando archivo local por defecto: '{file_path}'")

    # 2. Cargar Datos
    try:
        df = load_dataset(file_path)
        print(f"✅ Data Ingested: {len(df)} records loaded successfully.")
    except FileNotFoundError:
        sys.exit(f"❌ Error: No se encontró el archivo en: '{file_path}'.\n   -> Verifica la ruta o ejecuta generate_data.py primero.")
    except ValueError as e:
        sys.exit(f"❌ Error de Validación: {e}")
    
    # 3. Enriquecer con Métricas (Compa-Ratio)
    df = calculate_compa_ratio(df)
    
    # 4. Detectar Riesgos y Anomalías (IQR)
    outliers = detect_risks(df)
    
    # 5. Análisis Estadístico Avanzado (Regresión)
    print("\n🧠 --- RUNNING STATISTICAL REGRESSION MODEL ---")
    regression_results = run_regression_analysis(df)
    print(regression_results)
    
    # --- REPORTE EJECUTIVO EN CONSOLA ---
    print("\n🔎 --- PAY EQUITY HEALTH CHECK REPORT ---")
    
    # KPI 1: Distribución por Zonas de Riesgo
    risk_counts = df["risk_flag"].value_counts().reset_index()
    risk_counts.columns = ["Risk Zone", "Count"]
    print("\n1. Headcount by Risk Zone:")
    print(tabulate(risk_counts, headers="keys", tablefmt="simple_grid"))
    
    # KPI 2: Top Anomalías (Outliers)
    print("\n2. Critical Anomalies Detected (Sample of Statistical Outliers):")
    if not outliers.empty:
        cols_to_show = ["id", "job_family", "job_level", "annual_salary", "compa_ratio", "risk_flag"]
        print(tabulate(outliers[cols_to_show].head(5), headers="keys", tablefmt="fancy_grid"))
    else:
        print("No statistical outliers detected via IQR.")

    # 6. Guardar resultados en la MISMA dirección del archivo de entrada
    # Extraemos el directorio del archivo original
    directory = os.path.dirname(file_path)
    output_filename = "audit_final_results.csv"
    
    # Si directory está vacío (archivo local), usamos el nombre directo, si no, unimos rutas
    if directory:
        output_path = os.path.join(directory, output_filename)
    else:
        output_path = output_filename
        
    try:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Audit complete. Detailed results exported to:")
        print(f"   📂 {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"❌ Error al guardar resultados: {e}")


def load_dataset(path):
    """
    Función Obligatoria 1: Ingesta y Limpieza.
    """
    # Quitamos comillas extra si el usuario copió la ruta como "ruta"
    path = path.replace('"', '').replace("'", "")
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError
        
    required_cols = ["id", "job_level", "annual_salary", "job_family", "gender", "tenure_months", "performance_rating"]
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError("CSV is missing required columns for equity analysis.")
    
    df["annual_salary"] = pd.to_numeric(df["annual_salary"], errors="coerce")
    df = df.dropna(subset=["annual_salary"])
    
    return df


def calculate_compa_ratio(df):
    """
    Función Obligatoria 2: Lógica de Negocio (Compa-Ratio).
    """
    df = df.copy()
    df["market_midpoint"] = df["job_level"].map(MARKET_MIDPOINTS)
    df["compa_ratio"] = round(df["annual_salary"] / df["market_midpoint"], 2)
    
    conditions = [
        (df["compa_ratio"] < 0.80),
        (df["compa_ratio"] > 1.20)
    ]
    choices = ["Critical: Underpaid", "Alert: Overpaid"]
    df["risk_flag"] = np.select(conditions, choices, default="Equitable")
    
    return df


def detect_risks(df):
    """
    Función Obligatoria 3: Detección de Anomalías (IQR).
    """
    Q1 = df["annual_salary"].quantile(0.25)
    Q3 = df["annual_salary"].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df["annual_salary"] < lower_bound) | (df["annual_salary"] > upper_bound)]
    return outliers.sort_values(by="compa_ratio")


def run_regression_analysis(df):
    """
    Función Adicional (Advanced): Regresión Lineal Múltiple.
    Calcula la brecha ajustada controlando por Nivel, Familia, Antigüedad y Desempeño.
    """
    # Transformación Logarítmica para normalizar salarios (Mejor práctica en econometría)
    # Evita error de log(0) sumando 1 si fuera necesario, aunque ya filtramos >0
    df["log_salary"] = np.log(df["annual_salary"])
    
    # Fórmula: Salario ~ Nivel + Familia + Género + Antigüedad + Desempeño
    # Esto aísla el impacto puro del género.
    formula = 'log_salary ~ C(job_level) + C(job_family) + C(gender) + tenure_months + performance_rating'
    
    # Ajustar modelo OLS (Ordinary Least Squares)
    model = smf.ols(formula=formula, data=df).fit()
    
    # Retornamos el resumen estadístico que impresiona a los reclutadores
    return model.summary()

if __name__ == "__main__":
    main()