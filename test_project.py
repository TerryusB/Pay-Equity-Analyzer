import pytest
import pandas as pd
import os
from project import load_dataset, calculate_compa_ratio, detect_risks

# --- CONFIGURACIÓN DE DATOS DE PRUEBA (MOCKS) ---

@pytest.fixture
def mock_df_metrics():
    """
    Crea un DataFrame simulado para probar la lógica de Compa-Ratio.
    Usamos el Nivel N1 cuyo Midpoint es 20,000 según tu diccionario.
    """
    return pd.DataFrame({
        "id": ["E1", "E2", "E3"],
        "job_level": ["N1", "N1", "N1"], # Midpoint esperado: 20,000
        "annual_salary": [20000, 10000, 30000], # Equitable, Underpaid, Overpaid
        "job_family": ["IT", "HR", "Sales"]
    })

@pytest.fixture
def mock_df_outliers():
    """
    Crea un DataFrame diseñado para tener un outlier estadístico claro.
    Salarios muy estables (20k) y uno gigante (1M).
    """
    return pd.DataFrame({
        "id": ["E1", "E2", "E3", "E4", "E5", "OUTLIER"],
        "job_level": ["N1"] * 6,
        # Cinco salarios idénticos y uno absurdo
        "annual_salary": [20000, 20000, 20000, 20000, 20000, 1000000],
        "job_family": ["IT"] * 6,
        "compa_ratio": [1.0] * 6 # Dummy value para que no falle el sort
    })


# --- TEST 1: INGESTA DE DATOS (load_dataset) ---

def test_load_dataset_valid():
    """Verifica que cargue correctamente un CSV temporal."""
    # 1. Crear CSV temporal
    temp_csv = "temp_test_data.csv"
    df = pd.DataFrame({
        "id": ["T1"], "job_level": ["N1"], "annual_salary": [50000],
        "job_family": ["IT"], "gender": ["M"], "tenure_months": [12],
        "performance_rating": [3.0]
    })
    df.to_csv(temp_csv, index=False)

    # 2. Probar función
    loaded_df = load_dataset(temp_csv)
    
    # 3. Validaciones
    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]["id"] == "T1"
    
    # 4. Limpieza (Borrar archivo temporal)
    os.remove(temp_csv)

def test_load_dataset_file_not_found():
    """Verifica que lance el error correcto si el archivo no existe."""
    with pytest.raises(FileNotFoundError):
        load_dataset("archivo_imaginario_123.csv")

def test_load_dataset_missing_columns():
    """Verifica que lance error si faltan columnas críticas."""
    temp_csv = "bad_columns.csv"
    # CSV sin la columna 'annual_salary'
    df = pd.DataFrame({"id": ["T1"], "job_family": ["IT"]}) 
    df.to_csv(temp_csv, index=False)
    
    with pytest.raises(ValueError):
        load_dataset(temp_csv)
    
    os.remove(temp_csv)


# --- TEST 2: LÓGICA DE NEGOCIO (calculate_compa_ratio) ---

def test_calculate_compa_ratio(mock_df_metrics):
    """
    Prueba la matemática del Compa-Ratio y las banderas de riesgo.
    Nivel N1 -> Midpoint 20,000.
    """
    result = calculate_compa_ratio(mock_df_metrics)
    
    # Caso 1: Salario 20,000 (Exacto) -> Ratio 1.0 -> Equitable
    row_eq = result[result["id"] == "E1"].iloc[0]
    assert row_eq["compa_ratio"] == 1.0
    assert row_eq["risk_flag"] == "Equitable"
    
    # Caso 2: Salario 10,000 (Mitad) -> Ratio 0.5 -> Critical: Underpaid
    row_under = result[result["id"] == "E2"].iloc[0]
    assert row_under["compa_ratio"] == 0.5
    assert row_under["risk_flag"] == "Critical: Underpaid"
    
    # Caso 3: Salario 30,000 (1.5x) -> Ratio 1.5 -> Alert: Overpaid
    row_over = result[result["id"] == "E3"].iloc[0]
    assert row_over["compa_ratio"] == 1.5
    assert row_over["risk_flag"] == "Alert: Overpaid"


# --- TEST 3: ESTADÍSTICA (detect_risks) ---

def test_detect_risks_iqr(mock_df_outliers):
    """
    Verifica que el método IQR detecte el salario de 1,000,000
    cuando el resto son de 20,000.
    """
    outliers = detect_risks(mock_df_outliers)
    
    # Debería detectar al menos 1 outlier
    assert len(outliers) == 1
    # Ese outlier debe ser el ID 'OUTLIER'
    assert outliers.iloc[0]["id"] == "OUTLIER"
    assert outliers.iloc[0]["annual_salary"] == 1000000