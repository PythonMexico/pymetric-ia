"""
Utilidades y funciones auxiliares para el proyecto.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re
import json
from datetime import datetime


def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detecta tipos de datos más específicos para un DataFrame.
    
    Args:
        df: DataFrame de pandas
        
    Returns:
        Diccionario con tipos de datos detectados
    """
    type_mapping = {}
    
    for column in df.columns:
        col_data = df[column].dropna()
        
        if len(col_data) == 0:
            type_mapping[column] = "empty"
            continue
        
        # Detectar tipos específicos
        if pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data):
                type_mapping[column] = "integer"
            else:
                type_mapping[column] = "float"
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            type_mapping[column] = "datetime"
        elif pd.api.types.is_bool_dtype(col_data):
            type_mapping[column] = "boolean"
        else:
            # Intentar detectar patrones específicos
            sample_values = col_data.astype(str).head(100)
            
            # Detectar fechas en formato string
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            ]
            
            is_date = any(
                sample_values.str.match(pattern).sum() > len(sample_values) * 0.8
                for pattern in date_patterns
            )
            
            if is_date:
                type_mapping[column] = "date_string"
            elif col_data.nunique() / len(col_data) < 0.1:  # Baja cardinalidad
                type_mapping[column] = "categorical"
            else:
                type_mapping[column] = "text"
    
    return type_mapping


def suggest_chart_types(df: pd.DataFrame, x_col: str = None, y_col: str = None) -> List[Dict[str, Any]]:
    """
    Sugiere tipos de gráficos basados en los tipos de datos.
    
    Args:
        df: DataFrame
        x_col: Columna para eje X (opcional)
        y_col: Columna para eje Y (opcional)
        
    Returns:
        Lista de sugerencias de gráficos
    """
    suggestions = []
    data_types = detect_data_types(df)
    
    numeric_cols = [col for col, dtype in data_types.items() if dtype in ['integer', 'float']]
    categorical_cols = [col for col, dtype in data_types.items() if dtype in ['categorical', 'text']]
    datetime_cols = [col for col, dtype in data_types.items() if dtype in ['datetime', 'date_string']]
    
    # Sugerencias basadas en combinaciones de tipos
    if len(numeric_cols) >= 2:
        suggestions.append({
            "type": "scatter",
            "title": "Gráfico de Dispersión",
            "description": f"Relación entre {numeric_cols[0]} y {numeric_cols[1]}",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "confidence": 0.9
        })
        
        suggestions.append({
            "type": "heatmap",
            "title": "Mapa de Calor de Correlación",
            "description": "Correlaciones entre variables numéricas",
            "confidence": 0.8
        })
    
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        suggestions.append({
            "type": "bar",
            "title": "Gráfico de Barras",
            "description": f"{numeric_cols[0]} por {categorical_cols[0]}",
            "x": categorical_cols[0],
            "y": numeric_cols[0],
            "confidence": 0.9
        })
        
        suggestions.append({
            "type": "box",
            "title": "Diagrama de Caja",
            "description": f"Distribución de {numeric_cols[0]} por {categorical_cols[0]}",
            "x": categorical_cols[0],
            "y": numeric_cols[0],
            "confidence": 0.8
        })
    
    if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
        suggestions.append({
            "type": "line",
            "title": "Serie Temporal",
            "description": f"Evolución de {numeric_cols[0]} en el tiempo",
            "x": datetime_cols[0],
            "y": numeric_cols[0],
            "confidence": 0.95
        })
    
    if len(numeric_cols) >= 1:
        suggestions.append({
            "type": "histogram",
            "title": "Histograma",
            "description": f"Distribución de {numeric_cols[0]}",
            "x": numeric_cols[0],
            "confidence": 0.8
        })
    
    if len(categorical_cols) >= 1:
        # Verificar si es adecuado para gráfico de pastel
        if len(categorical_cols) >= 1:
            unique_values = df[categorical_cols[0]].nunique()
            if unique_values <= 8:  # Máximo 8 categorías para pie chart
                suggestions.append({
                    "type": "pie",
                    "title": "Gráfico de Pastel",
                    "description": f"Distribución de {categorical_cols[0]}",
                    "values": categorical_cols[0],
                    "confidence": 0.7
                })
    
    # Ordenar por confianza
    suggestions.sort(key=lambda x: x["confidence"], reverse=True)
    
    return suggestions


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia un DataFrame aplicando transformaciones básicas.
    
    Args:
        df: DataFrame a limpiar
        
    Returns:
        DataFrame limpio
    """
    df_clean = df.copy()
    
    # Convertir columnas de texto a datetime si es posible
    for col in df_clean.select_dtypes(include=['object']).columns:
        # Intentar conversión a datetime
        try:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='ignore')
        except:
            pass
    
    # Limpiar nombres de columnas
    df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_').str.lower()
    
    return df_clean


def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera un resumen completo de un DataFrame.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Diccionario con resumen completo
    """
    summary = {
        "basic_info": {
            "shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "dtypes": df.dtypes.to_dict()
        },
        "missing_data": {
            "total_missing": df.isnull().sum().sum(),
            "missing_by_column": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict()
        },
        "data_types": detect_data_types(df),
        "suggestions": suggest_chart_types(df)
    }
    
    # Estadísticas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    # Estadísticas categóricas
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_stats = {}
    for col in categorical_cols:
        categorical_stats[col] = {
            "unique_count": df[col].nunique(),
            "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
            "top_values": df[col].value_counts().head(5).to_dict()
        }
    
    if categorical_stats:
        summary["categorical_stats"] = categorical_stats
    
    return summary


def format_number(num: float, decimals: int = 2) -> str:
    """Formatea números para mostrar de manera legible."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.{decimals}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def extract_code_from_text(text: str) -> Optional[str]:
    """
    Extrae código Python de un texto que puede contener markdown.
    
    Args:
        text: Texto que puede contener código
        
    Returns:
        Código Python extraído o None
    """
    # Buscar bloques de código Python
    python_pattern = r'```python\n(.*?)\n```'
    match = re.search(python_pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # Buscar bloques de código genéricos
    code_pattern = r'```\n(.*?)\n```'
    match = re.search(code_pattern, text, re.DOTALL)
    
    if match:
        code = match.group(1).strip()
        # Verificar si parece código Python
        if any(keyword in code for keyword in ['import', 'def', 'plt.', 'px.', 'go.']):
            return code
    
    return None


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida un DataFrame y retorna información sobre posibles problemas.
    
    Args:
        df: DataFrame a validar
        
    Returns:
        Diccionario con información de validación
    """
    issues = []
    warnings = []
    
    # Verificar tamaño
    if len(df) == 0:
        issues.append("DataFrame está vacío")
    elif len(df) < 10:
        warnings.append("DataFrame tiene muy pocas filas para análisis estadístico")
    
    # Verificar columnas duplicadas
    if df.columns.duplicated().any():
        issues.append("Hay nombres de columnas duplicados")
    
    # Verificar filas completamente vacías
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        warnings.append(f"Hay {empty_rows} filas completamente vacías")
    
    # Verificar columnas completamente vacías
    empty_cols = df.isnull().all().sum()
    if empty_cols > 0:
        warnings.append(f"Hay {empty_cols} columnas completamente vacías")
    
    # Verificar alta cardinalidad en columnas categóricas
    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9:
            warnings.append(f"Columna '{col}' tiene alta cardinalidad ({unique_ratio:.1%})")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "recommendations": _generate_recommendations(df, issues, warnings)
    }


def _generate_recommendations(df: pd.DataFrame, issues: List[str], warnings: List[str]) -> List[str]:
    """Genera recomendaciones basadas en los problemas encontrados."""
    recommendations = []
    
    if "DataFrame está vacío" in issues:
        recommendations.append("Verificar el archivo de datos y el proceso de carga")
    
    if any("duplicados" in issue for issue in issues):
        recommendations.append("Renombrar columnas duplicadas antes del análisis")
    
    if any("completamente vacías" in warning for warning in warnings):
        recommendations.append("Considerar eliminar filas/columnas vacías")
    
    if any("alta cardinalidad" in warning for warning in warnings):
        recommendations.append("Considerar agrupar categorías o usar muestreo para columnas de alta cardinalidad")
    
    # Recomendaciones generales
    missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_percentage > 0.1:
        recommendations.append("Alto porcentaje de datos faltantes - considerar estrategias de imputación")
    
    return recommendations
