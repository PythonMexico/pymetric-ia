"""
Controlador para manejo de datos.
Gestiona la carga, procesamiento y análisis de datasets.
"""
import pandas as pd
import numpy as np
import io
import json
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import tempfile
import os

from models.data_models import DatasetInfo, DataFormat
from models.enhanced_data_store import EnhancedDataStore


class DataController:
    """Controlador para operaciones con datos."""
    
    def __init__(self):
        self.data_store = EnhancedDataStore()
    
    def diagnose_file_content(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Diagnostica el contenido de un archivo para identificar problemas."""
        try:
            # Información básica del archivo
            file_size = len(file_content)
            
            # Intentar decodificar como texto
            try:
                text_content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text_content = file_content.decode('latin-1')
                except:
                    text_content = file_content.decode('iso-8859-1', errors='ignore')
            
            # Analizar las primeras líneas
            lines = text_content.split('\n')[:10]
            
            return {
                "file_size": file_size,
                "total_lines": len(text_content.split('\n')),
                "first_lines": lines,
                "encoding_detected": "utf-8" if file_content.decode('utf-8', errors='ignore') else "other",
                "possible_separators": self._detect_separators(lines[0] if lines else "")
            }
        except Exception as e:
            return {"error": f"Error al diagnosticar archivo: {str(e)}"}
    
    def _detect_separators(self, first_line: str) -> Dict[str, int]:
        """Detecta posibles separadores en la primera línea."""
        separators = {',': 0, ';': 0, '\t': 0, '|': 0}
        for sep in separators:
            separators[sep] = first_line.count(sep)
        return separators
    
    async def load_dataset_from_file(self, file_content: bytes, filename: str, 
                                   description: Optional[str] = None) -> Dict[str, Any]:
        """
        Carga un dataset desde un archivo.
        
        Args:
            file_content: Contenido del archivo en bytes
            filename: Nombre del archivo
            description: Descripción opcional del dataset
            
        Returns:
            Dict con información del dataset cargado o error
        """
        try:
            # Determinar formato del archivo
            file_extension = Path(filename).suffix.lower()
            
            if file_extension == '.csv':
                # Intentar diferentes configuraciones para CSV
                try:
                    # Primero intentar con configuración estándar
                    df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
                except UnicodeDecodeError:
                    # Intentar con encoding latin-1
                    try:
                        df = pd.read_csv(io.BytesIO(file_content), encoding='latin-1')
                    except:
                        df = pd.read_csv(io.BytesIO(file_content), encoding='iso-8859-1')
                except pd.errors.EmptyDataError:
                    return {
                        "success": False,
                        "error": "El archivo CSV está vacío"
                    }
                except pd.errors.ParserError:
                    # Intentar con diferentes separadores
                    try:
                        df = pd.read_csv(io.BytesIO(file_content), sep=';', encoding='utf-8')
                    except:
                        try:
                            df = pd.read_csv(io.BytesIO(file_content), sep='\t', encoding='utf-8')
                        except:
                            return {
                                "success": False,
                                "error": "No se pudo determinar el formato del archivo CSV. Verifica que tenga columnas válidas."
                            }
                
                # Verificar que el DataFrame no esté vacío
                if df.empty:
                    return {
                        "success": False,
                        "error": "El archivo CSV no contiene datos válidos"
                    }
                
                data_format = DataFormat.CSV
            elif file_extension in ['.xlsx', '.xls']:
                try:
                    df = pd.read_excel(io.BytesIO(file_content))
                    if df.empty:
                        return {
                            "success": False,
                            "error": "El archivo Excel no contiene datos válidos"
                        }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Error al leer archivo Excel: {str(e)}"
                    }
                data_format = DataFormat.EXCEL
            elif file_extension == '.json':
                json_data = json.loads(file_content.decode('utf-8'))
                df = pd.json_normalize(json_data)
                data_format = DataFormat.JSON
            elif file_extension == '.parquet':
                df = pd.read_parquet(io.BytesIO(file_content))
                data_format = DataFormat.PARQUET
            else:
                return {
                    "success": False,
                    "error": f"Formato de archivo no soportado: {file_extension}"
                }
            
            # Crear información del dataset
            dataset_name = Path(filename).stem
            dataset_info = DatasetInfo(
                name=dataset_name,
                format=data_format,
                size=len(df),
                columns=df.columns.tolist(),
                column_types={col: str(df[col].dtype) for col in df.columns},
                memory_usage=df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                description=description
            )
            
            # Guardar en el almacén
            self.data_store.add_dataset(dataset_name, df, dataset_info)
            
            return {
                "success": True,
                "dataset_info": dataset_info.to_dict(),
                "preview": self._get_dataset_preview(df)
            }
            
        except Exception as e:
            # Incluir diagnóstico del archivo en caso de error
            diagnosis = self.diagnose_file_content(file_content, filename)
            return {
                "success": False,
                "error": f"Error al cargar el archivo: {str(e)}",
                "diagnosis": diagnosis
            }
    
    def get_dataset_summary(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene un resumen estadístico del dataset."""
        df = self.data_store.get_dataset(dataset_name)
        if df is None:
            return None
        
        try:
            summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_summary": {},
                "categorical_summary": {}
            }
            
            # Resumen de columnas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
            
            # Resumen de columnas categóricas
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                summary["categorical_summary"][col] = {
                    "unique_values": df[col].nunique(),
                    "top_values": df[col].value_counts().head().to_dict()
                }
            
            return summary
            
        except Exception as e:
            return {"error": f"Error al generar resumen: {str(e)}"}
    
    def get_column_analysis(self, dataset_name: str, column_name: str) -> Optional[Dict[str, Any]]:
        """Análisis detallado de una columna específica."""
        df = self.data_store.get_dataset(dataset_name)
        if df is None or column_name not in df.columns:
            return None
        
        try:
            col_data = df[column_name]
            analysis = {
                "name": column_name,
                "dtype": str(col_data.dtype),
                "total_count": len(col_data),
                "null_count": col_data.isnull().sum(),
                "null_percentage": (col_data.isnull().sum() / len(col_data)) * 100
            }
            
            if col_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                # Análisis numérico
                analysis.update({
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "mean": col_data.mean(),
                    "median": col_data.median(),
                    "std": col_data.std(),
                    "quartiles": {
                        "q25": col_data.quantile(0.25),
                        "q75": col_data.quantile(0.75)
                    }
                })
            else:
                # Análisis categórico
                analysis.update({
                    "unique_count": col_data.nunique(),
                    "most_frequent": col_data.mode().iloc[0] if not col_data.mode().empty else None,
                    "value_counts": col_data.value_counts().head(10).to_dict()
                })
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error al analizar columna: {str(e)}"}
    
    def filter_dataset(self, dataset_name: str, filters: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Aplica filtros a un dataset."""
        df = self.data_store.get_dataset(dataset_name)
        if df is None:
            return None
        
        try:
            filtered_df = df.copy()
            
            for column, filter_config in filters.items():
                if column not in df.columns:
                    continue
                
                filter_type = filter_config.get("type")
                value = filter_config.get("value")
                
                if filter_type == "equals":
                    filtered_df = filtered_df[filtered_df[column] == value]
                elif filter_type == "greater_than":
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif filter_type == "less_than":
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif filter_type == "between":
                    min_val, max_val = value
                    filtered_df = filtered_df[
                        (filtered_df[column] >= min_val) & 
                        (filtered_df[column] <= max_val)
                    ]
                elif filter_type == "contains":
                    filtered_df = filtered_df[
                        filtered_df[column].astype(str).str.contains(value, na=False)
                    ]
                elif filter_type == "in":
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
            
            return filtered_df
            
        except Exception as e:
            print(f"Error al filtrar dataset: {str(e)}")
            return df
    
    def get_suggested_visualizations(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Sugiere visualizaciones basadas en el tipo de datos."""
        df = self.data_store.get_dataset(dataset_name)
        if df is None:
            return []
        
        suggestions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Sugerencias basadas en tipos de columnas
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "scatter",
                "title": "Gráfico de dispersión",
                "description": "Relación entre variables numéricas",
                "columns": numeric_cols[:2]
            })
        
        if len(numeric_cols) >= 1:
            suggestions.append({
                "type": "histogram",
                "title": "Histograma",
                "description": "Distribución de variable numérica",
                "columns": [numeric_cols[0]]
            })
        
        if len(categorical_cols) >= 1:
            suggestions.append({
                "type": "bar",
                "title": "Gráfico de barras",
                "description": "Frecuencia de categorías",
                "columns": [categorical_cols[0]]
            })
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions.append({
                "type": "box",
                "title": "Diagrama de caja",
                "description": "Distribución por categoría",
                "columns": [categorical_cols[0], numeric_cols[0]]
            })
        
        if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions.append({
                "type": "line",
                "title": "Serie temporal",
                "description": "Evolución temporal",
                "columns": [datetime_cols[0], numeric_cols[0]]
            })
        
        return suggestions
    
    def _get_dataset_preview(self, df: pd.DataFrame, rows: int = 5) -> Dict[str, Any]:
        """Obtiene una vista previa del dataset."""
        return {
            "head": df.head(rows).to_dict('records'),
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict()
        }
    
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """Lista todos los datasets disponibles con su información."""
        datasets = []
        for name in self.data_store.list_datasets():
            metadata = self.data_store.get_dataset_metadata(name)
            if metadata:
                # Crear estructura compatible con el formato esperado
                dataset_info = {
                    'name': name,
                    'size': metadata.get('size', 0),
                    'columns': metadata.get('columns', []),
                    'format': metadata.get('format', 'Unknown')
                }
                datasets.append(dataset_info)
        return datasets
