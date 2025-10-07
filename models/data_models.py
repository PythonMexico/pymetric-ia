"""
Modelos para manejo de datos y visualizaciones.
Define las estructuras de datos principales del sistema.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import pandas as pd
import json
from datetime import datetime


class ChartType(Enum):
    """Tipos de gráficos soportados."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    PIE = "pie"
    AREA = "area"
    VIOLIN = "violin"
    SUNBURST = "sunburst"
    TREEMAP = "treemap"
    CANDLESTICK = "candlestick"


class DataFormat(Enum):
    """Formatos de datos soportados."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"


@dataclass
class DatasetInfo:
    """Información sobre un dataset."""
    name: str
    format: DataFormat
    size: int  # número de filas
    columns: List[str]
    column_types: Dict[str, str]
    memory_usage: float  # en MB
    upload_time: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            "name": self.name,
            "format": self.format.value,
            "size": self.size,
            "columns": self.columns,
            "column_types": self.column_types,
            "memory_usage": self.memory_usage,
            "upload_time": self.upload_time.isoformat(),
            "description": self.description
        }


@dataclass
class VisualizationRequest:
    """Solicitud de visualización del usuario."""
    query: str
    dataset_name: Optional[str] = None
    chart_type: Optional[ChartType] = None
    columns: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    custom_options: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "query": self.query,
            "dataset_name": self.dataset_name,
            "chart_type": self.chart_type.value if self.chart_type else None,
            "columns": self.columns,
            "filters": self.filters,
            "custom_options": self.custom_options
        }


@dataclass
class VisualizationResult:
    chart_type: ChartType
    title: str
    code: str  # Código Python generado
    figure_json: Optional[str] = None  # JSON del gráfico Plotly
    insights: Optional[List[str]] = None  # Insights encontrados
    suggestions: Optional[List[str]] = None  # Sugerencias de mejora
    execution_time: Optional[float] = None  # Tiempo de ejecución en segundos
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "chart_type": self.chart_type.value,
            "title": self.title,
            "code": self.code,
            "figure_json": self.figure_json,
            "insights": self.insights,
            "suggestions": self.suggestions,
            "execution_time": self.execution_time,
            "error": self.error,
            "created_at": self.created_at.isoformat()
        }
