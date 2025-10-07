"""
Modelos Pydantic para Metric Grapher IA
Integración con MongoDB para persistencia de datos
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId class for Pydantic compatibility."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class ChartTypeEnum(str, Enum):
    """Tipos de gráficos disponibles."""
    BAR = "bar"
    LINE = "line" 
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    AREA = "area"
    VIOLIN = "violin"
    UNKNOWN = "unknown"


class LogLevelEnum(str, Enum):
    """Niveles de logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogCategoryEnum(str, Enum):
    """Categorías de logs."""
    SYSTEM = "system"
    DATABASE = "database"
    VISUALIZATION = "visualization"
    USER_ACTION = "user_action"
    API_CALL = "api_call"
    FILE_UPLOAD = "file_upload"
    PATTERN_DETECTION = "pattern_detection"
    ERROR = "error"


class DatasetMetadata(BaseModel):
    """Metadatos de un dataset."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        json_encoders={ObjectId: str}
    )
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str = Field(..., description="Nombre del dataset")
    original_filename: str = Field(..., description="Nombre original del archivo")
    format: str = Field(..., description="Formato del archivo (CSV, Excel, etc.)")
    size: int = Field(..., description="Número de filas")
    columns: List[str] = Field(..., description="Lista de columnas")
    column_types: Dict[str, str] = Field(..., description="Tipos de datos por columna")
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    file_size_bytes: int = Field(..., description="Tamaño del archivo en bytes")
    summary_stats: Optional[Dict[str, Any]] = Field(None, description="Estadísticas resumen")


class VisualizationRecord(BaseModel):
    """Registro de una visualización generada."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        json_encoders={ObjectId: str}
    )
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    dataset_id: Optional[PyObjectId] = Field(None, description="ID del dataset utilizado")
    dataset_name: str = Field(..., description="Nombre del dataset")
    query: str = Field(..., description="Consulta original del usuario")
    chart_type: ChartTypeEnum = Field(..., description="Tipo de gráfico generado")
    title: str = Field(..., description="Título del gráfico")
    code: str = Field(..., description="Código Python generado")
    figure_json: Optional[str] = Field(None, description="JSON de la figura de Plotly")
    insights: List[str] = Field(default_factory=list, description="Insights encontrados")
    suggestions: List[str] = Field(default_factory=list, description="Sugerencias adicionales")
    execution_time: float = Field(..., description="Tiempo de ejecución en segundos")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = Field(..., description="Modelo de Claude utilizado")
    success: bool = Field(True, description="Si la visualización fue exitosa")
    error_message: Optional[str] = Field(None, description="Mensaje de error si falló")


class UserSession(BaseModel):
    """Sesión de usuario."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        json_encoders={ObjectId: str}
    )
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    session_id: str = Field(..., description="ID único de la sesión")
    current_model: str = Field(default="sonnet4", description="Modelo actual de Claude")
    current_dataset: Optional[str] = Field(None, description="Dataset actualmente seleccionado")
    messages_count: int = Field(default=0, description="Número de mensajes en la sesión")
    visualizations_count: int = Field(default=0, description="Número de visualizaciones creadas")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Preferencias del usuario")


class VisualizationRequest(BaseModel):
    """Solicitud de visualización."""
    query: str = Field(..., description="Consulta del usuario")
    dataset_name: Optional[str] = Field(None, description="Nombre del dataset a usar")
    chart_type_hint: Optional[ChartTypeEnum] = Field(None, description="Sugerencia de tipo de gráfico")
    additional_context: Optional[str] = Field(None, description="Contexto adicional")
    session_id: Optional[str] = Field(None, description="ID de la sesión")


class DatabaseConfig(BaseModel):
    """Configuración de la base de datos."""
    host: str = Field(default="localhost", description="Host de MongoDB")
    port: int = Field(default=27017, description="Puerto de MongoDB")
    database_name: str = Field(default="anthropic-db", description="Nombre de la base de datos")
    collection_name: str = Field(default="metric-grapher-ia", description="Nombre de la colección")
    connection_timeout: int = Field(default=5000, description="Timeout de conexión en ms")


class APIResponse(BaseModel):
    """Respuesta estándar de la API."""
    success: bool = Field(..., description="Si la operación fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo")
    data: Optional[Dict[str, Any]] = Field(None, description="Datos de respuesta")
    error: Optional[str] = Field(None, description="Mensaje de error si aplica")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LogEntry(BaseModel):
    """Entrada de log del sistema."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        json_encoders={ObjectId: str}
    )
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp del log")
    level: LogLevelEnum = Field(..., description="Nivel del log")
    category: LogCategoryEnum = Field(..., description="Categoría del log")
    message: str = Field(..., description="Mensaje principal del log")
    details: Optional[Dict[str, Any]] = Field(None, description="Detalles adicionales")
    session_id: Optional[str] = Field(None, description="ID de la sesión")
    user_id: Optional[str] = Field(None, description="ID del usuario")
    function_name: Optional[str] = Field(None, description="Función donde ocurrió el evento")
    file_name: Optional[str] = Field(None, description="Archivo donde ocurrió el evento")
    line_number: Optional[int] = Field(None, description="Línea donde ocurrió el evento")
    stack_trace: Optional[str] = Field(None, description="Stack trace si es un error")
    duration_ms: Optional[float] = Field(None, description="Duración de la operación en ms")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadatos adicionales")
