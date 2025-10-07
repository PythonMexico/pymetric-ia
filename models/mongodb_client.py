"""
Cliente MongoDB para Metric Grapher IA
Maneja la conexión y operaciones básicas con MongoDB
"""

import os
from typing import Optional, List, Dict, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from datetime import datetime
import logging

from .pydantic_models import DatabaseConfig, DatasetMetadata, VisualizationRecord, UserSession

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBClient:
    """Cliente singleton para MongoDB."""
    
    _instance = None
    _client = None
    _database = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config = DatabaseConfig()
            self.initialized = True
            self._connect()
    
    def _connect(self):
        """Establece conexión con MongoDB."""
        try:
            connection_string = f"mongodb://{self.config.host}:{self.config.port}/"
            self._client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=self.config.connection_timeout
            )
            
            # Verificar conexión
            self._client.server_info()
            self._database = self._client[self.config.database_name]
            
            logger.info(f"Conectado a MongoDB: {self.config.database_name}")
            
            # Crear índices si no existen
            self._create_indexes()
            
        except Exception as e:
            logger.error(f"Error conectando a MongoDB: {str(e)}")
            raise
    
    def _create_indexes(self):
        """Crea índices necesarios para optimizar consultas."""
        try:
            # Índices para datasets
            datasets_collection = self.get_collection("datasets")
            datasets_collection.create_index("name", unique=True)
            datasets_collection.create_index("upload_date")
            
            # Índices para visualizaciones
            viz_collection = self.get_collection("visualizations")
            viz_collection.create_index("dataset_id")
            viz_collection.create_index("created_at")
            viz_collection.create_index("chart_type")
            
            # Índices para sesiones
            sessions_collection = self.get_collection("sessions")
            sessions_collection.create_index("session_id", unique=True)
            sessions_collection.create_index("last_activity")
            
            logger.info("Índices creados correctamente")
            
        except Exception as e:
            logger.warning(f"Error creando índices: {str(e)}")
    
    def get_database(self) -> Database:
        """Retorna la instancia de la base de datos."""
        if self._database is None:
            self._connect()
        return self._database
    
    def get_collection(self, collection_name: str) -> Collection:
        """Retorna una colección específica."""
        return self.get_database()[collection_name]
    
    def is_connected(self) -> bool:
        """Verifica si la conexión está activa."""
        try:
            self._client.server_info()
            return True
        except Exception:
            return False
    
    def close_connection(self):
        """Cierra la conexión a MongoDB."""
        if self._client:
            self._client.close()
            logger.info("🔌 Conexión a MongoDB cerrada")


class DatasetRepository:
    """Repositorio para operaciones con datasets."""
    
    def __init__(self):
        self.client = MongoDBClient()
        self.collection = self.client.get_collection("datasets")
    
    async def save_dataset(self, dataset: DatasetMetadata) -> str:
        """Guarda un dataset en MongoDB."""
        try:
            dataset_dict = dataset.model_dump(by_alias=True, exclude_unset=True)
            result = self.collection.insert_one(dataset_dict)
            logger.info(f"📊 Dataset guardado: {dataset.name}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error guardando dataset: {str(e)}")
            raise
    
    async def get_dataset(self, dataset_name: str) -> Optional[DatasetMetadata]:
        """Obtiene un dataset por nombre."""
        try:
            doc = self.collection.find_one({"name": dataset_name})
            if doc:
                return DatasetMetadata(**doc)
            return None
        except Exception as e:
            logger.error(f"Error obteniendo dataset: {str(e)}")
            return None
    
    async def list_datasets(self) -> List[DatasetMetadata]:
        """Lista todos los datasets."""
        try:
            docs = list(self.collection.find().sort("upload_date", -1))
            return [DatasetMetadata(**doc) for doc in docs]
        except Exception as e:
            logger.error(f"Error listando datasets: {str(e)}")
            return []
    
    async def delete_dataset(self, dataset_name: str) -> bool:
        """Elimina un dataset."""
        try:
            result = self.collection.delete_one({"name": dataset_name})
            if result.deleted_count > 0:
                logger.info(f"🗑️ Dataset eliminado: {dataset_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error eliminando dataset: {str(e)}")
            return False


class VisualizationRepository:
    """Repositorio para operaciones con visualizaciones."""
    
    def __init__(self):
        self.client = MongoDBClient()
        self.collection = self.client.get_collection("visualizations")
    
    async def save_visualization(self, visualization: VisualizationRecord) -> str:
        """Guarda una visualización en MongoDB."""
        try:
            viz_dict = visualization.model_dump(by_alias=True, exclude_unset=True)
            result = self.collection.insert_one(viz_dict)
            logger.info(f"📈 Visualización guardada: {visualization.title}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error guardando visualización: {str(e)}")
            raise
    
    async def get_visualizations_by_dataset(self, dataset_name: str, limit: int = 10) -> List[VisualizationRecord]:
        """Obtiene visualizaciones por dataset."""
        try:
            docs = list(
                self.collection
                .find({"dataset_name": dataset_name})
                .sort("created_at", -1)
                .limit(limit)
            )
            return [VisualizationRecord(**doc) for doc in docs]
        except Exception as e:
            logger.error(f"Error obteniendo visualizaciones: {str(e)}")
            return []
    
    async def get_recent_visualizations(self, limit: int = 5) -> List[VisualizationRecord]:
        """Obtiene las visualizaciones más recientes."""
        try:
            docs = list(
                self.collection
                .find()
                .sort("created_at", -1)
                .limit(limit)
            )
            return [VisualizationRecord(**doc) for doc in docs]
        except Exception as e:
            logger.error(f"Error obteniendo visualizaciones recientes: {str(e)}")
            return []


class SessionRepository:
    """Repositorio para operaciones con sesiones de usuario."""
    
    def __init__(self):
        self.client = MongoDBClient()
        self.collection = self.client.get_collection("sessions")
    
    async def save_session(self, session: UserSession) -> str:
        """Guarda una sesión de usuario."""
        try:
            session_dict = session.model_dump(by_alias=True, exclude_unset=True)
            result = self.collection.replace_one(
                {"session_id": session.session_id},
                session_dict,
                upsert=True
            )
            return str(result.upserted_id) if result.upserted_id else session.session_id
        except Exception as e:
            logger.error(f"Error guardando sesión: {str(e)}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Obtiene una sesión por ID."""
        try:
            doc = self.collection.find_one({"session_id": session_id})
            if doc:
                return UserSession(**doc)
            return None
        except Exception as e:
            logger.error(f"Error obteniendo sesión: {str(e)}")
            return None
    
    async def update_session_activity(self, session_id: str):
        """Actualiza la última actividad de una sesión."""
        try:
            self.collection.update_one(
                {"session_id": session_id},
                {"$set": {"last_activity": datetime.utcnow()}}
            )
        except Exception as e:
            logger.error(f"Error actualizando actividad de sesión: {str(e)}")


class LogRepository:
    """Repositorio para operaciones con logs del sistema."""
    
    def __init__(self):
        self.client = MongoDBClient()
        self.collection = self.client.get_collection("logs")
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Crea índices para optimizar consultas de logs."""
        try:
            # Índice por timestamp (para consultas por fecha)
            self.collection.create_index([("timestamp", -1)])
            # Índice por nivel de log
            self.collection.create_index([("level", 1)])
            # Índice por categoría
            self.collection.create_index([("category", 1)])
            # Índice por sesión
            self.collection.create_index([("session_id", 1)])
            # Índice compuesto para consultas comunes
            self.collection.create_index([("level", 1), ("timestamp", -1)])
        except Exception as e:
            logger.error(f"Error creando índices de logs: {str(e)}")
    
    async def save_log(self, log_entry: 'LogEntry') -> str:
        """Guarda una entrada de log."""
        try:
            log_dict = log_entry.model_dump(by_alias=True, exclude_unset=True)
            result = self.collection.insert_one(log_dict)
            return str(result.inserted_id)
        except Exception as e:
            # En caso de error guardando logs, usar logging estándar como fallback
            import logging
            logging.error(f"Error guardando log en MongoDB: {str(e)}")
            return ""
    
    async def get_recent_logs(self, limit: int = 50, level: Optional[str] = None, 
                             category: Optional[str] = None) -> List['LogEntry']:
        """Obtiene logs recientes con filtros opcionales."""
        try:
            from .pydantic_models import LogEntry
            
            query = {}
            if level:
                query["level"] = level
            if category:
                query["category"] = category
            
            docs = list(
                self.collection
                .find(query)
                .sort("timestamp", -1)
                .limit(limit)
            )
            return [LogEntry(**doc) for doc in docs]
        except Exception as e:
            logger.error(f"Error obteniendo logs recientes: {str(e)}")
            return []
    
    async def get_logs_by_session(self, session_id: str, limit: int = 100) -> List['LogEntry']:
        """Obtiene logs de una sesión específica."""
        try:
            from .pydantic_models import LogEntry
            
            docs = list(
                self.collection
                .find({"session_id": session_id})
                .sort("timestamp", -1)
                .limit(limit)
            )
            return [LogEntry(**doc) for doc in docs]
        except Exception as e:
            logger.error(f"Error obteniendo logs de sesión: {str(e)}")
            return []
    
    async def get_error_logs(self, hours: int = 24, limit: int = 100) -> List['LogEntry']:
        """Obtiene logs de error de las últimas horas."""
        try:
            from .pydantic_models import LogEntry
            from datetime import timedelta
            
            since = datetime.utcnow() - timedelta(hours=hours)
            docs = list(
                self.collection
                .find({
                    "level": {"$in": ["error", "critical"]},
                    "timestamp": {"$gte": since}
                })
                .sort("timestamp", -1)
                .limit(limit)
            )
            return [LogEntry(**doc) for doc in docs]
        except Exception as e:
            logger.error(f"Error obteniendo logs de error: {str(e)}")
            return []
    
    async def get_log_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Obtiene estadísticas de logs."""
        try:
            from datetime import timedelta
            
            since = datetime.utcnow() - timedelta(hours=hours)
            
            # Contar por nivel
            level_stats = list(self.collection.aggregate([
                {"$match": {"timestamp": {"$gte": since}}},
                {"$group": {"_id": "$level", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]))
            
            # Contar por categoría
            category_stats = list(self.collection.aggregate([
                {"$match": {"timestamp": {"$gte": since}}},
                {"$group": {"_id": "$category", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]))
            
            # Total de logs
            total_logs = self.collection.count_documents({"timestamp": {"$gte": since}})
            
            return {
                "total_logs": total_logs,
                "by_level": {item["_id"]: item["count"] for item in level_stats},
                "by_category": {item["_id"]: item["count"] for item in category_stats},
                "period_hours": hours
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de logs: {str(e)}")
            return {}


# Instancias globales
mongodb_client = MongoDBClient()
dataset_repo = DatasetRepository()
visualization_repo = VisualizationRepository()
session_repo = SessionRepository()
log_repo = LogRepository()
