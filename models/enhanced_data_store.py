"""
DataStore mejorado con integración MongoDB y Pydantic
Mantiene compatibilidad con el código existente
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd

from .data_models import ChartType, VisualizationResult, VisualizationRequest
from .pydantic_models import (
    ChartTypeEnum, DatasetMetadata, VisualizationRecord, UserSession
)
from .mongodb_client import dataset_repo, visualization_repo, session_repo


class EnhancedDataStore:
    """DataStore mejorado con persistencia MongoDB."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnhancedDataStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # Cache en memoria para acceso rápido
            self.datasets: Dict[str, pd.DataFrame] = {}
            self.dataset_metadata: Dict[str, Dict[str, Any]] = {}
            self.visualization_history: List[Dict[str, Any]] = []
            
            # Session management
            self.current_session_id = str(uuid.uuid4())
            self.current_session: Optional[UserSession] = None
            
            self._initialized = True
            
            # Inicializar sesión de forma síncrona
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Si ya hay un loop corriendo, crear task
                    asyncio.create_task(self._initialize_session())
                else:
                    # Si no hay loop, ejecutar directamente
                    loop.run_until_complete(self._initialize_session())
            except Exception as e:
                print(f"Error inicializando sesión: {e}")
    
    async def _initialize_session(self):
        """Inicializa la sesión de usuario."""
        try:
            self.current_session = UserSession(
                session_id=self.current_session_id,
                current_model="sonnet4"
            )
            await session_repo.save_session(self.current_session)
            print(f"Sesión inicializada: {self.current_session_id}")
        except Exception as e:
            print(f"Error inicializando sesión: {e}")
    
    def add_dataset(self, name: str, df: pd.DataFrame, metadata: Dict[str, Any] = None):
        """Agrega un dataset (versión síncrona para compatibilidad)."""
        # Ejecutar versión async
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._add_dataset_async(name, df, metadata))
            else:
                loop.run_until_complete(self._add_dataset_async(name, df, metadata))
        except Exception as e:
            print(f"Error agregando dataset: {e}")
            # Fallback a almacenamiento local
            self._add_dataset_local(name, df, metadata)
    
    async def _add_dataset_async(self, name: str, df: pd.DataFrame, metadata: Dict[str, Any] = None):
        """Versión async para agregar dataset."""
        # Guardar en cache local primero
        self.datasets[name] = df.copy()
        
        if metadata is None:
            metadata = {}
        
        # Si metadata es un objeto DatasetInfo, convertirlo a dict
        if hasattr(metadata, 'to_dict'):
            metadata = metadata.to_dict()
        
        # Preparar metadata para MongoDB
        dataset_metadata = DatasetMetadata(
            name=name,
            original_filename=metadata.get('original_filename', name),
            format=metadata.get('format', 'Unknown'),
            size=len(df),
            columns=df.columns.tolist(),
            column_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            file_size_bytes=metadata.get('file_size_bytes', 0),
            summary_stats=self._generate_summary_stats(df)
        )
        
        try:
            # Guardar en MongoDB
            await dataset_repo.save_dataset(dataset_metadata)
            self.dataset_metadata[name] = dataset_metadata.model_dump()
            print(f"Dataset guardado en MongoDB: {name}")
            
            # Log de éxito
            try:
                from .log_manager import log_manager
                await log_manager.log_database_operation(
                    operation="save_dataset",
                    collection="datasets",
                    success=True,
                    details={"dataset_name": name, "size": len(df)}
                )
            except:
                pass  # No fallar si el logging falla
                
        except Exception as e:
            # Log detallado del error
            try:
                from .log_manager import log_manager
                error_details = {
                    "dataset_name": name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "full_error": str(e) if hasattr(e, '__dict__') else None
                }
                
                # Si es un error de MongoDB, agregar detalles específicos
                if hasattr(e, 'details'):
                    error_details["mongodb_details"] = e.details
                
                await log_manager.log_database_operation(
                    operation="save_dataset",
                    collection="datasets", 
                    success=False,
                    details=error_details
                )
            except:
                pass  # No fallar si el logging falla
            
            print(f"Error guardando en MongoDB: {e}")
            # Fallback a almacenamiento local
            self._add_dataset_local(name, df, metadata)
    
    def _add_dataset_local(self, name: str, df: pd.DataFrame, metadata: Dict[str, Any] = None):
        """Fallback para almacenamiento local."""
        self.datasets[name] = df.copy()
        
        if metadata is None:
            metadata = {}
        
        # Si metadata es un objeto DatasetInfo, convertirlo a dict
        if hasattr(metadata, 'to_dict'):
            metadata = metadata.to_dict()
        
        metadata.update({
            'name': name,
            'size': len(df),
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'added_at': datetime.now().isoformat()
        })
        
        self.dataset_metadata[name] = metadata
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Genera estadísticas resumen del dataset."""
        try:
            stats = {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'null_counts': df.isnull().sum().to_dict(),
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
            }
            
            # Estadísticas para columnas numéricas
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                stats['numeric_stats'] = df[numeric_cols].describe().to_dict()
            
            return stats
        except Exception as e:
            print(f"Error generando estadísticas: {e}")
            return {}
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Obtiene un dataset por nombre."""
        return self.datasets.get(name)
    
    def get_dataset_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtiene los metadatos de un dataset."""
        return self.dataset_metadata.get(name)
    
    def list_datasets(self) -> List[str]:
        """Lista todos los datasets disponibles."""
        return list(self.datasets.keys())
    
    def remove_dataset(self, name: str) -> bool:
        """Elimina un dataset (versión síncrona)."""
        if name in self.datasets:
            del self.datasets[name]
            if name in self.dataset_metadata:
                del self.dataset_metadata[name]
            
            # Eliminar de MongoDB de forma async
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(dataset_repo.delete_dataset(name))
                else:
                    loop.run_until_complete(dataset_repo.delete_dataset(name))
            except Exception as e:
                print(f"Error eliminando de MongoDB: {e}")
            
            return True
        return False
    
    def add_visualization(self, result: VisualizationResult, dataset_name: str = None, query: str = ""):
        """Agrega una visualización (versión síncrona)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._add_visualization_async(result, dataset_name, query))
            else:
                loop.run_until_complete(self._add_visualization_async(result, dataset_name, query))
        except Exception as e:
            print(f"Error agregando visualización: {e}")
            # Fallback a almacenamiento local
            self._add_visualization_local(result)
    
    async def _add_visualization_async(self, result: VisualizationResult, dataset_name: str = None, query: str = ""):
        """Versión async para agregar visualización."""
        # Convertir ChartType a ChartTypeEnum
        chart_type_enum = ChartTypeEnum(result.chart_type.value)
        
        # Crear registro para MongoDB
        viz_record = VisualizationRecord(
            dataset_id=None,  # Se podría obtener del dataset_repo si es necesario
            dataset_name=dataset_name or "unknown",
            query=query,
            chart_type=chart_type_enum,
            title=result.title,
            code=result.code,
            figure_json=result.figure_json,
            insights=result.insights or [],
            suggestions=result.suggestions or [],
            execution_time=result.execution_time or 0.0,
            model_used=self.current_session.current_model if self.current_session else "unknown",
            success=result.error is None,
            error_message=result.error
        )
        
        try:
            # Guardar en MongoDB
            await visualization_repo.save_visualization(viz_record)
            print(f"Visualización guardada en MongoDB: {result.title}")
        except Exception as e:
            print(f"Error guardando visualización en MongoDB: {e}")
        
        # Mantener cache local
        self._add_visualization_local(result)
    
    def _add_visualization_local(self, result: VisualizationResult):
        """Fallback para almacenamiento local de visualizaciones."""
        viz_dict = {
            'timestamp': datetime.now().isoformat(),
            'chart_type': result.chart_type.value,
            'title': result.title,
            'code': result.code,
            'insights': result.insights,
            'suggestions': result.suggestions,
            'execution_time': result.execution_time,
            'success': result.error is None,
            'error': result.error
        }
        
        self.visualization_history.append(viz_dict)
        
        # Mantener solo las últimas 50 visualizaciones en cache
        if len(self.visualization_history) > 50:
            self.visualization_history = self.visualization_history[-50:]
    
    def get_visualization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene el historial de visualizaciones."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Si hay un loop corriendo, usar cache local por ahora
                return self.visualization_history[-limit:] if self.visualization_history else []
            else:
                # Si no hay loop, intentar obtener de MongoDB
                return loop.run_until_complete(self._get_visualization_history_async(limit))
        except Exception as e:
            print(f"Error obteniendo historial: {e}")
            return self.visualization_history[-limit:] if self.visualization_history else []
    
    async def _get_visualization_history_async(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Versión async para obtener historial."""
        try:
            recent_viz = await visualization_repo.get_recent_visualizations(limit)
            if recent_viz:
                return [
                    {
                        'timestamp': viz.created_at.isoformat(),
                        'chart_type': viz.chart_type.value,
                        'title': viz.title,
                        'code': viz.code,
                        'insights': viz.insights,
                        'suggestions': viz.suggestions,
                        'execution_time': viz.execution_time,
                        'success': viz.success,
                        'error': viz.error_message
                    }
                    for viz in recent_viz
                ]
        except Exception as e:
            print(f"Error obteniendo historial de MongoDB: {e}")
        
        # Fallback a cache local
        return self.visualization_history[-limit:] if self.visualization_history else []
    
    def clear_history(self):
        """Limpia el historial de visualizaciones (solo cache local)."""
        self.visualization_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del almacén."""
        return {
            'total_datasets': len(self.datasets),
            'total_visualizations': len(self.visualization_history),
            'dataset_names': list(self.datasets.keys()),
            'current_session': self.current_session_id,
            'mongodb_connected': True,  # Se podría verificar la conexión real
            'memory_usage': {
                name: df.memory_usage(deep=True).sum() 
                for name, df in self.datasets.items()
            }
        }


# Crear instancia global para compatibilidad
DataStore = EnhancedDataStore
