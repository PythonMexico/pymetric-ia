"""
LogManager - Sistema centralizado de logging para Pymetric IA
Guarda logs en MongoDB y proporciona una interfaz unificada
"""

import inspect
import traceback
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from threading import Lock

from .pydantic_models import LogEntry, LogLevelEnum, LogCategoryEnum
from .mongodb_client import log_repo


class LogManager:
    """Gestor centralizado de logs con patrón Singleton."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LogManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.session_id = None
            self.user_id = None
            self._initialized = True
    
    def set_session_context(self, session_id: str, user_id: Optional[str] = None):
        """Establece el contexto de sesión para los logs."""
        self.session_id = session_id
        self.user_id = user_id
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """Obtiene información del caller (función, archivo, línea)."""
        try:
            # Obtener el frame del caller (saltamos 2 frames: este método y el método público)
            frame = inspect.currentframe().f_back.f_back
            return {
                "function_name": frame.f_code.co_name,
                "file_name": frame.f_code.co_filename.split('/')[-1],  # Solo el nombre del archivo
                "line_number": frame.f_lineno
            }
        except:
            return {}
    
    async def _log(self, level: LogLevelEnum, category: LogCategoryEnum, 
                   message: str, details: Optional[Dict[str, Any]] = None,
                   duration_ms: Optional[float] = None, 
                   metadata: Optional[Dict[str, Any]] = None):
        """Método interno para crear y guardar logs."""
        try:
            caller_info = self._get_caller_info()
            
            log_entry = LogEntry(
                level=level,
                category=category,
                message=message,
                details=details or {},
                session_id=self.session_id,
                user_id=self.user_id,
                function_name=caller_info.get("function_name"),
                file_name=caller_info.get("file_name"),
                line_number=caller_info.get("line_number"),
                duration_ms=duration_ms,
                metadata=metadata or {}
            )
            
            # Guardar en MongoDB de forma asíncrona
            await log_repo.save_log(log_entry)
            
        except Exception as e:
            # Fallback a logging estándar si falla MongoDB
            import logging
            logging.error(f"Error en LogManager: {str(e)}")
    
    # Métodos públicos para diferentes niveles de log
    
    async def debug(self, message: str, category: LogCategoryEnum = LogCategoryEnum.SYSTEM,
                   details: Optional[Dict[str, Any]] = None, **kwargs):
        """Log de debug."""
        await self._log(LogLevelEnum.DEBUG, category, message, details, **kwargs)
    
    async def info(self, message: str, category: LogCategoryEnum = LogCategoryEnum.SYSTEM,
                  details: Optional[Dict[str, Any]] = None, **kwargs):
        """Log de información."""
        await self._log(LogLevelEnum.INFO, category, message, details, **kwargs)
    
    async def warning(self, message: str, category: LogCategoryEnum = LogCategoryEnum.SYSTEM,
                     details: Optional[Dict[str, Any]] = None, **kwargs):
        """Log de advertencia."""
        await self._log(LogLevelEnum.WARNING, category, message, details, **kwargs)
    
    async def error(self, message: str, category: LogCategoryEnum = LogCategoryEnum.ERROR,
                   details: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None, **kwargs):
        """Log de error."""
        if exception:
            if not details:
                details = {}
            details.update({
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "stack_trace": traceback.format_exc()
            })
        
        await self._log(LogLevelEnum.ERROR, category, message, details, **kwargs)
    
    async def critical(self, message: str, category: LogCategoryEnum = LogCategoryEnum.ERROR,
                      details: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None, **kwargs):
        """Log crítico."""
        if exception:
            if not details:
                details = {}
            details.update({
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "stack_trace": traceback.format_exc()
            })
        
        await self._log(LogLevelEnum.CRITICAL, category, message, details, **kwargs)
    
    # Métodos de conveniencia para categorías específicas
    
    async def log_user_action(self, action: str, details: Optional[Dict[str, Any]] = None):
        """Log de acción de usuario."""
        await self.info(f"User action: {action}", LogCategoryEnum.USER_ACTION, details)
    
    async def log_database_operation(self, operation: str, collection: str, 
                                   success: bool = True, duration_ms: Optional[float] = None,
                                   details: Optional[Dict[str, Any]] = None):
        """Log de operación de base de datos."""
        level = LogLevelEnum.INFO if success else LogLevelEnum.ERROR
        message = f"Database {operation} on {collection}: {'SUCCESS' if success else 'FAILED'}"
        
        if not details:
            details = {}
        details.update({
            "operation": operation,
            "collection": collection,
            "success": success
        })
        
        await self._log(level, LogCategoryEnum.DATABASE, message, details, duration_ms)
    
    async def log_visualization_request(self, query: str, dataset_name: str, 
                                      success: bool = True, duration_ms: Optional[float] = None,
                                      chart_type: Optional[str] = None):
        """Log de solicitud de visualización."""
        level = LogLevelEnum.INFO if success else LogLevelEnum.ERROR
        message = f"Visualization request: {'SUCCESS' if success else 'FAILED'}"
        
        details = {
            "query": query,
            "dataset_name": dataset_name,
            "success": success
        }
        
        if chart_type:
            details["chart_type"] = chart_type
        
        await self._log(level, LogCategoryEnum.VISUALIZATION, message, details, duration_ms)
    
    async def log_file_upload(self, filename: str, file_size: int, 
                            success: bool = True, error_message: Optional[str] = None):
        """Log de carga de archivo."""
        level = LogLevelEnum.INFO if success else LogLevelEnum.ERROR
        message = f"File upload: {filename} ({'SUCCESS' if success else 'FAILED'})"
        
        details = {
            "filename": filename,
            "file_size_bytes": file_size,
            "success": success
        }
        
        if error_message:
            details["error_message"] = error_message
        
        await self._log(level, LogCategoryEnum.FILE_UPLOAD, message, details)
    
    async def log_api_call(self, endpoint: str, method: str, status_code: int,
                          duration_ms: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        """Log de llamada a API."""
        level = LogLevelEnum.INFO if 200 <= status_code < 400 else LogLevelEnum.ERROR
        message = f"API call: {method} {endpoint} -> {status_code}"
        
        if not details:
            details = {}
        details.update({
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code
        })
        
        await self._log(level, LogCategoryEnum.API_CALL, message, details, duration_ms)
    
    async def log_pattern_detection(self, query: str, detected_patterns: Dict[str, bool],
                                  selected_handler: str):
        """Log de detección de patrones."""
        message = f"Pattern detection: {selected_handler}"
        
        details = {
            "query": query,
            "detected_patterns": detected_patterns,
            "selected_handler": selected_handler
        }
        
        await self._log(LogLevelEnum.DEBUG, LogCategoryEnum.PATTERN_DETECTION, message, details)
    
    # Métodos síncronos para casos donde no se puede usar async
    
    def sync_error(self, message: str, exception: Optional[Exception] = None,
                  category: LogCategoryEnum = LogCategoryEnum.ERROR):
        """Log de error síncrono."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Si ya hay un loop corriendo, crear una tarea
                asyncio.create_task(self.error(message, category, exception=exception))
            else:
                # Si no hay loop, ejecutar directamente
                loop.run_until_complete(self.error(message, category, exception=exception))
        except:
            # Fallback final a logging estándar
            import logging
            logging.error(f"LogManager sync_error fallback: {message}")
            if exception:
                logging.error(f"Exception: {str(exception)}")
    
    def sync_info(self, message: str, category: LogCategoryEnum = LogCategoryEnum.SYSTEM):
        """Log de info síncrono."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.info(message, category))
            else:
                loop.run_until_complete(self.info(message, category))
        except:
            import logging
            logging.info(f"LogManager sync_info fallback: {message}")


# Instancia global del LogManager
log_manager = LogManager()
