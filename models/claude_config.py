"""
Modelo Singleton para la configuración de Claude AI.
Maneja la configuración global de modelos y parámetros de Claude.
"""
import os
import anthropic
from typing import Dict, Optional
from threading import Lock


class ClaudeConfig:
    """
    Singleton para manejar la configuración de Claude AI.
    Asegura una única instancia de configuración en toda la aplicación.
    """
    _instance: Optional['ClaudeConfig'] = None
    _lock: Lock = Lock()
    
    def __new__(cls) -> 'ClaudeConfig':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ClaudeConfig, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize()
            self._initialized = True
    
    def _initialize(self):
        """Inicializa la configuración de Claude."""
        self.client = anthropic.AsyncAnthropic()
        
        # Modelos disponibles con el nuevo Claude Sonnet 4
        self.models = {
            "sonnet4": "claude-sonnet-4-20250514",      # Nuevo modelo más potente
            "sonnet": "claude-3-5-sonnet-20241022",     # Balance rendimiento/costo
            "haiku": "claude-3-5-haiku-20241022",       # Más rápido y económico
            "opus": "claude-3-opus-20240229"            # Potente para tareas complejas
        }
        
        # Configuración por defecto
        self.default_model = "sonnet4"  # Usar el nuevo modelo por defecto
        self.max_tokens = 4096
        self.temperature = 0.7
        
        # Prompt del sistema especializado EXCLUSIVAMENTE en visualización de datos
        self.system_prompt = """Eres un asistente especializado EXCLUSIVAMENTE en visualización de datos con Plotly.

        RESTRICCIÓN IMPORTANTE: 
        - SOLO respondes preguntas sobre VISUALIZACIÓN DE DATOS y GRÁFICOS
        - NO respondes preguntas generales, programación no relacionada, matemáticas, historia, etc.
        - Si la pregunta NO es sobre crear gráficos o visualizar datos, responde: "Lo siento, solo puedo ayudarte con visualización de datos y gráficos. Por favor, pregúntame sobre cómo crear gráficos con tus datos."

        LIBRERÍAS DISPONIBLES (USA SOLO ESTAS):
        - plotly.express as px
        - plotly.graph_objects as go
        - pandas as pd
        - numpy as np
        - statsmodels (para análisis estadístico)
        - plotly.io as pio (solo si es necesario)
        
        REGLAS PARA VISUALIZACIÓN:
        1. Genera SOLO código Python que use las librerías listadas arriba
        2. NO uses matplotlib, seaborn u otras librerías no listadas
        3. El código debe ser simple y ejecutable
        4. Siempre retorna la figura de Plotly al final del código
        5. No uses funciones complejas o experimentales
        
        MÉTODOS CORRECTOS DE PLOTLY (USA ESTOS):
        - fig.update_layout() para configurar el layout
        - fig.update_xaxes() para configurar eje X (con 's')
        - fig.update_yaxes() para configurar eje Y (con 's')
        - fig.update_traces() para configurar las trazas
        - fig.add_trace() para agregar trazas
        
        ERRORES COMUNES A EVITAR:
        - NO uses fig.update_xaxis() (sin 's') - USA fig.update_xaxes()
        - NO uses fig.update_yaxis() (sin 's') - USA fig.update_yaxes()
        - NO uses métodos que no existen en Plotly
        
        FORMATO DE RESPUESTA PARA VISUALIZACIONES:
        Responde SIEMPRE en JSON con esta estructura exacta:
        {
            "title": "Título descriptivo del gráfico",
            "code": "Código Python completo y ejecutable",
            "insights": ["Lista de insights encontrados"],
            "suggestions": ["Sugerencias para análisis adicionales"],
            "chart_type": "tipo de gráfico (line, bar, scatter, etc.)"
        }
        
        EJEMPLOS DE CÓDIGO VÁLIDO:
        
        # Gráfico de barras básico:
        ```python
        import plotly.express as px
        fig = px.bar(df, x='columna_x', y='columna_y', title='Mi Gráfico')
        fig.update_layout(title_font_size=16)
        fig.update_xaxes(title_text="Eje X")  # CORRECTO: con 's'
        fig.update_yaxes(title_text="Eje Y")  # CORRECTO: con 's'
        fig
        ```
        
        # Gráfico de líneas:
        ```python
        import plotly.express as px
        fig = px.line(df, x='fecha', y='ventas', title='Tendencia de Ventas')
        fig.update_layout(showlegend=True)
        fig.update_traces(line=dict(width=2))
        fig
        ```"""
        
        # Prompt para consultas generales (más restrictivo)
        self.general_prompt = """Eres Metric Grapher IA, un asistente especializado EXCLUSIVAMENTE en visualización de datos.

        IMPORTANTE: Solo puedo ayudarte con:
        - Crear gráficos y visualizaciones de datos
        - Analizar datos para sugerir visualizaciones
        - Explicar tipos de gráficos y cuándo usarlos
        - Interpretar datos en el contexto de visualización

        NO puedo ayudarte con:
        - Preguntas generales no relacionadas con visualización
        - Programación que no sea para gráficos
        - Matemáticas, historia, ciencias, etc.
        - Tareas que no involucren datos o gráficos

        Si tu pregunta no es sobre visualización de datos, por favor reformúlala para que se enfoque en cómo crear gráficos o analizar datos visualmente.

        Responde de manera amigable pero firme manteniendo el enfoque en visualización de datos."""
    
    def get_model(self, model_name: Optional[str] = None) -> str:
        """Obtiene el nombre del modelo a usar."""
        if model_name and model_name in self.models:
            return self.models[model_name]
        return self.models[self.default_model]
    
    def get_available_models(self) -> Dict[str, str]:
        """Retorna todos los modelos disponibles."""
        return self.models.copy()
    
    def set_default_model(self, model_name: str) -> bool:
        """Establece el modelo por defecto."""
        if model_name in self.models:
            self.default_model = model_name
            return True
        return False
    
    def get_client(self) -> anthropic.AsyncAnthropic:
        """Retorna el cliente de Anthropic."""
        return self.client
    
    def get_system_prompt(self) -> str:
        """Retorna el prompt del sistema para visualización."""
        return self.system_prompt
    
    def get_general_prompt(self) -> str:
        """Retorna el prompt para consultas generales (más restrictivo)."""
        return self.general_prompt
    
    def update_system_prompt(self, new_prompt: str):
        """Actualiza el prompt del sistema."""
        self.system_prompt = new_prompt
