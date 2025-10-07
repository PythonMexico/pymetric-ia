"""
Controlador para generación de visualizaciones.
Integra Claude AI con librerías de visualización.
"""
import json
import re
import traceback
import asyncio
from typing import Dict, Any, Optional, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import base64

from models.claude_config import ClaudeConfig
from models.data_models import VisualizationRequest, VisualizationResult, ChartType
from models.enhanced_data_store import EnhancedDataStore
from controllers.data_controller import DataController


class VisualizationController:
    """Controlador para generación de visualizaciones con Claude AI."""
    
    def __init__(self):
        self.claude_config = ClaudeConfig()
        self.data_store = EnhancedDataStore()
        self.data_controller = DataController()
    
    async def generate_visualization(self, request: VisualizationRequest) -> VisualizationResult:
        """
        Genera una visualización basada en la solicitud del usuario.
        
        Args:
            request: Solicitud de visualización
            
        Returns:
            Resultado de la visualización generada
        """
        start_time = datetime.now()
        
        try:
            # Obtener el dataset si se especifica
            df = None
            dataset_context = ""
            
            if request.dataset_name:
                df = self.data_store.get_dataset(request.dataset_name)
                if df is None:
                    return VisualizationResult(
                        chart_type=ChartType.LINE,
                        title="Error",
                        code="",
                        error=f"Dataset '{request.dataset_name}' no encontrado"
                    )
                
                # Crear contexto del dataset para Claude
                dataset_context = self._create_dataset_context(df, request.dataset_name)
            
            # Generar código con Claude
            code_response = await self._generate_code_with_claude(request, dataset_context)
            
            if "error" in code_response:
                return VisualizationResult(
                    chart_type=ChartType.LINE,
                    title="Error en generación",
                    code="",
                    error=code_response["error"]
                )
            
            # Ejecutar el código generado
            execution_result = await self._execute_visualization_code(
                code_response["code"], 
                df, 
                request.dataset_name
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Crear resultado
            result = VisualizationResult(
                chart_type=self._detect_chart_type(code_response["code"]),
                title=code_response.get("title", "Visualización"),
                code=code_response["code"],
                figure_json=execution_result.get("figure_json"),
                insights=code_response.get("insights", []),
                suggestions=code_response.get("suggestions", []),
                execution_time=execution_time,
                error=execution_result.get("error")
            )
            
            # Guardar en historial
            self.data_store.add_visualization(result)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return VisualizationResult(
                chart_type=ChartType.LINE,
                title="Error",
                code="",
                error=f"Error inesperado: {str(e)}",
                execution_time=execution_time
            )
    
    async def _generate_code_with_claude(self, request: VisualizationRequest, 
                                       dataset_context: str) -> Dict[str, Any]:
        """Genera código Python usando Claude AI."""
        try:
            # Construir prompt para Claude
            prompt = self._build_claude_prompt(request, dataset_context)
            
            # Llamar a Claude
            client = self.claude_config.get_client()
            response = await client.messages.create(
                model=self.claude_config.get_model("sonnet4"),  # Usar Claude Sonnet 4
                system=self.claude_config.get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.claude_config.max_tokens,
                temperature=self.claude_config.temperature
            )
            
            # Parsear respuesta
            response_text = response.content[0].text
            return self._parse_claude_response(response_text)
            
        except Exception as e:
            return {"error": f"Error al comunicarse con Claude: {str(e)}"}
    
    def _build_claude_prompt(self, request: VisualizationRequest, 
                           dataset_context: str) -> str:
        """Construye el prompt para Claude."""
        prompt = f"""
        Necesito generar una visualización de datos usando Python con Plotly.

        SOLICITUD DEL USUARIO:
        {request.query}

        {dataset_context}

        INSTRUCCIONES:
        1. Genera código Python que use Plotly para crear la visualización
        2. El código debe ser ejecutable y completo
        3. Usa 'df' como nombre de la variable del DataFrame
        4. Incluye títulos, etiquetas y formato profesional
        5. Añade interactividad cuando sea apropiado
        6. El código debe retornar la figura de Plotly

        FORMATO DE RESPUESTA:
        Estructura tu respuesta en JSON con estos campos:
        {{
            "title": "Título descriptivo de la visualización",
            "code": "Código Python completo",
            "insights": ["Lista de insights encontrados en los datos"],
            "suggestions": ["Sugerencias para análisis adicionales"],
            "chart_type": "tipo de gráfico (line, bar, scatter, etc.)"
        }}

        EJEMPLO DE CÓDIGO:
        ```python
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Tu código aquí
        fig = px.scatter(df, x='columna_x', y='columna_y', title='Mi Gráfico')
        fig.update_layout(
            title_font_size=16,
            xaxis_title="Etiqueta X",
            yaxis_title="Etiqueta Y"
        )
        
        # Retornar la figura
        fig
        ```

        Responde SOLO con el JSON solicitado.
        """
        
        return prompt
    
    def _create_dataset_context(self, df: pd.DataFrame, dataset_name: str) -> str:
        """Crea contexto del dataset para Claude."""
        try:
            # Información básica
            context = f"""
            DATASET: {dataset_name}
            Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas
            
            COLUMNAS Y TIPOS:
            """
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                context += f"- {col}: {dtype} (valores nulos: {null_count})\n"
            
            # Vista previa de datos
            context += f"\nVISTA PREVIA (primeras 3 filas):\n"
            context += df.head(3).to_string()
            
            # Estadísticas básicas para columnas numéricas
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                context += f"\n\nESTADÍSTICAS NUMÉRICAS:\n"
                context += df[numeric_cols].describe().to_string()
            
            return context
            
        except Exception as e:
            return f"Error al crear contexto del dataset: {str(e)}"
    
    def _parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """Parsea la respuesta de Claude."""
        try:
            # Buscar JSON en la respuesta
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # Si no hay JSON, intentar extraer código Python
            code_match = re.search(r'```python\n(.*?)\n```', response_text, re.DOTALL)
            if code_match:
                return {
                    "title": "Visualización generada",
                    "code": code_match.group(1),
                    "insights": [],
                    "suggestions": [],
                    "chart_type": "unknown"
                }
            
            return {"error": "No se pudo parsear la respuesta de Claude"}
            
        except json.JSONDecodeError as e:
            return {"error": f"Error al parsear JSON: {str(e)}"}
        except Exception as e:
            return {"error": f"Error al procesar respuesta: {str(e)}"}
    
    def _fix_common_plotly_errors(self, code: str) -> str:
        """Corrige errores comunes en código de Plotly."""
        # Corregir métodos incorrectos
        fixes = {
            'update_xaxis(': 'update_xaxes(',
            'update_yaxis(': 'update_yaxes(',
            '.update_xaxis(': '.update_xaxes(',
            '.update_yaxis(': '.update_yaxes(',
            'fig.update_xaxis(': 'fig.update_xaxes(',
            'fig.update_yaxis(': 'fig.update_yaxes(',
            # Otros errores comunes
            'fig.show()': 'fig',  # Cambiar show() por retornar la figura
            'plt.show()': '',     # Remover matplotlib show
            'import matplotlib.pyplot as plt': '',  # Remover matplotlib
        }
        
        corrected_code = code
        for wrong, correct in fixes.items():
            corrected_code = corrected_code.replace(wrong, correct)
        
        # Asegurar que la figura se retorna al final
        if 'fig' in corrected_code and not corrected_code.strip().endswith('fig'):
            corrected_code += '\nfig'
        
        return corrected_code

    async def _execute_visualization_code(self, code: str, df: Optional[pd.DataFrame], 
                                        dataset_name: Optional[str]) -> Dict[str, Any]:
        """Ejecuta el código de visualización generado."""
        try:
            # Corregir errores comunes antes de ejecutar
            corrected_code = self._fix_common_plotly_errors(code)
            
            # Preparar entorno de ejecución
            import numpy as np
            import statsmodels.api as sm
            import plotly.io as pio
            from datetime import datetime
            import calendar
            
            exec_globals = {
                'pd': pd,
                'px': px,
                'go': go,
                'np': np,
                'sm': sm,
                'pio': pio,
                'datetime': datetime,
                'calendar': calendar,
                'df': df,
                'dataset_name': dataset_name,
                'plotly': pio
            }
            
            # Ejecutar código corregido
            exec(corrected_code, exec_globals)
            
            # Buscar la figura generada
            fig = None
            for var_name, var_value in exec_globals.items():
                if isinstance(var_value, (go.Figure, type(px.scatter(pd.DataFrame())))):
                    fig = var_value
                    break
            
            if fig is None:
                return {"error": "No se generó ninguna figura"}
            
            # Convertir figura a JSON
            figure_json = fig.to_json()
            
            return {
                "figure_json": figure_json,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Error al ejecutar código: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def _detect_chart_type(self, code: str) -> ChartType:
        """Detecta el tipo de gráfico del código generado."""
        code_lower = code.lower()
        
        if 'px.line' in code_lower or 'go.scatter' in code_lower and 'mode=\'lines\'' in code_lower:
            return ChartType.LINE
        elif 'px.bar' in code_lower or 'go.bar' in code_lower:
            return ChartType.BAR
        elif 'px.scatter' in code_lower or 'go.scatter' in code_lower:
            return ChartType.SCATTER
        elif 'px.histogram' in code_lower or 'go.histogram' in code_lower:
            return ChartType.HISTOGRAM
        elif 'px.box' in code_lower or 'go.box' in code_lower:
            return ChartType.BOX
        elif 'px.heatmap' in code_lower or 'go.heatmap' in code_lower:
            return ChartType.HEATMAP
        elif 'px.pie' in code_lower or 'go.pie' in code_lower:
            return ChartType.PIE
        elif 'px.area' in code_lower:
            return ChartType.AREA
        elif 'px.violin' in code_lower or 'go.violin' in code_lower:
            return ChartType.VIOLIN
        elif 'px.sunburst' in code_lower:
            return ChartType.SUNBURST
        elif 'px.treemap' in code_lower:
            return ChartType.TREEMAP
        else:
            return ChartType.LINE  # Por defecto
    
    async def get_chart_suggestions(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Obtiene sugerencias de gráficos para un dataset."""
        return self.data_controller.get_suggested_visualizations(dataset_name)
    
    def get_visualization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene el historial de visualizaciones."""
        history = self.data_store.get_visualization_history(limit)
        return [result.to_dict() for result in history]
