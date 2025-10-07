import os
import json
import asyncio
from typing import Optional, Dict, Any, List
import chainlit as cl
import plotly.graph_objects as go
import plotly.io as pio

# Importar controladores y modelos
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.claude_config import ClaudeConfig
from models.data_models import VisualizationRequest, ChartType
from models.enhanced_data_store import EnhancedDataStore
from models.pattern_manager import pattern_manager
from models.log_manager import log_manager
from models.pydantic_models import LogCategoryEnum
from controllers.data_controller import DataController
from controllers.visualization_controller import VisualizationController

# Inicializar componentes MVC con MongoDB
claude_config = ClaudeConfig()
data_controller = DataController()
visualization_controller = VisualizationController()
data_store = EnhancedDataStore()  # Usar el DataStore mejorado con MongoDB


@cl.on_chat_start
async def start_chat():
    """Inicializa la sesión de chat."""
    # Configurar sesión
    cl.user_session.set("messages", [])
    cl.user_session.set("current_model", "sonnet4")
    
    # Configurar contexto de logging
    session_id = cl.user_session.get("id", "unknown")
    log_manager.set_session_context(session_id)
    
    # Log de inicio de sesión
    await log_manager.info("Nueva sesión iniciada", LogCategoryEnum.SYSTEM, {
        "session_id": session_id,
        "initial_model": "sonnet4"
    })
    
    # Mensaje de bienvenida mejorado
    welcome_message = """**Pymetric IA** - Tu asistente especializado EXCLUSIVAMENTE en visualización de datos

**Capacidades:**
- Análisis automático de datasets (CSV, Excel, JSON, Parquet)
- Generación de visualizaciones con Plotly
- Powered by Claude Sonnet 4 (último modelo)
- Sugerencias inteligentes de gráficos
- Insights automáticos de datos

**IMPORTANTE:** Solo respondo preguntas sobre visualización de datos y gráficos. No puedo ayudarte con preguntas generales, programación no relacionada, matemáticas, historia, etc.

**Comandos disponibles:**
- `/datasets` - Ver datasets cargados
- `/ejemplo` - Cargar dataset de ejemplo
- `/codigo` - Mostrar/ocultar código Python
- `/modelo` - Ver modelo actual de Claude
- `/modelo [sonnet4|sonnet|haiku|opus]` - Cambiar modelo
- `/historial` - Ver historial de visualizaciones
- `/ayuda` - Mostrar esta ayuda

**Para empezar:**
1. Escribe `/ejemplo` para cargar datos de muestra
2. O sube tu propio archivo de datos (CSV, Excel, etc.)
3. Haz preguntas sobre tus datos
4. Solicita visualizaciones específicas

¿Qué datos quieres analizar hoy?"""
    
    await cl.Message(
        content=welcome_message,
        author="Pymetric IA"
    ).send()


async def handle_file_upload(files: list):
    """Maneja la carga de archivos de datos."""
    try:
        for file in files:
            # Leer contenido del archivo desde el path
            try:
                with open(file.path, 'rb') as f:
                    file_content = f.read()
                filename = file.name
                
                # Log de inicio de carga de archivo
                await log_manager.info(f"Iniciando carga de archivo: {filename}", 
                                     LogCategoryEnum.FILE_UPLOAD, {
                                         "filename": filename,
                                         "file_size_bytes": len(file_content)
                                     })
                
            except Exception as e:
                await log_manager.error(f"Error leyendo archivo {file.name}", 
                                      LogCategoryEnum.FILE_UPLOAD, 
                                      exception=e)
                await cl.Message(content=f"Error al leer archivo {file.name}: {str(e)}").send()
                continue
            
            # Procesar archivo con el controlador de datos
            result = await data_controller.load_dataset_from_file(
                file_content, filename
            )
            
            if result["success"]:
                dataset_info = result["dataset_info"]
                
                # Log de éxito en carga de archivo
                await log_manager.log_file_upload(
                    filename=filename,
                    file_size=len(file_content),
                    success=True
                )
                
                # Mensaje de confirmación sin vista previa
                success_msg = f"""**Dataset cargado exitosamente**

**Información del dataset:**
- **Nombre**: {dataset_info['name']}
- **Formato**: {dataset_info['format']}
- **Filas**: {dataset_info['size']:,}
- **Columnas**: {len(dataset_info['columns'])}
- **Tamaño**: {dataset_info['memory_usage']:.2f} MB

**Columnas detectadas:**
{', '.join(dataset_info['columns'])}

¡Ahora puedes hacer preguntas sobre estos datos o solicitar visualizaciones!"""

                await cl.Message(content=success_msg).send()
                
                # Obtener sugerencias de visualización
                suggestions = await visualization_controller.get_chart_suggestions(dataset_info['name'])
                if suggestions:
                    suggestions_text = "**Sugerencias de visualizaciones:**\n"
                    for i, suggestion in enumerate(suggestions[:3], 1):
                        suggestions_text += f"{i}. {suggestion['title']}: {suggestion['description']}\n"
                    
                    await cl.Message(content=suggestions_text).send()
            else:
                # Log de error en carga de archivo
                await log_manager.log_file_upload(
                    filename=filename,
                    file_size=len(file_content),
                    success=False,
                    error_message=result['error']
                )
                
                error_msg = f"Error al cargar archivo: {result['error']}"
                
                # Agregar información de diagnóstico si está disponible
                if 'diagnosis' in result and 'error' not in result['diagnosis']:
                    diagnosis = result['diagnosis']
                    error_msg += f"""

**Diagnóstico del archivo:**
- Tamaño: {diagnosis['file_size']} bytes
- Líneas totales: {diagnosis['total_lines']}
- Separadores detectados: {diagnosis['possible_separators']}

**Primeras líneas del archivo:**
```
{chr(10).join(diagnosis['first_lines'][:5])}
```

**Sugerencias:**
- Verifica que el archivo tenga encabezados de columna
- Asegúrate de que use el separador correcto (coma, punto y coma, tab)
- Confirma que el archivo no esté vacío o corrupto"""
                
                await cl.Message(content=error_msg).send()
                
    except Exception as e:
        await cl.Message(content=f"Error inesperado: {str(e)}").send()


def extract_dataset_from_query(query: str, available_datasets: List[str]) -> Optional[str]:
    """Extrae el nombre del dataset de una consulta natural."""
    query_lower = query.lower()
    
    # Buscar patrones como "usa dataset X", "con el dataset Y", "del dataset Z"
    dataset_patterns = [
        r'usa(?:r)?\s+(?:el\s+)?dataset\s+(\w+)',
        r'con\s+(?:el\s+)?dataset\s+(\w+)',
        r'del\s+dataset\s+(\w+)',
        r'en\s+(?:el\s+)?dataset\s+(\w+)',
        r'dataset\s+(\w+)',
        r'datos?\s+de\s+(\w+)',
        r'tabla\s+(\w+)'
    ]
    
    import re
    for pattern in dataset_patterns:
        match = re.search(pattern, query_lower)
        if match:
            dataset_name = match.group(1)
            # Verificar si el dataset existe (case insensitive)
            for available in available_datasets:
                if available.lower() == dataset_name.lower():
                    return available
    
    return None


def is_visualization_query(query: str) -> bool:
    """Detecta si una consulta es para CREAR/GENERAR gráficos específicos."""
    query_lower = query.lower()
    
    # Obtener patrones desde YAML
    educational_exclusion_patterns = pattern_manager.get_patterns("educational_exclusion_patterns")
    creation_keywords = pattern_manager.get_patterns("visualization_creation_patterns")
    chart_types = pattern_manager.get_patterns("chart_types")
    specific_patterns = pattern_manager.get_patterns("specific_chart_patterns")
    
    # Si es una pregunta educativa, NO es para generar gráfico
    if any(pattern in query_lower for pattern in educational_exclusion_patterns):
        return False
    
    # Debe tener una palabra de creación Y un tipo de gráfico
    has_creation = any(keyword in query_lower for keyword in creation_keywords)
    has_chart_type = any(chart_type in query_lower for chart_type in chart_types)
    
    # O patrones específicos de solicitud de gráfico
    has_specific_pattern = any(pattern in query_lower for pattern in specific_patterns)
    
    return (has_creation and has_chart_type) or has_specific_pattern


def is_dataset_query(query: str) -> bool:
    """Detecta si una consulta es sobre describir o analizar el dataset."""
    query_lower = query.lower()
    
    # Obtener patrones desde YAML
    dataset_patterns = pattern_manager.get_dataset_patterns()
    
    # Verificar patrones específicos primero
    if any(pattern in query_lower for pattern in dataset_patterns["specific"]):
        return True
    
    # Patrones más generales
    has_dataset = any(word in query_lower for word in dataset_patterns["dataset_words"])
    has_action = any(word in query_lower for word in dataset_patterns["action_words"])
    has_request = any(word in query_lower for word in dataset_patterns["request_words"])
    
    # Debe tener al menos dataset + (acción O solicitud)
    return has_dataset and (has_action or has_request)


def is_system_query(query: str) -> bool:
    """Detecta si una consulta es sobre el sistema o configuración."""
    query_lower = query.lower()
    
    # Obtener patrones desde YAML
    system_patterns = pattern_manager.get_patterns("system_patterns")
    
    return any(pattern in query_lower for pattern in system_patterns)


def is_educational_visualization_query(query: str) -> bool:
    """Detecta si una consulta es educativa sobre visualización de datos."""
    query_lower = query.lower()
    
    # Obtener patrones desde YAML
    educational_viz_patterns = pattern_manager.get_patterns("educational_viz_patterns")
    educational_exclusion_patterns = pattern_manager.get_patterns("educational_exclusion_patterns")
    
    # Debe ser una pregunta educativa Y contener términos de visualización
    has_educational_pattern = any(pattern in query_lower for pattern in educational_viz_patterns)
    has_exclusion_pattern = any(pattern in query_lower for pattern in educational_exclusion_patterns)
    
    return has_educational_pattern or has_exclusion_pattern


def is_visualization_recommendation_query(query: str) -> bool:
    """Detecta si una consulta es para pedir recomendaciones de visualización."""
    query_lower = query.lower()
    
    # Obtener patrones desde YAML
    recommendation_patterns = pattern_manager.get_patterns("visualization_recommendation_patterns")
    
    return any(pattern in query_lower for pattern in recommendation_patterns)


async def call_claude_for_general_query(query: str, model_name: str = None):
    """Llama a Claude para consultas generales (no de visualización)."""
    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": query})

    try:
        client = claude_config.get_client()
        model = claude_config.get_model(model_name)
        
        msg = cl.Message(content="", author="Claude")

        stream = await client.messages.create(
            model=model,
            system=claude_config.get_general_prompt(),  # Usar prompt restrictivo
            messages=messages,
            max_tokens=claude_config.max_tokens,
            temperature=claude_config.temperature,
            stream=True,
        )

        async for data in stream:
            if data.type == "content_block_delta":
                await msg.stream_token(data.delta.text)

        await msg.send()
        messages.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("messages", messages)
        
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()


async def handle_dataset_query(query: str):
    """Maneja consultas sobre descripción del dataset."""
    try:
        # Debug: Confirmar que la función se está ejecutando
        await cl.Message(content="Analizando dataset...", author="Sistema").send()
        
        datasets = data_controller.list_available_datasets()
        
        if not datasets:
            await cl.Message(
                content="No hay datasets cargados. Usa `/ejemplo` para cargar datos de muestra o sube tu propio archivo.",
                author="Sistema"
            ).send()
            return
        
        # Usar dataset activo seleccionado por el usuario
        current_dataset = cl.user_session.get("current_dataset", None)
        dataset_names = [d['name'] for d in datasets]
        
        if current_dataset and current_dataset in dataset_names:
            dataset_name = current_dataset
            dataset_info = next(d for d in datasets if d['name'] == dataset_name)
        else:
            # Si no hay dataset seleccionado, usar el primero y establecerlo como activo
            dataset_info = datasets[0]
            dataset_name = dataset_info['name']
            cl.user_session.set("current_dataset", dataset_name)
        
        # Confirmar qué dataset se está analizando
        await cl.Message(content=f"Analizando dataset activo: **{dataset_name}**", author="Sistema").send()
        
        # Obtener el DataFrame
        df = data_controller.data_store.get_dataset(dataset_name)
        if df is None:
            await cl.Message(content="Error: No se pudo acceder al dataset.").send()
            return
        
        # Crear descripción simple sin vista previa
        description = f"""**Descripción del Dataset: {dataset_name}**

**Información General:**
- **Filas**: {len(df):,} registros
- **Columnas**: {len(df.columns)} variables

**Columnas disponibles:**
{', '.join(df.columns.tolist())}

**Tipos de datos:**
{dict(df.dtypes)}

**¿Qué puedes hacer ahora?**
- "Crea un gráfico de barras de [columna]"
- "Muestra la tendencia de [columna]"
- "Compara [columna1] vs [columna2]"
"""

        await cl.Message(content=description, author="Análisis de Datos").send()
        
    except Exception as e:
        await cl.Message(content=f"Error al describir dataset: {str(e)}").send()


async def handle_visualization_recommendation_query(query: str):
    """Maneja consultas de recomendación de visualización."""
    try:
        await cl.Message(content="Analizando datos para generar recomendaciones...", author="Sistema").send()
        
        datasets = data_controller.list_available_datasets()
        
        if not datasets:
            await cl.Message(
                content="Para recomendarte visualizaciones necesito que primero cargues datos. Usa `/ejemplo` para cargar datos de muestra o sube tu propio archivo.",
                author="Sistema"
            ).send()
            return
        
        # Usar dataset activo
        current_dataset = cl.user_session.get("current_dataset", None)
        dataset_names = [d['name'] for d in datasets]
        
        if current_dataset and current_dataset in dataset_names:
            dataset_name = current_dataset
        else:
            dataset_name = datasets[0]['name']
            cl.user_session.set("current_dataset", dataset_name)
        
        # Obtener sugerencias del controlador de visualización
        suggestions = await visualization_controller.get_chart_suggestions(dataset_name)
        
        if suggestions:
            recommendations_msg = f"""**Recomendaciones de Visualización para: {dataset_name}**

**Basándome en el análisis de tus datos, te recomiendo:**

"""
            for i, suggestion in enumerate(suggestions[:5], 1):
                recommendations_msg += f"{i}. **{suggestion['title']}**\n   {suggestion['description']}\n\n"
            
            recommendations_msg += """**¿Cómo proceder?**
- Elige una de las recomendaciones y pídemela: "Crea un [tipo de gráfico]"
- O sé más específico: "Muestra la correlación entre [columna1] y [columna2]"
- También puedes preguntar: "¿Por qué recomiendas un gráfico de barras?"

¿Cuál te interesa más?"""
            
            await cl.Message(content=recommendations_msg, author="Recomendaciones IA").send()
        else:
            # Fallback: análisis básico del dataset
            df = data_controller.data_store.get_dataset(dataset_name)
            if df is not None:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                fallback_msg = f"""**Recomendaciones Básicas para: {dataset_name}**

**Columnas numéricas detectadas:** {len(numeric_cols)}
{', '.join(numeric_cols[:5]) if numeric_cols else 'Ninguna'}

**Columnas categóricas detectadas:** {len(categorical_cols)}
{', '.join(categorical_cols[:5]) if categorical_cols else 'Ninguna'}

**Mis recomendaciones:**
"""
                
                if numeric_cols:
                    fallback_msg += f"1. **Histograma** - Para ver la distribución de {numeric_cols[0]}\n"
                    if len(numeric_cols) > 1:
                        fallback_msg += f"2. **Gráfico de dispersión** - Para correlación entre {numeric_cols[0]} y {numeric_cols[1]}\n"
                
                if categorical_cols:
                    fallback_msg += f"3. **Gráfico de barras** - Para contar categorías en {categorical_cols[0]}\n"
                
                if numeric_cols and categorical_cols:
                    fallback_msg += f"4. **Box plot** - Para comparar {numeric_cols[0]} por {categorical_cols[0]}\n"
                
                fallback_msg += "\n¿Te interesa alguna de estas opciones?"
                
                await cl.Message(content=fallback_msg, author="Recomendaciones IA").send()
        
    except Exception as e:
        await cl.Message(content=f"Error generando recomendaciones: {str(e)}").send()


async def handle_visualization_request(query: str, dataset_name: str = None):
    """Maneja solicitudes de visualización."""
    try:
        # Crear solicitud de visualización
        request = VisualizationRequest(
            query=query,
            dataset_name=dataset_name
        )
        
        # Mostrar mensaje de procesamiento
        processing_msg = await cl.Message(
            content="Generando visualización...", 
            author="Sistema"
        ).send()
        
        # Generar visualización
        result = await visualization_controller.generate_visualization(request)
        
        # Eliminar mensaje de procesamiento
        await processing_msg.remove()
        
        if result.error:
            await cl.Message(content=f"Error: {result.error}").send()
            return
        
        # Mostrar código solo si el usuario lo ha activado
        show_code = cl.user_session.get("show_code", False)
        if show_code:
            code_msg = f"""**Código generado:**
```python
{result.code}
```"""
            await cl.Message(content=code_msg).send()
        
        # Mostrar visualización si se generó correctamente
        if result.figure_json:
            try:
                fig_dict = json.loads(result.figure_json)
                fig = go.Figure(fig_dict)
                
                # Enviar gráfico a Chainlit
                elements = [cl.Plotly(name="visualization", figure=fig, display="inline")]
                await cl.Message(content="**Visualización generada:**", elements=elements).send()
                
                # Mostrar insights si los hay
                if result.insights:
                    insights_msg = "**Insights encontrados:**\n"
                    for insight in result.insights:
                        insights_msg += f"• {insight}\n"
                    await cl.Message(content=insights_msg).send()
                
                # Mostrar sugerencias si las hay
                if result.suggestions:
                    suggestions_msg = "**Sugerencias para análisis adicionales:**\n"
                    for suggestion in result.suggestions:
                        suggestions_msg += f"• {suggestion}\n"
                    await cl.Message(content=suggestions_msg).send()
                    
            except Exception as e:
                await cl.Message(content=f"Error al mostrar gráfico: {str(e)}").send()
        
    except Exception as e:
        await cl.Message(content=f"Error inesperado: {str(e)}").send()


@cl.on_message
async def chat(message: cl.Message):
    """Manejador principal de mensajes."""
    # Manejar archivos adjuntos si los hay
    if message.elements:
        files = []
        for element in message.elements:
            # Verificar si es un archivo
            if hasattr(element, 'path') and hasattr(element, 'name'):
                files.append(element)
        if files:
            await handle_file_upload(files)
            return
    
    content = message.content.strip()
    
    # Debug: Mostrar qué mensaje se recibió (temporal)
    # await cl.Message(content=f"Debug: Recibí el mensaje: '{content}'", author="Debug").send()
    
    # Comandos del sistema
    if content == "/reset":
        await log_manager.log_user_action("reset_session")
        
        cl.user_session.set("messages", [])
        data_store.clear_history()
        await cl.Message(
            content="Sesión reiniciada. Historial y contexto borrados.",
            author="Sistema"
        ).send()
        return
    
    elif content == "/datasets":
        datasets = data_controller.list_available_datasets()
        current_dataset = cl.user_session.get("current_dataset", None)
        
        if datasets:
            datasets_msg = "**Datasets disponibles:**\n"
            for dataset in datasets:
                active_marker = " **ACTIVO**" if dataset['name'] == current_dataset else ""
                datasets_msg += f"• **{dataset['name']}** ({dataset['format']}) - {dataset['size']:,} filas{active_marker}\n"
            
            if current_dataset:
                datasets_msg += f"\n**Dataset actual:** {current_dataset}\n"
                datasets_msg += "**Cambiar dataset:** `/usar [nombre_dataset]`"
            else:
                datasets_msg += "\n**Tip:** Usa `/usar [nombre_dataset]` para seleccionar un dataset"
        else:
            datasets_msg = "No hay datasets cargados. Sube un archivo para empezar."
        await cl.Message(content=datasets_msg).send()
        return
    
    elif content == "/codigo":
        # Toggle para mostrar/ocultar código
        show_code = cl.user_session.get("show_code", False)
        cl.user_session.set("show_code", not show_code)
        status = "activado" if not show_code else "desactivado"
        await cl.Message(
            content=f"Mostrar código Python: **{status}**",
            author="Sistema"
        ).send()
        return
    
    elif content == "/ejemplo":
        # Cargar archivo de ejemplo
        try:
            await log_manager.log_user_action("load_sample_dataset")
            await cl.Message(content="Cargando archivo de ejemplo...", author="Sistema").send()
            
            sample_file_path = "/Users/psf/Github/metric-grapher-ia/sample_data.csv"
            with open(sample_file_path, 'rb') as f:
                file_content = f.read()
            
            result = await data_controller.load_dataset_from_file(
                file_content, "sample_data.csv"
            )
            
            if result["success"]:
                await cl.Message(content="Archivo de ejemplo cargado exitosamente!").send()
                dataset_info = result["dataset_info"]
                
                # Mostrar información del dataset
                info_msg = f"""**Dataset de ejemplo cargado:**
- **Nombre**: {dataset_info['name']}
- **Filas**: {dataset_info['size']:,}
- **Columnas**: {', '.join(dataset_info['columns'])}

¡Ahora puedes hacer preguntas sobre estos datos!"""
                await cl.Message(content=info_msg).send()
            else:
                await cl.Message(content=f"Error al cargar ejemplo: {result['error']}").send()
        except Exception as e:
            await cl.Message(content=f"Error al cargar archivo de ejemplo: {str(e)}").send()
        return
    
    elif content == "/historial":
        history = visualization_controller.get_visualization_history(5)
        if history:
            history_msg = "**Historial de visualizaciones:**\n"
            for i, viz in enumerate(history, 1):
                history_msg += f"{i}. {viz['title']} ({viz['chart_type']})\n"
        else:
            history_msg = "No hay visualizaciones en el historial."
        await cl.Message(content=history_msg).send()
        return
    
    elif content == "/test":
        # Comando de debug temporal
        test_queries = [
            "Me puedes describir el dataset?",
            "¿Qué métricas me recomiendas graficar?",
            "Crea un gráfico de barras",
            "¿Qué es un histograma?"
        ]
        
        debug_msg = "**Test de Detección de Consultas:**\n\n"
        
        for query in test_queries:
            result1 = is_dataset_query(query)
            result2 = is_visualization_query(query)
            result3 = is_educational_visualization_query(query)
            result4 = is_visualization_recommendation_query(query)
            result5 = is_system_query(query)
            
            debug_msg += f"**Consulta:** \"{query}\"\n"
            debug_msg += f"- Dataset: {result1} | Visualización: {result2} | Educativa: {result3}\n"
            debug_msg += f"- Recomendación: {result4} | Sistema: {result5}\n\n"
        
        await cl.Message(content=debug_msg, author="Debug").send()
        return
    
    elif content == "/mongodb":
        # Comando para probar MongoDB
        try:
            from models.mongodb_client import mongodb_client
            
            # Verificar conexión
            is_connected = mongodb_client.is_connected()
            
            # Obtener estadísticas
            stats = data_store.get_stats()
            
            mongodb_msg = f"""**Estado de MongoDB:**

**Conexión:** {'Conectado' if is_connected else 'Desconectado'}
**Base de datos:** anthropic-db
**Colección:** metric-grapher-ia
**Sesión actual:** {stats.get('current_session', 'N/A')}

**Estadísticas:**
- **Datasets en memoria:** {stats.get('total_datasets', 0)}
- **Visualizaciones en cache:** {stats.get('total_visualizations', 0)}
- **Datasets disponibles:** {', '.join(stats.get('dataset_names', []))}

**Prueba de escritura:** Intentando guardar datos de prueba..."""
            
            await cl.Message(content=mongodb_msg, author="MongoDB").send()
            
        except Exception as e:
            await cl.Message(content=f"Error probando MongoDB: {str(e)}", author="MongoDB").send()
        return
    
    elif content.startswith("/logs"):
        # Comando para ver logs
        try:
            from models.mongodb_client import log_repo
            
            # Parsear parámetros opcionales
            parts = content.split()
            limit = 20
            level_filter = None
            
            if len(parts) > 1:
                try:
                    limit = int(parts[1])
                except ValueError:
                    level_filter = parts[1].lower()
                    if len(parts) > 2:
                        try:
                            limit = int(parts[2])
                        except ValueError:
                            pass
            
            # Obtener logs
            logs = await log_repo.get_recent_logs(limit=limit, level=level_filter)
            
            if logs:
                logs_msg = f"""**Logs Recientes** (últimos {len(logs)} registros)
{f"**Filtro:** {level_filter}" if level_filter else ""}

"""
                for log in logs:
                    timestamp = log.timestamp.strftime("%H:%M:%S")
                    level_emoji = {
                        "debug": "🔍",
                        "info": "ℹ️",
                        "warning": "⚠️",
                        "error": "❌",
                        "critical": "🚨"
                    }.get(log.level, "📝")
                    
                    logs_msg += f"{level_emoji} **{timestamp}** [{log.level.upper()}] {log.category}: {log.message}\n"
                    
                    if log.details and len(str(log.details)) < 100:
                        logs_msg += f"   └─ {log.details}\n"
                    
                    logs_msg += "\n"
                
                # Obtener estadísticas
                stats = await log_repo.get_log_stats(hours=24)
                if stats:
                    logs_msg += f"""**Estadísticas (24h):**
- **Total logs:** {stats.get('total_logs', 0)}
- **Por nivel:** {stats.get('by_level', {})}
- **Por categoría:** {stats.get('by_category', {})}"""
                
            else:
                logs_msg = "No se encontraron logs recientes."
            
            await cl.Message(content=logs_msg, author="Logs").send()
            
        except Exception as e:
            await log_manager.error("Error obteniendo logs", LogCategoryEnum.SYSTEM, exception=e)
            await cl.Message(content=f"Error obteniendo logs: {str(e)}", author="Logs").send()
        return
    
    elif content == "/patrones":
        # Comando para mostrar información de patrones
        try:
            version = pattern_manager.get_version()
            languages = pattern_manager.get_supported_languages()
            
            patterns_info = f"""**Información de Patrones - Pymetric IA**

**Versión:** {version}
**Idiomas soportados:** {', '.join(languages)}
**Archivo:** config/patterns.yaml

**Tipos de patrones disponibles:**
- `system_patterns`: {len(pattern_manager.get_patterns('system_patterns'))} patrones
- `educational_viz_patterns`: {len(pattern_manager.get_patterns('educational_viz_patterns'))} patrones
- `visualization_creation_patterns`: {len(pattern_manager.get_patterns('visualization_creation_patterns'))} patrones
- `visualization_recommendation_patterns`: {len(pattern_manager.get_patterns('visualization_recommendation_patterns'))} patrones
- `chart_types`: {len(pattern_manager.get_patterns('chart_types'))} patrones
- `specific_chart_patterns`: {len(pattern_manager.get_patterns('specific_chart_patterns'))} patrones

**Comandos:**
- `/patrones reload` - Recargar patrones desde YAML
- `/patrones add [tipo] [patrón]` - Agregar nuevo patrón

**Ejemplo:** `/patrones add visualization_recommendation_patterns "qué gráficos sugieres"`"""
            
            await cl.Message(content=patterns_info, author="Sistema").send()
        except Exception as e:
            await cl.Message(content=f"Error obteniendo información de patrones: {str(e)}", author="Sistema").send()
        return
    
    elif content.startswith("/patrones "):
        # Comandos de gestión de patrones
        parts = content.split(" ", 2)
        if len(parts) >= 2:
            action = parts[1]
            
            if action == "reload":
                pattern_manager.reload_patterns()
                await cl.Message(content="🔄 **Patrones recargados desde YAML**", author="Sistema").send()
            
            elif action == "add" and len(parts) == 4:
                pattern_type = parts[2]
                pattern = parts[3].strip('"\'')
                success = pattern_manager.add_pattern_to_yaml(pattern_type, pattern)
                if success:
                    await cl.Message(content=f"✅ **Patrón agregado:** '{pattern}' -> {pattern_type}", author="Sistema").send()
                else:
                    await cl.Message(content=f"❌ **Error agregando patrón**", author="Sistema").send()
            else:
                await cl.Message(content="Uso: `/patrones reload` o `/patrones add [tipo] [patrón]`", author="Sistema").send()
        return
    
    elif content == "/ayuda":
        help_msg = """**Ayuda - Pymetric IA**

**Comandos disponibles:**
- `/datasets` - Ver datasets cargados
- `/usar [nombre]` - Seleccionar dataset activo
- `/ejemplo` - Cargar dataset de ejemplo
- `/codigo` - Mostrar/ocultar código Python
- `/modelo` - Ver modelo actual de Claude
- `/modelo [sonnet4|sonnet|haiku|opus]` - Cambiar modelo
- `/historial` - Ver historial de visualizaciones
- `/mongodb` - Verificar estado de MongoDB
- `/patrones` - Ver información de patrones YAML
- `/patrones reload` - Recargar patrones
- `/logs [nivel] [cantidad]` - Ver logs del sistema
- `/test` - Probar detección de consultas
- `/reset` - Reiniciar sesión
- `/ayuda` - Mostrar esta ayuda

**Ejemplos de consultas:**
- "Crea un gráfico de barras de las ventas por mes"
- "Muestra la correlación entre precio y ventas"
- "Haz un histograma de la edad de los clientes"
- "Analiza las tendencias en los datos"

**Formatos soportados:**
CSV, Excel (.xlsx, .xls), JSON, Parquet

**Solución de problemas:**
Si tienes problemas cargando un archivo CSV, verifica que:
- Tenga encabezados de columna en la primera fila
- Use comas (,) como separador (o punto y coma ;)
- Esté codificado en UTF-8
- No esté vacío o corrupto"""
        await cl.Message(content=help_msg).send()
        return
    
    elif content.startswith("/modelo "):
        model_name = content.replace("/modelo ", "").strip()
        available_models = claude_config.get_available_models()
        
        if model_name in available_models:
            cl.user_session.set("current_model", model_name)
            await cl.Message(
                content=f"Modelo cambiado a: **{available_models[model_name]}**",
                author="Sistema"
            ).send()
        else:
            models_list = ", ".join(available_models.keys())
            await cl.Message(
                content=f"Modelo no válido. Disponibles: {models_list}",
                author="Sistema"
            ).send()
        return
    
    elif content == "/modelo":
        current_model = cl.user_session.get("current_model", "sonnet4")
        model_name = claude_config.get_model(current_model)
        await cl.Message(
            content=f"Modelo actual: **{model_name}**",
            author="Sistema"
        ).send()
        return
    
    elif content.startswith("/usar "):
        # Comando para seleccionar dataset activo
        dataset_name = content[6:].strip()  # Remover "/usar "
        
        if not dataset_name:
            await cl.Message(
                content="Especifica el nombre del dataset. Ejemplo: `/usar sample_data`",
                author="Sistema"
            ).send()
            return
        
        # Verificar que el dataset existe
        datasets = data_controller.list_available_datasets()
        dataset_names = [d['name'] for d in datasets]
        
        if dataset_name not in dataset_names:
            available = ", ".join(dataset_names) if dataset_names else "ninguno"
            await cl.Message(
                content=f"Dataset '{dataset_name}' no encontrado.\nDatasets disponibles: {available}",
                author="Sistema"
            ).send()
            return
        
        # Establecer dataset activo
        cl.user_session.set("current_dataset", dataset_name)
        
        # Obtener info del dataset
        dataset_info = next(d for d in datasets if d['name'] == dataset_name)
        
        await cl.Message(
            content=f"Dataset activo cambiado a: **{dataset_name}**\n{dataset_info['size']:,} filas, {len(dataset_info['columns'])} columnas",
            author="Sistema"
        ).send()
        return
    
    # Debug: Verificar detección de consultas (comentado para limpiar interfaz)
    # is_viz = is_visualization_query(content)
    # is_dataset = is_dataset_query(content)
    # is_educational = is_educational_visualization_query(content)
    # 
    # await cl.Message(
    #     content=f"**Debug Detección:**\n- Visualización: {is_viz}\n- Dataset: {is_dataset}\n- Educativa: {is_educational}",
    #     author="Debug"
    # ).send()
    
    # Detectar tipo de consulta y hacer logging
    detected_patterns = {
        "system": is_system_query(content),
        "visualization_recommendation": is_visualization_recommendation_query(content),
        "visualization": is_visualization_query(content),
        "dataset": is_dataset_query(content),
        "educational": is_educational_visualization_query(content)
    }
    
    # Determinar el handler seleccionado
    selected_handler = "unknown"
    
    # Verificar el tipo de consulta
    if is_system_query(content):
        selected_handler = "system_query"
        
        # Log de detección de patrones
        await log_manager.log_pattern_detection(content, detected_patterns, selected_handler)
        
        # Es una pregunta sobre el modelo/sistema
        current_model = cl.user_session.get("current_model", "sonnet4")
        available_models = claude_config.get_available_models()
        model_name = available_models.get(current_model, "Desconocido")
        
        system_info = f"""**Información del Sistema - Pymetric IA**

**Modelo actual:** {model_name} (`{current_model}`)
**Especialización:** Visualización de datos exclusivamente
**Librerías:** Plotly, Pandas, NumPy
**Base de datos:** MongoDB (anthropic-db)

**Modelos disponibles:**
- `sonnet4` - Claude Sonnet 4 (más potente)
- `sonnet` - Claude 3.5 Sonnet (balance)
- `haiku` - Claude 3.5 Haiku (rápido)
- `opus` - Claude 3 Opus (complejo)

**Cambiar modelo:** `/modelo [nombre]`
**Ver comandos:** `/ayuda`

¿Te gustaría que analice algún dataset o cree una visualización?"""
        
        await cl.Message(content=system_info, author="Sistema").send()
        
    elif is_visualization_recommendation_query(content):
        selected_handler = "visualization_recommendation"
        
        # Log de detección de patrones
        await log_manager.log_pattern_detection(content, detected_patterns, selected_handler)
        
        # Es una consulta pidiendo recomendaciones de visualización
        await handle_visualization_recommendation_query(content)
        
    elif is_visualization_query(content):
        selected_handler = "visualization_request"
        
        # Log de detección de patrones
        await log_manager.log_pattern_detection(content, detected_patterns, selected_handler)
        # Es una consulta para CREAR gráficos específicos
        await cl.Message(content="Procesando solicitud de visualización...", author="Sistema").send()
        
        datasets = data_controller.list_available_datasets()
        
        if not datasets:
            await cl.Message(
                content="Para crear visualizaciones necesito que primero cargues datos. Usa `/ejemplo` para cargar datos de muestra o sube tu propio archivo.",
                author="Sistema"
            ).send()
            return
        
        # 1. Intentar extraer dataset de la consulta natural
        dataset_names = [d['name'] for d in datasets]
        extracted_dataset = extract_dataset_from_query(content, dataset_names)
        
        if extracted_dataset:
            # Si se detectó un dataset en la consulta, usarlo y guardarlo como activo
            dataset_name = extracted_dataset
            cl.user_session.set("current_dataset", dataset_name)
            await cl.Message(
                content=f"Detecté que quieres usar el dataset: **{dataset_name}**",
                author="Sistema"
            ).send()
        else:
            # 2. Usar dataset seleccionado por el usuario o el primero disponible
            current_dataset = cl.user_session.get("current_dataset", None)
            
            if current_dataset and current_dataset in dataset_names:
                dataset_name = current_dataset
            else:
                # Si no hay dataset seleccionado o no existe, usar el primero
                dataset_name = datasets[0]['name']
                cl.user_session.set("current_dataset", dataset_name)
                
                if len(datasets) > 1:
                    await cl.Message(
                        content=f"Usando dataset: **{dataset_name}** (tienes {len(datasets)} datasets disponibles)\nUsa `/usar [nombre]` para cambiar de dataset",
                        author="Sistema"
                    ).send()
        
        # Log de solicitud de visualización
        await log_manager.log_visualization_request(
            query=content,
            dataset_name=dataset_name,
            success=True  # Se actualizará en handle_visualization_request si hay error
        )
        
        await handle_visualization_request(content, dataset_name)
        
    elif is_dataset_query(content):
        selected_handler = "dataset_query"
        
        # Log de detección de patrones
        await log_manager.log_pattern_detection(content, detected_patterns, selected_handler)
        
        # Es una consulta sobre describir el dataset
        await cl.Message(content="Detecté una consulta sobre dataset...", author="Debug").send()
        await handle_dataset_query(content)
        
    elif is_educational_visualization_query(content):
        selected_handler = "educational_query"
        
        # Log de detección de patrones
        await log_manager.log_pattern_detection(content, detected_patterns, selected_handler)
        
        # Es una pregunta educativa sobre visualización - responder con Claude
        await cl.Message(content="Procesando pregunta educativa...", author="Sistema").send()
        current_model = cl.user_session.get("current_model", "sonnet4")
        await call_claude_for_general_query(content, current_model)
        
    else:
        selected_handler = "restricted_query"
        
        # Log de detección de patrones
        await log_manager.log_pattern_detection(content, detected_patterns, selected_handler)
        
        # No es una consulta relacionada con visualización - responder con restricción mejorada
        restriction_msg = """**No puedo responder a tu pregunta**

**Pymetric IA** es un asistente especializado **exclusivamente en visualización de datos**.

**SÍ puedo ayudarte con:**
- Crear gráficos (barras, líneas, scatter, histogramas, etc.)
- Analizar y describir datasets
- Sugerir visualizaciones apropiadas
- Explicar tipos de gráficos y cuándo usarlos
- Interpretar tendencias y patrones en datos
- Preguntas sobre el modelo actual (`/modelo`)

**NO puedo ayudarte con:**
- Preguntas generales no relacionadas con gráficos
- Programación fuera del contexto de visualización
- Matemáticas, historia, ciencias, noticias, etc.
- Conversación casual o temas no relacionados con datos

**Ejemplos de preguntas que SÍ puedo responder:**
- "Crea un gráfico de barras de las ventas por mes"
- "¿Qué modelo estás usando?"
- "Describe mi dataset"
- "¿Cuándo usar un gráfico de líneas?"
- "¿Qué tipos de gráficos existen?"

**Para empezar:**
1. Carga datos: `/ejemplo` o sube un archivo
2. Haz preguntas sobre visualización de esos datos
3. Usa `/ayuda` para ver todos los comandos

¿Te gustaría reformular tu pregunta enfocándote en visualización de datos?"""
        
        await cl.Message(content=restriction_msg, author="Sistema").send()