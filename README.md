# Pymetric IA

Una aplicación inteligente de visualización de datos que combina **Claude Sonnet 4** con **Chainlit** para generar gráficos automáticamente a partir de consultas en lenguaje natural.

## Características

- **IA Avanzada**: Powered by Claude Sonnet 4 (último modelo de Anthropic)
- **Visualizaciones Automáticas**: Genera código Python y gráficos con Plotly
- **Interfaz Conversacional**: Chat intuitivo con Chainlit
- **Múltiples Formatos**: Soporte para CSV, Excel, JSON, Parquet
- **Arquitectura MVC**: Código organizado y escalable
- **Patrón Singleton**: Gestión eficiente de configuración
- **Sugerencias Inteligentes**: Recomendaciones automáticas de visualizaciones
- **Persistencia MongoDB**: Almacenamiento robusto con Pydantic
- **Selección de Dataset**: Gestión inteligente de múltiples datasets

## Arquitectura

El proyecto implementa el patrón **Modelo-Vista-Controlador (MVC)** con **Singleton** para una arquitectura limpia y escalable:

```
metric-grapher-ia/
├── models/                 # Modelos de datos
│   ├── claude_config.py    # Patron Singleton para configuración de Claude
│   ├── data_models.py      # Estructuras de datos y almacén
│   ├── enhanced_data_store.py  # DataStore con MongoDB
│   ├── mongodb_client.py   # Cliente MongoDB con repositorios
│   └── pydantic_models.py  # Modelos Pydantic para validación
├── controllers/            # Lógica de negocio
│   ├── data_controller.py  # Procesamiento de datos
│   └── visualization_controller.py  # Generación de gráficos
├── views/                  # Interfaz de usuario
│   └── app.py             # Aplicación Chainlit
└── utils/                  # Utilidades
    └── helpers.py         # Funciones auxiliares
```

### Componentes Principales

#### **Claude Sonnet 4 Integration**
- Modelo más avanzado de Anthropic (claude-sonnet-4-20250514)
- Genera código Python optimizado para visualizaciones
- Análisis inteligente de datos y sugerencias

#### **Plotly Visualizations**
- Gráficos interactivos y profesionales
- Soporte completo para tipos de gráficos
- Exportación y personalización avanzada

#### **Chainlit Interface**
- Chat conversacional intuitivo
- Carga de archivos drag-and-drop
- Visualización integrada de gráficos

#### **MongoDB + Pydantic**
- Persistencia robusta de datasets y visualizaciones
- Validación automática de datos con Pydantic
- Gestión de sesiones de usuario
- Fallback automático a almacenamiento local

## Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/metric-grapher-ia.git
cd metric-grapher-ia
```

2. **Crear entorno virtual**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**
```bash
pip install -e .
```

4. **Configurar API Key**
```bash
export ANTHROPIC_API_KEY="tu-api-key-aqui"
```

5. **Configurar MongoDB (Opcional)**
```bash
# Instalar MongoDB localmente o usar MongoDB Atlas
# La aplicación funciona con fallback local si MongoDB no está disponible
```

6. **Ejecutar la aplicación**
```bash
chainlit run views/app.py
```

## Uso

### Comandos Disponibles

- `/datasets` - Ver datasets cargados
- `/usar [nombre]` - Seleccionar dataset activo
- `/ejemplo` - Cargar dataset de ejemplo
- `/modelo` - Ver modelo actual de Claude
- `/modelo [sonnet4|sonnet|haiku|opus]` - Cambiar modelo
- `/historial` - Ver historial de visualizaciones
- `/mongodb` - Verificar estado de MongoDB
- `/codigo` - Mostrar/ocultar código Python
- `/reset` - Reiniciar sesión
- `/ayuda` - Mostrar ayuda

### Ejemplos de Consultas

```
"Crea un gráfico de barras de las ventas por mes"
"Muestra la correlación entre precio y ventas"
"Haz un histograma de la edad de los clientes"
"Analiza las tendencias en los datos de ventas"
"Genera un scatter plot de altura vs peso"
"Usa el dataset data_sinthetic para crear un gráfico"
```

### Formatos Soportados

- **CSV** (.csv)
- **Excel** (.xlsx, .xls)
- **JSON** (.json)
- **Parquet** (.parquet)

## Tipos de Gráficos

- **Líneas**: Series temporales y tendencias
- **Barras**: Comparaciones categóricas
- **Dispersión**: Correlaciones y relaciones
- **Histogramas**: Distribuciones
- **Box Plots**: Estadísticas por grupos
- **Heatmaps**: Matrices de correlación
- **Pie Charts**: Proporciones
- **Área**: Evolución acumulativa
- **Violin**: Distribuciones detalladas

## Configuración Avanzada

### Modelos de Claude

```python
# Configuración en models/claude_config.py
CLAUDE_MODELS = {
    "sonnet4": "claude-sonnet-4-20250514",      # Más potente (nuevo)
    "sonnet": "claude-3-5-sonnet-20241022",     # Balance rendimiento/costo
    "haiku": "claude-3-5-haiku-20241022",       # Rápido y económico
    "opus": "claude-3-opus-20240229"            # Tareas complejas
}
```

### Configuración de MongoDB

```python
# En models/mongodb_client.py
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 27017,
    "database": "anthropic-db",
    "collection": "metric-grapher-ia"
}
```

### Personalización de Prompts

```python
# En models/claude_config.py
system_prompt = """Tu prompt personalizado aquí..."""
```

## Desarrollo

### Estructura del Código

#### Modelos (models/)
- `ClaudeConfig`: Singleton para configuración de IA
- `EnhancedDataStore`: Almacén de datos con MongoDB
- `MongoDBClient`: Cliente MongoDB con repositorios
- `PydanticModels`: Modelos de validación de datos
- `VisualizationRequest/Result`: Estructuras de datos

#### Controladores (controllers/)
- `DataController`: Carga y procesamiento de datos
- `VisualizationController`: Generación de gráficos con IA

#### Vistas (views/)
- `app.py`: Interfaz Chainlit y manejo de eventos

### Extender Funcionalidad

1. **Nuevos tipos de gráficos**: Agregar en `ChartType` enum
2. **Nuevos formatos**: Extender `DataController.load_dataset_from_file()`
3. **Nuevos comandos**: Agregar en `chat()` message handler
4. **Nuevos modelos Pydantic**: Agregar en `pydantic_models.py`

## Ejemplos de Uso

### Gestión de Múltiples Datasets
```bash
# Cargar datasets
/ejemplo
# Subir archivo CSV/Excel

# Ver datasets disponibles
/datasets

# Seleccionar dataset específico
/usar data_sinthetic

# Usar dataset en consulta natural
"Crea un gráfico con el dataset sample_data"
```

### Generación de Gráficos
```python
# Solicitud natural:
"Crea un gráfico de ventas por región"

# El sistema:
# 1. Analiza los datos del dataset activo
# 2. Genera código Python optimizado
# 3. Ejecuta y muestra el gráfico interactivo
# 4. Proporciona insights automáticos
# 5. Guarda en MongoDB para persistencia
```

## Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Distribuido bajo la Licencia MIT. Ver `LICENSE` para más información.

## Autor

**Hugo Ramirez** - [hughpythoneer@gmail.com](mailto:hughpythoneer@gmail.com)

---

**¡Dale una estrella si te gusta el proyecto!**