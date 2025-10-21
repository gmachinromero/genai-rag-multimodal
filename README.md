# GenAI RAG Multimodal

Un sistema de análisis multimodal que utiliza modelos de lenguaje grandes (LLMs) para procesar y analizar contenido multimedia (imágenes, audio y video) y generar metadatos estructurados para sistemas RAG (Retrieval-Augmented Generation).

## 🚀 Características

- **Análisis multimodal**: Soporte para imágenes, audio y video
- **Transcripción automática**: Uso de Whisper para transcripción de audio/video
- **Análisis inteligente**: Extracción de metadatos usando GPT-4 Vision y GPT-4
- **Estructura modular**: Arquitectura limpia y mantenible
- **Salida estructurada**: Datos en formato CSV listos para RAG

## 📁 Estructura del Proyecto

```
genai-rag-multimodal/
├── data/
│   ├── raw/           # Archivos multimedia de entrada
│   └── processed/     # Resultados del análisis (CSV)
├── genai_rag_multimodal/
│   ├── analyze_assets/
│   │   ├── __init__.py
│   │   ├── config.py      # Configuración y constantes
│   │   ├── core.py        # Lógica principal de análisis
│   │   ├── models.py      # Modelos de datos (Pydantic)
│   │   └── utils.py       # Funciones auxiliares
│   ├── examples/
│   │   ├── analyze_assets_ollama_qwen.py  # Versión alternativa con Ollama
│   │   └── analyze_assets_openai.py       # Versión con OpenAI directo
│   └── scripts/
│       └── run_analysis.py # Punto de entrada
├── tests/
│   └── __init__.py
├── .env                 # Variables de entorno (API keys)
├── .gitignore          # Archivos ignorados por Git
├── pyproject.toml      # Configuración del proyecto (Poetry)
├── poetry.lock         # Lockfile de dependencias
└── README.md           # Este archivo
```

## 🛠️ Instalación

### Prerrequisitos

- Python 3.11.10
- Poetry (gestor de dependencias)
- API Key de OpenAI

### Instalación

1. **Clona el repositorio:**
   ```bash
   git clone <url-del-repositorio>
   cd genai-rag-multimodal
   ```

2. **Instala las dependencias:**
   ```bash
   poetry install
   ```

3. **Configura las variables de entorno:**
   ```bash
   touch .env
   # Edita .env y agrega tu OPENAI_API_KEY
   ```

## 📊 Formato de Datos

### Entrada
Coloca tus archivos multimedia en `data/raw/`:
- **Imágenes**: JPG, PNG, GIF, BMP, WebP
- **Audio**: MP3, WAV, FLAC, M4A, OGG
- **Video**: MP4, AVI, MOV, MKV, WebM, FLV

### Salida
El sistema genera un archivo CSV en `data/processed/assets_analysis.csv` con la siguiente estructura:

| Columna | Descripción |
|---------|-------------|
| `id` | Identificador único del archivo |
| `nombre_archivo` | Nombre original del archivo |
| `tipo_asset` | Tipo: 'imagen', 'audio', o 'video' |
| `formato_asset` | Extensión del archivo (mp3, jpg, etc.) |
| `idioma` | Idioma detectado (código ISO 639-1) |
| `etiquetas` | Lista de 3-5 etiquetas separadas por `\|` |
| `transcripción` | Texto transcrito (para audio/video con voz) |
| `resumen` | Resumen breve del contenido |

## 🚀 Uso

### Ejecución básica

```bash
# Desde la raíz del proyecto
poetry run python genai_rag_multimodal/scripts/run_analysis.py
```

### Ejecución con opciones personalizadas

El script procesa automáticamente todos los archivos en `data/raw/` y guarda los resultados en `data/processed/assets_analysis.csv`.

## 🔧 Configuración

### Variables de entorno (.env)

```bash
OPENAI_API_KEY=tu_clave_api_aqui
```

### Configuración del sistema

Edita `genai_rag_multimodal/analyze_assets/config.py` para modificar:

- Modelos de LLM utilizados
- Extensiones de archivo soportadas
- Rutas de directorios
- Parámetros de procesamiento

## 🏗️ Arquitectura

### Módulos principales

- **`config.py`**: Configuración centralizada y constantes
- **`models.py`**: Modelos de datos con validación (Pydantic)
- **`utils.py`**: Funciones auxiliares para procesamiento multimedia
- **`core.py`**: Lógica de negocio y orquestación del análisis

### Flujo de procesamiento

1. **Detección**: Identifica tipo de archivo por extensión
2. **Extracción**: Procesa el contenido según el tipo
   - Imágenes: Análisis directo con GPT-4 Vision
   - Audio: Transcripción con Whisper + análisis con GPT-4
   - Video: Extracción de frames + transcripción audio + análisis
3. **Análisis**: Genera metadatos estructurados
4. **Validación**: Asegura integridad de datos con Pydantic
5. **Almacenamiento**: Guarda resultados en CSV
