# analyze_assets/config.py

# Centraliza configuración, constantes y carga de entorno

# --- Librerías ---
import os
from pathlib import Path
from dotenv import load_dotenv


# --- Carga de entorno ---
load_dotenv()


# --- Constantes de configuración ---
DIRECTORIO_DATOS_RAW = Path("data/raw")
DIRECTORIO_DATOS_PROCESADOS = Path("data/processed")
ARCHIVO_SALIDA_CSV = DIRECTORIO_DATOS_PROCESADOS / "assets_analysis.csv"

MODELO_LLM_OPENAI = "gpt-4o-mini"
MODELO_WHISPER_OPENAI = "whisper-1"

EXTENSIONES_IMAGEN = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}
EXTENSIONES_AUDIO = {"mp3", "wav", "flac", "m4a", "ogg"}
EXTENSIONES_VIDEO = {"mp4", "avi", "mov", "mkv", "webm", "flv"}


# --- API Key ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
