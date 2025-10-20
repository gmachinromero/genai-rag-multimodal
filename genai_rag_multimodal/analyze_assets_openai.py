# Librerías
import base64
import logging
import os
from pathlib import Path
from typing import List, Literal, Optional

import cv2
from dotenv import load_dotenv
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field



# --- Configuración de Logging ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)



# --- Configuración Inicial ---

# Cargar variables
load_dotenv()

# Constantes
DIRECTORIO_DATOS_RAW = Path("data/raw")
DIRECTORIO_DATOS_PROCESADOS = Path("data/processed")
ARCHIVO_SALIDA_CSV = DIRECTORIO_DATOS_PROCESADOS / "assets_analysis.csv"
MODELO_LLM_OPENAI = "gpt-4o-mini"
MODELO_WHISPER_OPENAI = "whisper-1"

# Extensiones soportadas
EXTENSIONES_IMAGEN = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}
EXTENSIONES_AUDIO = {"mp3", "wav", "flac", "m4a", "ogg"}
EXTENSIONES_VIDEO = {"mp4", "avi", "mov", "mkv", "webm", "flv"}



# --- Esquema de Datos con Pydantic ---

class AnalisisActivo(BaseModel):
    """Estructura de datos para el análisis de un activo multimedia."""
    
    idioma: str = Field(
        default=None,
        description="Idioma principal del contenido (código ISO 639-1: 'es', 'en', etc.). None si no hay texto."
    )
    
    etiquetas: List[str] = Field(
        description="Lista de 3 a 5 etiquetas descriptivas del contenido.",
        min_length=3,
        max_length=5
    )
    
    transcripcion: Optional[str] = Field(
        default=None,
        description="Transcripción del texto. None si no hay texto."
    )
    
    resumen: str = Field(
        description="Resumen conciso en 1-2 frases del contenido."
    )



# --- Funciones Auxiliares ---

def obtener_tipo_activo(ruta_archivo: Path) -> tuple[Optional[Literal["imagen", "audio", "video"]], str]:
    """
    Determina el tipo de activo según la extensión del archivo.
    
    Args:
        ruta_archivo: Ruta al archivo.
    
    Returns:
        Tupla (tipo_activo, extension) donde tipo_activo es 'imagen', 'audio', 'video' o None.
    """
    extension = ruta_archivo.suffix[1:].lower()
    
    if extension in EXTENSIONES_IMAGEN:
        return "imagen", extension
    if extension in EXTENSIONES_AUDIO:
        return "audio", extension
    if extension in EXTENSIONES_VIDEO:
        return "video", extension
    
    return None, extension


def codificar_imagen_base64(ruta_imagen: Path) -> str:
    """Codifica una imagen en base64."""

    with open(ruta_imagen, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    

def extraer_frames_video(ruta_video: Path, num_frames: int) -> List[str]:
    """
    Extrae frames clave de un video y los devuelve en base64.
    
    Args:
        ruta_video: Ruta al archivo de video.
        num_frames: Número de frames a extraer (distribuidos uniformemente).
    
    Returns:
        Lista de frames en formato base64.
    """

    cap = cv2.VideoCapture(str(ruta_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError(f"No se pudieron leer frames del video: {ruta_video.name}")
    
    # Calcular posiciones de frames distribuidas uniformemente
    posiciones = [int(total_frames * i / (num_frames + 1)) for i in range(1, num_frames + 1)]
    frames_base64 = []
    
    for pos in posiciones:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        
        if ret:
            # Convertir frame a JPEG y luego a base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            frames_base64.append(frame_base64)
    
    cap.release()
    return frames_base64



# --- Funciones de Análisis ---

def analizar_imagen_con_llm(ruta_imagen: Path, llm_estructurado: ChatOpenAI) -> AnalisisActivo:
    """
    Analiza una imagen usando un LLM con capacidad de visión.
    
    Args:
        ruta_imagen: Ruta al archivo de imagen.
        llm_estructurado: Modelo LLM configurado con salida estructurada.
    
    Returns:
        Objeto AnalisisActivo con los datos extraídos.
    """

    logger.info(f"Analizando imagen: {ruta_imagen.name}")
    
    datos_imagen_b64 = codificar_imagen_base64(ruta_imagen)
    extension = ruta_imagen.suffix[1:]
    
    mensaje = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Analiza esta imagen y extrae:\n"
                    "- El idioma de cualquier texto visible\n"
                    "- 3-5 etiquetas descriptivas\n"
                    "- Transcripción completa del texto (si existe)\n"
                    "- Un resumen breve\n\n"
                    "Responde SOLO con el formato JSON estructurado."
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{extension};base64,{datos_imagen_b64}"}
            }
        ]
    )
    
    return llm_estructurado.invoke([mensaje])


def analizar_audio_con_llm(ruta_audio: Path, cliente_openai: OpenAI, llm_estructurado: ChatOpenAI) -> AnalisisActivo:
    """
    Analiza un archivo de audio mediante transcripción y análisis LLM.
    
    Args:
        ruta_audio: Ruta al archivo de audio.
        cliente_openai: Cliente de OpenAI para Whisper.
        llm_estructurado: Modelo LLM configurado con salida estructurada.
    
    Returns:
        Objeto AnalisisActivo con los datos extraídos.
    """

    logger.info(f"Transcribiendo audio: {ruta_audio.name}")
    
    # Transcripción con Whisper
    with open(ruta_audio, "rb") as archivo_audio:
        transcripcion_whisper = cliente_openai.audio.transcriptions.create(
            model=MODELO_WHISPER_OPENAI,
            file=archivo_audio,
            response_format="text"
        )
    
    logger.info(f"Analizando transcripción de: {ruta_audio.name}")
    
    # Análisis con LLM
    mensaje = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Analiza esta transcripción de audio y extrae:\n"
                    "- El idioma del audio\n"
                    "- 3-5 etiquetas descriptivas del contenido\n"
                    "- Un resumen breve\n\n"
                    f"Transcripción:\n{transcripcion_whisper}\n\n"
                    "Responde SOLO con el formato JSON estructurado."
                )
            }
        ]
    )
    
    analisis = llm_estructurado.invoke([mensaje])
    
    # Preservar la transcripción original de Whisper
    analisis.transcripcion = str(transcripcion_whisper)
    
    return analisis


def analizar_video_con_llm(ruta_video: Path, cliente_openai: OpenAI, llm_estructurado: ChatOpenAI, num_frames: int) -> AnalisisActivo:
    """
    Analiza un archivo de video extrayendo frames y audio.
    
    Args:
        ruta_video: Ruta al archivo de video.
        cliente_openai: Cliente de OpenAI para Whisper.
        llm_estructurado: Modelo LLM configurado con salida estructurada.
        num_frames: Número de frames a extraer del video.
    
    Returns:
        Objeto AnalisisActivo con los datos extraídos.
    """

    logger.info(f"Procesando video: {ruta_video.name}")
    
    # 1. Extraer frames del video
    logger.info(f"  → Extrayendo {num_frames} frames...")
    frames_base64 = extraer_frames_video(ruta_video, num_frames)
    
    # 2. Transcribir audio del video
    transcripcion_whisper = ""
    logger.info(f"  → Transcribiendo audio del video...")
    try:
        # Whisper acepta archivos de video directamente (extrae el audio automáticamente)
        with open(ruta_video, "rb") as archivo_video:
            transcripcion_whisper = cliente_openai.audio.transcriptions.create(
                model=MODELO_WHISPER_OPENAI,
                file=archivo_video,
                response_format="text"
            )
    except Exception as e:
        logger.warning(f"  ⚠ No se pudo transcribir audio: {e}")
        transcripcion_whisper = ""
    
    # 3. Construir mensaje para el LLM con frames y transcripción
    logger.info(f"  → Analizando contenido del video...")
    
    contenido_mensaje = [
        {
            "type": "text",
            "text": (
                "Analiza este video basándote en los frames extraídos"
                f"{' y la transcripción del audio' if transcripcion_whisper else ''}.\n\n"
                "Extrae:\n"
                "- El idioma del contenido (audio o texto visible)\n"
                "- 3-5 etiquetas descriptivas del video\n"
                "- Un resumen breve del contenido\n\n"
            )
        }
    ]
    
    # Añadir frames como imágenes
    for i, frame_base64 in enumerate(frames_base64, 1):
        contenido_mensaje.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
        })
    
    # Añadir transcripción si existe
    if transcripcion_whisper:
        contenido_mensaje[0]["text"] += f"\nTranscripción del audio:\n{transcripcion_whisper}\n\n"
    
    contenido_mensaje[0]["text"] += "Responde SOLO con el formato JSON estructurado."
    
    mensaje = HumanMessage(content=contenido_mensaje)
    analisis = llm_estructurado.invoke([mensaje])
    
    # Preservar la transcripción original de Whisper
    if transcripcion_whisper:
        analisis.transcripcion = str(transcripcion_whisper)
    
    return analisis



# --- Procesamiento Principal ---

def procesar_directorio(directorio_datos: Path, llm_estructurado: ChatOpenAI, cliente_openai: OpenAI) -> pd.DataFrame:
    """
    Procesa todos los archivos multimedia en un directorio.
    
    Args:
        directorio_datos: Directorio con los archivos a procesar.
        llm_estructurado: Modelo LLM estructurado.
        cliente_openai: Cliente de OpenAI.
    
    Returns:
        DataFrame con los resultados del análisis.
    """
    
    if not directorio_datos.exists():
        raise FileNotFoundError(f"El directorio no existe: {directorio_datos}")
    
    resultados = []
    id_archivo = 1
    
    archivos = sorted(directorio_datos.iterdir())
    logger.info(f"Encontrados {len(archivos)} archivos para procesar")
    
    for archivo in archivos:
        # Omitir directorios y archivos ocultos
        if archivo.is_dir() or archivo.name.startswith("."):
            continue
        
        tipo_activo, formato_activo = obtener_tipo_activo(archivo)
        
        if tipo_activo is None:
            logger.warning(f"Tipo no soportado: {archivo.name} (extensión: {formato_activo})")
            continue
        
        try:
            if tipo_activo == "imagen":
                analisis = analizar_imagen_con_llm(archivo, llm_estructurado)
            elif tipo_activo == "audio":
                analisis = analizar_audio_con_llm(archivo, cliente_openai, llm_estructurado)
            elif tipo_activo == "video":
                analisis = analizar_video_con_llm(archivo, cliente_openai, llm_estructurado, num_frames=10)
            else:
                continue
            
            # Convertir a diccionario y añadir metadatos
            datos_fila = analisis.model_dump()
            datos_fila.update({
                "id": id_archivo,
                "nombre_archivo": archivo.name,
                "tipo_asset": tipo_activo,
                "formato_asset": formato_activo
            })
            
            resultados.append(datos_fila)
            id_archivo += 1
            logger.info(f"✓ Procesado exitosamente: {archivo.name}")
            
        except Exception as e:
            logger.error(f"✗ Error procesando {archivo.name}: {str(e)}")
            continue
    
    return pd.DataFrame(resultados)




# --- Función Principal ---

def main():
    """Punto de entrada del script."""
    
    # Leer API Key
    logger.info("Leyendo API Key...")
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Inicializar clientes
    logger.info("Inicializando clientes de API...")
    llm = ChatOpenAI(model=MODELO_LLM_OPENAI, temperature=0, api_key=api_key)
    llm_estructurado = llm.with_structured_output(AnalisisActivo)
    cliente_openai = OpenAI(api_key=api_key)
    
    # Procesar archivos
    logger.info(f"Procesando archivos en: {DIRECTORIO_DATOS_RAW.absolute()}")
    df_resultados = procesar_directorio(
        DIRECTORIO_DATOS_RAW,
        llm_estructurado,
        cliente_openai
    )
    
    # Guardar resultados
    if df_resultados.empty:
        logger.warning("No se procesaron archivos o no se obtuvieron resultados.")
        return
    
    # Crear directorio de salida
    DIRECTORIO_DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)
    
    # Reordenar columnas
    columnas_ordenadas = [
        'id', 'nombre_archivo', 'tipo_asset', 'formato_asset',
        'idioma', 'etiquetas', 'transcripcion', 'resumen'
    ]
    df_resultados = df_resultados[[c for c in columnas_ordenadas if c in df_resultados.columns]]

    # Convertir listas a strings con separador de pipe para mejor legibilidad
    df_resultados['etiquetas'] = df_resultados['etiquetas'].apply(
        lambda x: ' | '.join(x) if isinstance(x, list) else x
    )
    
    # Limpiar saltos de línea en transcripciones y resúmenes
    # Reemplazar saltos de línea por espacios para mantener el texto en una sola línea CSV
    for columna in ['transcripcion', 'resumen']:
        if columna in df_resultados.columns:
            df_resultados[columna] = df_resultados[columna].apply(
                lambda x: x.replace('\n', ' ').replace('\r', ' ').strip() if pd.notna(x) and isinstance(x, str) else x
            )
    
    # Guardar CSV con configuración estándar
    df_resultados.to_csv(
        ARCHIVO_SALIDA_CSV, 
        index=False, 
        encoding='utf-8-sig'
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Análisis completado exitosamente")
    logger.info(f"✓ Archivos procesados: {len(df_resultados)}")
    logger.info(f"✓ Resultados guardados en: {ARCHIVO_SALIDA_CSV.absolute()}")
    logger.info(f"{'='*60}\n")
    


if __name__ == "__main__":
    main()