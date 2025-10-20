# analyze_assets/core.py

#  Lógica principal

# --- Librerías ---
import pandas as pd
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

from .models import AnalisisActivo
from .utils import instanciar_logger, obtener_tipo_activo, codificar_imagen_base64, extraer_frames_video
from .config import (
    DIRECTORIO_DATOS_RAW, DIRECTORIO_DATOS_PROCESADOS, ARCHIVO_SALIDA_CSV,
    MODELO_LLM_OPENAI, MODELO_WHISPER_OPENAI, OPENAI_API_KEY
)


# --- Instanciar logger ---

logger = instanciar_logger(__name__)



# --- Funciones de análisis ---

def analizar_imagen_con_llm(ruta_imagen: Path, llm_estructurado: ChatOpenAI) -> AnalisisActivo:
    logger.info(f"Analizando imagen: {ruta_imagen.name}")
    datos_imagen_b64 = codificar_imagen_base64(ruta_imagen)
    extension = ruta_imagen.suffix[1:]
    mensaje = HumanMessage(
        content=[
            {"type": "text", "text": (
                "Analiza esta imagen y extrae:\n"
                "- El idioma de cualquier texto visible\n"
                "- 3-5 etiquetas descriptivas\n"
                "- Transcripción completa del texto (si existe)\n"
                "- Un resumen breve\n\n"
                "Responde SOLO con el formato JSON estructurado."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/{extension};base64,{datos_imagen_b64}"}}
        ]
    )

    return llm_estructurado.invoke([mensaje])


def analizar_audio_con_llm(ruta_audio: Path, cliente_openai: OpenAI, llm_estructurado: ChatOpenAI) -> AnalisisActivo:
    logger.info(f"Transcribiendo audio: {ruta_audio.name}")

    # Transcribir audio
    with open(ruta_audio, "rb") as archivo_audio:
        transcripcion = cliente_openai.audio.transcriptions.create(
            model=MODELO_WHISPER_OPENAI,
            file=archivo_audio,
            response_format="text"
        )

    # Analizar audio
    mensaje = HumanMessage(content=[{
        "type": "text",
        "text": (
            "Analiza esta transcripción de audio y extrae:\n"
            "- El idioma del audio\n"
            "- 3-5 etiquetas descriptivas del contenido\n"
            "- Un resumen breve\n\n"
            f"Transcripción:\n{transcripcion}\n\n"
            "Responde SOLO con el formato JSON estructurado."
        )
    }])
    analisis = llm_estructurado.invoke([mensaje])
    analisis.transcripcion = str(transcripcion)
    
    return analisis


def analizar_video_con_llm(ruta_video: Path, cliente_openai: OpenAI, llm_estructurado: ChatOpenAI, num_frames: int = 10) -> AnalisisActivo:
    logger.info(f"Procesando video: {ruta_video.name}")

    # Obtener frames
    frames_base64 = extraer_frames_video(ruta_video, num_frames)

    # Transcribir audio
    try:
        with open(ruta_video, "rb") as archivo_video:
            transcripcion = cliente_openai.audio.transcriptions.create(
                model=MODELO_WHISPER_OPENAI,
                file=archivo_video,
                response_format="text"
            )
    except Exception as e:
        logger.warning(f"⚠ No se pudo transcribir audio: {e}")
        transcripcion = ""
    
    # Analizar frames y transcripción del audio
    contenido = [{
        "type": "text",
        "text": (
            "Analiza este video basándote en los frames extraídos"
            f"{' y la transcripción del audio' if transcripcion else ''}.\n\n"
            "Extrae:\n"
            "- El idioma del contenido (audio o texto visible)\n"
            "- 3-5 etiquetas descriptivas del video\n"
            "- Un resumen breve del contenido\n\n"
        )
    }]

    for frame in frames_base64:
        contenido.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
    
    if transcripcion:
        contenido[0]["text"] += f"\nTranscripción del audio:\n{transcripcion}\n\n"
        
    contenido[0]["text"] += "Responde SOLO con el formato JSON estructurado."

    analisis = llm_estructurado.invoke([HumanMessage(content=contenido)])
    analisis.transcripcion = str(transcripcion)
    
    return analisis



# --- Procesamiento de directorio ---

def procesar_directorio(directorio: Path, llm_estructurado: ChatOpenAI, cliente_openai: OpenAI) -> pd.DataFrame:
    if not directorio.exists():
        raise FileNotFoundError(f"El directorio no existe: {directorio}")
    resultados = []
    id_archivo = 1
    archivos = sorted(directorio.iterdir())
    logger.info(f"Encontrados {len(archivos)} archivos para procesar")

    for archivo in archivos:
        if archivo.is_dir() or archivo.name.startswith("."):
            continue
        tipo, formato = obtener_tipo_activo(archivo)
        if tipo is None:
            logger.warning(f"Tipo no soportado: {archivo.name}")
            continue

        try:
            if tipo == "imagen":
                analisis = analizar_imagen_con_llm(archivo, llm_estructurado)
            elif tipo == "audio":
                analisis = analizar_audio_con_llm(archivo, cliente_openai, llm_estructurado)
            elif tipo == "video":
                analisis = analizar_video_con_llm(archivo, cliente_openai, llm_estructurado)
            else:
                continue

            fila = analisis.model_dump()
            fila.update({
                "id": id_archivo,
                "nombre_archivo": archivo.name,
                "tipo_asset": tipo,
                "formato_asset": formato
            })
            resultados.append(fila)
            id_archivo += 1
            logger.info(f"✓ Procesado exitosamente: {archivo.name}")

        except Exception as e:
            logger.error(f"✗ Error procesando {archivo.name}: {e}")
            continue

    return pd.DataFrame(resultados)



# --- Main ---

def main():

    # Inicializar clientes
    logger.info("Inicializando clientes OpenAI...")
    llm = ChatOpenAI(model=MODELO_LLM_OPENAI, temperature=0, api_key=OPENAI_API_KEY)
    llm_estructurado = llm.with_structured_output(AnalisisActivo)
    cliente_openai = OpenAI(api_key=OPENAI_API_KEY)

    # Procesar archivos
    logger.info(f"Procesando archivos en: {DIRECTORIO_DATOS_RAW.absolute()}")
    df = procesar_directorio(DIRECTORIO_DATOS_RAW, llm_estructurado, cliente_openai)

    if df.empty:
        logger.warning("No se procesaron archivos o no se obtuvieron resultados.")
        return

    # Comprobar ruta destino
    DIRECTORIO_DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)

    # Ordenar columnas
    columnas = ['id', 'nombre_archivo', 'tipo_asset', 'formato_asset', 'idioma', 'etiquetas', 'transcripcion', 'resumen']
    df = df[columnas]

    # Pasar de campo lista, a string con pipes
    df['etiquetas'] = df['etiquetas'].str.join(' | ')

    # Limpiar transcripcion y resumen
    for col in ['transcripcion', 'resumen']:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.replace(r"[\n\r]+", " ", regex=True)
                .str.strip()
            )

    # Guardar fichero
    df.to_csv(ARCHIVO_SALIDA_CSV, index=False, encoding='utf-8-sig')

    logger.info(f"✓ Resultados guardados en: {ARCHIVO_SALIDA_CSV.absolute()}")
