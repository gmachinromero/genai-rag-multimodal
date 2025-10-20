# analyze_assets/utils.py

# Funciones auxiliares reutilizables

# --- Librerías ---
import base64
import cv2
from pathlib import Path
from typing import List, Literal, Optional

from .config import EXTENSIONES_IMAGEN, EXTENSIONES_AUDIO, EXTENSIONES_VIDEO


# --- Funciones Auxiliares ---

def obtener_tipo_activo(ruta_archivo: Path) -> tuple[Optional[Literal["imagen", "audio", "video"]], str]:
    """Determina el tipo de activo según la extensión del archivo."""
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
    """Extrae frames clave de un video y los devuelve en base64."""
    cap = cv2.VideoCapture(str(ruta_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError(f"No se pudieron leer frames del video: {ruta_video.name}")
    
    posiciones = [int(total_frames * i / (num_frames + 1)) for i in range(1, num_frames + 1)]
    frames_base64 = []
    
    for pos in posiciones:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            frames_base64.append(base64.b64encode(buffer).decode('utf-8'))
    
    cap.release()
    return frames_base64
