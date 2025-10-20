# analyze_assets/models.py

# Define estructuras de datos con Pydantic

# --- Librerías ---
from typing import List, Optional
from pydantic import BaseModel, Field


# --- Clase Pydantic ---
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
