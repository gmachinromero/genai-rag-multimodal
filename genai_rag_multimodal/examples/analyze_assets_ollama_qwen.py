import os
import pandas as pd
import ollama

# Usar Ollama para análisis de imágenes

def obtener_tipo_y_formato_activo(nombre_archivo):
    """Determinar tipo de activo y formato desde el nombre del archivo."""
    ext = nombre_archivo.split('.')[-1].lower()
    if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
        return 'imagen', ext
    else:
        return 'desconocido', ext

def analizar_imagen_con_llm(ruta_imagen):
    """Analizar imagen con Ollama qwen2.5vl:3b."""
    with open(ruta_imagen, 'rb') as f:
        datos_imagen = f.read()

    respuesta = ollama.chat(model='qwen2.5vl:3b', messages=[
        {
            'role': 'user',
            'content': 'Analiza esta imagen y proporciona: idioma de cualquier texto, etiquetas (separadas por comas, máximo 5), transcripción de cualquier texto, y un resumen. Responde en español.',
            'images': [datos_imagen]
        }
    ])

    resultado = respuesta['message']['content']

    # Parsear la respuesta
    lineas = resultado.split('\n')
    idioma = 'desconocido'
    etiquetas = []
    transcripcion = ''
    resumen = ''

    for linea in lineas:
        if 'idioma' in linea.lower():
            idioma = linea.split(':')[-1].strip()
        elif 'etiquetas' in linea.lower():
            etiquetas = [tag.strip() for tag in linea.split(':')[-1].split(',')][:5]  # Máximo 5 etiquetas
        elif 'transcripción' in linea.lower():
            transcripcion = linea.split(':')[-1].strip()
        elif 'resumen' in linea.lower():
            resumen = linea.split(':')[-1].strip()

    return idioma, etiquetas, transcripcion, resumen

def procesar_archivo(ruta_archivo, id_archivo):
    """Procesar un archivo de imagen individual y devolver diccionario de datos."""
    nombre_archivo = os.path.basename(ruta_archivo)
    tipo_activo, formato_activo = obtener_tipo_y_formato_activo(nombre_archivo)

    if tipo_activo == 'imagen':
        idioma, etiquetas, transcripcion, resumen = analizar_imagen_con_llm(ruta_archivo)
    else:
        return None  # Saltar archivos no imagen

    return {
        'id': id_archivo,
        'tipo_asset': tipo_activo,
        'formato_asset': formato_activo,
        'idioma': idioma,
        'etiquetas': etiquetas,
        'transcripción': transcripcion,
        'resumen': resumen
    }

def main():
    directorio_datos = 'data/raw'
    resultados = []

    id_archivo = 1
    for nombre_archivo in os.listdir(directorio_datos):
        if nombre_archivo.startswith('.'):  # Saltar archivos ocultos
            continue
        ruta_archivo = os.path.join(directorio_datos, nombre_archivo)
        if os.path.isfile(ruta_archivo):
            resultado = procesar_archivo(ruta_archivo, id_archivo)
            if resultado:
                resultados.append(resultado)
                id_archivo += 1

    df = pd.DataFrame(resultados)
    print(df)
    # Guardar en CSV en data/processed
    df.to_csv('data/processed/assets_analysis.csv', index=False)

if __name__ == '__main__':
    main()