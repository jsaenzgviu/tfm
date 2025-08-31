#!/usr/bin/env python3
"""
Script para redimensionar imágenes a 224x224 manteniendo proporción.
Código totalmente dinámico y generalizable.
"""

import os
import sys
from PIL import Image
import glob
from pathlib import Path
import argparse
from datetime import datetime
import json


def get_edge_color(image, sample_size=10):
    """
    Obtiene el color predominante de los bordes de la imagen.
    Para machine learning, es mejor usar un color promedio de los bordes
    que negro o blanco arbitrarios.

    Args:
        image: PIL Image object
        sample_size: Número de píxeles a muestrear por borde

    Returns:
        tuple: Color RGB promedio de los bordes
    """
    width, height = image.size

    # Muestrear píxeles de los bordes
    edge_pixels = []

    # Borde superior e inferior
    for i in range(0, width, max(1, width // sample_size)):
        edge_pixels.append(image.getpixel((i, 0)))  # Superior
        edge_pixels.append(image.getpixel((i, height - 1)))  # Inferior

    # Borde izquierdo y derecho
    for i in range(0, height, max(1, height // sample_size)):
        edge_pixels.append(image.getpixel((0, i)))  # Izquierdo
        edge_pixels.append(image.getpixel((width - 1, i)))  # Derecho

    # Calcular color promedio
    if edge_pixels:
        avg_r = sum(pixel[0] for pixel in edge_pixels) // len(edge_pixels)
        avg_g = sum(pixel[1] for pixel in edge_pixels) // len(edge_pixels)
        avg_b = sum(pixel[2] for pixel in edge_pixels) // len(edge_pixels)
        return (avg_r, avg_g, avg_b)

    # Fallback: color gris neutral para machine learning
    return (128, 128, 128)


def generate_processing_report(source_dir, output_dir, target_size, quality, processed_count, error_count, processing_time):
    """
    Genera un informe detallado del procesamiento para trazabilidad en ciencia de datos.

    Args:
        source_dir: Directorio fuente
        output_dir: Directorio de salida
        target_size: Tamaño objetivo
        quality: Calidad JPEG
        processed_count: Número de imágenes procesadas exitosamente
        error_count: Número de errores
        processing_time: Tiempo de procesamiento en segundos
    """
    # Obtener nombre del directorio fuente para el informe
    source_dir_name = os.path.basename(os.path.abspath(source_dir))

    # Generar timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Nombre del archivo de informe con trazabilidad
    report_filename = f"resize_report_{source_dir_name}_{timestamp}.txt"
    report_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), report_filename)

    # Crear informe detallado
    report_content = f"""
=== INFORME DE REDIMENSIONADO DE IMÁGENES ===
Generado para trazabilidad en ciencia de datos

INFORMACIÓN GENERAL:
- Fecha y hora: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Directorio procesado: {source_dir_name}
- Script utilizado: {os.path.basename(__file__)}
- Usuario/Sistema: {os.getenv("USER", "unknown")}@{os.getenv("HOSTNAME", "unknown")}

CONFIGURACIÓN DE PROCESAMIENTO:
- Directorio fuente: {source_dir}
- Directorio destino: {output_dir}
- Tamaño objetivo: {target_size[0]}x{target_size[1]} píxeles
- Algoritmo de redimensionado: Lanczos
- Calidad JPEG: {quality}%
- Formato de salida: JPEG
- Relleno: Color promedio de bordes

RESULTADOS:
- Imágenes procesadas exitosamente: {processed_count}
- Errores encontrados: {error_count}
- Total de archivos analizados: {processed_count + error_count}
- Tasa de éxito: {(processed_count / (processed_count + error_count) * 100):.1f}%
- Tiempo de procesamiento: {processing_time:.2f} segundos
- Promedio por imagen: {(processing_time / max(1, processed_count)):.3f} segundos

TRANSFORMACIONES APLICADAS:
1. Redimensionado proporcional con algoritmo Lanczos
2. Padding con color promedio de bordes para mantener aspecto cuadrado
3. Conversión de PNG a JPEG (si aplica)
4. Compresión JPEG optimizada
5. Normalización a formato estándar para machine learning

ARCHIVOS GENERADOS:
- Directorio de salida: {output_dir}
- Formato: Todos los archivos convertidos a .jpg
- Dimensiones: Todas las imágenes son exactamente {target_size[0]}x{target_size[1]} píxeles

TRAZABILIDAD:
- Esta transformación es parte del preprocesamiento del dataset
- Mantiene la calidad visual mientras estandariza dimensiones
- Compatible con frameworks de deep learning (PyTorch, TensorFlow)
- Conserva las proporciones originales sin distorsión

NOTAS PARA REPRODUCIBILIDAD:
- Comando usado: resize_image.py --source {source_dir_name} --output {os.path.basename(output_dir)} --size {target_size[0]} {target_size[1]} --quality {quality}
- Versión de PIL/Pillow: {Image.__version__ if hasattr(Image, "__version__") else "N/A"}
- Python: {sys.version.split()[0]}

=== FIN DEL INFORME ===
"""

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\n📋 Informe de trazabilidad generado: {report_filename}")
        print(f"   Ubicación: {report_path}")
        return report_path
    except Exception as e:
        print(f"⚠️  Error generando informe: {e}")
        return None


def generate_json_metadata(source_dir, output_dir, target_size, quality, processed_count, error_count, processing_time):
    """
    Genera metadata en formato JSON para integración con pipelines de ML.

    Args:
        source_dir: Directorio fuente
        output_dir: Directorio de salida
        target_size: Tamaño objetivo
        quality: Calidad JPEG
        processed_count: Número de imágenes procesadas
        error_count: Número de errores
        processing_time: Tiempo de procesamiento
    """
    source_dir_name = os.path.basename(os.path.abspath(source_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metadata_filename = f"resize_metadata_{source_dir_name}_{timestamp}.json"
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), metadata_filename)

    metadata = {
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "source_directory": source_dir,
            "source_directory_name": source_dir_name,
            "output_directory": output_dir,
            "script_name": os.path.basename(__file__),
        },
        "configuration": {
            "target_size": {"width": target_size[0], "height": target_size[1]},
            "resampling_algorithm": "Lanczos",
            "jpeg_quality": quality,
            "output_format": "JPEG",
            "padding_method": "edge_color_average",
        },
        "results": {
            "images_processed": processed_count,
            "errors": error_count,
            "total_files": processed_count + error_count,
            "success_rate": round((processed_count / (processed_count + error_count) * 100), 2) if (processed_count + error_count) > 0 else 0,
            "processing_time_seconds": round(processing_time, 3),
            "average_time_per_image": round((processing_time / max(1, processed_count)), 4),
        },
        "transformations": [
            "proportional_resize_lanczos",
            "square_padding_edge_color",
            "png_to_jpeg_conversion",
            "jpeg_optimization",
            "ml_standardization",
        ],
        "reproducibility": {
            "command": f"resize_image.py --source {source_dir_name} --output {os.path.basename(output_dir)} --size {target_size[0]} {target_size[1]} --quality {quality}",
            "pillow_version": getattr(Image, "__version__", "N/A"),
            "python_version": sys.version.split()[0],
        },
    }

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"📊 Metadata JSON generado: {metadata_filename}")
        return metadata_path
    except Exception as e:
        print(f"⚠️  Error generando metadata JSON: {e}")
        return None


def resize_image_with_padding(image_path, target_size=(224, 224), quality=96):
    """
    Redimensiona una imagen manteniendo la proporción y rellenando con color de borde.

    Args:
        image_path: Ruta a la imagen
        target_size: Tamaño objetivo (ancho, alto)
        quality: Calidad de compresión para JPEG

    Returns:
        PIL.Image: Imagen redimensionada
    """
    try:
        # Abrir imagen
        with Image.open(image_path) as img:
            # Convertir a RGB si es necesario (para PNGs con transparencia)
            if img.mode in ("RGBA", "LA"):
                # Para machine learning, el mejor fondo es un color neutral
                # que no introduzca bias. Usamos el color promedio de los bordes
                # o gris neutral si no se puede calcular
                background_color = (128, 128, 128)  # Gris neutral
                rgb_img = Image.new("RGB", img.size, background_color)
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = rgb_img
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Calcular dimensiones manteniendo proporción
            original_width, original_height = img.size
            target_width, target_height = target_size

            # Calcular factor de escala para mantener proporción
            scale = min(target_width / original_width, target_height / original_height)

            # Nuevas dimensiones
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            # Redimensionar con Lanczos
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Crear imagen final con padding
            final_img = Image.new("RGB", target_size, get_edge_color(resized_img))

            # Calcular posición para centrar
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2

            # Pegar imagen redimensionada en el centro
            final_img.paste(resized_img, (paste_x, paste_y))

            return final_img

    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None


def process_directory(source_dir, output_dir, target_size=(224, 224), quality=96, generate_reports=True):
    """
    Procesa todas las imágenes de un directorio.
    Función completamente dinámica - no hay hardcoding.

    Args:
        source_dir: Directorio fuente
        output_dir: Directorio de salida
        target_size: Tamaño objetivo
        quality: Calidad JPEG
        generate_reports: Si generar informes de trazabilidad
    """
    # Registrar tiempo de inicio para trazabilidad
    start_time = datetime.now()

    # Verificar que el directorio fuente existe
    if not os.path.exists(source_dir):
        print(f"Error: El directorio fuente '{source_dir}' no existe.")
        return False

    # Crear directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Obtener nombre del directorio fuente para uso dinámico
    source_dir_name = os.path.basename(os.path.abspath(source_dir))

    # Patrones de archivos de imagen (dinámico)
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]

    processed_count = 0
    error_count = 0

    print(f"Procesando imágenes del directorio: {source_dir_name}")
    print(f"Directorio fuente: {source_dir}")
    print(f"Directorio destino: {output_dir}")
    print(f"Tamaño objetivo: {target_size}")
    print(f"Calidad JPEG: {quality}")
    print("-" * 60)

    # Procesar cada tipo de archivo
    for extension in image_extensions:
        # Patrón dinámico usando el directorio fuente
        pattern = os.path.join(source_dir, extension)
        files = glob.glob(pattern)

        for file_path in files:
            try:
                # Obtener nombre del archivo sin extensión
                filename = os.path.splitext(os.path.basename(file_path))[0]

                # Procesar imagen
                resized_img = resize_image_with_padding(file_path, target_size, quality)

                if resized_img:
                    # Guardar como JPEG con nombre dinámico
                    output_path = os.path.join(output_dir, f"{filename}.jpg")
                    resized_img.save(output_path, "JPEG", quality=quality, optimize=True)
                    print(f"✓ Procesado: {os.path.basename(file_path)} -> {os.path.basename(output_path)}")
                    processed_count += 1
                else:
                    print(f"✗ Error procesando: {os.path.basename(file_path)}")
                    error_count += 1

            except Exception as e:
                print(f"✗ Error procesando {os.path.basename(file_path)}: {e}")
                error_count += 1

    # Calcular tiempo de procesamiento
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    print("-" * 60)
    print("Resumen del procesamiento:")
    print(f"- Directorio procesado: {source_dir_name}")
    print(f"- Imágenes procesadas exitosamente: {processed_count}")
    print(f"- Errores: {error_count}")
    print(f"- Total archivos: {processed_count + error_count}")
    print(f"- Tiempo de procesamiento: {processing_time:.2f} segundos")
    print(f"- Promedio por imagen: {(processing_time / max(1, processed_count)):.3f} segundos")

    # Generar informes de trazabilidad si está habilitado
    if generate_reports and processed_count > 0:
        print("\n🔍 Generando informes de trazabilidad...")

        # Generar informe detallado en texto
        report_path = generate_processing_report(source_dir, output_dir, target_size, quality, processed_count, error_count, processing_time)

        # Generar metadata en JSON
        metadata_path = generate_json_metadata(source_dir, output_dir, target_size, quality, processed_count, error_count, processing_time)

        if report_path or metadata_path:
            print("✅ Informes de trazabilidad generados exitosamente")
        else:
            print("⚠️  Hubo problemas generando los informes")

    return processed_count > 0


def main():
    """Función principal con configuración dinámica."""
    parser = argparse.ArgumentParser(description="Redimensiona imágenes a 224x224 manteniendo proporción")
    parser.add_argument("--source", "-s", default="./same2/tomato_mosaic_virus", help="Directorio fuente (por defecto: test_size)")
    parser.add_argument("--output", "-o", default="./same3/train/tomato_mosaic_virus", help="Directorio de salida (por defecto: resized_output)")
    parser.add_argument("--size", "-sz", type=int, nargs=2, default=[224, 224], help="Tamaño objetivo [ancho alto] (por defecto: 224 224)")
    parser.add_argument("--quality", "-q", type=int, default=96, help="Calidad JPEG (1-100, por defecto: 96)")
    parser.add_argument("--no-reports", action="store_true", help="Deshabilitar generación de informes de trazabilidad")

    args = parser.parse_args()

    # Obtener directorio del script para rutas relativas
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construir rutas dinámicamente (no hardcodeadas)
    if os.path.isabs(args.source):
        source_directory = args.source
    else:
        source_directory = os.path.join(script_dir, args.source)

    if os.path.isabs(args.output):
        output_directory = args.output
    else:
        output_directory = os.path.join(script_dir, args.output)

    target_size = tuple(args.size)

    # Validar parámetros
    if not (1 <= args.quality <= 100):
        print("Error: La calidad debe estar entre 1 y 100")
        return 1

    if target_size[0] <= 0 or target_size[1] <= 0:
        print("Error: Las dimensiones deben ser positivas")
        return 1

    # Mostrar configuración
    print("=== REDIMENSIONADOR DE IMÁGENES DINÁMICO ===")
    print("Configuración:")
    print(f"  - Directorio fuente: {source_directory}")
    print(f"  - Directorio salida: {output_directory}")
    print(f"  - Tamaño objetivo: {target_size}")
    print(f"  - Calidad JPEG: {args.quality}")
    print(f"  - Informes de trazabilidad: {'Deshabilitados' if args.no_reports else 'Habilitados'}")
    print()

    # Procesar directorio
    success = process_directory(source_directory, output_directory, target_size, args.quality, generate_reports=not args.no_reports)

    if success:
        print("\n✓ Procesamiento completado exitosamente")
        if not args.no_reports:
            print("📋 Consulta los informes generados para trazabilidad del dataset")
        return 0
    else:
        print("\n✗ No se procesaron imágenes")
        return 1


if __name__ == "__main__":
    sys.exit(main())
