#!/usr/bin/env python3
"""
Analizador de tamaños de imágenes - Versión dinámico y generalizable
Genera un informe detallado de los tamaños de imágenes en cualquier directorio.

IMPORTANTE: Este script es completamente dinámico y NO está hardcodeado para ningún directorio específico.
"""

import os
import sys
from pathlib import Path
from PIL import Image
from collections import Counter
from datetime import datetime

# ==================================================================================
# CONFIGURACIÓN - MODIFICAR ESTAS VARIABLES SEGÚN NECESIDADES
# ==================================================================================

# Directorio a analizar (relativo al directorio del script o ruta absoluta)
TARGET_DIRECTORY = "tomato_dataset"

# Extensiones de imagen a buscar
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}

# Prefijo para el nombre del archivo de reporte
REPORT_PREFIX = "image_sizes_report"

# Directorio donde guardar el reporte (None = directorio padre del target)
REPORT_OUTPUT_DIR = None

# Mostrar progreso cada N imágenes procesadas
PROGRESS_INTERVAL = 1000

# Incluir listado detallado de todas las imágenes en el reporte
INCLUDE_DETAILED_LIST = True

# Ordenar imágenes por (opciones: 'name', 'size', 'pixels')
SORT_IMAGES_BY = "name"

# Mostrar solo los N tamaños más comunes en el resumen (0 = mostrar todos)
TOP_SIZES_LIMIT = 0

# ==================================================================================
# FIN DE CONFIGURACIÓN
# ==================================================================================


class ImageSizeAnalyzer:
    """Analizador dinámico de tamaños de imágenes que funciona con cualquier directorio."""

    def __init__(self):
        """Inicializa el analizador usando las variables de configuración."""
        # Determinar el directorio objetivo
        if os.path.isabs(TARGET_DIRECTORY):
            self.target_directory = TARGET_DIRECTORY
        else:
            # Ruta relativa al directorio del script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.target_directory = os.path.join(script_dir, TARGET_DIRECTORY)

        # Configurar directorio de salida del reporte
        if REPORT_OUTPUT_DIR is None:
            self.report_output_dir = os.path.dirname(self.target_directory)
        else:
            self.report_output_dir = REPORT_OUTPUT_DIR

        self.image_extensions = IMAGE_EXTENSIONS
        self.results = {"images_found": [], "size_counts": Counter(), "total_images": 0, "errors": []}
        self.processed_count = 0

    def is_image_file(self, file_path):
        """Verifica si un archivo es una imagen basándose en su extensión."""
        return file_path.suffix.lower() in self.image_extensions

    def get_image_size(self, image_path):
        """
        Obtiene el tamaño de una imagen.

        Returns:
            tuple: (width, height) o None si hay error
        """
        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            self.results["errors"].append(f"Error al leer {image_path}: {str(e)}")
            return None

    def scan_directory(self):
        """Escanea recursivamente el directorio objetivo buscando imágenes."""
        print(f"Escaneando directorio: {self.target_directory}")
        print(f"Extensiones de imagen: {', '.join(sorted(self.image_extensions))}")

        if not os.path.exists(self.target_directory):
            raise FileNotFoundError(f"El directorio {self.target_directory} no existe")

        target_path = Path(self.target_directory)

        # Escaneo recursivo
        for image_file in target_path.rglob("*"):
            if image_file.is_file() and self.is_image_file(image_file):
                size = self.get_image_size(image_file)
                if size:
                    width, height = size
                    size_str = f"{width}x{height}"

                    self.results["images_found"].append(
                        {
                            "path": str(image_file),
                            "relative_path": str(image_file.relative_to(target_path)),
                            "size": size_str,
                            "width": width,
                            "height": height,
                            "pixels": width * height,
                        }
                    )

                    self.results["size_counts"][size_str] += 1
                    self.results["total_images"] += 1
                    self.processed_count += 1

                    # Mostrar progreso
                    if PROGRESS_INTERVAL > 0 and self.processed_count % PROGRESS_INTERVAL == 0:
                        print(f"Procesadas {self.processed_count} imágenes...")

        print(f"Análisis completado. Encontradas {self.results['total_images']} imágenes")

    def generate_report(self):
        """Genera un informe completo en formato texto y Markdown."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = os.path.basename(self.target_directory)

        # Nombre del archivo de reporte dinámico usando configuración
        report_filename = f"{REPORT_PREFIX}_{dir_name}_{timestamp}.txt"
        report_path = os.path.join(self.report_output_dir, report_filename)

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_text_report(f, dir_name, timestamp)
            self._write_markdown_table(f)

        print(f"Reporte generado: {report_path}")
        return report_path

    def _write_text_report(self, f, dir_name, timestamp):
        """Escribe la sección de texto del reporte."""
        f.write("=" * 80 + "\n")
        f.write("REPORTE DE ANÁLISIS DE TAMAÑOS DE IMÁGENES\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Directorio analizado: {self.target_directory}\n")
        f.write(f"Nombre del directorio: {dir_name}\n")
        f.write(f"Fecha y hora del análisis: {timestamp}\n")
        f.write(f"Total de imágenes encontradas: {self.results['total_images']}\n")
        f.write(f"Extensiones procesadas: {', '.join(sorted(self.image_extensions))}\n\n")

        # Resumen de tamaños únicos
        f.write("RESUMEN DE TAMAÑOS ENCONTRADOS:\n")
        f.write("-" * 40 + "\n")

        # Aplicar límite si está configurado
        size_items = self.results["size_counts"].most_common()
        if TOP_SIZES_LIMIT > 0:
            size_items = size_items[:TOP_SIZES_LIMIT]
            f.write(f"(Mostrando los {TOP_SIZES_LIMIT} tamaños más comunes)\n")

        for size, count in size_items:
            f.write(f"{size:<15} : {count:>4} imagen(es)\n")

        f.write(f"\nTotal de tamaños únicos: {len(self.results['size_counts'])}\n")

        if TOP_SIZES_LIMIT > 0 and len(self.results["size_counts"]) > TOP_SIZES_LIMIT:
            f.write(f"(Se muestran solo los primeros {TOP_SIZES_LIMIT})\n")

        f.write("\n")

        # Errores si los hay
        if self.results["errors"]:
            f.write("ERRORES ENCONTRADOS:\n")
            f.write("-" * 20 + "\n")
            for error in self.results["errors"]:
                f.write(f"- {error}\n")
            f.write("\n")

    def _write_markdown_table(self, f):
        """Escribe la tabla en formato Markdown."""
        f.write("# TABLA DETALLADA DE IMÁGENES (Formato Markdown)\n\n")

        # Tabla de resumen por tamaños
        f.write("## Resumen por Tamaños\n\n")
        f.write("| Tamaño (píxeles) | Cantidad | Megapíxeles Aprox |\n")
        f.write("|------------------|----------|-------------------|\n")

        # Aplicar límite si está configurado
        size_items = self.results["size_counts"].most_common()
        if TOP_SIZES_LIMIT > 0:
            size_items = size_items[:TOP_SIZES_LIMIT]

        for size, count in size_items:
            width, height = size.split("x")
            megapixels = round((int(width) * int(height)) / 1_000_000, 2)
            f.write(f"| {size:<15} | {count:>8} | {megapixels:>17} |\n")

        f.write("\n")

        # Tabla detallada de todas las imágenes (solo si está habilitado)
        if INCLUDE_DETAILED_LIST:
            f.write("## Listado Detallado de Imágenes\n\n")
            f.write("| # | Archivo | Tamaño | Ancho | Alto | Megapíxeles |\n")
            f.write("|---|---------|--------|-------|------|-------------|\n")

            # Ordenar imágenes según configuración
            if SORT_IMAGES_BY == "name":
                sorted_images = sorted(self.results["images_found"], key=lambda x: x["relative_path"])
            elif SORT_IMAGES_BY == "size":
                sorted_images = sorted(self.results["images_found"], key=lambda x: x["size"])
            elif SORT_IMAGES_BY == "pixels":
                sorted_images = sorted(self.results["images_found"], key=lambda x: x["pixels"], reverse=True)
            else:
                sorted_images = self.results["images_found"]

            for i, img in enumerate(sorted_images, 1):
                megapixels = round(img["pixels"] / 1_000_000, 2)
                f.write(f"| {i:>3} | {img['relative_path']:<30} | {img['size']:<10} | {img['width']:>5} | {img['height']:>4} | {megapixels:>11} |\n")
        else:
            f.write("## Listado Detallado de Imágenes\n\n")
            f.write("*Listado detallado deshabilitado en configuración (INCLUDE_DETAILED_LIST = False)*\n\n")


def main():
    """Función principal que usa las configuraciones definidas al inicio del script."""
    print("=" * 60)
    print("ANALIZADOR DE TAMAÑOS DE IMÁGENES")
    print("=" * 60)
    print(f"Directorio objetivo: {TARGET_DIRECTORY}")
    print(f"Extensiones de imagen: {', '.join(sorted(IMAGE_EXTENSIONS))}")
    print(f"Prefijo de reporte: {REPORT_PREFIX}")

    if REPORT_OUTPUT_DIR:
        print(f"Directorio de salida: {REPORT_OUTPUT_DIR}")
    else:
        print("Directorio de salida: Directorio padre del target")

    print(f"Incluir listado detallado: {'Sí' if INCLUDE_DETAILED_LIST else 'No'}")
    print(f"Ordenar imágenes por: {SORT_IMAGES_BY}")

    if TOP_SIZES_LIMIT > 0:
        print(f"Mostrar solo top {TOP_SIZES_LIMIT} tamaños")
    else:
        print("Mostrar todos los tamaños")

    print("-" * 60)

    try:
        analyzer = ImageSizeAnalyzer()
        analyzer.scan_directory()
        report_path = analyzer.generate_report()

        print("\n" + "=" * 50)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 50)
        print(f"Directorio analizado: {analyzer.target_directory}")
        print(f"Imágenes encontradas: {analyzer.results['total_images']}")
        print(f"Tamaños únicos: {len(analyzer.results['size_counts'])}")
        print(f"Reporte guardado en: {report_path}")

        if analyzer.results["errors"]:
            print(f"Errores encontrados: {len(analyzer.results['errors'])}")

    except Exception as e:
        print(f"Error durante el análisis: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
