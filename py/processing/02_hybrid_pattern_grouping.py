#!/usr/bin/env python3
"""
PASO 2 MEJORADO: AGRUPACIÓN HÍBRIDA CON VERIFICACIÓN VISUAL SELECTIVA
Combina patrones descobertos + verificación SSIM para grupos candidatos.
Resuelve casos como lb_269.png vs lb_269.jpg y archivos con mismo patrón pero contenido diferente.
"""

import os
import re
import json
import cv2
import numpy as np
from collections import defaultdict
import time
import csv
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# ==================== CONFIGURACIÓN PRINCIPAL ====================
# 🎯 DIRECTORIO A ANALIZAR - CAMBIAR AQUÍ FÁCILMENTE:
DEFAULT_TARGET_DIRECTORY = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/same3/train/tomato_yellow_leaf_curl_virus"


def load_discovered_patterns():
    """Carga los 1,412 patrones descobertos del dataset completo."""
    try:
        with open("patterns_for_algorithm.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        patterns = data["frequent_patterns"]
        print(f"✅ Cargados {len(patterns)} patrones descobertos del dataset completo")
        return patterns
    except FileNotFoundError:
        print("❌ Archivo patterns_for_algorithm.json no encontrado")
        print("   Ejecute primero: python 01_extract_all_real_patterns.py")
        return {}


def extract_base_identifier_ultimate(filename, target_directory, patterns_data=None):
    """
    Versión definitiva que extrae el identificador base removiendo transformaciones
    en el orden correcto para evitar conflictos con transformaciones compuestas.
    INCLUYE CASOS ESPECIALES PARA DIRECTORIO HEALTHY
    """
    # CASO ESPECIAL: Verificar primero si es un patrón semántico de HEALTHY
    healthy_identifier = extract_healthy_special_cases(filename, target_directory)
    if healthy_identifier:
        return healthy_identifier

    name_without_ext = os.path.splitext(filename)[0].lower()

    # PASO 1: Remover transformaciones conocidas (ORDEN IMPORTANTE)
    clean_name = name_without_ext
    transformations = [
        "new30degfliplr",  # MÁS ESPECÍFICO PRIMERO
        "change_90",
        "change_180",
        "change_270",
        "change_360",
        "change",
        "mirror_vertical",
        "mirror_horizontal",
        "mirror",  # DESPUÉS de mirror_vertical/horizontal
        "new30deg",  # DESPUÉS de new30degfliplr
        "hight",
        "lower",
        "fliptb",
        "fliplr",
        "flip",  # DESPUÉS de fliptb/fliplr
        "180deg",
        "90deg",
        "270deg",
        "30deg",
        "newpixel25",
        "bright",
        "dark",
        "blur",
        "sharp",
        "noise",
        "contrast",
    ]

    for transform in transformations:
        clean_name = clean_name.replace(f"_{transform}", "")

    # PASO 2: Buscar patrones específicos descobertos
    # 2a. Patrones _lbXX, _pmXX específicos
    lb_match = re.search(r"_lb(\d+)", clean_name)
    if lb_match:
        return f"lb_{lb_match.group(1)}"

    pm_match = re.search(r"_pm(\d+)", clean_name)
    if pm_match:
        return f"pm_{pm_match.group(1)}"

    # PASO 3: Extraer partes significativas (ignorando hashes)
    parts = clean_name.split("_")

    # Filtrar hashes hexadecimales pero conservar números importantes
    significant_parts = []
    for part in parts:
        if len(part) >= 8 and re.match(r"^[a-f0-9]+$", part):
            continue  # Ignorar hashes hexadecimales válidos
        elif len(part) >= 2 or part.isdigit():  # CAMBIO: >= 2 en lugar de > 2
            significant_parts.append(part)

    # PASO 3.5: IDENTIFICADORES ÚNICOS PARA img_XX_3
    for i in range(len(significant_parts) - 2):
        if (
            significant_parts[i] == "img"
            and i + 1 < len(significant_parts)
            and significant_parts[i + 1].isdigit()
            and i + 2 < len(significant_parts)
            and significant_parts[i + 2] == "3"
        ):
            img_number = significant_parts[i + 1]
            return f"img_{img_number}_3"

    # PASO 3.6: IDENTIFICADORES PARA figure_X_Y (SOLUCIÓN PARA REGRESIÓN)
    for i in range(len(significant_parts) - 2):
        if (
            significant_parts[i] == "figure"
            and i + 1 < len(significant_parts)
            and significant_parts[i + 1].isdigit()
            and i + 2 < len(significant_parts)
            and significant_parts[i + 2].isdigit()
        ):
            x_number = significant_parts[i + 1]
            y_number = significant_parts[i + 2]
            return f"figure_{x_number}_{y_number}"

    # PASO 4: Retornar partes significativas para análisis posterior
    if len(significant_parts) >= 2:
        return "_".join(significant_parts[-2:])
    elif len(significant_parts) == 1:
        return significant_parts[0]

    # Último recurso
    return f"unique_{name_without_ext[:20]}"


def extract_healthy_special_cases(filename, target_directory):
    """
    CASO ESPECIAL PARA DIRECTORIO HEALTHY:
    Maneja patrones semánticos específicos donde variantes de la misma muestra biológica
    deben agruparse juntas, independientemente de diferencias en nomenclatura.

    PATRONES DETECTADOS EN HEALTHY:
    - Tipo A: hl_382.jpg ↔ gh_hl_leaf_382_1.jpg, gh_hl_leaf_382_2.jpg
    - Tipo B: gh_hl_leaf_429.jpg ↔ gh_hl_leaf_429_1.jpg
    """
    # Solo aplicar en directorio healthy
    if not target_directory.endswith("/healthy"):
        return None

    name_without_ext = os.path.splitext(filename)[0].lower()

    # PATRÓN TIPO A: hl_XXX ↔ gh_hl_leaf_XXX_Y
    # Buscar hl_número
    hl_match = re.search(r"_hl_(\d+)$", name_without_ext)
    if hl_match:
        number = hl_match.group(1)
        return f"healthy_leaf_{number}"

    # Buscar gh_hl_leaf_número_variante o gh_hl_leaf_número
    gh_hl_match = re.search(r"_gh_hl_leaf_(\d+)(?:_\d+)?$", name_without_ext)
    if gh_hl_match:
        number = gh_hl_match.group(1)
        return f"healthy_leaf_{number}"

    # PATRÓN TIPO B: rs_hl_XXXX ↔ variantes similares
    rs_hl_match = re.search(r"_rs_hl_(\d+)", name_without_ext)
    if rs_hl_match:
        number = rs_hl_match.group(1)
        return f"healthy_rs_{number}"

    return None  # No es un caso especial de healthy


def load_and_resize_image(image_path, size=(64, 64)):
    """Carga y redimensiona imagen para comparación rápida."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convertir a escala de grises y redimensionar
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, size)
        return resized
    except Exception as e:
        print(f"❌ Error cargando {image_path}: {e}")
        return None


def calculate_similarity(img1, img2):
    """Calcula similitud SSIM entre dos imágenes."""
    try:
        # Calcular SSIM
        similarity_index = ssim(img1, img2, data_range=255)
        return similarity_index
    except Exception as e:
        print(f"❌ Error calculando similitud: {e}")
        return 0.0


def should_apply_visual_verification(group_files):
    """
    Determina si un grupo necesita verificación visual SSIM.
    Solo aplica a casos ambiguos: mismo nombre base con diferentes extensiones/hashes
    """
    if len(group_files) < 2:
        return False

    # Extraer nombres base sin extensión y extensiones
    file_info = []
    for file_path in group_files:
        filename = os.path.basename(file_path)
        # Separar nombre base y extensión
        base_name, extension = os.path.splitext(filename)

        # Remover el hash del principio (formato: hash_nombrearchivo)
        clean_name = base_name
        if "_" in base_name:
            parts = base_name.split("_", 1)
            if len(parts) > 1 and re.match(r"^[a-f0-9]{8,}$", parts[0]):
                clean_name = parts[1]

        file_info.append({"path": file_path, "clean_name": clean_name, "extension": extension, "full_base": base_name})

    # Buscar patrones que indican transformaciones obvias
    transformation_suffixes = [
        "_change_90",
        "_change_180",
        "_change_270",
        "_change_360",
        "_change",
        "_mirror_vertical",
        "_mirror_horizontal",
        "_mirror",
        "_hight",
        "_lower",
        "_flip",
        "_fliptb",
        "_fliplr",
        "_180deg",
        "_90deg",
        "_270deg",
        "_30deg",
        "_new30deg",
        "_new30degfliplr",
        "_newpixel25",
        "_bright",
        "_dark",
        "_blur",
        "_sharp",
        "_noise",
        "_contrast",
    ]

    # Si algún archivo tiene sufijos de transformación, NO aplicar SSIM
    for info in file_info:
        for suffix in transformation_suffixes:
            if suffix in info["clean_name"]:
                return False

    # CASO 1: Mismo nombre limpio con diferentes extensiones (ej: lb_269.jpg vs lb_269.png)
    clean_names = [info["clean_name"] for info in file_info]
    extensions = [info["extension"] for info in file_info]

    if len(set(clean_names)) == 1 and len(set(extensions)) > 1:
        print(f"  → SSIM requerido: Mismo nombre '{clean_names[0]}' con diferentes extensiones")
        return True

    # CASO 2: Mismo nombre limpio con diferentes hashes (ej: hash1_img_1767_3.jpg vs hash2_img_1767_3.jpg)
    if len(set(clean_names)) == 1 and len(set(extensions)) == 1:
        full_bases = [info["full_base"] for info in file_info]
        if len(set(full_bases)) > 1:
            print(f"  → SSIM requerido: Mismo nombre '{clean_names[0]}' con diferentes hashes")
            return True

    return False


def verify_visual_similarity(group_files, threshold=0.85):
    """
    Verifica similitud visual entre archivos de un grupo usando SSIM
    Solo aplica verificación en casos ambiguos
    """
    if not should_apply_visual_verification(group_files):
        return True  # Aceptar grupo sin verificación SSIM

    if len(group_files) < 2:
        return True

    print(f"  → Aplicando verificación SSIM (caso ambiguo detectado)")
    similarities = []
    for i, file1 in enumerate(group_files):
        for j, file2 in enumerate(group_files[i + 1 :], i + 1):
            try:
                # CARGAR LAS IMÁGENES ANTES DE COMPARARLAS
                img1 = load_and_resize_image(file1)
                img2 = load_and_resize_image(file2)

                if img1 is None or img2 is None:
                    print(f"  ❌ Error cargando imágenes: {os.path.basename(file1)} o {os.path.basename(file2)}")
                    return False

                similarity = calculate_similarity(img1, img2)
                similarities.append(similarity)
                print(f"  SSIM entre {os.path.basename(file1)} y {os.path.basename(file2)}: {similarity:.3f}")
            except Exception as e:
                print(f"  ❌ Error calculando SSIM: {e}")
                return False

    # Si todas las similitudes están por encima del umbral, es un grupo válido
    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity >= threshold
    return True


def get_transformation_type(filename):
    """Identifica el tipo de transformación."""
    name_lower = filename.lower()

    # Detectar transformaciones complejas primero (más específicas)
    if "new30degfliplr" in name_lower:
        return "rotate_30_flip_horizontal"
    elif "change_90" in name_lower:
        return "rotate_90"
    elif "change_180" in name_lower:
        return "rotate_180"
    elif "change_270" in name_lower:
        return "rotate_270"
    elif "mirror_vertical" in name_lower:
        return "mirror_vertical"
    elif "mirror_horizontal" in name_lower or "mirror" in name_lower:
        return "mirror_horizontal"
    elif "hight" in name_lower:
        return "brightness_high"
    elif "lower" in name_lower:
        return "brightness_low"
    elif "fliptb" in name_lower:
        return "flip_vertical"
    elif "fliplr" in name_lower:
        return "flip_horizontal"
    elif "flip" in name_lower:
        return "flip"
    elif "180deg" in name_lower:
        return "rotate_180"
    elif "90deg" in name_lower:
        return "rotate_90"
    elif "270deg" in name_lower:
        return "rotate_270"
    elif "new30deg" in name_lower or "30deg" in name_lower:
        return "rotate_30"
    elif "newpixel" in name_lower:
        return "pixel_transform"
    else:
        return "identity"


def hybrid_group_analysis(target_directory):
    """
    Análisis híbrido: patrones + verificación visual selectiva.
    """
    start_time = time.time()

    # Cargar patrones descobertos
    discovered_patterns = load_discovered_patterns()
    if not discovered_patterns:
        print("❌ No se pudieron cargar los patrones. Abortando.")
        return None, None, 0

    # Lista de imágenes
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    all_files = [f for f in os.listdir(target_directory) if f.lower().endswith(extensions)]

    print(f"🚀 ANÁLISIS HÍBRIDO CON VERIFICACIÓN VISUAL SELECTIVA")
    print(f"Directorio: {target_directory}")
    print(f"Total de archivos: {len(all_files)}")

    # PASO 1: Agrupar por patrones (rápido)
    pattern_groups = defaultdict(list)

    for filename in all_files:
        base_id = extract_base_identifier_ultimate(filename, target_directory, discovered_patterns)
        pattern_groups[base_id].append(filename)

    print(f"🔍 Grupos iniciales por patrón: {len(pattern_groups)}")

    # PASO 2: Verificación visual selectiva para grupos candidatos
    final_groups = {}
    files_without_transformations = []
    visual_verifications = 0
    grouped_files = 0

    for base_id, candidate_files in pattern_groups.items():
        if len(candidate_files) == 1:
            # Un solo archivo, sin transformaciones
            files_without_transformations.append(candidate_files[0])
        else:
            # Múltiples candidatos, verificar si necesitan análisis visual
            print(f"\n🔍 Analizando grupo '{base_id}' con {len(candidate_files)} candidatos:")

            # Verificar si el grupo pasa la verificación visual
            if verify_visual_similarity(candidate_files):
                # Grupo válido - todos los archivos son del mismo conjunto
                base_file = min(candidate_files, key=lambda f: len(f))
                related_files = [f for f in candidate_files if f != base_file]

                final_groups[base_file] = {
                    "related_files": related_files,
                    "transformations_found": list(set(get_transformation_type(f) for f in candidate_files)),
                    "group_identifier": base_id,
                    "total_files": len(candidate_files),
                    "verification_method": "pattern_confirmed" if not should_apply_visual_verification(candidate_files) else "visual_confirmed",
                }
                grouped_files += len(candidate_files)
            else:
                # Grupo rechazado por verificación visual - son archivos únicos
                visual_verifications += 1
                for file_path in candidate_files:
                    files_without_transformations.append(file_path)

    analysis_time = time.time() - start_time

    print(f"\n✅ ANÁLISIS HÍBRIDO COMPLETADO EN {analysis_time:.3f}s")
    print(f"📊 Estadísticas FINALES:")
    print(f"   - Verificaciones visuales realizadas: {visual_verifications}")
    print(f"   - Grupos verificados: {len(final_groups)}")
    print(f"   - Archivos agrupados: {sum(group['total_files'] for group in final_groups.values())}")
    print(f"   - Archivos únicos: {len(files_without_transformations)}")

    return final_groups, files_without_transformations, analysis_time


def generate_hybrid_report(transformation_groups, files_without_transformations, analysis_time, target_dir):
    """Genera reporte del análisis híbrido"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.basename(target_dir)
    report_file = f"hibrid_analysis_{dir_name}_{timestamp}.txt"
    csv_file = f"hibrid_analysis_{dir_name}_{timestamp}.csv"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("ANÁLISIS HÍBRIDO CON VERIFICACIÓN VISUAL SELECTIVA\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Directorio: {target_dir}\n")
        f.write(f"Tiempo de análisis: {analysis_time:.3f}s\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        total_with_transforms = sum(group["total_files"] for group in transformation_groups.values())
        total_files = total_with_transforms + len(files_without_transformations)

        f.write("ESTADÍSTICAS FINALES:\n")
        f.write(f"- Total de archivos: {total_files}\n")
        f.write(f"- Grupos verificados: {len(transformation_groups)}\n")
        f.write(f"- Archivos agrupados: {total_with_transforms}\n")
        f.write(f"- Archivos únicos: {len(files_without_transformations)}\n")
        f.write(f"- Eficiencia: {(total_with_transforms / total_files) * 100:.1f}% archivos agrupados\n\n")

        f.write("GRUPOS VERIFICADOS VISUALMENTE:\n")
        f.write("-" * 50 + "\n")
        for base_file, info in transformation_groups.items():
            all_files = [base_file] + info["related_files"]
            f.write(f"\nGrupo: {info['group_identifier']} ({info['total_files']} archivos)\n")
            f.write(f"Método: {info['verification_method']}\n")
            f.write(f"Transformaciones: {', '.join(info['transformations_found'])}\n")
            f.write("Archivos:\n")
            for file in all_files:
                f.write(f"  - {file}\n")

        if files_without_transformations:
            f.write(f"\nARCHIVOS ÚNICOS ({len(files_without_transformations)} total):\n")
            f.write("-" * 40 + "\n")
            for file in files_without_transformations:
                f.write(f"- {file}\n")

    # CSV detallado
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "group_id", "has_transformations", "group_size", "transform_type", "verification_method"])

        for base_file, info in transformation_groups.items():
            all_files = [base_file] + info["related_files"]
            for file in all_files:
                transform_type = get_transformation_type(file)
                writer.writerow([file, info["group_identifier"], "YES", info["total_files"], transform_type, info["verification_method"]])

        for file in files_without_transformations:
            base_id = f"unique_{file[:10]}"
            transform_type = get_transformation_type(file)
            writer.writerow([file, base_id, "NO", 1, transform_type, "unique_confirmed"])

    return report_file, csv_file


if __name__ == "__main__":
    import sys

    # Directorio de destino para análisis híbrido
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        target_directory = DEFAULT_TARGET_DIRECTORY
        print(f"⚠️  Usando directorio por defecto: {target_directory}")

    print("🔥 INICIANDO ANÁLISIS HÍBRIDO CON VERIFICACIÓN VISUAL...")

    # Análisis híbrido
    result = hybrid_group_analysis(target_directory)
    if result[0] is not None:
        transformation_groups, files_without_transformations, analysis_time = result

        # Generar reportes
        report_file, csv_file = generate_hybrid_report(transformation_groups, files_without_transformations, analysis_time, target_directory)

        print(f"\n🎉 PROCESO HÍBRIDO COMPLETADO")
        print(f"📄 Reporte híbrido: {report_file}")
        print(f"📊 CSV híbrido: {csv_file}")
        print(f"⚡ Tiempo total: {analysis_time:.3f}s")

        # Mostrar resumen ejecutivo
        total_files = sum(group["total_files"] for group in transformation_groups.values()) + len(files_without_transformations)
        grouped_files = sum(group["total_files"] for group in transformation_groups.values())
        efficiency = (grouped_files / total_files) * 100

        print(f"\n📈 RESUMEN EJECUTIVO:")
        print(f"   🎯 Eficiencia: {efficiency:.1f}% archivos agrupados")
        print(f"   📊 {len(transformation_groups)} grupos vs {len(files_without_transformations)} únicos")
        print(f"   ⚡ Velocidad: {total_files / analysis_time:.0f} archivos/segundo")
    else:
        print("❌ Análisis fallido. Verifique que exista patterns_for_algorithm.json")
