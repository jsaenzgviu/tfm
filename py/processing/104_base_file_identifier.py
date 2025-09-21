#!/usr/bin/env python3
"""
PASO 2B: IDENTIFICACIÓN DE ARCHIVOS BASE EN GRUPOS
Basado en 02_hybrid_pattern_grouping.py, este script identifica en cada grupo
cuál es el archivo base (sin transformaciones) del cual se derivan todas las variantes.
Los archivos base identificados serán tratados como "únicos" para su posterior movimiento.
"""

import os
import re
import json
import cv2
from collections import defaultdict
import time
import csv
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# ==================================================================================
# CONFIGURACIÓN - MODIFICAR ESTAS VARIABLES SEGÚN NECESIDADES
# ==================================================================================

# Directorio objetivo (relativo al script o ruta absoluta)
TARGET_DIRECTORY = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/same4/train/bacterial_spot"

# Configuraciones de análisis
ENABLE_VISUAL_VERIFICATION = True
VISUAL_SIMILARITY_THRESHOLD = 0.85
SHOW_DETAILED_OUTPUT = True

# ==================================================================================
# FIN DE CONFIGURACIÓN
# ==================================================================================


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
    SOLUCIÓN MEJORADA: Maneja correctamente hashes únicos de 16 caracteres
    """
    # CASO ESPECIAL: Verificar primero si es un patrón semántico de HEALTHY
    healthy_identifier = extract_healthy_special_cases(filename, target_directory)
    if healthy_identifier:
        return healthy_identifier

    name_without_ext = os.path.splitext(filename)[0].lower()

    # PASO 0: REMOVER HASH ÚNICO DE 16 CARACTERES AL INICIO
    # Los primeros 16 caracteres son hash único del archivo - remover para agrupación
    parts_initial = name_without_ext.split("_")
    if len(parts_initial) > 0 and len(parts_initial[0]) == 16 and re.match(r"^[a-f0-9]+$", parts_initial[0]):
        # Remover el hash único inicial
        name_without_hash = "_".join(parts_initial[1:])
        print(f"   🔧 Removiendo hash único: {parts_initial[0]} -> {name_without_hash}")
    else:
        name_without_hash = name_without_ext

    # PASO 1: Remover transformaciones conocidas (ORDEN IMPORTANTE)
    clean_name = name_without_hash
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
        "jpg_rf",  # Añadido para patrones RF
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

    # PASO 3: Extraer partes significativas (ignorando hashes adicionales)
    parts = clean_name.split("_")

    # Filtrar hashes hexadecimales adicionales pero conservar números importantes
    significant_parts = []
    for part in parts:
        if len(part) >= 8 and re.match(r"^[a-f0-9]+$", part):
            continue  # Ignorar hashes hexadecimales válidos adicionales
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

    # Último recurso: usar el nombre sin hash pero sin generar ID único
    # Esto evita que archivos similares se separen por tener hashes diferentes
    if name_without_hash:
        # Tomar las primeras partes significativas del nombre sin hash
        clean_parts = name_without_hash.split("_")[:3]  # Primeras 3 partes
        return "_".join(clean_parts) if clean_parts else "common_pattern"

    return "common_pattern"


def extract_healthy_special_cases(filename, target_directory):
    """
    CASO ESPECIAL PARA DIRECTORIO HEALTHY:
    Maneja patrones semánticos específicos donde variantes de la misma muestra biológica
    deben agruparse juntas, independientemente de diferencias en nomenclatura.
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


def detect_transformation_type(filename):
    """
    Detecta qué tipo de transformación tiene aplicada un archivo.
    Retorna tuple (has_transformation, transformation_type, priority)
    """
    name_lower = filename.lower()

    # Lista de transformaciones con prioridades (menor = más prioritario para ser base)
    transformations = [
        # Transformaciones simples y comunes (mayor prioridad para ser base)
        ("jpg_rf", 10),  # Casos RF especiales
        # Transformaciones de brillo/contraste (prioridad intermedia)
        ("hight", 20),
        ("lower", 20),
        ("bright", 20),
        ("dark", 20),
        ("contrast", 20),
        # Transformaciones geométricas (menor prioridad para ser base)
        ("new30degfliplr", 30),
        ("change_90", 30),
        ("change_180", 30),
        ("change_270", 30),
        ("change_360", 30),
        ("change", 25),  # change genérico tiene menos prioridad que específicos
        ("mirror_vertical", 30),
        ("mirror_horizontal", 30),
        ("mirror", 25),  # mirror genérico
        ("new30deg", 30),
        ("fliptb", 30),
        ("fliplr", 30),
        ("flip", 25),  # flip genérico
        ("180deg", 30),
        ("90deg", 30),
        ("270deg", 30),
        ("30deg", 30),
        ("newpixel25", 30),
        # Efectos (menor prioridad para ser base)
        ("blur", 40),
        ("sharp", 40),
        ("noise", 40),
    ]

    # Buscar transformaciones en orden de especificidad
    for transform, priority in transformations:
        if f"_{transform}" in name_lower:
            return True, transform, priority

    # Si no se encontró ninguna transformación, es archivo base
    return False, "identity", 0


def identify_base_file_in_group(group_files):
    """
    Identifica cuál archivo es la base del grupo (sin transformaciones).
    Retorna el archivo que más probablemente sea el original.
    """
    file_analysis = []

    for file_path in group_files:
        filename = os.path.basename(file_path)
        has_transform, transform_type, priority = detect_transformation_type(filename)

        # Calcular score: archivos sin transformación tienen score 0 (mejor)
        # Archivos con transformaciones tienen scores > 0 (peor)
        score = priority

        # Factor adicional: longitud del nombre (archivos más cortos tienden a ser base)
        name_length_factor = len(filename) * 0.1
        score += name_length_factor

        file_analysis.append(
            {
                "filename": filename,
                "full_path": file_path,
                "has_transformation": has_transform,
                "transformation_type": transform_type,
                "priority": priority,
                "score": score,
            }
        )

    # Ordenar por score (menor score = más probable que sea base)
    file_analysis.sort(key=lambda x: x["score"])

    if SHOW_DETAILED_OUTPUT and len(file_analysis) > 1:
        print(f"  📋 Análisis de grupo ({len(file_analysis)} archivos):")
        for i, analysis in enumerate(file_analysis[:3]):  # Mostrar top 3
            status = "🎯 BASE" if i == 0 else f"  #{i + 1}"
            print(f"     {status}: {analysis['filename']} (score: {analysis['score']:.1f}, transform: {analysis['transformation_type']})")

    # Retornar el archivo con menor score (más probable base)
    return file_analysis[0]["filename"]


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

    # CASO 1: Mismo nombre limpio con diferentes extensiones
    clean_names = [info["clean_name"] for info in file_info]
    extensions = [info["extension"] for info in file_info]

    if len(set(clean_names)) == 1 and len(set(extensions)) > 1:
        return True

    # CASO 2: Mismo nombre limpio con diferentes hashes
    if len(set(clean_names)) == 1 and len(set(extensions)) == 1:
        full_bases = [info["full_base"] for info in file_info]
        if len(set(full_bases)) > 1:
            return True

    return False


def verify_visual_similarity(group_files, threshold=None):
    """
    Verifica similitud visual entre archivos de un grupo usando SSIM
    Solo aplica verificación en casos ambiguos
    """
    if threshold is None:
        threshold = VISUAL_SIMILARITY_THRESHOLD

    if not should_apply_visual_verification(group_files):
        return True  # No necesita verificación, asumir válido

    if len(group_files) < 2:
        return True

    print("  → Aplicando verificación SSIM (caso ambiguo detectado)")
    similarities = []
    for i, file1 in enumerate(group_files):
        img1_path = file1 if os.path.isabs(file1) else os.path.join(TARGET_DIRECTORY, file1)
        img1 = load_and_resize_image(img1_path)
        if img1 is None:
            continue

        for j, file2 in enumerate(group_files[i + 1 :], i + 1):
            img2_path = file2 if os.path.isabs(file2) else os.path.join(TARGET_DIRECTORY, file2)
            img2 = load_and_resize_image(img2_path)
            if img2 is None:
                continue

            similarity = calculate_similarity(img1, img2)
            similarities.append(similarity)

    # Si todas las similitudes están por encima del umbral, es un grupo válido
    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity >= threshold
    return True


def base_file_analysis(target_directory):
    """
    Análisis principal: identifica archivos base en cada grupo.
    """
    start_time = time.time()

    # Cargar patrones descobertos
    discovered_patterns = load_discovered_patterns()
    if not discovered_patterns:
        print("⚠️  Continuando sin patrones descobertos...")

    # Lista de imágenes
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    all_files = [f for f in os.listdir(target_directory) if f.lower().endswith(extensions)]

    print("🚀 IDENTIFICACIÓN DE ARCHIVOS BASE EN GRUPOS")
    print(f"Directorio: {target_directory}")
    print(f"Total de archivos: {len(all_files)}")

    # PASO 1: Agrupar por patrones (rápido)
    pattern_groups = defaultdict(list)

    for filename in all_files:
        base_id = extract_base_identifier_ultimate(filename, target_directory, discovered_patterns)
        pattern_groups[base_id].append(filename)

    print(f"🔍 Grupos iniciales por patrón: {len(pattern_groups)}")

    # PASO 2: Identificar archivo base en cada grupo
    base_files = []
    group_analysis = {}
    visual_verifications = 0

    for base_id, candidate_files in pattern_groups.items():
        if len(candidate_files) == 1:
            # Grupo con un solo archivo - es base por defecto
            base_files.append(candidate_files[0])
            group_analysis[base_id] = {
                "base_file": candidate_files[0],
                "total_files": 1,
                "group_files": candidate_files,
                "method": "single_file",
                "verified": True,
            }
            continue

        # Grupo con múltiples archivos - identificar base
        if ENABLE_VISUAL_VERIFICATION and should_apply_visual_verification(candidate_files):
            # Verificación visual requerida
            visual_verifications += 1
            full_paths = [os.path.join(target_directory, f) for f in candidate_files]
            is_valid_group = verify_visual_similarity(full_paths)

            if not is_valid_group:
                # Grupo no válido - tratar todos como únicos
                base_files.extend(candidate_files)
                group_analysis[base_id] = {
                    "base_file": "MULTIPLE_BASES",
                    "total_files": len(candidate_files),
                    "group_files": candidate_files,
                    "method": "visual_verification_failed",
                    "verified": False,
                }
                continue

        # Identificar archivo base del grupo
        base_file = identify_base_file_in_group(candidate_files)
        base_files.append(base_file)

        group_analysis[base_id] = {
            "base_file": base_file,
            "total_files": len(candidate_files),
            "group_files": candidate_files,
            "method": "pattern_analysis",
            "verified": True,
        }

    analysis_time = time.time() - start_time

    print(f"\n✅ ANÁLISIS DE ARCHIVOS BASE COMPLETADO EN {analysis_time:.3f}s")
    print("📊 Estadísticas FINALES:")
    print(f"   - Verificaciones visuales realizadas: {visual_verifications}")
    print(f"   - Grupos analizados: {len(group_analysis)}")
    print(f"   - Archivos base identificados: {len(base_files)}")
    print(f"   - Total archivos procesados: {len(all_files)}")

    return base_files, group_analysis, analysis_time


def generate_base_files_report(base_files, group_analysis, analysis_time, target_dir):
    """Genera reporte del análisis de archivos base"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.basename(target_dir)

    # Generar archivos de reporte
    report_file = f"base_files_analysis_{dir_name}_{timestamp}.txt"
    csv_file = f"base_files_analysis_{dir_name}_{timestamp}.csv"

    # Reporte de texto
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("ANÁLISIS DE ARCHIVOS BASE EN GRUPOS\n")
        f.write("================================================================================\n\n")
        f.write(f"Directorio: {target_dir}\n")
        f.write(f"Tiempo de análisis: {analysis_time:.3f}s\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("ESTADÍSTICAS FINALES:\n")
        f.write(f"- Total de archivos: {sum(group['total_files'] for group in group_analysis.values())}\n")
        f.write(f"- Grupos analizados: {len(group_analysis)}\n")
        f.write(f"- Archivos base identificados: {len(base_files)}\n")

        # Estadísticas por método
        methods = {}
        for group in group_analysis.values():
            method = group["method"]
            methods[method] = methods.get(method, 0) + 1

        f.write("\nMÉTODOS UTILIZADOS:\n")
        for method, count in methods.items():
            f.write(f"- {method}: {count} grupos\n")

        f.write("\nGRUPOS ANALIZADOS:\n")
        f.write("-" * 50 + "\n")

        for group_id, info in group_analysis.items():
            f.write(f"\nGrupo: {group_id} ({info['total_files']} archivos)\n")
            f.write(f"Método: {info['method']}\n")
            f.write(f"Archivo base: {info['base_file']}\n")
            f.write("Archivos del grupo:\n")
            for file in info["group_files"]:
                marker = "🎯" if file == info["base_file"] else "  "
                f.write(f"  {marker} {file}\n")

    # CSV para compatibilidad con 03_move_unique_files.py
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "has_transformations", "group_id", "is_base_file", "method"])

        for group_id, info in group_analysis.items():
            for file in info["group_files"]:
                is_base = file == info["base_file"]
                has_transformations = "NO" if is_base else "YES"
                writer.writerow([file, has_transformations, group_id, is_base, info["method"]])

    return report_file, csv_file


def main():
    """Función principal del análisis de archivos base."""
    print("🚀 INICIANDO IDENTIFICACIÓN DE ARCHIVOS BASE")
    print("============================================")

    print("\n🎯 CONFIGURACIÓN:")
    print(f"   - Directorio objetivo: {TARGET_DIRECTORY}")
    print(f"   - Verificación visual: {'Habilitada' if ENABLE_VISUAL_VERIFICATION else 'Deshabilitada'}")
    print(f"   - Umbral similitud: {VISUAL_SIMILARITY_THRESHOLD}")
    print(f"   - Salida detallada: {'Habilitada' if SHOW_DETAILED_OUTPUT else 'Deshabilitada'}")

    # Verificar que el directorio existe
    if not os.path.exists(TARGET_DIRECTORY):
        print(f"❌ El directorio {TARGET_DIRECTORY} no existe")
        return

    # Realizar análisis
    base_files, group_analysis, analysis_time = base_file_analysis(TARGET_DIRECTORY)

    # Generar reportes
    report_file, csv_file = generate_base_files_report(base_files, group_analysis, analysis_time, TARGET_DIRECTORY)

    # Resumen final
    print("\n🎉 ANÁLISIS COMPLETADO")
    print(f"📄 Reporte generado: {report_file}")
    print(f"📊 CSV generado: {csv_file}")
    print(f"🎯 Archivos base identificados: {len(base_files)}")

    if SHOW_DETAILED_OUTPUT:
        print("\n📋 PRIMEROS 10 ARCHIVOS BASE:")
        for i, base_file in enumerate(base_files[:10], 1):
            print(f"   {i}. {base_file}")


if __name__ == "__main__":
    main()
