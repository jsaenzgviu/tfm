#!/usr/bin/env python3
"""
ANÁLISIS EXHAUSTIVO DE PATRONES - IGNORANDO HASHES ÚNICOS
Analiza TODOS los 33,261 archivos para encontrar TODOS los patrones después de hashes únicos.
"""

import os
import re
from collections import defaultdict, Counter
import json

# directorio base del dataset para análisis
BASE_DIR = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/same"


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


def extract_pattern_after_hashes(filename, discovered_patterns=None, target_directory=None):
    """
    Extrae patrones significativos USANDO LA MISMA LÓGICA QUE EL HÍBRIDO.
    Sincronizado con extract_base_identifier_ultimate() de 02_hybrid_pattern_grouping.py
    INCLUYE CASOS ESPECIALES PARA DIRECTORIO HEALTHY
    """
    # CASO ESPECIAL: Verificar primero si es un patrón semántico de HEALTHY
    if target_directory:
        healthy_identifier = extract_healthy_special_cases(filename, target_directory)
        if healthy_identifier:
            return [healthy_identifier]

    name_without_ext = os.path.splitext(filename)[0].lower()

    # PASO 1: Remover transformaciones conocidas (ORDEN IMPORTANTE: más específicos primero)
    clean_name = name_without_ext
    transformations = [
        "new30degfliplr",  # ← MÁS ESPECÍFICO PRIMERO
        "change_90",
        "change_180",
        "change_270",
        "change_360",
        "change",
        "mirror_vertical",
        "mirror_horizontal",
        "mirror",  # ← ESTE debe estar DESPUÉS de mirror_vertical/horizontal
        "new30deg",  # ← DESPUÉS de new30degfliplr
        "hight",
        "lower",
        "fliptb",
        "fliplr",
        "flip",  # ← DESPUÉS de fliptb/fliplr
        "180deg",
        "90deg",
        "270deg",
        "30deg",
        "newpixel25",
    ]

    for transform in transformations:
        if f"_{transform}" in clean_name:
            clean_name = clean_name.replace(f"_{transform}", "")

    # PASO 2: Buscar patrones específicos descobertos (igual que híbrido)
    lb_match = re.search(r"_lb(\d+)", clean_name)
    if lb_match:
        return [f"lb_{lb_match.group(1)}"]

    # Buscar pm seguido de número (con o sin guión bajo)
    pm_match = re.search(r"[_]?pm(\d+)", clean_name)
    if pm_match:
        return [f"pm_{pm_match.group(1)}"]

    # PASO 3: Extraer partes significativas (ignorando hashes) - IGUAL QUE HÍBRIDO
    parts = clean_name.split("_")

    # Filtrar hashes hexadecimales pero conservar números importantes
    significant_parts = []
    for part in parts:
        if len(part) >= 8 and re.match(r"^[a-f0-9]+$", part):
            continue  # Ignorar hashes hexadecimales válidos
        elif len(part) > 2 or part.isdigit():  # Partes significativas O números - IGUAL QUE HÍBRIDO
            significant_parts.append(part)

    # PASO 3.5: IDENTIFICADORES ÚNICOS PARA img_XX_3 - IGUAL QUE HÍBRIDO
    for i in range(len(significant_parts) - 2):
        if (
            significant_parts[i] == "img"
            and i + 1 < len(significant_parts)
            and significant_parts[i + 1].isdigit()
            and i + 2 < len(significant_parts)
            and significant_parts[i + 2] == "3"
        ):
            img_number = significant_parts[i + 1]
            return [f"img_{img_number}_3"]

    # PASO 3.6: IDENTIFICADORES PARA figure_X_Y (SOLUCIÓN PARA REGRESIÓN) - IGUAL QUE HÍBRIDO
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
            return [f"figure_{x_number}_{y_number}"]

    # PASO 4: Retornar partes significativas para análisis posterior
    return significant_parts


def find_all_suffix_patterns():
    """
    Encuentra TODOS los patrones de sufijos en TODO el dataset.
    SINCRONIZADO con la lógica de 02_hybrid_pattern_grouping.py
    """
    print("🔍 ANALIZANDO TODOS LOS 33,261 ARCHIVOS PARA PATRONES DE SUFIJOS...")

    suffix_patterns = Counter()
    all_patterns = []
    transformation_suffixes = Counter()
    discovered_patterns = {}  # Se va llenando progresivamente

    total_files = 0

    for split in ["train", "valid"]:
        split_path = os.path.join(BASE_DIR, split)
        if not os.path.exists(split_path):
            continue

        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if not os.path.isdir(category_path):
                continue

            files_in_category = [f for f in os.listdir(category_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            for filename in files_in_category:
                total_files += 1
                if total_files % 5000 == 0:
                    print(f"  Procesados: {total_files}")

                # Extraer partes usando LÓGICA SINCRONIZADA con híbrido
                significant_parts = extract_pattern_after_hashes(filename, discovered_patterns, category_path)

                # Si retorna un patrón específico (lb_XX, pm_XX, img_XX_3), guardarlo directamente
                if len(significant_parts) == 1 and isinstance(significant_parts[0], str):
                    pattern = significant_parts[0]
                    suffix_patterns[pattern] += 1
                    all_patterns.append(
                        {"filename": filename, "pattern": pattern, "full_parts": significant_parts, "category": category, "split": split}
                    )
                elif len(significant_parts) >= 2:
                    # Tomar las últimas 2-4 partes como patrón (igual que antes)
                    for length in [2, 3, 4]:
                        if len(significant_parts) >= length:
                            pattern = "_".join(significant_parts[-length:])
                            suffix_patterns[pattern] += 1

                            # Guardar patrón con contexto
                            all_patterns.append(
                                {"filename": filename, "pattern": pattern, "full_parts": significant_parts, "category": category, "split": split}
                            )

                    # PASO 5 del híbrido: Buscar patrones semánticos en posiciones intermedias
                    for i in range(len(significant_parts) - 1):
                        for length in [3, 2]:
                            if i + length <= len(significant_parts):
                                combo = "_".join(significant_parts[i : i + length])
                                if re.search(r"\d+$", combo):
                                    suffix_patterns[combo] += 1

                # Buscar sufijos de transformación específicos (igual que antes)
                name_lower = filename.lower()
                transforms = [
                    "change_90",
                    "change_180",
                    "change_270",
                    "mirror",
                    "mirror_vertical",
                    "mirror_horizontal",
                    "hight",
                    "lower",
                    "flip",
                    "fliptb",
                    "fliplr",
                    "180deg",
                    "90deg",
                    "270deg",
                    "30deg",
                    "new30deg",
                    "new30degfliplr",
                    "newpixel25",
                ]

                for transform in transforms:
                    if transform in name_lower:
                        transformation_suffixes[transform] += 1

    print(f"\n✅ PROCESADOS {total_files} ARCHIVOS")
    return suffix_patterns, all_patterns, transformation_suffixes


def find_base_patterns(suffix_patterns, min_count=2):
    """
    Encuentra patrones base que aparecen múltiples veces.
    """
    print(f"\n🔍 BUSCANDO PATRONES QUE APARECEN AL MENOS {min_count} VECES...")

    # Filtrar patrones que aparecen múltiples veces
    repeated_patterns = {pattern: count for pattern, count in suffix_patterns.items() if count >= min_count}

    # Ordenar por frecuencia
    sorted_patterns = sorted(repeated_patterns.items(), key=lambda x: x[1], reverse=True)

    print(f"✅ ENCONTRADOS {len(sorted_patterns)} PATRONES REPETIDOS")

    return sorted_patterns


def generate_comprehensive_report(sorted_patterns, transformation_suffixes):
    """
    Genera reporte completo de TODOS los patrones encontrados.
    """
    print("\n📊 GENERANDO REPORTE COMPLETO...")

    with open("all_patterns_comprehensive.txt", "w", encoding="utf-8") as f:
        f.write("ANÁLISIS EXHAUSTIVO DE TODOS LOS PATRONES DEL DATASET\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total de patrones únicos encontrados: {len(sorted_patterns)}\n\n")

        f.write("PATRONES MÁS FRECUENTES (top 100):\n")
        f.write("-" * 50 + "\n")
        for pattern, count in sorted_patterns[:100]:
            f.write(f"{pattern}: {count} archivos\n")

        f.write(f"\nTRANSFORMACIONES ENCONTRADAS:\n")
        f.write("-" * 30 + "\n")
        for transform, count in transformation_suffixes.most_common(20):
            f.write(f"{transform}: {count} archivos\n")

        if len(sorted_patterns) > 100:
            f.write(f"\nTODOS LOS PATRONES ({len(sorted_patterns)} total):\n")
            f.write("-" * 40 + "\n")
            for pattern, count in sorted_patterns:
                f.write(f"{pattern}: {count}\n")

    # Generar JSON con patrones para usar en el algoritmo
    patterns_for_algorithm = {
        "frequent_patterns": dict(sorted_patterns[:200]),  # Top 200
        "transformation_suffixes": dict(transformation_suffixes),
        "all_patterns_count": len(sorted_patterns),
    }

    with open("patterns_for_algorithm.json", "w", encoding="utf-8") as f:
        json.dump(patterns_for_algorithm, f, indent=2, ensure_ascii=False)

    return patterns_for_algorithm


def main():
    print("🚀 ANÁLISIS EXHAUSTIVO DE PATRONES - IGNORANDO HASHES ÚNICOS")
    print("=" * 80)

    # Paso 1: Extraer todos los patrones de sufijos
    suffix_patterns, all_patterns, transformation_suffixes = find_all_suffix_patterns()

    # Paso 2: Encontrar patrones que se repiten
    sorted_patterns = find_base_patterns(suffix_patterns, min_count=2)

    # Paso 3: Generar reporte
    patterns_data = generate_comprehensive_report(sorted_patterns, transformation_suffixes)

    print(f"\n✅ ANÁLISIS COMPLETADO!")
    print(f"📄 Reporte completo: all_patterns_comprehensive.txt")
    print(f"📊 Patrones para algoritmo: patterns_for_algorithm.json")
    print(f"🔍 {len(sorted_patterns)} patrones únicos encontrados")
    print(f"⭐ Top 10 patrones más frecuentes:")

    for i, (pattern, count) in enumerate(sorted_patterns[:10], 1):
        print(f"   {i}. {pattern}: {count} archivos")


if __name__ == "__main__":
    main()
