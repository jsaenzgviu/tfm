#!/usr/bin/env python3
"""
PASO 4: CREAR SUBCONJUNTO DE VALIDACIÓN
Mueve archivos desde train/[clase]/base_files/ y train/[clase]/unique_files/ hacia valid/[clase]/
- Detección automática de clases
- Búsqueda en ambas carpetas (base_files y unique_files)
- Movimiento directo sin análisis de grupos
- Configuración simple por número de archivos
- Compatibilidad dinámica con cualquier directorio
"""

import os
import shutil
import random
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN PRINCIPAL
# =============================================================================
TRAIN_BASE_DIR = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/same4/train"
TARGET_FILES_PER_CLASS = 370  # Número de archivos a mover por clase
RANDOM_SEED = 42  # Semilla para reproducibilidad
VALID_SUBDIR = "valid"  # Nombre del subdirectorio de validación


def discover_classes(train_base_dir):
    """
    Descubre automáticamente las clases disponibles en el directorio train.
    Busca subdirectorios que contengan carpetas base_files y/o unique_files.
    """
    classes_found = []

    if not os.path.exists(train_base_dir):
        print(f"❌ Directorio base no encontrado: {train_base_dir}")
        return classes_found

    for item in os.listdir(train_base_dir):
        class_path = os.path.join(train_base_dir, item)
        base_files_path = os.path.join(class_path, "base_files")
        unique_files_path = os.path.join(class_path, "unique_files")

        if os.path.isdir(class_path) and (os.path.exists(base_files_path) or os.path.exists(unique_files_path)):
            # Recopilar archivos de ambas carpetas
            all_image_files = []
            base_count = 0
            unique_count = 0

            # Archivos de base_files
            if os.path.exists(base_files_path):
                base_files = [f for f in os.listdir(base_files_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
                all_image_files.extend([(f, "base_files") for f in base_files])
                base_count = len(base_files)

            # Archivos de unique_files
            if os.path.exists(unique_files_path):
                unique_files = [f for f in os.listdir(unique_files_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
                all_image_files.extend([(f, "unique_files") for f in unique_files])
                unique_count = len(unique_files)

            if all_image_files:
                total_files = len(all_image_files)
                classes_found.append(
                    {
                        "name": item,
                        "path": class_path,
                        "base_files_path": base_files_path,
                        "unique_files_path": unique_files_path,
                        "all_files": all_image_files,
                        "file_count": total_files,
                        "base_count": base_count,
                        "unique_count": unique_count,
                    }
                )
                print(f"📂 Clase encontrada: {item} (base: {base_count}, unique: {unique_count}, total: {total_files} archivos)")

    return classes_found


def create_valid_structure(train_base_dir, valid_subdir):
    """
    Crea la estructura de directorios valid paralela a train.
    """
    # El directorio valid debe estar al mismo nivel que train
    train_parent = os.path.dirname(train_base_dir)
    valid_base_dir = os.path.join(train_parent, valid_subdir)

    os.makedirs(valid_base_dir, exist_ok=True)
    print(f"📁 Directorio base de validación: {valid_base_dir}")

    return valid_base_dir


def select_and_move_files(class_info, valid_base_dir, target_files):
    """
    Selecciona y mueve archivos de una clase específica desde ambas carpetas (base_files y unique_files).
    """
    class_name = class_info["name"]
    available_files = class_info["file_count"]
    base_count = class_info["base_count"]
    unique_count = class_info["unique_count"]
    all_files = class_info["all_files"]  # Lista de tuplas (filename, source_folder)

    print(f"\n🎯 Procesando clase: {class_name}")
    print(f"📊 Archivos disponibles: {available_files} (base: {base_count}, unique: {unique_count})")

    # Crear directorio de destino para esta clase
    class_valid_dir = os.path.join(valid_base_dir, class_name)
    os.makedirs(class_valid_dir, exist_ok=True)

    # Determinar cuántos archivos seleccionar
    files_to_move = min(target_files, len(all_files))

    if files_to_move == 0:
        print(f"⚠️  No hay archivos para mover en {class_name}")
        return [], []

    # Seleccionar archivos aleatoriamente de ambas carpetas
    selected_files = random.sample(all_files, files_to_move)

    print(f"🎲 Seleccionados {files_to_move} de {len(all_files)} archivos")

    # Mostrar distribución de la selección
    base_selected = sum(1 for _, folder in selected_files if folder == "base_files")
    unique_selected = sum(1 for _, folder in selected_files if folder == "unique_files")
    print(f"   📂 De base_files: {base_selected}, de unique_files: {unique_selected}")

    # Mover archivos
    moved_files = []
    failed_files = []

    for filename, source_folder in selected_files:
        if source_folder == "base_files":
            source_path = os.path.join(class_info["base_files_path"], filename)
        else:  # unique_files
            source_path = os.path.join(class_info["unique_files_path"], filename)

        target_path = os.path.join(class_valid_dir, filename)

        if move_single_file(source_path, target_path, filename):
            moved_files.append(filename)
        else:
            failed_files.append(filename)

    print(f"✅ Movidos exitosamente: {len(moved_files)}")
    if failed_files:
        print(f"❌ Errores: {len(failed_files)}")

    return moved_files, failed_files


def move_single_file(source_path, target_path, filename):
    """
    Mueve un archivo individual con manejo de errores.
    """
    if not os.path.exists(source_path):
        print(f"❌ Archivo no encontrado: {filename}")
        return False

    try:
        shutil.move(source_path, target_path)
        print(f"✅ Movido: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error moviendo {filename}: {str(e)}")
        return False


def generate_validation_report(classes_results, train_base_dir, valid_base_dir):
    """
    Genera reporte detallado de la operación de validación.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"validation_subset_report_{timestamp}.txt"

    total_moved = sum(len(result["moved_files"]) for result in classes_results.values())
    total_failed = sum(len(result["failed_files"]) for result in classes_results.values())

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("REPORTE DE CREACIÓN DE SUBCONJUNTO DE VALIDACIÓN\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directorio origen (train): {train_base_dir}\n")
        f.write(f"Directorio destino (valid): {valid_base_dir}\n")
        f.write(f"Semilla aleatoria: {RANDOM_SEED}\n")
        f.write(f"Archivos objetivo por clase: {TARGET_FILES_PER_CLASS}\n\n")

        f.write("RESUMEN POR CLASE:\n")
        f.write("-" * 40 + "\n")
        for class_name, result in classes_results.items():
            f.write(f"Clase: {class_name}\n")
            f.write(f"  - Archivos disponibles: {result['available_files']}\n")
            f.write(f"  - Archivos movidos: {len(result['moved_files'])}\n")
            f.write(f"  - Errores: {len(result['failed_files'])}\n")
            f.write(f"  - Directorio destino: {result['target_dir']}\n\n")

        f.write("ESTADÍSTICAS GLOBALES:\n")
        f.write("-" * 30 + "\n")
        f.write(f"- Clases procesadas: {len(classes_results)}\n")
        f.write(f"- Total archivos movidos: {total_moved}\n")
        f.write(f"- Total errores: {total_failed}\n")
        f.write(f"- Tasa de éxito: {(total_moved / (total_moved + total_failed) * 100):.1f}%\n\n")

        # Detalles de archivos movidos por clase
        f.write("ARCHIVOS MOVIDOS POR CLASE:\n")
        f.write("-" * 40 + "\n")
        for class_name, result in classes_results.items():
            if result["moved_files"]:
                f.write(f"\n{class_name}:\n")
                for filename in result["moved_files"]:
                    f.write(f"  ✅ {filename}\n")

        # Errores si los hay
        if total_failed > 0:
            f.write("\nERRORES EN MOVIMIENTO:\n")
            f.write("-" * 30 + "\n")
            for class_name, result in classes_results.items():
                if result["failed_files"]:
                    f.write(f"\n{class_name}:\n")
                    for filename in result["failed_files"]:
                        f.write(f"  ❌ {filename}\n")

    return report_file


def verify_validation_structure(classes_results, valid_base_dir):
    """
    Verifica que la estructura de validación se creó correctamente.
    """
    print("\n🔍 VERIFICANDO ESTRUCTURA DE VALIDACIÓN...")

    verification_ok = True
    total_expected = 0
    total_found = 0

    for class_name, result in classes_results.items():
        class_valid_dir = result["target_dir"]
        expected_files = len(result["moved_files"])
        total_expected += expected_files

        if os.path.exists(class_valid_dir):
            found_files = len([f for f in os.listdir(class_valid_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))])
            total_found += found_files

            print(f"� {class_name}: {found_files}/{expected_files} archivos")

            if found_files != expected_files:
                print(f"⚠️  Discrepancia en clase {class_name}")
                verification_ok = False
        else:
            print(f"❌ Directorio no encontrado para clase {class_name}")
            verification_ok = False

    print(f"\n� Total: {total_found}/{total_expected} archivos")

    if verification_ok and total_found == total_expected:
        print("✅ Verificación exitosa: Todos los archivos están en su lugar")
        return True
    else:
        print("⚠️ Se encontraron discrepancias en la verificación")
        return False


def main():
    """
    Función principal para crear subconjunto de validación.
    """
    print("🚀 INICIANDO CREACIÓN DE SUBCONJUNTO DE VALIDACIÓN")
    print("=" * 60)

    # Establecer semilla para reproducibilidad
    random.seed(RANDOM_SEED)

    print(f"📂 Directorio base train: {TRAIN_BASE_DIR}")
    print(f"🎯 Archivos objetivo por clase: {TARGET_FILES_PER_CLASS}")
    print(f"🎲 Semilla aleatoria: {RANDOM_SEED}")

    # Descubrir clases disponibles
    classes = discover_classes(TRAIN_BASE_DIR)

    if not classes:
        print("❌ No se encontraron clases con archivos")
        print("   Verifique que existen directorios train/[clase]/base_files/ o train/[clase]/unique_files/")
        return

    print(f"\n📋 Clases detectadas: {len(classes)}")
    for class_info in classes:
        print(f"  - {class_info['name']}: {class_info['file_count']} archivos")

    # Crear estructura de validación
    valid_base_dir = create_valid_structure(TRAIN_BASE_DIR, VALID_SUBDIR)

    # Confirmar operación
    total_files_to_move = sum(min(TARGET_FILES_PER_CLASS, cls["file_count"]) for cls in classes)

    print("\n⚠️ CONFIRMACIÓN REQUERIDA:")
    print(f"   Se moverán aproximadamente {total_files_to_move} archivos")
    print("   Desde: train/[clase]/base_files/ y train/[clase]/unique_files/")
    print(f"   Hacia: {VALID_SUBDIR}/[clase]/")

    confirm = input("\n¿Continuar con la creación del subconjunto? (s/N): ").lower().strip()
    if confirm not in ["s", "si", "sí", "y", "yes"]:
        print("❌ Operación cancelada por el usuario")
        return

    # Procesar cada clase
    classes_results = {}

    for class_info in classes:
        class_name = class_info["name"]
        moved_files, failed_files = select_and_move_files(class_info, valid_base_dir, TARGET_FILES_PER_CLASS)

        classes_results[class_name] = {
            "available_files": class_info["file_count"],
            "moved_files": moved_files,
            "failed_files": failed_files,
            "target_dir": os.path.join(valid_base_dir, class_name),
        }

    # Verificar estructura
    verification_ok = verify_validation_structure(classes_results, valid_base_dir)

    # Generar reporte
    report_file = generate_validation_report(classes_results, TRAIN_BASE_DIR, valid_base_dir)

    # Resumen final
    total_moved = sum(len(result["moved_files"]) for result in classes_results.values())
    total_failed = sum(len(result["failed_files"]) for result in classes_results.values())

    print("\n🎉 CREACIÓN DE SUBCONJUNTO COMPLETADA")
    print(f"📄 Reporte generado: {report_file}")
    print(f"✅ Archivos movidos: {total_moved}")
    print(f"❌ Errores: {total_failed}")
    print(f"📁 Directorio validación: {valid_base_dir}")

    if total_failed == 0 and verification_ok:
        print("\n🏆 ¡OPERACIÓN 100% EXITOSA!")
        print("� Subconjunto de validación creado correctamente")
    else:
        print(f"\n⚠️ Operación completada con {total_failed} errores")


if __name__ == "__main__":
    main()
