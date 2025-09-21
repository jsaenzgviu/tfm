#!/usr/bin/env python3
"""
PASO 3B: MOVER ARCHIVOS BASE IDENTIFICADOS
Mueve físicamente los archivos base (identificados por 02_base_file_identifier.py)
a un directorio separado. Los archivos base son los originales de los cuales se
derivaron todas las transformaciones.
"""

import os
import shutil
import csv
from datetime import datetime
import glob

# ==================================================================================
# CONFIGURACIÓN - MODIFICAR ESTAS VARIABLES SEGÚN NECESIDADES
# ==================================================================================

# Directorio objetivo (relativo al script o ruta absoluta)
TARGET_DIRECTORY = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/same4/train/bacterial_spot"

# Configuraciones de movimiento
DESTINATION_SUBDIR = "base_files"  # Subdirectorio donde mover archivos base
CONFIRM_BEFORE_MOVE = True  # Pedir confirmación antes de mover
SHOW_DETAILED_OUTPUT = True  # Mostrar información detallada

# ==================================================================================
# FIN DE CONFIGURACIÓN
# ==================================================================================


def load_latest_base_analysis_results(source_directory):
    """Carga los resultados del último análisis de archivos base realizado."""
    # Directorio donde está este script (donde también están los archivos de análisis)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Extraer el nombre del directorio para buscar los archivos CSV correspondientes
    dir_name = os.path.basename(source_directory)

    # Buscar el CSV más reciente de análisis de archivos base
    base_csv_files = glob.glob(os.path.join(script_dir, f"base_files_analysis_{dir_name}_*.csv"))

    if base_csv_files:
        latest_csv = max(base_csv_files, key=os.path.getctime)
        print(f"📊 Usando análisis de archivos base: {os.path.basename(latest_csv)}")
    else:
        print("❌ No se encontraron archivos de análisis de archivos base CSV")
        print(f"   Patrón buscado: base_files_analysis_{dir_name}_*.csv")
        print("   Ejecute primero: python 02_base_file_identifier.py")
        return None

    # Leer resultados
    base_files = []
    transformed_files = []
    total_groups = 0
    methods_used = {}

    with open(latest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["is_base_file"] == "True":
                base_files.append(row["filename"])
            else:
                transformed_files.append(row["filename"])

            # Estadísticas adicionales
            method = row.get("method", "unknown")
            methods_used[method] = methods_used.get(method, 0) + 1

    # Contar grupos únicos
    unique_groups = set()
    with open(latest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unique_groups.add(row["group_id"])
    total_groups = len(unique_groups)

    return base_files, transformed_files, latest_csv, total_groups, methods_used


def create_destination_directory(source_dir, subdir_name):
    """Crea directorio DENTRO del directorio fuente para organizar archivos base."""
    dest_dir = os.path.join(source_dir, subdir_name)
    os.makedirs(dest_dir, exist_ok=True)
    return dest_dir


def move_base_files(source_dir, target_dir, base_files):
    """Mueve archivos base al directorio destino."""
    moved_files = []
    failed_files = []

    print(f"\n🚀 INICIANDO MOVIMIENTO DE {len(base_files)} ARCHIVOS BASE")
    print(f"Origen: {source_dir}")
    print(f"Destino: {target_dir}")

    for filename in base_files:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)

        if not os.path.exists(source_path):
            failed_files.append(f"{filename} - No encontrado")
            continue

        try:
            # Mover archivo
            shutil.move(source_path, target_path)
            moved_files.append(filename)
            if SHOW_DETAILED_OUTPUT:
                print(f"✅ Movido: {filename}")

        except Exception as e:
            failed_files.append(f"{filename} - Error: {str(e)}")
            print(f"❌ Error moviendo {filename}: {str(e)}")

    return moved_files, failed_files


def generate_movement_report(moved_files, failed_files, source_dir, target_dir, analysis_file, total_groups, methods_used):
    """Genera reporte del movimiento de archivos base."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Obtener nombre del directorio dinámicamente para trazabilidad
    dir_name = os.path.basename(source_dir)
    report_file = f"movement_report_base_files_{dir_name}_{timestamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("REPORTE DE MOVIMIENTO DE ARCHIVOS BASE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Análisis base: {os.path.basename(analysis_file)}\n")
        f.write("Método: Identificación de archivos base por patrón\n")
        f.write(f"Directorio origen: {source_dir}\n")
        f.write(f"Directorio destino: {target_dir}\n")
        f.write(f"\nGrupos analizados: {total_groups}\n")
        f.write(f"Archivos base movidos exitosamente: {len(moved_files)}\n")
        f.write(f"Archivos con errores: {len(failed_files)}\n\n")

        f.write("MÉTODOS UTILIZADOS EN EL ANÁLISIS:\n")
        for method, count in methods_used.items():
            f.write(f"- {method}: {count} archivos\n")

        if moved_files:
            f.write(f"\nARCHIVOS BASE MOVIDOS EXITOSAMENTE ({len(moved_files)}):\n")
            f.write("-" * 50 + "\n")
            for filename in moved_files:
                f.write(f"✅ {filename}\n")

        if failed_files:
            f.write(f"\nARCHIVOS CON ERRORES ({len(failed_files)}):\n")
            f.write("-" * 40 + "\n")
            for error in failed_files:
                f.write(f"❌ {error}\n")

        # Estadísticas
        total_attempted = len(moved_files) + len(failed_files)
        success_rate = (len(moved_files) / total_attempted * 100) if total_attempted > 0 else 0
        f.write("\nESTADÍSTICAS:\n")
        f.write(f"- Total grupos analizados: {total_groups}\n")
        f.write(f"- Archivos base identificados: {total_attempted}\n")
        f.write(f"- Exitosamente movidos: {len(moved_files)}\n")
        f.write(f"- Fallidos: {len(failed_files)}\n")
        f.write(f"- Tasa de éxito: {success_rate:.1f}%\n")
        f.write(f"- Eficiencia: {(len(moved_files) / total_groups * 100):.1f}% (archivos base/grupos)\n")

    return report_file


def verify_movement(target_dir, moved_files):
    """Verifica que los archivos se movieron correctamente."""
    print("\n🔍 VERIFICANDO MOVIMIENTO...")
    verified = 0
    missing = 0

    for filename in moved_files:
        target_path = os.path.join(target_dir, filename)
        if os.path.exists(target_path):
            verified += 1
        else:
            missing += 1
            print(f"⚠️  Archivo faltante: {filename}")

    print(f"✅ Verificados: {verified}/{len(moved_files)} archivos")
    if missing > 0:
        print(f"❌ Faltantes: {missing} archivos")

    return verified, missing


def show_movement_preview(base_files, transformed_files, total_groups):
    """Muestra un preview de lo que se va a mover."""
    print("\n📋 PREVIEW DEL MOVIMIENTO:")
    print(f"   - Total de grupos analizados: {total_groups}")
    print(f"   - Archivos base a mover: {len(base_files)}")
    print(f"   - Archivos transformados que permanecen: {len(transformed_files)}")
    print(f"   - Eficiencia: {(len(base_files) / total_groups * 100):.1f}% (un archivo base por grupo)")

    if SHOW_DETAILED_OUTPUT and len(base_files) <= 20:
        print("\n📄 ARCHIVOS BASE A MOVER:")
        for i, filename in enumerate(base_files, 1):
            print(f"   {i}. {filename}")
    elif len(base_files) > 20:
        print("\n📄 PRIMEROS 10 ARCHIVOS BASE A MOVER:")
        for i, filename in enumerate(base_files[:10], 1):
            print(f"   {i}. {filename}")
        print(f"   ... y {len(base_files) - 10} más")


def main():
    """Función principal del script de movimiento de archivos base."""
    print("🚀 INICIANDO MOVIMIENTO DE ARCHIVOS BASE")
    print("=======================================")

    print("\n🎯 CONFIGURACIÓN:")
    print(f"   - Directorio objetivo: {TARGET_DIRECTORY}")
    print(f"   - Subdirectorio destino: {DESTINATION_SUBDIR}")
    print(f"   - Confirmación requerida: {'Sí' if CONFIRM_BEFORE_MOVE else 'No'}")
    print(f"   - Salida detallada: {'Habilitada' if SHOW_DETAILED_OUTPUT else 'Deshabilitada'}")

    # Verificar que el directorio existe
    if not os.path.exists(TARGET_DIRECTORY):
        print(f"❌ El directorio {TARGET_DIRECTORY} no existe")
        return

    # Cargar resultados del análisis
    result = load_latest_base_analysis_results(TARGET_DIRECTORY)
    if result is None:
        return

    base_files, transformed_files, analysis_file, total_groups, methods_used = result

    print("\n📊 RESUMEN DEL ANÁLISIS:")
    print(f"   - Archivos base identificados: {len(base_files)}")
    print(f"   - Archivos transformados: {len(transformed_files)}")
    print(f"   - Total grupos: {total_groups}")
    print(f"   - Total archivos: {len(base_files) + len(transformed_files)}")

    if len(base_files) == 0:
        print("\n❓ No hay archivos base identificados para mover.")
        print("   Verificar que el análisis se ejecutó correctamente.")
        return

    # Mostrar preview
    show_movement_preview(base_files, transformed_files, total_groups)

    # Crear directorio destino
    dest_dir = create_destination_directory(TARGET_DIRECTORY, DESTINATION_SUBDIR)

    # Confirmar operación si está habilitado
    if CONFIRM_BEFORE_MOVE:
        print("\n⚠️  CONFIRMACIÓN REQUERIDA:")
        print(f"   Se moverán {len(base_files)} archivos base")
        print(f"   Desde: {TARGET_DIRECTORY}")
        print(f"   Hacia: {dest_dir}")
        print(f"   Los archivos transformados ({len(transformed_files)}) permanecerán en el directorio original")

        confirm = input("\n¿Continuar con el movimiento? (s/N): ").lower().strip()
        if confirm not in ["s", "si", "sí", "y", "yes"]:
            print("❌ Operación cancelada por el usuario")
            return

    # Realizar movimiento
    moved_files, failed_files = move_base_files(TARGET_DIRECTORY, dest_dir, base_files)

    # Verificar movimiento
    verified, missing = verify_movement(dest_dir, moved_files)

    # Generar reporte
    report_file = generate_movement_report(moved_files, failed_files, TARGET_DIRECTORY, dest_dir, analysis_file, total_groups, methods_used)

    # Resumen final
    print("\n🎉 MOVIMIENTO DE ARCHIVOS BASE COMPLETADO")
    print(f"📄 Reporte generado: {report_file}")
    print(f"✅ Archivos base movidos: {len(moved_files)}")
    print(f"❌ Errores: {len(failed_files)}")
    print(f"📂 Directorio destino: {dest_dir}")
    print(f"📁 Archivos transformados restantes: {len(transformed_files)}")

    if len(failed_files) == 0:
        print("\n🏆 ¡OPERACIÓN 100% EXITOSA!")
        print(f"🎯 Se identificó correctamente un archivo base por cada uno de los {total_groups} grupos")
    else:
        print(f"\n⚠️  Operación completada con {len(failed_files)} errores")


if __name__ == "__main__":
    main()
