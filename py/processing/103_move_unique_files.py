#!/usr/bin/env python3
"""
PASO 3: MOVER ARCHIVOS ÚNICOS (VERSIÓN HÍBRIDA)
Mueve físicamente los archivos únicos (sin transformaciones) a un directorio separado.
Usa los resultados del análisis híbrido con verificación visual selectiva.
Prioriza resultados HYBRID sobre FINAL para mayor precisión.
"""

import os
import json
import shutil
import csv
from datetime import datetime
import glob


def load_latest_analysis_results(source_directory):
    """Carga los resultados del último análisis híbrido realizado."""
    # Directorio donde está este script (donde también están los archivos de análisis)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Extraer el nombre del directorio para buscar los archivos CSV correspondientes
    dir_name = os.path.basename(source_directory)

    # Buscar el CSV más reciente (solo HYBRID, ya no hay FINAL)
    hybrid_csv_files = glob.glob(os.path.join(script_dir, f"hibrid_analysis_{dir_name}_*.csv"))

    if hybrid_csv_files:
        latest_csv = max(hybrid_csv_files, key=os.path.getctime)
        print(f"📊 Usando análisis híbrido: {latest_csv}")
    else:
        print("❌ No se encontraron archivos de análisis CSV")
        print(f"   Patrón buscado: hibrid_analysis_{dir_name}_*.csv")
        print("   Ejecute primero: python 02_hybrid_pattern_grouping.py")
        return None

    # Leer resultados
    unique_files = []
    grouped_files = []

    with open(latest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["has_transformations"] == "NO":
                unique_files.append(row["filename"])
            else:
                grouped_files.append(row["filename"])

    return unique_files, grouped_files, latest_csv


def create_directories(source_dir):
    """Crea directorio DENTRO del directorio fuente para organizar archivos."""
    unique_dir = os.path.join(source_dir, "unique_files")

    os.makedirs(unique_dir, exist_ok=True)

    return unique_dir


def move_unique_files(source_dir, target_dir, unique_files):
    """Mueve archivos únicos al directorio destino."""
    moved_files = []
    failed_files = []

    print(f"\n🚀 INICIANDO MOVIMIENTO DE {len(unique_files)} ARCHIVOS ÚNICOS")
    print(f"Origen: {source_dir}")
    print(f"Destino: {target_dir}")

    for filename in unique_files:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)

        if not os.path.exists(source_path):
            failed_files.append(f"{filename} - No encontrado")
            continue

        try:
            # Mover archivo
            shutil.move(source_path, target_path)
            moved_files.append(filename)
            print(f"✅ Movido: {filename}")

        except Exception as e:
            failed_files.append(f"{filename} - Error: {str(e)}")
            print(f"❌ Error moviendo {filename}: {str(e)}")

    return moved_files, failed_files


def generate_movement_report(moved_files, failed_files, source_dir, target_dir, analysis_file):
    """Genera reporte del movimiento de archivos (versión híbrida)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Obtener nombre del directorio dinámicamente para trazabilidad
    dir_name = os.path.basename(source_dir)
    report_file = f"movement_report_hybrid_{dir_name}_{timestamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("REPORTE DE MOVIMIENTO DE ARCHIVOS ÚNICOS (ANÁLISIS HÍBRIDO)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Análisis base: {analysis_file}\n")
        f.write("Método: Verificación visual selectiva + patrones\n")
        f.write(f"Directorio origen: {source_dir}\n")
        f.write(f"Directorio destino: {target_dir}\n")
        f.write(f"\nArchivos movidos exitosamente: {len(moved_files)}\n")
        f.write(f"Archivos con errores: {len(failed_files)}\n\n")

        if moved_files:
            f.write("ARCHIVOS MOVIDOS EXITOSAMENTE:\n")
            f.write("-" * 40 + "\n")
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
        f.write(f"- Total intentados: {total_attempted}\n")
        f.write(f"- Exitosos: {len(moved_files)}\n")
        f.write(f"- Fallidos: {len(failed_files)}\n")
        f.write(f"- Tasa de éxito: {success_rate:.1f}%\n")

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


def main():
    """Función principal del script de movimiento híbrido."""
    print("🚀 INICIANDO PASO 3: MOVIMIENTO DE ARCHIVOS ÚNICOS (HÍBRIDO)")
    print("============================================================")

    # Configuración del directorio fuente con los archivos únicos a mover
    source_directory = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/same3/train/tomato_yellow_leaf_curl_virus"

    # Cargar resultados del análisis
    result = load_latest_analysis_results(source_directory)
    if result is None:
        return

    unique_files, grouped_files, analysis_file = result

    print("\n📊 RESUMEN DEL ANÁLISIS:")
    print(f"   - Archivos únicos: {len(unique_files)}")
    print(f"   - Archivos agrupados: {len(grouped_files)}")
    print(f"   - Total: {len(unique_files) + len(grouped_files)}")

    if len(unique_files) == 0:
        print("\n🎉 ¡EXCELENTE! No hay archivos únicos que mover.")
        print("   Todos los archivos tienen transformaciones detectadas.")
        return

    # Crear directorios DENTRO del directorio fuente
    unique_dir = create_directories(source_directory)

    # Confirmar operación
    print("\n⚠️  CONFIRMACIÓN REQUERIDA:")
    print(f"   Se moverán {len(unique_files)} archivos únicos")
    print(f"   Desde: {source_directory}")
    print(f"   Hacia: {unique_dir}")

    confirm = input("\n¿Continuar con el movimiento? (s/N): ").lower().strip()
    if confirm not in ["s", "si", "sí", "y", "yes"]:
        print("❌ Operación cancelada por el usuario")
        return

    # Realizar movimiento
    moved_files, failed_files = move_unique_files(source_directory, unique_dir, unique_files)

    # Verificar movimiento
    verified, missing = verify_movement(unique_dir, moved_files)

    # Generar reporte
    report_file = generate_movement_report(moved_files, failed_files, source_directory, unique_dir, analysis_file)

    # Resumen final
    print("\n🎉 MOVIMIENTO COMPLETADO")
    print(f"📄 Reporte generado: {report_file}")
    print(f"✅ Archivos movidos: {len(moved_files)}")
    print(f"❌ Errores: {len(failed_files)}")
    print(f"📂 Directorio destino: {unique_dir}")

    if len(failed_files) == 0:
        print("\n🏆 ¡OPERACIÓN 100% EXITOSA!")
    else:
        print(f"\n⚠️  Operación completada con {len(failed_files)} errores")


if __name__ == "__main__":
    main()
