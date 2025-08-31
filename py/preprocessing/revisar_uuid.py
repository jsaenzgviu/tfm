import os
from collections import defaultdict
from datetime import datetime


def extract_uuid_from_filename(filename):
    """
    Extrae los primeros 16 caracteres del nombre del archivo
    """
    return filename[:16] if len(filename) >= 16 else filename


def find_files_with_duplicate_uuid(root_path):
    """
    Busca archivos que tengan los mismos 16 primeros dígitos en el nombre
    """
    files_by_uuid = defaultdict(list)

    # Recorrer todos los directorios y subdirectorios
    for root, dirs, files in os.walk(root_path):
        for filename in files:
            # Extraer los primeros 16 caracteres
            uuid_prefix = extract_uuid_from_filename(filename)

            # Solo considerar archivos que tengan al menos 16 caracteres
            if len(filename) >= 16:
                full_path = os.path.join(root, filename)

                # Guardar información del archivo
                files_by_uuid[uuid_prefix].append(
                    {
                        "filename": filename,
                        "full_path": full_path,
                        "directory": root,
                        "size": os.path.getsize(full_path)
                        if os.path.exists(full_path)
                        else 0,
                    }
                )

    return files_by_uuid


def generate_uuid_report(
    files_by_uuid, root_path, output_file="informe_uuid_duplicados.txt"
):
    """
    Genera un informe con los archivos que tienen los mismos 16 primeros dígitos
    """
    duplicates = {
        uuid_prefix: files
        for uuid_prefix, files in files_by_uuid.items()
        if len(files) > 1
    }

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("INFORME DE ARCHIVOS CON UUID DUPLICADOS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directorio analizado: {root_path}\n")
        f.write(f"Total de UUIDs únicos: {len(files_by_uuid)}\n")
        f.write(f"UUIDs con duplicados: {len(duplicates)}\n")
        f.write("=" * 80 + "\n\n")

        if duplicates:
            f.write("ARCHIVOS CON UUID DUPLICADOS (primeros 16 caracteres):\n")
            f.write("-" * 60 + "\n")

            for uuid_prefix, files in duplicates.items():
                f.write(f"\nUUID: {uuid_prefix}\n")
                f.write(f"Cantidad de archivos con este UUID: {len(files)}\n")
                f.write("Archivos encontrados:\n")

                for i, file_info in enumerate(files, 1):
                    f.write(f"  {i}. {file_info['filename']}\n")
                    f.write(f"     Ruta completa: {file_info['full_path']}\n")
                    f.write(f"     Directorio: {file_info['directory']}\n")
                    f.write(f"     Tamaño: {file_info['size']} bytes\n")

                f.write("-" * 60 + "\n")
        else:
            f.write("✓ No se encontraron archivos con UUIDs duplicados.\n")
            f.write(
                "Todos los archivos tienen UUIDs únicos en los primeros 16 caracteres.\n"
            )

        # Estadísticas adicionales
        f.write("\nESTADÍSTICAS DETALLADAS:\n")
        f.write("-" * 30 + "\n")
        total_files = sum(len(files) for files in files_by_uuid.values())
        f.write(f"Total de archivos analizados: {total_files}\n")
        f.write(f"Archivos con nombres >= 16 caracteres: {total_files}\n")

        if duplicates:
            total_duplicated_files = sum(len(files) for files in duplicates.values())
            f.write(
                f"Total de archivos con UUIDs duplicados: {total_duplicated_files}\n"
            )
            f.write(
                f"Porcentaje de archivos duplicados: {(total_duplicated_files / total_files) * 100:.2f}%\n"
            )

            f.write("\nRESUMEN DE DUPLICADOS POR UUID:\n")
            for uuid_prefix, files in sorted(duplicates.items()):
                f.write(f"- {uuid_prefix}: {len(files)} archivos\n")


def main():
    # Ruta a analizar
    root_path = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/same/"

    # Verificar que la ruta existe
    if not os.path.exists(root_path):
        print(f"Error: La ruta {root_path} no existe.")
        return

    print(f"Analizando directorio: {root_path}")
    print("Buscando archivos con UUIDs duplicados (primeros 16 caracteres)...")
    print("=" * 70)

    # Buscar archivos con UUIDs duplicados
    files_by_uuid = find_files_with_duplicate_uuid(root_path)

    # Filtrar solo los duplicados
    duplicates = {
        uuid_prefix: files
        for uuid_prefix, files in files_by_uuid.items()
        if len(files) > 1
    }

    # Generar informe
    output_file = "informe_uuid_duplicados.txt"
    generate_uuid_report(files_by_uuid, root_path, output_file)

    # Mostrar resumen en consola
    total_files = sum(len(files) for files in files_by_uuid.values())

    print("Análisis completado:")
    print(f"- Total de archivos analizados: {total_files}")
    print(f"- UUIDs únicos encontrados: {len(files_by_uuid)}")
    print(f"- UUIDs con duplicados: {len(duplicates)}")

    if duplicates:
        print("\n⚠ DUPLICADOS ENCONTRADOS:")
        total_duplicated = sum(len(files) for files in duplicates.values())
        print(f"- Total de archivos con UUIDs duplicados: {total_duplicated}")
        print(
            f"- Porcentaje de duplicados: {(total_duplicated / total_files) * 100:.2f}%"
        )

        print("\n" + "=" * 80)
        print("REPORTE DETALLADO DE DUPLICADOS:")
        print("=" * 80)

        for uuid_prefix, files in duplicates.items():
            print(f"\nUUID: {uuid_prefix}")
            print(f"Cantidad de archivos: {len(files)}")
            print("-" * 60)

            for i, file_info in enumerate(files, 1):
                print(f"  {i}. Archivo: {file_info['filename']}")
                print(f"     Ruta completa: {file_info['full_path']}")
                print(f"     Tamaño: {file_info['size']} bytes")
                print()

            print("-" * 60)

    else:
        print("\n✓ No se encontraron UUIDs duplicados.")
        print("Todos los archivos tienen identificadores únicos.")

    print(f"\nInforme detallado guardado en: {output_file}")


if __name__ == "__main__":
    main()
