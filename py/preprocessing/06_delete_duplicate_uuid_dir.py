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


def delete_duplicates_same_directory(files_by_uuid):
    """
    Elimina archivos duplicados que estén en el mismo directorio.
    Mantiene el primer archivo encontrado y elimina los demás.
    """
    deleted_files = []
    preserved_files = []
    different_dir_duplicates = []

    for uuid_prefix, files in files_by_uuid.items():
        if len(files) > 1:
            # Agrupar archivos por directorio
            files_by_dir = defaultdict(list)
            for file_info in files:
                files_by_dir[file_info["directory"]].append(file_info)

            # Procesar cada directorio
            for directory, dir_files in files_by_dir.items():
                if len(dir_files) > 1:
                    # Hay duplicados en el mismo directorio
                    # Preservar el primero, eliminar el resto
                    preserved_files.append(dir_files[0])

                    for file_to_delete in dir_files[1:]:
                        try:
                            os.remove(file_to_delete["full_path"])
                            deleted_files.append(file_to_delete)
                            print(f"Eliminado: {file_to_delete['full_path']}")
                        except Exception as e:
                            print(
                                f"Error eliminando {file_to_delete['full_path']}: {e}"
                            )
                else:
                    # Solo un archivo en este directorio, pero puede haber otros en otros directorios
                    if len(files_by_dir) > 1:
                        different_dir_duplicates.extend(dir_files)
                    else:
                        preserved_files.extend(dir_files)

    return deleted_files, preserved_files, different_dir_duplicates


def generate_deletion_report(
    deleted_files,
    preserved_files,
    different_dir_duplicates,
    root_path,
    output_file="informe_eliminacion_duplicados.txt",
):
    """
    Genera un informe detallado de las eliminaciones realizadas
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("INFORME DE ELIMINACIÓN DE ARCHIVOS DUPLICADOS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directorio procesado: {root_path}\n")
        f.write(f"Archivos eliminados: {len(deleted_files)}\n")
        f.write(f"Archivos preservados: {len(preserved_files)}\n")
        f.write(
            f"Duplicados en directorios diferentes: {len(different_dir_duplicates)}\n"
        )
        f.write("=" * 80 + "\n\n")

        # Archivos eliminados
        if deleted_files:
            f.write("ARCHIVOS ELIMINADOS (duplicados en el mismo directorio):\n")
            f.write("-" * 60 + "\n")

            for i, file_info in enumerate(deleted_files, 1):
                f.write(f"{i}. {file_info['filename']}\n")
                f.write(f"   UUID: {file_info['filename'][:16]}\n")
                f.write(f"   Ruta: {file_info['full_path']}\n")
                f.write(f"   Tamaño: {file_info['size']} bytes\n\n")
        else:
            f.write(
                "✓ No se eliminaron archivos (no hay duplicados en el mismo directorio).\n\n"
            )

        # Duplicados en directorios diferentes
        if different_dir_duplicates:
            f.write("DUPLICADOS PRESERVADOS (en directorios diferentes):\n")
            f.write("-" * 60 + "\n")

            # Agrupar por UUID
            duplicates_by_uuid = defaultdict(list)
            for file_info in different_dir_duplicates:
                uuid_prefix = file_info["filename"][:16]
                duplicates_by_uuid[uuid_prefix].append(file_info)

            for uuid_prefix, files in duplicates_by_uuid.items():
                f.write(f"\nUUID: {uuid_prefix}\n")
                f.write("Archivos en directorios diferentes:\n")
                for i, file_info in enumerate(files, 1):
                    f.write(f"  {i}. {file_info['filename']}\n")
                    f.write(f"     Directorio: {file_info['directory']}\n")
                    f.write(f"     Ruta completa: {file_info['full_path']}\n")
                    f.write(f"     Tamaño: {file_info['size']} bytes\n")
                f.write("-" * 40 + "\n")
        else:
            f.write("✓ No hay duplicados en directorios diferentes.\n\n")

        # Estadísticas
        f.write("ESTADÍSTICAS FINALES:\n")
        f.write("-" * 25 + "\n")
        total_processed = (
            len(deleted_files) + len(preserved_files) + len(different_dir_duplicates)
        )
        f.write(f"Total de archivos procesados: {total_processed}\n")
        f.write(f"Archivos eliminados: {len(deleted_files)}\n")
        f.write(
            f"Archivos preservados: {len(preserved_files) + len(different_dir_duplicates)}\n"
        )
        if total_processed > 0:
            f.write(
                f"Porcentaje eliminado: {(len(deleted_files) / total_processed) * 100:.2f}%\n"
            )


def main():
    # Ruta a analizar
    root_path = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/same/"

    # Verificar que la ruta existe
    if not os.path.exists(root_path):
        print(f"Error: La ruta {root_path} no existe.")
        return

    print(f"Procesando directorio: {root_path}")
    print(
        "Buscando y eliminando archivos con UUIDs duplicados en el mismo directorio..."
    )
    print("=" * 80)

    # Buscar archivos con UUIDs duplicados
    files_by_uuid = find_files_with_duplicate_uuid(root_path)

    # Filtrar solo los duplicados
    duplicates = {
        uuid_prefix: files
        for uuid_prefix, files in files_by_uuid.items()
        if len(files) > 1
    }

    if not duplicates:
        print("✓ No se encontraron archivos con UUIDs duplicados.")
        return

    print(f"Encontrados {len(duplicates)} UUIDs con duplicados.")

    # Confirmar eliminación
    response = input("¿Desea proceder con la eliminación de duplicados? (s/N): ")
    if response.lower() != "s":
        print("Operación cancelada.")
        return

    # Eliminar duplicados en el mismo directorio
    deleted_files, preserved_files, different_dir_duplicates = (
        delete_duplicates_same_directory(files_by_uuid)
    )

    # Generar informe de eliminación
    output_file = "informe_eliminacion_duplicados.txt"
    generate_deletion_report(
        deleted_files, preserved_files, different_dir_duplicates, root_path, output_file
    )

    # Mostrar resumen en consola
    print("\n" + "=" * 80)
    print("RESUMEN DE ELIMINACIÓN:")
    print("=" * 80)
    print(f"✓ Archivos eliminados: {len(deleted_files)}")
    print(
        f"✓ Archivos preservados: {len(preserved_files) + len(different_dir_duplicates)}"
    )

    if deleted_files:
        print("\nARCHIVOS ELIMINADOS:")
        for file_info in deleted_files[:10]:  # Mostrar solo los primeros 10
            print(f"  - {file_info['filename']} ({file_info['directory']})")
        if len(deleted_files) > 10:
            print(f"  ... y {len(deleted_files) - 10} archivos más")

    if different_dir_duplicates:
        print("\nDUPLICADOS PRESERVADOS (en directorios diferentes):")
        duplicates_by_uuid = defaultdict(list)
        for file_info in different_dir_duplicates:
            uuid_prefix = file_info["filename"][:16]
            duplicates_by_uuid[uuid_prefix].append(file_info)

        for uuid_prefix, files in list(duplicates_by_uuid.items())[:5]:
            print(
                f"  UUID {uuid_prefix}: {len(files)} archivos en directorios diferentes"
            )
        if len(duplicates_by_uuid) > 5:
            print(f"  ... y {len(duplicates_by_uuid) - 5} UUIDs más con duplicados")

    print(f"\nInforme detallado guardado en: {output_file}")


if __name__ == "__main__":
    main()
