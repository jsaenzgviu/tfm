import os
import hashlib
from collections import defaultdict
from datetime import datetime


def calculate_sha256(file_path):
    """
    Calcula el hash SHA256 de un archivo
    """
    sha256_hash = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:
            # Leer el archivo en chunks para manejar archivos grandes
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error calculando hash para {file_path}: {e}")
        return None


def find_files_with_duplicate_hash(root_path):
    """
    Busca archivos que tengan el mismo hash SHA256
    """
    files_by_hash = defaultdict(list)
    total_files = 0
    processed_files = 0

    print("Calculando hashes SHA256...")

    # Recorrer todos los directorios y subdirectorios
    for root, dirs, files in os.walk(root_path):
        total_files += len(files)

    # Procesar archivos
    for root, dirs, files in os.walk(root_path):
        for filename in files:
            full_path = os.path.join(root, filename)
            processed_files += 1

            # Mostrar progreso
            if processed_files % 100 == 0 or processed_files == total_files:
                print(f"Procesando: {processed_files}/{total_files} archivos...")

            # Calcular hash del archivo
            file_hash = calculate_sha256(full_path)

            if file_hash:
                # Obtener información del archivo
                try:
                    file_size = os.path.getsize(full_path)
                    files_by_hash[file_hash].append(
                        {
                            "filename": filename,
                            "full_path": full_path,
                            "directory": root,
                            "size": file_size,
                            "hash": file_hash,
                        }
                    )
                except Exception as e:
                    print(f"Error obteniendo información de {full_path}: {e}")

    return files_by_hash


def generate_hash_duplicate_report(
    files_by_hash, root_path, output_file="informe_hash_duplicados.txt"
):
    """
    Genera un informe detallado de archivos con hashes duplicados
    """
    duplicates = {
        file_hash: files for file_hash, files in files_by_hash.items() if len(files) > 1
    }

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("INFORME DE ARCHIVOS CON HASH SHA256 DUPLICADOS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directorio analizado: {root_path}\n")
        f.write(f"Total de hashes únicos: {len(files_by_hash)}\n")
        f.write(f"Hashes con duplicados: {len(duplicates)}\n")
        f.write("=" * 80 + "\n\n")

        if duplicates:
            f.write("ARCHIVOS CON HASH SHA256 DUPLICADOS:\n")
            f.write("-" * 60 + "\n")

            for file_hash, files in duplicates.items():
                f.write(f"\nHASH SHA256: {file_hash}\n")
                f.write(f"Cantidad de archivos con este hash: {len(files)}\n")
                f.write(f"Tamaño del archivo: {files[0]['size']} bytes\n")
                f.write("Archivos duplicados:\n")

                for i, file_info in enumerate(files, 1):
                    f.write(f"  {i}. {file_info['filename']}\n")
                    f.write(f"     Ruta completa: {file_info['full_path']}\n")
                    f.write(f"     Directorio: {file_info['directory']}\n")

                f.write("-" * 60 + "\n")
        else:
            f.write("✓ No se encontraron archivos con hashes duplicados.\n")
            f.write("Todos los archivos son únicos según su contenido.\n")

        # Estadísticas adicionales
        f.write("\nESTADÍSTICAS DETALLADAS:\n")
        f.write("-" * 30 + "\n")
        total_files = sum(len(files) for files in files_by_hash.values())
        f.write(f"Total de archivos analizados: {total_files}\n")

        if duplicates:
            total_duplicated_files = sum(len(files) for files in duplicates.values())
            unique_files = total_files - total_duplicated_files + len(duplicates)
            f.write(f"Total de archivos duplicados: {total_duplicated_files}\n")
            f.write(f"Archivos únicos: {unique_files}\n")
            f.write(
                f"Porcentaje de archivos duplicados: {(total_duplicated_files / total_files) * 100:.2f}%\n"
            )

            # Calcular espacio desperdiciado
            wasted_space = 0
            for files in duplicates.values():
                file_size = files[0]["size"]
                wasted_space += file_size * (len(files) - 1)

            f.write(
                f"Espacio desperdiciado por duplicados: {wasted_space} bytes ({wasted_space / (1024 * 1024):.2f} MB)\n"
            )

            f.write("\nRESUMEN DE DUPLICADOS POR HASH:\n")
            for file_hash, files in sorted(
                duplicates.items(), key=lambda x: len(x[1]), reverse=True
            ):
                f.write(
                    f"- {file_hash[:16]}...: {len(files)} archivos ({files[0]['size']} bytes cada uno)\n"
                )


def delete_duplicate_files(files_by_hash):
    """
    Elimina archivos duplicados basado en el hash SHA256.
    Mantiene el primer archivo encontrado y elimina todos los demás duplicados.
    """
    deleted_files = []
    preserved_files = []
    errors = []

    for file_hash, files in files_by_hash.items():
        if len(files) > 1:
            # Preservar el primer archivo
            preserved_files.append(files[0])

            # Eliminar todos los duplicados
            for file_to_delete in files[1:]:
                try:
                    os.remove(file_to_delete["full_path"])
                    deleted_files.append(file_to_delete)
                    print(f"Eliminado: {file_to_delete['full_path']}")
                except Exception as e:
                    error_msg = f"Error eliminando {file_to_delete['full_path']}: {e}"
                    errors.append(error_msg)
                    print(error_msg)
        else:
            # Archivo único, preservar
            preserved_files.extend(files)

    return deleted_files, preserved_files, errors


def generate_deletion_report(
    deleted_files,
    preserved_files,
    errors,
    root_path,
    output_file="informe_eliminacion_hash_duplicados.txt",
):
    """
    Genera un informe detallado de las eliminaciones realizadas
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("INFORME DE ELIMINACIÓN DE ARCHIVOS DUPLICADOS POR HASH\n")
        f.write("=" * 80 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directorio procesado: {root_path}\n")
        f.write(f"Archivos eliminados: {len(deleted_files)}\n")
        f.write(f"Archivos preservados: {len(preserved_files)}\n")
        f.write(f"Errores durante eliminación: {len(errors)}\n")
        f.write("=" * 80 + "\n\n")

        # Archivos eliminados
        if deleted_files:
            f.write("ARCHIVOS ELIMINADOS (duplicados por hash SHA256):\n")
            f.write("-" * 60 + "\n")

            for i, file_info in enumerate(deleted_files, 1):
                f.write(f"{i}. {file_info['filename']}\n")
                f.write(f"   Hash SHA256: {file_info['hash']}\n")
                f.write(f"   Ruta: {file_info['full_path']}\n")
                f.write(f"   Tamaño: {file_info['size']} bytes\n\n")
        else:
            f.write("✓ No se eliminaron archivos (no hay duplicados).\n\n")

        # Errores
        if errors:
            f.write("ERRORES DURANTE LA ELIMINACIÓN:\n")
            f.write("-" * 40 + "\n")
            for i, error in enumerate(errors, 1):
                f.write(f"{i}. {error}\n")
            f.write("\n")

        # Estadísticas de espacio liberado
        if deleted_files:
            total_space_freed = sum(file_info["size"] for file_info in deleted_files)
            f.write("ESPACIO LIBERADO:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total de bytes liberados: {total_space_freed}\n")
            f.write(
                f"Espacio liberado en MB: {total_space_freed / (1024 * 1024):.2f} MB\n"
            )
            f.write(
                f"Espacio liberado en GB: {total_space_freed / (1024 * 1024 * 1024):.2f} GB\n\n"
            )

        # Estadísticas finales
        f.write("ESTADÍSTICAS FINALES:\n")
        f.write("-" * 25 + "\n")
        total_processed = len(deleted_files) + len(preserved_files)
        f.write(f"Total de archivos procesados: {total_processed}\n")
        f.write(f"Archivos eliminados: {len(deleted_files)}\n")
        f.write(f"Archivos preservados: {len(preserved_files)}\n")
        f.write(f"Errores: {len(errors)}\n")
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
    print("Buscando y eliminando archivos con hashes SHA256 duplicados...")
    print("=" * 80)

    # Buscar archivos con hashes duplicados
    files_by_hash = find_files_with_duplicate_hash(root_path)

    # Filtrar solo los duplicados
    duplicates = {
        file_hash: files for file_hash, files in files_by_hash.items() if len(files) > 1
    }

    if not duplicates:
        print("✓ No se encontraron archivos duplicados.")
        return

    # Mostrar resumen antes de eliminar
    total_files = sum(len(files) for files in files_by_hash.values())
    total_duplicated = sum(len(files) for files in duplicates.values())

    print("\nRESUMEN ANTES DE ELIMINAR:")
    print(f"- Total de archivos: {total_files}")
    print(f"- Archivos duplicados: {total_duplicated}")
    print(f"- Grupos de duplicados: {len(duplicates)}")

    # Calcular espacio a liberar
    space_to_free = 0
    for files in duplicates.values():
        file_size = files[0]["size"]
        space_to_free += file_size * (len(files) - 1)

    print(f"- Espacio a liberar: {space_to_free / (1024 * 1024):.2f} MB")

    # Confirmar eliminación
    response = input("\n¿Desea proceder con la eliminación de duplicados? (s/N): ")
    if response.lower() != "s":
        print("Operación cancelada.")
        return

    print("\nEliminando archivos duplicados...")
    print("-" * 40)

    # Eliminar archivos duplicados
    deleted_files, preserved_files, errors = delete_duplicate_files(files_by_hash)

    # Generar informe de eliminación
    output_file = "informe_eliminacion_hash_duplicados.txt"
    generate_deletion_report(
        deleted_files, preserved_files, errors, root_path, output_file
    )

    # Mostrar resumen final en consola
    print("\n" + "=" * 80)
    print("RESUMEN DE ELIMINACIÓN:")
    print("=" * 80)
    print(f"✓ Archivos eliminados: {len(deleted_files)}")
    print(f"✓ Archivos preservados: {len(preserved_files)}")

    if errors:
        print(f"⚠ Errores: {len(errors)}")

    if deleted_files:
        total_space_freed = sum(file_info["size"] for file_info in deleted_files)
        print(f"✓ Espacio liberado: {total_space_freed / (1024 * 1024):.2f} MB")

        print("\nPRIMEROS ARCHIVOS ELIMINADOS:")
        for file_info in deleted_files[:10]:
            print(f"  - {file_info['filename']} ({file_info['size']} bytes)")
        if len(deleted_files) > 10:
            print(f"  ... y {len(deleted_files) - 10} archivos más")

    if errors:
        print("\nERRORES ENCONTRADOS:")
        for error in errors[:5]:
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... y {len(errors) - 5} errores más")

    print(f"\nInforme detallado guardado en: {output_file}")


if __name__ == "__main__":
    main()
