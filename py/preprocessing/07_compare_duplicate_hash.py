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


def main():
    # Ruta a analizar
    root_path = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/same/"

    # Verificar que la ruta existe
    if not os.path.exists(root_path):
        print(f"Error: La ruta {root_path} no existe.")
        return

    print(f"Analizando directorio: {root_path}")
    print("Buscando archivos con hashes SHA256 duplicados...")
    print("=" * 70)

    # Buscar archivos con hashes duplicados
    files_by_hash = find_files_with_duplicate_hash(root_path)

    # Filtrar solo los duplicados
    duplicates = {
        file_hash: files for file_hash, files in files_by_hash.items() if len(files) > 1
    }

    # Generar informe
    output_file = "informe_hash_duplicados.txt"
    generate_hash_duplicate_report(files_by_hash, root_path, output_file)

    # Mostrar resumen en consola
    total_files = sum(len(files) for files in files_by_hash.values())

    print("\nAnálisis completado:")
    print(f"- Total de archivos analizados: {total_files}")
    print(f"- Hashes únicos encontrados: {len(files_by_hash)}")
    print(f"- Hashes con duplicados: {len(duplicates)}")

    if duplicates:
        print("\n⚠ DUPLICADOS ENCONTRADOS:")
        total_duplicated = sum(len(files) for files in duplicates.values())
        unique_files = total_files - total_duplicated + len(duplicates)
        print(f"- Total de archivos duplicados: {total_duplicated}")
        print(f"- Archivos únicos: {unique_files}")
        print(
            f"- Porcentaje de duplicados: {(total_duplicated / total_files) * 100:.2f}%"
        )

        # Calcular espacio desperdiciado
        wasted_space = 0
        for files in duplicates.values():
            file_size = files[0]["size"]
            wasted_space += file_size * (len(files) - 1)

        print(f"- Espacio desperdiciado: {wasted_space / (1024 * 1024):.2f} MB")

        print("\n" + "=" * 80)
        print("REPORTE DETALLADO DE DUPLICADOS:")
        print("=" * 80)

        for file_hash, files in list(duplicates.items())[
            :5
        ]:  # Mostrar solo los primeros 5
            print(f"\nHASH: {file_hash}")
            print(f"Tamaño: {files[0]['size']} bytes")
            print(f"Cantidad de duplicados: {len(files)}")
            print("Archivos:")

            for i, file_info in enumerate(files, 1):
                print(f"  {i}. {file_info['filename']}")
                print(f"     Ruta: {file_info['full_path']}")

            print("-" * 60)

        if len(duplicates) > 5:
            print(
                f"\n... y {len(duplicates) - 5} grupos de duplicados más (ver informe completo)"
            )

    else:
        print("\n✓ No se encontraron archivos duplicados.")
        print("Todos los archivos son únicos según su contenido.")

    print(f"\nInforme detallado guardado en: {output_file}")


if __name__ == "__main__":
    main()
