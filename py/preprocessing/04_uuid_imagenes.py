import os
import hashlib
from pathlib import Path


def count_files_recursive(root_path):
    """
    Cuenta todos los archivos de manera recursiva en un directorio
    """
    count = 0
    for root, dirs, files in os.walk(root_path):
        count += len(files)
    return count


def generate_16_char_uuid_from_file(file_path):
    """
    Genera un UUID de 16 caracteres basado en el hash SHA256 del archivo
    """
    sha256_hash = hashlib.sha256()

    # Leer el archivo en chunks para manejar archivos grandes
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    # Obtener los primeros 16 caracteres del hash
    return sha256_hash.hexdigest()[:16]


def rename_files_with_uuid(root_path):
    """
    Renombra todos los archivos agregando un UUID de 16 caracteres al comienzo
    """
    renamed_count = 0
    errors = []

    # Recorrer todos los directorios y subdirectorios
    for root, dirs, files in os.walk(root_path):
        for filename in files:
            try:
                # Ruta completa del archivo original
                old_path = os.path.join(root, filename)

                # Generar UUID de 16 caracteres basado en el contenido del archivo
                file_uuid = generate_16_char_uuid_from_file(old_path)

                # Crear nuevo nombre con UUID al comienzo
                new_filename = f"{file_uuid}_{filename}"
                new_path = os.path.join(root, new_filename)

                # Renombrar el archivo
                os.rename(old_path, new_path)
                renamed_count += 1

                print(f"Renombrado: {filename} -> {new_filename}")

            except Exception as e:
                error_msg = f"Error renombrando {filename}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)

    return renamed_count, errors


def main():
    # Ruta a procesar
    root_path = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/same/"

    # Verificar que la ruta existe
    if not os.path.exists(root_path):
        print(f"Error: La ruta {root_path} no existe.")
        return

    print(f"Procesando directorio: {root_path}")
    print("=" * 60)

    # 1. Contar archivos antes del renombrado
    print("1. Contando archivos antes del renombrado...")
    initial_count = count_files_recursive(root_path)
    print(f"   Número de archivos encontrados: {initial_count}")

    if initial_count == 0:
        print("No hay archivos para procesar.")
        return

    print("\n2. Renombrando archivos con UUID...")
    print("-" * 40)

    # 2. Renombrar archivos con UUID
    renamed_count, errors = rename_files_with_uuid(root_path)

    print(f"\n   Archivos renombrados exitosamente: {renamed_count}")
    if errors:
        print(f"   Errores encontrados: {len(errors)}")
        for error in errors:
            print(f"   - {error}")

    # 3. Contar archivos después del renombrado
    print("\n3. Contando archivos después del renombrado...")
    final_count = count_files_recursive(root_path)
    print(f"   Número de archivos final: {final_count}")

    # Verificación
    print("\n" + "=" * 60)
    print("RESUMEN:")
    print(f"- Archivos iniciales: {initial_count}")
    print(f"- Archivos renombrados: {renamed_count}")
    print(f"- Archivos finales: {final_count}")
    print(f"- Errores: {len(errors)}")

    if initial_count == final_count:
        print("✓ El número de archivos se mantuvo constante.")
    else:
        print("⚠ El número de archivos cambió durante el proceso.")

    if renamed_count == initial_count - len(errors):
        print("✓ Todos los archivos posibles fueron renombrados correctamente.")
    else:
        print("⚠ Algunos archivos no pudieron ser renombrados.")


if __name__ == "__main__":
    main()
