import os
from collections import defaultdict
from datetime import datetime


def find_files_with_same_name(root_path):
    """
    Busca archivos con el mismo nombre (sin extensión) de manera recursiva
    """
    files_by_name = defaultdict(list)

    # Recorrer todos los directorios y subdirectorios
    for root, dirs, files in os.walk(root_path):
        for file in files:
            # Obtener el nombre sin extensión
            name_without_ext = os.path.splitext(file)[0]
            full_path = os.path.join(root, file)

            # Guardar información del archivo
            files_by_name[name_without_ext].append(
                {
                    "full_path": full_path,
                    "filename": file,
                    "directory": root,
                    "extension": os.path.splitext(file)[1],
                }
            )

    return files_by_name


def generate_report(
    files_by_name, root_path, output_file="informe_archivos_duplicados.txt"
):
    """
    Genera un informe con los archivos que tienen el mismo nombre
    """
    duplicates = {
        name: files for name, files in files_by_name.items() if len(files) > 1
    }

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("INFORME DE ARCHIVOS CON EL MISMO NOMBRE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directorio analizado: {root_path}\n")
        f.write(f"Total de nombres únicos: {len(files_by_name)}\n")
        f.write(f"Nombres con duplicados: {len(duplicates)}\n")
        f.write("=" * 80 + "\n\n")

        if duplicates:
            f.write("ARCHIVOS CON EL MISMO NOMBRE:\n")
            f.write("-" * 40 + "\n")

            for name, files in duplicates.items():
                f.write(f"\nNombre base: {name}\n")
                f.write(f"Cantidad de archivos: {len(files)}\n")
                f.write("Archivos encontrados:\n")

                for i, file_info in enumerate(files, 1):
                    f.write(f"  {i}. {file_info['filename']}\n")
                    f.write(f"     Ruta: {file_info['full_path']}\n")
                    f.write(f"     Extensión: {file_info['extension']}\n")

                f.write("-" * 40 + "\n")
        else:
            f.write("No se encontraron archivos con nombres duplicados.\n")

        # Estadísticas adicionales
        f.write("\nESTADÍSTICAS:\n")
        f.write("-" * 20 + "\n")
        total_files = sum(len(files) for files in files_by_name.values())
        f.write(f"Total de archivos analizados: {total_files}\n")

        if duplicates:
            f.write("\nRESUMEN DE DUPLICADOS:\n")
            for name, files in duplicates.items():
                extensions = [f["extension"] for f in files]
                f.write(f"- {name}: {len(files)} archivos {extensions}\n")


def main():
    # Ruta a analizar
    root_path = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/same/"

    # Verificar que la ruta existe
    if not os.path.exists(root_path):
        print(f"Error: La ruta {root_path} no existe.")
        return

    print(f"Analizando directorio: {root_path}")
    print("Buscando archivos con el mismo nombre...")

    # Buscar archivos
    files_by_name = find_files_with_same_name(root_path)

    # Generar informe
    output_file = "informe_archivos_duplicados.txt"
    generate_report(files_by_name, root_path, output_file)

    # Mostrar resumen en consola
    duplicates = {
        name: files for name, files in files_by_name.items() if len(files) > 1
    }
    total_files = sum(len(files) for files in files_by_name.values())

    print("\nAnálisis completado:")
    print(f"- Total de archivos analizados: {total_files}")
    print(f"- Nombres únicos encontrados: {len(files_by_name)}")
    print(f"- Nombres con duplicados: {len(duplicates)}")
    print(f"- Informe guardado en: {output_file}")

    if duplicates:
        print("\nArchivos con nombres duplicados:")
        for name, files in list(duplicates.items())[:5]:  # Mostrar solo los primeros 5
            print(f"  - {name}: {len(files)} archivos")
        if len(duplicates) > 5:
            print(f"  ... y {len(duplicates) - 5} más (ver informe completo)")


if __name__ == "__main__":
    main()
