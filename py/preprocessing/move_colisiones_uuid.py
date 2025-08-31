import os
import shutil
from datetime import datetime

# Definir las rutas de los directorios
base_dir = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "valid")
collisions_dir = os.path.join(base_dir, "colisiones_a_revisar")

# Crear el directorio de colisiones si no existe
os.makedirs(collisions_dir, exist_ok=True)

# Crear el archivo de informe
log_file = "informe_colisiones.txt"
with open(log_file, "w", encoding="utf-8") as log:
    # Escribir encabezado con fecha y hora
    log.write(
        f"Informe de colisiones generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    log.write("=" * 50 + "\n\n")

    # Leer el archivo de informe de colisiones
    with open("colisiones_reporte.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Nombre de base conflictivo:"):
            # Obtener el nombre de base del conflicto
            base_name = line.split(":")[1].strip()
            log.write(f"Procesando colisión: {base_name}\n")
            i += 1

            # Buscar la sección de TRAIN
            while i < len(lines) and not lines[i].strip().startswith(
                "Presente en TRAIN:"
            ):
                i += 1

            if i < len(lines):
                i += 1  # Saltar la línea "Presente en TRAIN:"

                # Recolectar los archivos de TRAIN
                train_files = []
                while i < len(lines) and lines[i].strip().startswith("- "):
                    file_path = lines[i].strip()[2:]  # Quitar "- " del inicio
                    train_files.append(file_path)
                    i += 1

                # Buscar la sección de VALIDATION
                while i < len(lines) and not lines[i].strip().startswith(
                    "Presente en VALIDATION:"
                ):
                    i += 1

                if i < len(lines):
                    i += 1  # Saltar la línea "Presente en VALIDATION:"

                    # Recolectar los archivos de VALIDATION
                    val_files = []
                    while i < len(lines) and lines[i].strip().startswith("- "):
                        file_path = lines[i].strip()[2:]  # Quitar "- " del inicio
                        val_files.append(file_path)
                        i += 1

                    # Mover los archivos de TRAIN a colisiones_a_revisar/train/
                    for file_path in train_files:
                        subdir = file_path.split("/")[0]
                        train_collision_path = os.path.join(
                            collisions_dir, "train", subdir
                        )
                        os.makedirs(train_collision_path, exist_ok=True)

                        src = os.path.join(train_dir, file_path)
                        dst = os.path.join(collisions_dir, "train", file_path)

                        if os.path.exists(src):
                            shutil.move(src, dst)
                            log.write(f"Movido de TRAIN: {file_path}\n")
                        else:
                            log.write(f"Archivo no encontrado en TRAIN: {src}\n")

                    # Mover los archivos de VALIDATION a colisiones_a_revisar/valid/
                    for file_path in val_files:
                        subdir = file_path.split("/")[0]
                        valid_collision_path = os.path.join(
                            collisions_dir, "valid", subdir
                        )
                        os.makedirs(valid_collision_path, exist_ok=True)

                        src = os.path.join(val_dir, file_path)
                        dst = os.path.join(collisions_dir, "valid", file_path)

                        if os.path.exists(src):
                            shutil.move(src, dst)
                            log.write(f"Movido de VALIDATION: {file_path}\n")
                        else:
                            log.write(f"Archivo no encontrado en VALIDATION: {src}\n")
        else:
            i += 1

    log.write("\n" + "=" * 50 + "\n")
    log.write(
        "Proceso completado. Todos los archivos con colisiones han sido movidos a 'colisiones_a_revisar'.\n"
    )
