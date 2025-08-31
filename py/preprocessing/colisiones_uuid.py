import os
from collections import defaultdict

# Definir la lista de sufijos de transformación
transformation_suffixes = [
    "_mirror",
    "_change_180",
    "_change_270",
    "_change_90",
    "_hight",
    "_lower",
    "_fliptb",
    "_fliplr",
    "_new30degfliplr",
    "_newpixel25",
    "_180deg",
    "_270deg",
    "_90deg",
    "_mirror_vertical",
    # Agregar más sufijos si es necesario
]


def get_base_name(filename):
    """
    Extraer el nombre de base de un archivo según su tipo.
    - Si el archivo tiene un sufijo de transformación, devuelve el nombre antes del sufijo.
    - Si no tiene sufijo de transformación, devuelve el nombre completo incluyendo la extensión.
    """
    fn, ext = os.path.splitext(filename)
    for suffix in transformation_suffixes:
        if fn.endswith(suffix):
            return fn[: -len(suffix)]
    return filename  # Incluye la extensión si no hay sufijo de transformación


def find_collisions(train_dir, val_dir):
    """
    Detectar colisiones entre los directorios train y validation.
    - Agrupa archivos por su nombre de base.
    - Si hay archivos con el mismo nombre de base en ambos directorios, es una colisión.
    """
    # Paso 1: Mapear nombres de base a archivos en train
    train_base_map = defaultdict(list)
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                base_name = get_base_name(filename)
                train_base_map[base_name].append((class_name, filename))

    # Paso 2: Mapear nombres de base a archivos en validation
    val_base_map = defaultdict(list)
    for class_name in os.listdir(val_dir):
        class_dir = os.path.join(val_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                base_name = get_base_name(filename)
                val_base_map[base_name].append((class_name, filename))

    # Paso 3: Encontrar colisiones (nombres de base presentes en ambos train y validation)
    collisions = [
        base_name for base_name in train_base_map if base_name in val_base_map
    ]

    # Paso 4: Generar reporte y guardarlo en archivo
    output_file = "colisiones_reporte.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Reporte de Colisiones ===\n")
        f.write(f"Colisiones detectadas: {len(collisions)}\n\n")
        for base_name in collisions:
            f.write(f"Nombre de base conflictivo: {base_name}\n")
            f.write("  Presente en TRAIN:\n")
            for class_name, filename in train_base_map[base_name]:
                f.write(f"    - {os.path.join(class_name, filename)}\n")
            f.write("  Presente en VALIDATION:\n")
            for class_name, filename in val_base_map[base_name]:
                f.write(f"    - {os.path.join(class_name, filename)}\n")
            f.write("\n")

    print(f"Reporte guardado en: {output_file}")
    print(f"Colisiones detectadas: {len(collisions)}")

    return collisions


# Uso
train_dir = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/train"
val_dir = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/valid"
collisions = find_collisions(train_dir, val_dir)
