import os

# Ruta al directorio del dataset
dataset_path = "./valid"

# Obtener nombres de las clases (carpetas)
classes = [
    d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))
]

# Contar imágenes por clase
class_distribution = {}
for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    num_images = len(
        [
            f
            for f in os.listdir(class_path)
            if f.lower().endswith(".jpg")
            or f.lower().endswith(".png")
            or f.lower().endswith(".jpeg")
        ]
    )
    class_distribution[cls] = num_images

# Calcular el total de imágenes
total_images = sum(class_distribution.values())

# Imprimir distribución en formato tabla Markdown
print("# Distribución de Imágenes por Clase\n")
print("| Clase | Número de Imágenes |")
print("|-------|-------------------|")

# Ordenar por número de imágenes (descendente)
for cls, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
    print(f"| {cls} | {count:,} |")

print(f"| **TOTAL** | **{total_images:,}** |")

print(f"\n**Total de imágenes en el dataset:** {total_images:,}")
print(f"**Número de clases:** {len(class_distribution)}")
