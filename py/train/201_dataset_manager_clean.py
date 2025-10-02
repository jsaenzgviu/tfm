#!/usr/bin/env python3
"""
Gestión y organización de datos para el dataset de enfermedades de tomate
Incluye split de datos y análisis de distribución
"""

import os
import shutil
import json
from sklearn.model_selection import train_test_split

# ==================================================================================
# CONFIGURACIÓN - MODIFICAR ESTAS VARIABLES SEGÚN NECESIDADES
# ==================================================================================

# Directorios base
SOURCE_DATASET_DIR = "dataset"  # Directorio original con train y test
TARGET_DATASET_DIR = "dataset_final"  # Directorio final con train/valid/test

# Configuración de splits
VALIDATION_SPLIT = 0.2  # 20% del train original se convertirá en validation
RANDOM_STATE = 42  # Para reproducibilidad

# Configuración de imágenes
TARGET_SIZE = (224, 224)  # Tamaño objetivo para las imágenes
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# ==================================================================================
# FIN DE CONFIGURACIÓN
# ==================================================================================


class DatasetManager:
    """
    Gestor del dataset con funcionalidades de split y análisis de distribución
    """

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.source_dir = os.path.join(self.script_dir, SOURCE_DATASET_DIR)
        self.target_dir = os.path.join(self.script_dir, TARGET_DATASET_DIR)

        self.class_names = []
        self.class_distribution = {}

        self._verify_source_dataset()

    def _verify_source_dataset(self):
        """Verificar que el dataset fuente existe y tiene la estructura correcta"""
        source_train = os.path.join(self.source_dir, "train")
        source_test = os.path.join(self.source_dir, "test")

        if not os.path.exists(source_train):
            raise FileNotFoundError(f"❌ No se encontró directorio: {source_train}")

        if not os.path.exists(source_test):
            raise FileNotFoundError(f"❌ No se encontró directorio: {source_test}")

        # Obtener clases del directorio train
        self.class_names = sorted([d for d in os.listdir(source_train) if os.path.isdir(os.path.join(source_train, d))])

        print("✅ Dataset fuente verificado:")
        print(f"   - Train: {source_train}")
        print(f"   - Test: {source_test}")
        print(f"   - Clases: {len(self.class_names)}")

    def analyze_dataset_distribution(self):
        """Analizar distribución actual del dataset"""
        print("\n📊 Analizando distribución del dataset...")

        source_train = os.path.join(self.source_dir, "train")
        source_test = os.path.join(self.source_dir, "test")

        train_distribution = {}
        test_distribution = {}

        # Contar imágenes en train
        for class_name in self.class_names:
            train_class_path = os.path.join(source_train, class_name)
            if os.path.exists(train_class_path):
                count = len([f for f in os.listdir(train_class_path) if any(f.lower().endswith(ext.lower()) for ext in VALID_EXTENSIONS)])
                train_distribution[class_name] = count
            else:
                train_distribution[class_name] = 0

        # Contar imágenes en test
        for class_name in self.class_names:
            test_class_path = os.path.join(source_test, class_name)
            if os.path.exists(test_class_path):
                count = len([f for f in os.listdir(test_class_path) if any(f.lower().endswith(ext.lower()) for ext in VALID_EXTENSIONS)])
                test_distribution[class_name] = count
            else:
                test_distribution[class_name] = 0

        # Imprimir tabla de distribución
        print("\n| Clase | Train Original | Test |")
        print("|-------|----------------|------|")

        total_train = 0
        total_test = 0

        for class_name in self.class_names:
            train_count = train_distribution[class_name]
            test_count = test_distribution[class_name]
            total_train += train_count
            total_test += test_count
            print(f"| {class_name:<35} | {train_count:6,d} | {test_count:4,d} |")

        print(f"| **TOTAL** | **{total_train:6,d}** | **{total_test:4,d}** |")

        # Guardar distribución
        distribution_data = {
            "train_distribution": train_distribution,
            "test_distribution": test_distribution,
            "total_train": total_train,
            "total_test": total_test,
            "class_names": self.class_names,
        }

        with open(os.path.join(self.script_dir, "dataset_distribution_original.json"), "w") as f:
            json.dump(distribution_data, f, indent=2)

        return distribution_data

    def create_train_validation_split(self):
        """Crear split de train/validation manteniendo distribución por clase"""
        print(f"\n🔄 Creando split train/validation ({int((1 - VALIDATION_SPLIT) * 100)}%/{int(VALIDATION_SPLIT * 100)}%)...")

        source_train = os.path.join(self.source_dir, "train")
        target_train = os.path.join(self.target_dir, "train")
        target_valid = os.path.join(self.target_dir, "valid")

        # Crear directorios objetivo
        os.makedirs(target_train, exist_ok=True)
        os.makedirs(target_valid, exist_ok=True)

        train_counts = {}
        valid_counts = {}

        for class_name in self.class_names:
            print(f"   Procesando clase: {class_name}")

            # Directorios de origen y destino
            source_class_dir = os.path.join(source_train, class_name)
            target_train_class = os.path.join(target_train, class_name)
            target_valid_class = os.path.join(target_valid, class_name)

            os.makedirs(target_train_class, exist_ok=True)
            os.makedirs(target_valid_class, exist_ok=True)

            # Obtener lista de imágenes
            images = [f for f in os.listdir(source_class_dir) if any(f.lower().endswith(ext.lower()) for ext in VALID_EXTENSIONS)]

            if len(images) == 0:
                print(f"     ⚠️  No se encontraron imágenes en {class_name}")
                continue

            # Split estratificado
            train_images, valid_images = train_test_split(images, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, shuffle=True)

            # Copiar imágenes a directorios correspondientes
            for img in train_images:
                src_path = os.path.join(source_class_dir, img)
                dst_path = os.path.join(target_train_class, img)
                shutil.copy2(src_path, dst_path)

            for img in valid_images:
                src_path = os.path.join(source_class_dir, img)
                dst_path = os.path.join(target_valid_class, img)
                shutil.copy2(src_path, dst_path)

            train_counts[class_name] = len(train_images)
            valid_counts[class_name] = len(valid_images)

            print(f"     ✅ {len(train_images)} train, {len(valid_images)} valid")

        # Imprimir resumen del split
        print("\n📈 Resumen del split train/validation:")
        print("| Clase | Train | Valid | Total Original |")
        print("|-------|-------|-------|----------------|")

        total_train_new = 0
        total_valid_new = 0

        for class_name in self.class_names:
            train_count = train_counts.get(class_name, 0)
            valid_count = valid_counts.get(class_name, 0)
            total_original = train_count + valid_count

            total_train_new += train_count
            total_valid_new += valid_count

            print(f"| {class_name:<20} | {train_count:5,d} | {valid_count:5,d} | {total_original:6,d} |")

        print(f"| **TOTAL** | **{total_train_new:5,d}** | **{total_valid_new:5,d}** | **{total_train_new + total_valid_new:6,d}** |")

        # Guardar información del split
        split_info = {
            "validation_split": VALIDATION_SPLIT,
            "random_state": RANDOM_STATE,
            "train_counts": train_counts,
            "valid_counts": valid_counts,
            "total_train": total_train_new,
            "total_valid": total_valid_new,
        }

        with open(os.path.join(self.target_dir, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)

        return split_info

    def copy_test_set(self):
        """Copiar el conjunto de test al directorio final"""
        print("\n📋 Copiando conjunto de test...")

        source_test = os.path.join(self.source_dir, "test")
        target_test = os.path.join(self.target_dir, "test")

        if os.path.exists(target_test):
            shutil.rmtree(target_test)

        shutil.copytree(source_test, target_test)

        # Contar imágenes copiadas
        test_counts = {}
        total_test = 0

        for class_name in self.class_names:
            test_class_path = os.path.join(target_test, class_name)
            if os.path.exists(test_class_path):
                count = len([f for f in os.listdir(test_class_path) if any(f.lower().endswith(ext.lower()) for ext in VALID_EXTENSIONS)])
                test_counts[class_name] = count
                total_test += count
            else:
                test_counts[class_name] = 0

        print(f"✅ Test set copiado: {total_test:,} imágenes")

        return test_counts

    def create_final_dataset(self):
        """Crear dataset final con estructura train/valid/test"""
        print("\n🏗️  Creando estructura final del dataset...")

        # Analizar distribución original
        self.analyze_dataset_distribution()

        # Crear split train/validation
        split_info = self.create_train_validation_split()

        # Copiar test set
        test_counts = self.copy_test_set()

        # Crear resumen final
        self._create_final_summary(split_info, test_counts)

        print("\n🎉 ¡Dataset final creado exitosamente!")
        print(f"   📁 Dataset: {self.target_dir}")

    def _create_final_summary(self, split_info, test_counts):
        """Crear resumen final del dataset procesado"""
        print("\n📋 Resumen final del dataset:")
        print("| Clase | Train | Valid | Test |")
        print("|-------|-------|-------|------|")

        total_train = 0
        total_valid = 0
        total_test = 0

        for class_name in self.class_names:
            train_count = split_info["train_counts"].get(class_name, 0)
            valid_count = split_info["valid_counts"].get(class_name, 0)
            test_count = test_counts.get(class_name, 0)

            total_train += train_count
            total_valid += valid_count
            total_test += test_count

            print(f"| {class_name:<25} | {train_count:5,d} | {valid_count:5,d} | {test_count:4,d} |")

        print(f"| **TOTAL** | **{total_train:5,d}** | **{total_valid:5,d}** | **{total_test:4,d}** |")

        # Guardar resumen final
        final_summary = {
            "dataset_structure": {"train": split_info["train_counts"], "validation": split_info["valid_counts"], "test": test_counts},
            "totals": {"train": total_train, "validation": total_valid, "test": total_test, "grand_total": total_train + total_valid + total_test},
            "configuration": {"validation_split": VALIDATION_SPLIT, "random_state": RANDOM_STATE, "target_size": TARGET_SIZE},
        }

        summary_path = os.path.join(self.target_dir, "dataset_final_summary.json")
        with open(summary_path, "w") as f:
            json.dump(final_summary, f, indent=2)

        print(f"\n📊 Resumen guardado en: {summary_path}")


def main():
    """Función principal"""
    print("🍅 Gestor de Dataset - Enfermedades de Tomate")
    print("=" * 50)

    try:
        # Crear instancia del gestor
        manager = DatasetManager()

        # Crear dataset final
        manager.create_final_dataset()

    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
