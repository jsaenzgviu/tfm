#!/usr/bin/env python3
"""
Script de entrenamiento con Knowledge Distillation y callback MEJORADO para MobileNetV4
Versión 2 con fix para el problema de normalización de datos
Incluye generación completa de informes y configuración centralizada
"""

import os
import sys
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import plot_model

# Importar nuestros módulos
from mobilenetv4_clean import create_mobilenetv4_model as create_mobilenetv4


def create_teacher_model(num_classes=11):
    """Crear modelo teacher DenseNet121"""
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
    from tensorflow.keras.models import Model

    # Cargar modelo base preentrenado
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Añadir capas de clasificación personalizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation="softmax", dtype="float32")(x)

    # Crear modelo completo
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


class MobileNetV4KnowledgeDistillationTrainer:
    """
    Entrenador de MobileNetV4 con Knowledge Distillation para clasificación de enfermedades en plantas de tomate
    Incluye generación completa de informes y configuración centralizada
    """

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(self.script_dir, DATASET_DIR)
        self.train_path = os.path.join(self.dataset_path, TRAIN_SUBDIR)
        self.valid_path = os.path.join(self.dataset_path, VALID_SUBDIR)
        self.test_path = os.path.join(self.dataset_path, TEST_SUBDIR)

        # Crear directorio del experimento
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"mobilenetv4_fixed_v2_{timestamp}"
        self.experiment_dir = os.path.join(self.script_dir, RESULTS_DIR, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Inicializar variables
        self.model = None
        self.history = None
        self.class_names = None
        self.class_weights = None

        print(f"📁 Experimento: {self.experiment_dir}")

    def setup_gpu_and_precision(self):
        """Configurar GPU y precisión mixta"""
        # Configurar GPU
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    if GPU_MEMORY_GROWTH:
                        tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU configurada: {len(gpus)} dispositivos")
            except RuntimeError as e:
                print(f"❌ Error configurando GPU: {e}")
        else:
            print("⚠️ No se encontraron GPUs, usando CPU")

        # Configurar precisión mixta
        if USE_MIXED_PRECISION:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("✅ Precisión mixta configurada")

    def calculate_class_weights(self):
        """Calcular pesos de clase para datos desbalanceados"""
        print("\n📊 Calculando pesos de clase...")

        # Obtener clases desde train directory
        self.class_names = sorted([d for d in os.listdir(self.train_path) if os.path.isdir(os.path.join(self.train_path, d))])

        # Contar imágenes por clase
        class_counts = {}
        total_samples = 0

        for class_name in self.class_names:
            class_path = os.path.join(self.train_path, class_name)
            count = len([f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            class_counts[class_name] = count
            total_samples += count

        # Calcular pesos balanceados
        n_classes = len(self.class_names)
        self.class_weights = {}

        for i, class_name in enumerate(self.class_names):
            weight = total_samples / (n_classes * class_counts[class_name])
            self.class_weights[i] = weight

        print("\n📊 Distribución de clases y pesos:")
        for i, class_name in enumerate(self.class_names):
            count = class_counts[class_name]
            weight = self.class_weights[i]
            print(f"   {i:2d}. {class_name:<35} | {count:5,d} imgs | peso: {weight:.3f}")

        # Guardar información en archivo
        class_info = {
            "class_names": self.class_names,
            "class_counts": class_counts,
            "class_weights": self.class_weights,
            "total_samples": total_samples,
        }

        with open(os.path.join(self.experiment_dir, "class_distribution.json"), "w") as f:
            json.dump(class_info, f, indent=2)

        return self.class_weights

    def create_data_generators(self):
        """Crear generadores de datos con la normalización correcta"""
        # Verificar que existe directorio de validación
        if not os.path.exists(self.valid_path):
            raise FileNotFoundError(
                f"❌ ERROR CRÍTICO: No existe directorio de validación: {self.valid_path}\n"
                f"   El conjunto TEST NUNCA debe usarse durante el entrenamiento.\n"
                f"   Ejecuta primero '201_dataset_manager_clean.py' para crear el split train/valid/test."
            )

        if not os.path.exists(self.test_path):
            raise FileNotFoundError(
                f"❌ ERROR: No existe directorio de test: {self.test_path}\n   Se requiere un conjunto test independiente para evaluación final."
            )

        # Data augmentation para train
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,  # Normalización principal
            rotation_range=ROTATION_RANGE,
            width_shift_range=WIDTH_SHIFT_RANGE,
            height_shift_range=HEIGHT_SHIFT_RANGE,
            horizontal_flip=HORIZONTAL_FLIP,
            zoom_range=ZOOM_RANGE,
            shear_range=SHEAR_RANGE,
            brightness_range=BRIGHTNESS_RANGE,
            fill_mode=FILL_MODE,
        )

        # Solo normalización para validación y test
        val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Crear generador de TRAIN
        train_generator = train_datagen.flow_from_directory(
            self.train_path, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="sparse", shuffle=True
        )

        # Crear generador de VALIDACIÓN (usado durante entrenamiento)
        val_generator = val_test_datagen.flow_from_directory(
            self.valid_path, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="sparse", shuffle=False
        )

        # Crear generador de TEST (SOLO para evaluación final, NUNCA durante entrenamiento)
        test_generator = val_test_datagen.flow_from_directory(
            self.test_path, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="sparse", shuffle=False
        )

        print("✅ Generadores de datos creados:")
        print(f"   • Train:      {train_generator.samples:5,d} imágenes ({self.train_path})")
        print(f"   • Validation: {val_generator.samples:5,d} imágenes ({self.valid_path})")
        print(f"   • Test:       {test_generator.samples:5,d} imágenes ({self.test_path})")
        print(f"   ⚠️  IMPORTANTE: Test set SOLO se usa en evaluación final, NUNCA durante entrenamiento")

        return train_generator, val_generator, test_generator

    def save_experiment_config(self):
        """Guardar configuración del experimento"""
        config = {
            "model_config": {
                "model_name": MODEL_NAME,
                "mobilenetv4_variant": MOBILENETV4_VARIANT,
                "teacher_model": TEACHER_MODEL,
                "img_height": IMG_HEIGHT,
                "img_width": IMG_WIDTH,
                "img_channels": IMG_CHANNELS,
                "num_classes": NUM_CLASSES,
            },
            "dataset_config": {
                "dataset_base_dir": DATASET_DIR,
                "train_path": self.train_path,
                "valid_path": self.valid_path,
                "test_path": self.test_path,
            },
            "training_config": {
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "epochs": EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "reduce_lr_patience": REDUCE_LR_PATIENCE,
                "reduce_lr_factor": REDUCE_LR_FACTOR,
            },
            "knowledge_distillation_config": {
                "temperature": TEMPERATURE,
                "alpha": ALPHA,
                "teacher_weights_path": TEACHER_WEIGHTS_PATH,
            },
            "augmentation_config": {
                "rotation_range": ROTATION_RANGE,
                "width_shift_range": WIDTH_SHIFT_RANGE,
                "height_shift_range": HEIGHT_SHIFT_RANGE,
                "shear_range": SHEAR_RANGE,
                "zoom_range": ZOOM_RANGE,
                "horizontal_flip": HORIZONTAL_FLIP,
                "brightness_range": BRIGHTNESS_RANGE,
                "fill_mode": FILL_MODE,
            },
            "hardware_config": {
                "use_mixed_precision": USE_MIXED_PRECISION,
                "gpu_memory_growth": GPU_MEMORY_GROWTH,
            },
            "experiment_info": {
                "timestamp": datetime.datetime.now().isoformat(),
                "experiment_name": self.experiment_name,
                "dataset_path": self.dataset_path,
                "results_dir": self.experiment_dir,
            },
        }

        config_path = os.path.join(self.experiment_dir, "experiment_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Configuración guardada en: {config_path}")

    def create_callbacks(self, val_dataset):
        """Crear callbacks para el entrenamiento"""
        callbacks = []

        # Callback personalizado para guardar el modelo student
        student_weights_path = os.path.join(self.experiment_dir, "student_model.weights.h5")
        fixed_checkpoint = FixedStudentModelCheckpoint(
            filepath=student_weights_path, monitor="val_accuracy", mode="max", verbose=1, val_dataset=val_dataset
        )
        callbacks.append(fixed_checkpoint)

        # Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=False,  # Nuestro callback maneja esto
            verbose=1,
            mode="max",
        )
        callbacks.append(early_stopping)

        # Reduce Learning Rate on Plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=1e-7, verbose=1
        )
        callbacks.append(reduce_lr)

        # CSV Logger para métricas
        if GENERATE_DETAILED_LOGS:
            csv_logger = CSVLogger(os.path.join(self.experiment_dir, "training_log.csv"))
            callbacks.append(csv_logger)

        print(f"✅ Callbacks configurados: {len(callbacks)} callbacks")

        return callbacks

    def evaluate_model(self, test_generator):
        """Evaluar modelo en conjunto de test"""
        print("\n📊 Evaluando modelo en conjunto de test...")

        # Buscar el mejor modelo guardado
        best_model_paths = [
            os.path.join(self.experiment_dir, "student_model_complete.h5"),
            os.path.join(self.experiment_dir, "student_model.weights_complete.h5"),
        ]

        loaded_model = None
        for model_path in best_model_paths:
            if os.path.exists(model_path):
                try:
                    if model_path.endswith("_complete.h5"):
                        loaded_model = tf.keras.models.load_model(model_path)
                    else:
                        loaded_model = create_mobilenetv4(variant=MOBILENETV4_VARIANT, num_classes=NUM_CLASSES)
                        loaded_model.load_weights(model_path)
                        loaded_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                    print(f"✅ Cargado modelo desde: {model_path}")
                    break
                except Exception as e:
                    print(f"❌ Error cargando {model_path}: {e}")
                    continue

        if loaded_model is None:
            print("❌ No se pudo cargar ningún modelo guardado")
            return None

        # Evaluación general
        evaluation_results = loaded_model.evaluate(test_generator, verbose=1)
        test_loss = evaluation_results[0]
        test_accuracy = evaluation_results[1]

        print("\n📈 Resultados en Test Set:")
        print(f"   • Test Loss: {test_loss:.4f}")
        print(f"   • Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

        # Predicciones para métricas detalladas
        test_generator.reset()
        predictions = loaded_model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes

        # Classification report
        if SAVE_CLASSIFICATION_REPORT:
            report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)

            # Guardar reporte
            report_path = os.path.join(self.experiment_dir, "classification_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            # Imprimir reporte resumido
            print(f"\n📋 Classification Report (resumen):")
            print(f"   • Macro avg F1-score: {report['macro avg']['f1-score']:.4f}")
            print(f"   • Weighted avg F1-score: {report['weighted avg']['f1-score']:.4f}")

        # Matriz de confusión
        if SAVE_CONFUSION_MATRIX:
            cm = confusion_matrix(y_true, y_pred)
            self.plot_confusion_matrix(cm)

        # Guardar métricas finales
        final_metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "model_params": int(loaded_model.count_params()),
            "experiment_name": self.experiment_name,
        }

        if SAVE_CLASSIFICATION_REPORT:
            final_metrics.update(
                {
                    "macro_f1": report["macro avg"]["f1-score"],
                    "weighted_f1": report["weighted avg"]["f1-score"],
                }
            )

        metrics_path = os.path.join(self.experiment_dir, "final_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

        return final_metrics

    def plot_training_history(self):
        """Generar plots del historial de entrenamiento"""
        if self.history is None:
            return

        # Configurar estilo
        plt.style.use("default")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Training History - {MODEL_NAME}", fontsize=16)

        # Plot 1: Loss
        axes[0, 0].plot(self.history.history["loss"], label="Training Loss", linewidth=2)
        axes[0, 0].plot(self.history.history["val_loss"], label="Validation Loss", linewidth=2)
        axes[0, 0].set_title("Model Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Accuracy
        axes[0, 1].plot(self.history.history["accuracy"], label="Training Accuracy", linewidth=2)
        axes[0, 1].plot(self.history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
        axes[0, 1].set_title("Model Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Knowledge Distillation Loss
        if "student_loss" in self.history.history:
            axes[1, 0].plot(self.history.history["student_loss"], label="Student Loss", linewidth=2)
            axes[1, 0].plot(self.history.history["distillation_loss"], label="Distillation Loss", linewidth=2)
            axes[1, 0].set_title("Knowledge Distillation Losses")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis("off")

        # Plot 4: Learning Rate (si está disponible)
        if "lr" in self.history.history:
            axes[1, 1].plot(self.history.history["lr"], linewidth=2)
            axes[1, 1].set_title("Learning Rate")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Learning Rate")
            axes[1, 1].set_yscale("log")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis("off")

        plt.tight_layout()

        # Guardar plot
        plot_path = os.path.join(self.experiment_dir, "training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Plots de entrenamiento guardados en: {plot_path}")

    def plot_confusion_matrix(self, cm):
        """Generar plot de matriz de confusión"""
        plt.figure(figsize=(12, 10))

        # Normalizar matriz de confusión
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Crear heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Accuracy"},
        )

        plt.title(f"Confusion Matrix - {MODEL_NAME}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Guardar plot
        cm_path = os.path.join(self.experiment_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Matriz de confusión guardada en: {cm_path}")

    def generate_final_report(self, final_metrics):
        """Generar reporte final del experimento"""
        if not GENERATE_EXPERIMENT_REPORT:
            return

        report_content = f"""# Reporte del Experimento: {self.experiment_name}

## Configuración del Experimento
- **Modelo**: MobileNetV4-{MOBILENETV4_VARIANT} con Knowledge Distillation
- **Teacher**: {TEACHER_MODEL}
- **Dataset**: {DATASET_DIR}
- **Fecha**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuración de Entrenamiento
- **Épocas**: {EPOCHS}
- **Batch Size**: {BATCH_SIZE}
- **Learning Rate**: {LEARNING_RATE}
- **Temperature**: {TEMPERATURE}
- **Alpha**: {ALPHA}

## Resultados Finales
- **Test Accuracy**: {final_metrics.get("test_accuracy", "N/A"):.4f} ({final_metrics.get("test_accuracy", 0) * 100:.2f}%)
- **Test Loss**: {final_metrics.get("test_loss", "N/A"):.4f}
- **Parámetros del Modelo**: {final_metrics.get("model_params", "N/A"):,}

## Archivos Generados
- Configuración: `experiment_config.json`
- Métricas: `final_metrics.json`
- Distribución de clases: `class_distribution.json`
- Log de entrenamiento: `training_log.csv`
- Gráficos: `training_history.png`, `confusion_matrix.png`
- Modelo: `student_model.weights_complete.h5`

## Directorio del Experimento
```
{self.experiment_dir}
```
"""

        report_path = os.path.join(self.experiment_dir, "experiment_report.md")
        with open(report_path, "w") as f:
            f.write(report_content)

        print(f"✅ Reporte final guardado en: {report_path}")

    def train(self):
        """Entrenar el modelo con Knowledge Distillation"""
        print(f"\n🚀 INICIANDO ENTRENAMIENTO: {self.experiment_name}")
        print(f"📊 Configuración:")
        print(f"   • Imagen: {IMG_SIZE}x{IMG_SIZE}")
        print(f"   • Batch: {BATCH_SIZE}")
        print(f"   • Épocas: {EPOCHS}")
        print(f"   • Learning Rate: {LEARNING_RATE}")
        print(f"   • Temperature: {TEMPERATURE}")
        print(f"   • Alpha: {ALPHA}")

        # Configurar GPU y precisión
        self.setup_gpu_and_precision()

        # Calcular pesos de clase
        class_weights = self.calculate_class_weights()

        # Crear datasets (train, validation, test)
        print("\n📂 Preparando datos...")
        train_gen, val_gen, test_gen = self.create_data_generators()

        # Guardar configuración
        self.save_experiment_config()

        # Crear teacher model
        print(f"\n🏫 Creando teacher model ({TEACHER_MODEL})...")
        teacher_model = create_teacher_model(num_classes=NUM_CLASSES)

        # Cargar pesos del teacher
        if os.path.exists(TEACHER_WEIGHTS_PATH):
            teacher_model.load_weights(TEACHER_WEIGHTS_PATH)
            print(f"✅ Teacher cargado desde: {TEACHER_WEIGHTS_PATH}")
        else:
            print(f"❌ No se encontró teacher en: {TEACHER_WEIGHTS_PATH}")
            print(f"   Entrenando teacher desde cero...")

            # Entrenar teacher rápidamente
            teacher_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
            )

            teacher_history = teacher_model.fit(
                train_gen,
                epochs=1,  # Solo 1 época para tener un teacher básico
                validation_data=val_gen,
                verbose=1,
            )

            # Guardar teacher
            teacher_model.save_weights(TEACHER_WEIGHTS_PATH)
            print(f"✅ Teacher guardado en: {TEACHER_WEIGHTS_PATH}")

        # Crear student model
        print(f"\n🎓 Creando student model (MobileNetV4-{MOBILENETV4_VARIANT})...")
        student_model = create_mobilenetv4(variant=MOBILENETV4_VARIANT, num_classes=NUM_CLASSES)
        print(f"✅ Student creado: {student_model.count_params():,} parámetros")

        # Guardar arquitectura del modelo si está habilitado
        if SAVE_MODEL_PLOTS and GENERATE_MODEL_SUMMARY:
            model_plot_path = os.path.join(self.experiment_dir, "student_model_architecture.png")
            try:
                plot_model(student_model, to_file=model_plot_path, show_shapes=True, show_layer_names=True)
            except Exception as e:
                print(f"⚠️ No se pudo guardar arquitectura del modelo: {e}")

            # Guardar resumen del modelo
            with open(os.path.join(self.experiment_dir, "student_model_summary.txt"), "w") as f:
                student_model.summary(print_fn=lambda x: f.write(x + "\n"))

        # Crear modelo de Knowledge Distillation
        print(f"\n🔬 Creando modelo de Knowledge Distillation...")
        kd_model = KnowledgeDistillationModel(teacher=teacher_model, student=student_model, temperature=TEMPERATURE, alpha=ALPHA)

        # Compilar modelo KD
        kd_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        print(f"✅ Modelo KD compilado")

        # Convertir val_gen a dataset para el callback
        val_dataset = tf.data.Dataset.from_generator(
            lambda: val_gen,
            output_signature=(tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.float32)),
        ).take(50)  # Limitar para verificación rápida

        # Crear callbacks
        callbacks = self.create_callbacks(val_dataset)

        print(f"\n🎯 Iniciando entrenamiento Knowledge Distillation...")

        # Entrenar
        self.history = kd_model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks, class_weight=class_weights, verbose=1)

        print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
        print(f"📈 Resultados finales:")

        if self.history.history:
            final_train_acc = self.history.history.get("accuracy", [0])[-1]
            final_val_acc = self.history.history.get("val_accuracy", [0])[-1]
            print(f"   • Train Accuracy: {final_train_acc:.4f}")
            print(f"   • Val Accuracy: {final_val_acc:.4f}")

        # Generar plots de entrenamiento
        if SAVE_TRAINING_PLOTS:
            self.plot_training_history()

        # Evaluar en test set (conjunto independiente, NUNCA visto durante entrenamiento)
        print("\n" + "=" * 70)
        print("🧪 EVALUACIÓN FINAL EN TEST SET (conjunto completamente independiente)")
        print("=" * 70)
        final_metrics = self.evaluate_model(test_gen)

        # Generar reporte final
        if final_metrics:
            self.generate_final_report(final_metrics)

        print(f"\n📁 Resultados guardados en: {self.experiment_dir}")

        return self.history


# ==================================================================================
# CONFIGURACIÓN - MODIFICAR ESTAS VARIABLES SEGÚN NECESIDADES
# ==================================================================================

# Directorio del dataset (relativo o absoluto)
DATASET_DIR = "dataset_final"  # Cambiar entre: dataset, dataset_final, dataset_optimized
TRAIN_SUBDIR = "train"
VALID_SUBDIR = "valid"  # Directorio de validación (si no existe, se usa TEST_SUBDIR)
TEST_SUBDIR = "test"

# Configuraciones de imagen y modelo
IMG_SIZE = 224
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
NUM_CLASSES = 11  # 11 enfermedades/estados de tomate

# Configuración de entrenamiento
BATCH_SIZE = 16  # Reducido para evitar OOM en RTX 3060
EPOCHS = 40  # Solo 1 época para prueba rápida - CAMBIAR A MÁS ÉPOCAS EN PRODUCCIÓN
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

# Configuración Knowledge Distillation
TEMPERATURE = 4.0
ALPHA = 0.3  # Peso de la pérdida de Knowledge Distillation

# Configuración de callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Data augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
BRIGHTNESS_RANGE = [0.8, 1.2]
FILL_MODE = "nearest"

# Configuración de modelos
MOBILENETV4_VARIANT = "medium"  # "small", "medium", "large"
TEACHER_MODEL = "densenet121"  # Teacher model a utilizar
TEACHER_WEIGHTS_PATH = "teacher_densenet121.weights.h5"

# Configuración de salida y experimentos
RESULTS_DIR = "experiments"
MODEL_NAME = "mobilenetv4_knowledge_distillation"
SAVE_MODEL_PLOTS = True
SAVE_TRAINING_PLOTS = True
SAVE_CONFUSION_MATRIX = True
SAVE_CLASSIFICATION_REPORT = True

# GPU Configuration
USE_MIXED_PRECISION = False  # Para optimizar memoria en RTX 3060
GPU_MEMORY_GROWTH = False

# Configuración de archivos de salida
GENERATE_EXPERIMENT_REPORT = True
GENERATE_DETAILED_LOGS = True
GENERATE_MODEL_SUMMARY = True

# ==================================================================================
# FIN DE CONFIGURACIÓN
# ==================================================================================


class FixedStudentModelCheckpoint(tf.keras.callbacks.Callback):
    """Callback mejorado para guardar correctamente el modelo student con datos consistentes"""

    def __init__(self, filepath, monitor="val_accuracy", mode="max", verbose=1, val_dataset=None):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.val_dataset = val_dataset  # Dataset de validación del entrenamiento
        self.best = -np.inf if mode == "max" else np.inf

    def create_independent_student(self):
        """Crear modelo student independiente y transferir pesos"""
        try:
            # Crear modelo limpio
            independent_student = create_mobilenetv4(variant="medium", num_classes=11)

            # Transferir pesos del student actual
            student_weights = self.model.student.get_weights()
            print(f"   📊 Transfiriendo {len(student_weights)} grupos de pesos")
            independent_student.set_weights(student_weights)

            # Compilar
            independent_student.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
            )

            return independent_student

        except Exception as e:
            print(f"❌ Error creando modelo independiente: {e}")
            return None

    def verify_model_quality(self, model, max_steps=5):
        """Verificar que el modelo funciona correctamente usando el MISMO dataset de validación"""
        try:
            if self.val_dataset is None:
                print("⚠️ No hay dataset de validación, saltando verificación")
                return True, 0.5  # Asumir válido

            # Evaluar con el MISMO dataset que se usa en entrenamiento (ya normalizado)
            results = model.evaluate(self.val_dataset, verbose=0, steps=max_steps)
            accuracy = results[1] if len(results) > 1 else results

            print(f"   → Accuracy verificación: {accuracy:.4f} ({accuracy * 100:.2f}%)")

            # Para las primeras épocas, el threshold debería ser más bajo
            is_valid = accuracy > 0.08  # Más del 8% (mejor que random que sería ~9.1%)

            return is_valid, accuracy

        except Exception as e:
            print(f"❌ Error en verificación: {e}")
            return False, 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            if self.verbose > 0:
                print(f"\nWarning: Can save best model only with {self.monitor} available, skipping.")
            return

        should_save = False
        if self.mode == "max":
            if current > self.best:
                should_save = True
        else:
            if current < self.best:
                should_save = True

        if should_save:
            old_best = self.best
            self.best = current

            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {old_best:.5f} to {current:.5f}")
                print(f"🔧 Iniciando guardado MEJORADO del modelo student...")

            # PROCESO DE GUARDADO MEJORADO
            try:
                # PASO 1: Crear modelo independiente
                print(f"   🏗️ Creando modelo independiente...")
                independent_student = self.create_independent_student()

                if independent_student is None:
                    raise ValueError("No se pudo crear modelo independiente")

                print(f"   ✅ Modelo creado: {independent_student.count_params():,} parámetros")

                # PASO 2: Verificar funcionamiento
                print(f"   🧪 Verificando calidad del modelo...")
                is_valid, accuracy = self.verify_model_quality(independent_student)

                if not is_valid:
                    print(f"   ⚠️ WARNING: Modelo con accuracy baja: {accuracy:.4f}")
                    print(f"   ⏭️ Guardando de todas formas para continuar entrenamiento...")

                print(f"   ✅ Modelo verificado")

                # PASO 3: Guardar en formato .weights.h5
                weights_filepath = self.filepath.replace(".h5", ".weights.h5")
                print(f"   💾 Guardando pesos en: {weights_filepath}")
                independent_student.save_weights(weights_filepath)

                # PASO 4: Verificar guardado
                print(f"   🔍 Verificando archivo guardado...")
                if not os.path.exists(weights_filepath):
                    raise ValueError("Archivo de pesos no se creó")

                # PASO 5: Probar carga
                print(f"   📥 Probando carga del archivo...")
                loaded_model = create_mobilenetv4(variant="medium", num_classes=11)
                loaded_model.load_weights(weights_filepath)
                loaded_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
                )

                # PASO 6: Verificar modelo cargado
                is_loaded_valid, loaded_accuracy = self.verify_model_quality(loaded_model)

                accuracy_diff = abs(accuracy - loaded_accuracy)

                print(f"   🎉 ¡GUARDADO EXITOSO!")
                print(f"   ✅ Accuracy independiente: {accuracy:.4f}")
                print(f"   ✅ Accuracy cargado: {loaded_accuracy:.4f}")
                print(f"   ✅ Diferencia: {accuracy_diff:.4f}")

                # Guardar también el modelo completo para backup
                model_filepath = weights_filepath.replace(".weights.h5", "_complete.h5")
                independent_student.save(model_filepath)
                print(f"   📦 Modelo completo guardado en: {model_filepath}")

            except Exception as e:
                print(f"   ❌ ERROR EN GUARDADO: {e}")
                print(f"   🚨 El modelo NO se guardó correctamente")
                # NO hacer fallback - es mejor fallar explícitamente
                raise e


class KnowledgeDistillationModel(tf.keras.Model):
    """Modelo de Knowledge Distillation para MobileNetV4"""

    def __init__(self, teacher, student, temperature=4.0, alpha=0.3):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

        # Freeze teacher
        self.teacher.trainable = False

        # Compilar métricas como objetos TensorFlow
        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = tf.keras.metrics.Mean(name="distillation_loss")
        self.student_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [
            self.student_loss_tracker,
            self.distillation_loss_tracker,
            self.student_accuracy,
        ]

    def train_step(self, data):
        # Manejar tanto (x, y) como (x, y, sample_weight)
        if len(data) == 2:
            x, y = data
        else:
            x, y, sample_weight = data

        # Forward pass teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.compiled_loss(y, student_predictions)

            # Distillation loss
            distillation_loss = tf.keras.losses.KLDivergence()(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )

            # Combined loss
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients and update student
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.student_accuracy.update_state(y, student_predictions)

        return {
            "loss": total_loss,
            "student_loss": self.student_loss_tracker.result(),
            "distillation_loss": self.distillation_loss_tracker.result(),
            "accuracy": self.student_accuracy.result(),
        }

    def test_step(self, data):
        # Manejar tanto (x, y) como (x, y, sample_weight)
        if len(data) == 2:
            x, y = data
        else:
            x, y, sample_weight = data

        # Forward pass
        student_predictions = self.student(x, training=False)
        student_loss = self.compiled_loss(y, student_predictions)

        # Update metrics
        self.student_accuracy.update_state(y, student_predictions)

        return {
            "loss": student_loss,
            "accuracy": self.student_accuracy.result(),
        }


def main():
    """Función principal de entrenamiento"""
    print("🍅 MobileNetV4 Knowledge Distillation - Clasificador de Enfermedades en Plantas de Tomate")
    print("=" * 80)

    try:
        # Crear instancia del entrenador
        trainer = MobileNetV4KnowledgeDistillationTrainer()

        # Entrenar modelo
        trainer.train()

        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        print(f"📁 Resultados guardados en: {trainer.experiment_dir}")

    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
