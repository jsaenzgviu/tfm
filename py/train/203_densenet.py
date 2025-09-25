#!/usr/bin/env python3
"""
Entrenamiento de modelo DenseNet121 para clasificación de enfermedades en plantas de tomate
Implementación completa siguiendo mejores prácticas de MLOps y configuración centralizada
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

# ==================================================================================
# CONFIGURACIÓN - MODIFICAR ESTAS VARIABLES SEGÚN NECESIDADES
# ==================================================================================

# Directorio del dataset (relativo o absoluto)
DATASET_DIR = "dataset_final"  # Cambiar entre: dataset_final, dataset_optimized, dataset_enhanced, dataset_smart

# Estructura de directorios del dataset
TRAIN_DIR = "train"
VALID_DIR = "valid"  # Opcional: si no existe, se creará split automático
TEST_DIR = "test"

# Configuraciones de entrenamiento
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
NUM_CLASSES = 11  # 11 enfermedades/estados de tomate

# Hiperparámetros de entrenamiento
BATCH_SIZE = 32  # Reducir a 16 si hay problemas de memoria en RTX 3060
INITIAL_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100

# División de datos (se creará automáticamente validation split)
VALIDATION_SPLIT = 0.2  # 20% del train se usará para validación

# Configuración de callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Data augmentation parameters
ROTATION_RANGE = 45
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.3
HORIZONTAL_FLIP = True
BRIGHTNESS_RANGE = [0.8, 1.2]

# Configuración de fine-tuning
FREEZE_BASE_LAYERS = 50  # Número de capas a congelar inicialmente
UNFREEZE_AFTER_EPOCH = 20  # Época después de la cual descongelar más capas

# Configuración de salida
RESULTS_DIR = "experiments"
MODEL_NAME = "densenet121_tomato_disease"
SAVE_MODEL_PLOTS = True
SAVE_TRAINING_PLOTS = True

# GPU Configuration
USE_MIXED_PRECISION = True  # Para optimizar memoria en RTX 3060
GPU_MEMORY_GROWTH = True

# ==================================================================================
# FIN DE CONFIGURACIÓN
# ==================================================================================


class TomatoDiseaseClassifier:
    """
    Clasificador de enfermedades en plantas de tomate usando DenseNet121
    """

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(self.script_dir, DATASET_DIR)
        self.train_path = os.path.join(self.dataset_path, TRAIN_DIR)
        self.test_path = os.path.join(self.dataset_path, TEST_DIR)
        self.valid_path = os.path.join(self.dataset_path, VALID_DIR)

        # AUTO-DETECTAR tipo de dataset
        self.has_manual_validation = os.path.exists(self.valid_path)
        self.dataset_type = "manual_split" if self.has_manual_validation else "auto_split"

        print(f"🔍 Tipo de dataset detectado: {self.dataset_type}")
        if self.has_manual_validation:
            print(f"   ✅ Directorio valid/ encontrado: Split manual")
        else:
            print(f"   ⚙️  Sin directorio valid/: Split automático con Keras")

        # Crear directorio de resultados
        self.results_dir = os.path.join(self.script_dir, RESULTS_DIR)
        self.experiment_dir = self._create_experiment_dir()

        self.model = None
        self.history = None
        self.class_names = None
        self.class_weights = None

        self._setup_gpu()
        self._verify_dataset()

    def _setup_gpu(self):
        """Configurar GPU para optimizar memoria"""
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Habilitar crecimiento de memoria
                if GPU_MEMORY_GROWTH:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                # Habilitar mixed precision si está configurado
                if USE_MIXED_PRECISION:
                    policy = tf.keras.mixed_precision.Policy("mixed_float16")
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print("✅ Mixed precision habilitada para optimizar memoria GPU")

                print(f"✅ GPU configurada: {len(gpus)} GPU(s) detectada(s)")

            except RuntimeError as e:
                print(f"❌ Error configurando GPU: {e}")
        else:
            print("⚠️  No se detectó GPU, usando CPU")

    def _verify_dataset(self):
        """Verificar que el dataset existe y tiene la estructura correcta"""
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"❌ No se encontró directorio de entrenamiento: {self.train_path}")

        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"❌ No se encontró directorio de test: {self.test_path}")

        # Verificación específica según el tipo de dataset
        if self.has_manual_validation:
            if not os.path.exists(self.valid_path):
                raise FileNotFoundError(f"❌ No se encontró directorio de validación: {self.valid_path}")
            print("   ✅ Verificación de split manual: train/, valid/, test/ encontrados")
        else:
            print("   ⚙️  Configuración de split automático: train/ y test/ encontrados")

        # Obtener nombres de clases
        self.class_names = sorted([d for d in os.listdir(self.train_path) if os.path.isdir(os.path.join(self.train_path, d))])

        if len(self.class_names) != NUM_CLASSES:
            print(f"⚠️  ADVERTENCIA: Se esperaban {NUM_CLASSES} clases, se encontraron {len(self.class_names)}")

        print("✅ Dataset verificado:")
        print(f"   - Directorio train: {self.train_path}")
        if self.has_manual_validation:
            print(f"   - Directorio valid: {self.valid_path}")
        print(f"   - Directorio test: {self.test_path}")
        print(f"   - Clases encontradas: {len(self.class_names)}")

    def _create_experiment_dir(self):
        """Crear directorio único para este experimento"""
        os.makedirs(self.results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{MODEL_NAME}_{timestamp}"
        experiment_path = os.path.join(self.results_dir, experiment_name)

        os.makedirs(experiment_path, exist_ok=True)
        print(f"✅ Directorio de experimento creado: {experiment_path}")

        return experiment_path

    def calculate_class_weights(self):
        """Calcular pesos de clase para balancear el dataset desbalanceado"""
        class_counts = {}
        total_samples = 0

        for class_name in self.class_names:
            class_path = os.path.join(self.train_path, class_name)
            count = len([f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            class_counts[class_name] = count
            total_samples += count

        # Calcular pesos usando la fórmula: weight_i = total_samples / (n_classes * n_samples_i)
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
        """Crear generadores de datos con auto-detección del tipo de split"""

        if self.has_manual_validation:
            # CASO 1: Dataset con split manual (dataset_final/ o dataset_segmented/)
            print("🔄 Usando split manual existente")

            # Para split manual, NO usar validation_split en ImageDataGenerator
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                rotation_range=ROTATION_RANGE,
                width_shift_range=WIDTH_SHIFT_RANGE,
                height_shift_range=HEIGHT_SHIFT_RANGE,
                shear_range=SHEAR_RANGE,
                zoom_range=ZOOM_RANGE,
                horizontal_flip=HORIZONTAL_FLIP,
                brightness_range=BRIGHTNESS_RANGE,
                fill_mode="nearest",
                # validation_split NO se usa aquí
            )

            # Para test y validation solo rescaling (sin augmentation)
            test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

            # Generador de entrenamiento
            train_generator = train_datagen.flow_from_directory(
                self.train_path,
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                class_mode="categorical",
                # subset NO se usa para split manual
            )

            # Generador de validación desde directorio separado
            validation_generator = test_datagen.flow_from_directory(
                self.valid_path,  # ¡Directorio de validación separado!
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                class_mode="categorical",
                shuffle=False,  # Sin shuffle para validación
            )

        else:
            # CASO 2: Dataset original (dataset/) - Split automático con Keras
            print("🔄 Usando split automático de Keras")

            # Data augmentation para entrenamiento con validation_split
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                rotation_range=ROTATION_RANGE,
                width_shift_range=WIDTH_SHIFT_RANGE,
                height_shift_range=HEIGHT_SHIFT_RANGE,
                shear_range=SHEAR_RANGE,
                zoom_range=ZOOM_RANGE,
                horizontal_flip=HORIZONTAL_FLIP,
                brightness_range=BRIGHTNESS_RANGE,
                fill_mode="nearest",
                validation_split=VALIDATION_SPLIT,  # ✅ Split automático activo
            )

            # Para test solo rescaling (sin augmentation)
            test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

            # Generador de entrenamiento
            train_generator = train_datagen.flow_from_directory(
                self.train_path,
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                class_mode="categorical",
                subset="training",  # ✅ Subset para split automático
            )

            # Generador de validación desde el mismo directorio train
            validation_generator = train_datagen.flow_from_directory(
                self.train_path,
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                class_mode="categorical",
                subset="validation",  # ✅ Subset para split automático
            )

        # Generador de test (común para ambos casos)
        test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,  # Importante para evaluación
        )

        print("✅ Generadores de datos creados:")
        print(f"   - Train: {train_generator.samples} imágenes")
        print(f"   - Validation: {validation_generator.samples} imágenes")
        print(f"   - Test: {test_generator.samples} imágenes")
        print(f"   - Tipo de split: {'Manual' if self.has_manual_validation else 'Automático'}")

        # Verificar que las clases están en el mismo orden
        assert train_generator.class_indices == validation_generator.class_indices == test_generator.class_indices

        return train_generator, validation_generator, test_generator

    def build_model(self):
        """Construir modelo DenseNet121 con fine-tuning"""
        # Cargar modelo base preentrenado
        base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

        # Congelar las primeras capas
        for i, layer in enumerate(base_model.layers):
            if i < FREEZE_BASE_LAYERS:
                layer.trainable = False
            else:
                layer.trainable = True

        print(f"✅ Modelo base DenseNet121 cargado:")
        print(f"   - Capas congeladas: {FREEZE_BASE_LAYERS}")
        print(f"   - Capas entrenables: {len(base_model.layers) - FREEZE_BASE_LAYERS}")

        # Añadir capas de clasificación personalizadas
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation="relu", name="dense_features")(x)
        x = Dropout(0.6)(x)
        x = Dense(256, activation="relu", name="dense_classifier")(x)
        x = Dropout(0.4)(x)

        # Capa de salida
        if USE_MIXED_PRECISION:
            # Para mixed precision, usar float32 en la última capa
            predictions = Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)
        else:
            predictions = Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

        # Crear modelo completo
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Compilar modelo
        optimizer = AdamW(learning_rate=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Usar métricas compatibles con todas las versiones de TensorFlow
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")],
        )

        print("✅ Modelo compilado:")
        print(f"   - Parámetros totales: {self.model.count_params():,}")
        print(f"   - Parámetros entrenables: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")

        # Guardar arquitectura del modelo
        if SAVE_MODEL_PLOTS:
            model_plot_path = os.path.join(self.experiment_dir, "model_architecture.png")
            plot_model(self.model, to_file=model_plot_path, show_shapes=True, show_layer_names=True)

            # Guardar resumen del modelo
            with open(os.path.join(self.experiment_dir, "model_summary.txt"), "w") as f:
                self.model.summary(print_fn=lambda x: f.write(x + "\n"))

        return self.model

    def create_callbacks(self):
        """Crear callbacks para el entrenamiento"""
        callbacks = []

        # Early Stopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
        callbacks.append(early_stopping)

        # Model Checkpoint (guardar mejor modelo)
        checkpoint_path = os.path.join(self.experiment_dir, "best_model.h5")
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, save_weights_only=False, verbose=1)
        callbacks.append(model_checkpoint)

        # Reduce Learning Rate on Plateau
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=1e-7, verbose=1)
        callbacks.append(reduce_lr)

        # CSV Logger para métricas
        csv_logger = CSVLogger(os.path.join(self.experiment_dir, "training_log.csv"))
        callbacks.append(csv_logger)

        print(f"✅ Callbacks configurados: {len(callbacks)} callbacks")

        return callbacks

    def save_experiment_config(self):
        """Guardar configuración del experimento"""
        config = {
            "model_config": {
                "model_name": MODEL_NAME,
                "img_height": IMG_HEIGHT,
                "img_width": IMG_WIDTH,
                "num_classes": NUM_CLASSES,
                "freeze_base_layers": FREEZE_BASE_LAYERS,
            },
            "dataset_config": {
                "dataset_base_dir": DATASET_DIR,
                "dataset_type": self.dataset_type,
                "has_manual_validation": self.has_manual_validation,
                "validation_split": VALIDATION_SPLIT if not self.has_manual_validation else "N/A (manual split)",
                "train_path": self.train_path,
                "valid_path": self.valid_path if self.has_manual_validation else "N/A (auto split)",
                "test_path": self.test_path,
            },
            "training_config": {
                "batch_size": BATCH_SIZE,
                "initial_learning_rate": INITIAL_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "epochs": EPOCHS,
                "validation_split": VALIDATION_SPLIT,
            },
            "augmentation_config": {
                "rotation_range": ROTATION_RANGE,
                "width_shift_range": WIDTH_SHIFT_RANGE,
                "height_shift_range": HEIGHT_SHIFT_RANGE,
                "shear_range": SHEAR_RANGE,
                "zoom_range": ZOOM_RANGE,
                "horizontal_flip": HORIZONTAL_FLIP,
                "brightness_range": BRIGHTNESS_RANGE,
            },
            "hardware_config": {"use_mixed_precision": USE_MIXED_PRECISION, "gpu_memory_growth": GPU_MEMORY_GROWTH},
            "experiment_info": {"timestamp": datetime.now().isoformat(), "dataset_path": self.dataset_path, "results_dir": self.experiment_dir},
        }

        config_path = os.path.join(self.experiment_dir, "experiment_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Configuración guardada en: {config_path}")

    def train(self):
        """Entrenar el modelo"""
        print("\n🚀 Iniciando entrenamiento...")

        # Calcular pesos de clase
        class_weights = self.calculate_class_weights()

        # Crear generadores de datos
        train_gen, val_gen, test_gen = self.create_data_generators()

        # Construir modelo
        self.build_model()

        # Crear callbacks
        callbacks = self.create_callbacks()

        # Guardar configuración
        self.save_experiment_config()

        # Entrenar modelo
        steps_per_epoch = train_gen.samples // BATCH_SIZE
        validation_steps = val_gen.samples // BATCH_SIZE

        print(f"\n📈 Comenzando entrenamiento:")
        print(f"   - Épocas: {EPOCHS}")
        print(f"   - Steps per epoch: {steps_per_epoch}")
        print(f"   - Validation steps: {validation_steps}")
        print(f"   - Usando class weights: {'Sí' if class_weights else 'No'}")

        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        print("✅ Entrenamiento completado!")

        # Evaluar en test set
        self.evaluate_model(test_gen)

        # Generar plots de entrenamiento
        if SAVE_TRAINING_PLOTS:
            self.plot_training_history()

        return self.history

    def evaluate_model(self, test_generator):
        """Evaluar modelo en conjunto de test"""
        print("\n📊 Evaluando modelo en conjunto de test...")

        # Cargar mejor modelo
        best_model_path = os.path.join(self.experiment_dir, "best_model.h5")
        if os.path.exists(best_model_path):
            self.model.load_weights(best_model_path)
            print("✅ Cargado mejor modelo guardado")

        # Evaluación general
        evaluation_results = self.model.evaluate(test_generator, verbose=1)
        test_loss = evaluation_results[0]
        test_accuracy = evaluation_results[1]
        test_top3_accuracy = evaluation_results[2]

        print("\n📈 Resultados en Test Set:")
        print(f"   - Test Loss: {test_loss:.4f}")
        print(f"   - Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
        print(f"   - Top-3 Accuracy: {test_top3_accuracy:.4f} ({test_top3_accuracy * 100:.2f}%)")

        # Predicciones para métricas detalladas
        test_generator.reset()
        predictions = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes

        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)

        # Guardar reporte
        report_path = os.path.join(self.experiment_dir, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Imprimir reporte resumido
        print(f"\n📋 Classification Report (resumen):")
        print(f"   - Macro avg F1-score: {report['macro avg']['f1-score']:.4f}")
        print(f"   - Weighted avg F1-score: {report['weighted avg']['f1-score']:.4f}")

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm)

        # Guardar métricas finales
        final_metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "test_top3_accuracy": float(test_top3_accuracy),
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }

        metrics_path = os.path.join(self.experiment_dir, "final_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

        return final_metrics

    def plot_training_history(self):
        """Generar plots del historial de entrenamiento"""
        if self.history is None:
            return

        # Configurar estilo
        plt.style.use("seaborn-v0_8")
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

        # Plot 3: Top-3 Accuracy
        if "top_3_accuracy" in self.history.history:
            axes[1, 0].plot(self.history.history["top_3_accuracy"], label="Training Top-3 Accuracy", linewidth=2)
            axes[1, 0].plot(self.history.history["val_top_3_accuracy"], label="Validation Top-3 Accuracy", linewidth=2)
            axes[1, 0].set_title("Model Top-3 Accuracy")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Top-3 Accuracy")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

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


def main():
    """Función principal"""
    print("🍅 Clasificador de Enfermedades en Plantas de Tomate")
    print("=" * 60)

    try:
        # Crear instancia del clasificador
        classifier = TomatoDiseaseClassifier()

        # Entrenar modelo
        classifier.train()

        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        print(f"📁 Resultados guardados en: {classifier.experiment_dir}")

    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
