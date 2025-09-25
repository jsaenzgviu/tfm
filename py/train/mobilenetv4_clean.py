#!/usr/bin/env python3
"""
MobileNetV4 + Knowledge Distillation - IMPLEMENTACIÓN LIMPIA Y FUNCIONAL
Versión completamente reescrita para funcionar de forma robusta
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==================================================================================
# CONFIGURACIÓN - MODIFICAR ESTAS VARIABLES SEGÚN NECESIDADES
# ==================================================================================

# Directorio del dataset (relativo al script)
DATASET_PATH = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/dataset_final"

# Configuración del modelo
MODEL_VARIANT = "medium"  # "small", "medium", "large", "hybrid" - CAMBIADO A LARGE PARA 95-96%
INPUT_SIZE = (224, 224, 3)
NUM_CLASSES = 11

# Configuración de entrenamiento
EPOCHS = 24  # Solo 1 época para debugging
BATCH_SIZE = 16  # Reducido para modelo Large con Mixed Precision
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

# Knowledge Distillation
TEACHER_ALPHA = 0.3  # Peso de la pérdida de Knowledge Distillation
TEACHER_TEMPERATURE = 4.0  # CAMBIADO de 4.0 a 3.0 para mejor destilación
USE_TEACHER_MODEL = True

# Configuración de callbacks
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 8
REDUCE_LR_FACTOR = 0.5

# Configuración de salida
SAVE_BEST_ONLY = True
SAVE_WEIGHTS_ONLY = False
VERBOSE_TRAINING = 1

# GPU Configuration - IGUAL QUE DENSENET.PY
USE_MIXED_PRECISION = False  # Para optimizar memoria en RTX 3060
GPU_MEMORY_GROWTH = False

# ==================================================================================
# FIN DE CONFIGURACIÓN
# ==================================================================================


def setup_gpu():
    """Configurar GPU para uso óptimo - IGUAL QUE DENSENET.PY"""
    # Habilitar Mixed Precision para optimizar memoria
    if USE_MIXED_PRECISION:
        try:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed Precision habilitado (float16) para optimizar memoria")
        except Exception as e:
            print(f"⚠️ Error configurando Mixed Precision: {e}")

    # Configurar GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                # Habilitar memory growth
                if GPU_MEMORY_GROWTH:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Limitar memoria GPU a 3.2GB para dejar espacio al sistema
                tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3200)])
            print(f"✅ GPU configurada con memory growth y límite optimizado: {len(gpus)} dispositivo(s)")
        except RuntimeError as e:
            print(f"⚠️ Error configurando GPU: {e}")
    else:
        print("ℹ️ Usando CPU")


def create_mobilenetv4_model(variant="large", input_shape=(224, 224, 3), num_classes=11):
    """
    Crear modelo MobileNetV4 limpio y funcional
    """
    # Configuraciones por variante (optimizadas para memoria)
    configs = {
        "small": {"width_mult": 0.5, "depth_mult": 0.6},
        "medium": {"width_mult": 0.75, "depth_mult": 0.8},
        "large": {"width_mult": 1.0, "depth_mult": 1.0},
        "hybrid": {"width_mult": 1.2, "depth_mult": 1.1},
    }

    config = configs.get(variant, configs["large"])

    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")

    # Preprocessing
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Stem convolution
    x = tf.keras.layers.Conv2D(filters=int(32 * config["width_mult"]), kernel_size=3, strides=2, padding="same", use_bias=False, name="stem_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(6.0, name="stem_relu")(x)

    # MobileNetV4 blocks (simplificado pero funcional)
    block_configs = [
        # (filters, kernel_size, strides, expand_ratio)
        (int(64 * config["width_mult"]), 3, 1, 4),
        (int(96 * config["width_mult"]), 3, 2, 6),
        (int(128 * config["width_mult"]), 3, 1, 4),
        (int(160 * config["width_mult"]), 3, 2, 6),
        (int(192 * config["width_mult"]), 3, 1, 6),
        (int(320 * config["width_mult"]), 3, 1, 6),
    ]

    for i, (filters, kernel_size, strides, expand_ratio) in enumerate(block_configs):
        # Repetir bloques según depth_mult
        num_repeats = max(1, int(2 * config["depth_mult"]))
        for j in range(num_repeats):
            stride = strides if j == 0 else 1
            x = mobilenet_block(x, filters, kernel_size, stride, expand_ratio, f"block_{i}_{j}")

    # Head
    x = tf.keras.layers.Conv2D(filters=int(1280 * config["width_mult"]), kernel_size=1, padding="same", use_bias=False, name="head_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.ReLU(6.0, name="head_relu")(x)

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dropout
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)

    # Classification head - Compatible con Mixed Precision
    if USE_MIXED_PRECISION:
        # Para Mixed Precision, la capa final debe ser float32
        x = tf.keras.layers.Dense(num_classes, name="predictions_logits")(x)
        outputs = tf.keras.layers.Activation("softmax", dtype="float32", name="predictions")(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"MobileNetV4_{variant}")
    return model


def mobilenet_block(x, filters, kernel_size, strides, expand_ratio, block_name):
    """
    Bloque MobileNet básico pero funcional
    """
    input_channels = x.shape[-1]
    expanded_channels = int(input_channels * expand_ratio)

    # Expand
    if expand_ratio != 1:
        x = tf.keras.layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False, name=f"{block_name}_expand_conv")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{block_name}_expand_bn")(x)
        x = tf.keras.layers.ReLU(6.0, name=f"{block_name}_expand_relu")(x)

    # Depthwise
    x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False, name=f"{block_name}_depthwise_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{block_name}_depthwise_bn")(x)
    x = tf.keras.layers.ReLU(6.0, name=f"{block_name}_depthwise_relu")(x)

    # Project
    x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{block_name}_project_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{block_name}_project_bn")(x)

    # Residual connection si es posible
    if strides == 1 and input_channels == filters:
        x = tf.keras.layers.Add(name=f"{block_name}_add")([x, x])

    return x


def load_teacher_model():
    """
    Cargar modelo profesor (DenseNet121 preentrenado)
    """
    try:
        teacher = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=INPUT_SIZE)

        # Añadir head de clasificación
        x = teacher.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

        teacher_model = tf.keras.Model(inputs=teacher.input, outputs=x)
        print("✅ Modelo profesor (DenseNet121) cargado exitosamente")
        return teacher_model
    except Exception as e:
        print(f"⚠️ Error cargando modelo profesor: {e}")
        return None


# Callback personalizado para guardar el modelo student
class StudentModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = -np.inf if mode == "max" else np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            if self.verbose > 0:
                print(f"\nWarning: Can save best model only with {self.monitor} available, skipping.")
            return

        if self.mode == "max":
            if current > self.best:
                old_best = self.best
                self.best = current
                if self.verbose > 0:
                    print(
                        f"\nEpoch {epoch + 1}: {self.monitor} improved from {old_best:.5f} to {current:.5f}, saving student model to {self.filepath}"
                    )

                # 🎯 SOLUCIÓN: Crear un modelo independiente con los pesos actuales del student
                print(f"\n� Creando modelo student independiente...")
                try:
                    from mobilenetv4_clean import create_mobilenetv4_model

                    # Crear un nuevo modelo student independiente
                    independent_student = create_mobilenetv4_model(variant="medium", input_shape=(224, 224, 3), num_classes=11)

                    # Copiar los pesos exactos del student entrenado
                    student_weights = self.model.student.get_weights()
                    independent_student.set_weights(student_weights)

                    # Compilar el modelo independiente
                    independent_student.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

                    print(f"   ✅ Modelo independiente creado con {independent_student.count_params():,} parámetros")

                    # Verificar con datos de test rápido
                    import os
                    from tensorflow.keras.preprocessing.image import ImageDataGenerator

                    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
                    test_gen = test_datagen.flow_from_directory(
                        "dataset/test", target_size=(224, 224), batch_size=8, class_mode="sparse", shuffle=False
                    )

                    # Evaluar modelo original vs independiente
                    print(f"   🔍 Verificando consistencia...")
                    original_results = self.model.student.evaluate(test_gen, verbose=0)
                    test_gen.reset()
                    independent_results = independent_student.evaluate(test_gen, verbose=0)

                    print(f"   → Student original accuracy: {original_results[1]:.4f}")
                    print(f"   → Student independiente accuracy: {independent_results[1]:.4f}")
                    print(f"   → Diferencia: {abs(original_results[1] - independent_results[1]):.6f}")

                    if abs(original_results[1] - independent_results[1]) < 0.001:
                        print(f"   ✅ Pesos copiados correctamente")

                        # Guardar el modelo independiente
                        independent_student.save(self.filepath)
                        print(f"   💾 Modelo independiente guardado en: {self.filepath}")

                        # Verificar el guardado
                        test_gen.reset()
                        loaded_model = tf.keras.models.load_model(self.filepath)
                        loaded_results = loaded_model.evaluate(test_gen, verbose=0)
                        print(f"   → Modelo cargado accuracy: {loaded_results[1]:.4f}")

                        if abs(independent_results[1] - loaded_results[1]) < 0.001:
                            print(f"   🎉 ¡ÉXITO! Modelo guardado correctamente con precisión preservada")
                        else:
                            print(f"   ⚠️ Problema en el guardado: diferencia {abs(independent_results[1] - loaded_results[1]):.6f}")
                    else:
                        print(f"   ❌ Error en copia de pesos: diferencia {abs(original_results[1] - independent_results[1]):.6f}")
                        # Fallback al método original
                        self.model.student.save(self.filepath)
                        print(f"   📦 Fallback: Guardado con método original")

                except Exception as e:
                    print(f"   ❌ Error en guardado avanzado: {e}")
                    # Fallback al método original
                    try:
                        self.model.student.save(self.filepath)
                        print(f"   📦 Fallback: Student model guardado con método original")
                    except Exception as e2:
                        print(f"   ❌ Error en fallback: {e2}")

                except Exception as e:
                    print(f"   ❌ Error en verificación: {e}")
                    # Continuar con guardado normal
                    try:
                        self.model.student.save(self.filepath)
                        print(f"   ✅ Student model saved successfully (sin verificación)")
                    except Exception as e2:
                        print(f"   ❌ Error saving student model: {e2}")

            else:
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f} (current: {current:.5f})")
        else:
            if current < self.best:
                old_best = self.best
                self.best = current
                if self.verbose > 0:
                    print(
                        f"\nEpoch {epoch + 1}: {self.monitor} improved from {old_best:.5f} to {current:.5f}, saving student model to {self.filepath}"
                    )
                try:
                    self.model.student.save(self.filepath)
                    print(f"✅ Student model saved successfully")
                except Exception as e:
                    print(f"❌ Error saving student model: {e}")
            else:
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f}")


# Callback para restaurar los mejores pesos del student
class StudentModelRestoreBest(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor="val_accuracy", mode="max", verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = -np.inf if mode == "max" else np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        if self.mode == "max":
            if current > self.best:
                self.best = current
                self.best_weights = self.model.student.get_weights()
        else:
            if current < self.best:
                self.best = current
                self.best_weights = self.model.student.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            if self.verbose > 0:
                print(f"\nRestoring best student weights from epoch with {self.monitor}={self.best:.5f}")
            self.model.student.set_weights(self.best_weights)


class KnowledgeDistillationModel(tf.keras.Model):
    """
    Modelo de Knowledge Distillation limpio y funcional
    """

    def __init__(self, student, teacher=None, alpha=0.3, temperature=4.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature

    def compile(self, optimizer, metrics=None):
        super().compile(optimizer=optimizer)
        self.student_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()
        self.metrics_list = metrics or []

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # Predicciones del estudiante
            student_predictions = self.student(x, training=True)

            # Pérdida del estudiante
            student_loss = self.student_loss_fn(y, student_predictions)

            # Knowledge Distillation si hay profesor
            if self.teacher is not None:
                teacher_predictions = self.teacher(x, training=False)

                # Suavizar las predicciones
                teacher_soft = tf.nn.softmax(teacher_predictions / self.temperature)
                student_soft = tf.nn.softmax(student_predictions / self.temperature)

                # Pérdida de destilación
                distillation_loss = self.distillation_loss_fn(teacher_soft, student_soft)

                # Pérdida total
                total_loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss * (self.temperature**2)
            else:
                total_loss = student_loss
                distillation_loss = 0.0

        # Actualizar pesos
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Actualizar métricas
        results = {"loss": total_loss, "student_loss": student_loss}
        if self.teacher is not None:
            results["distillation_loss"] = distillation_loss

        for metric in self.metrics_list:
            metric.update_state(y, student_predictions)
            results[metric.name] = metric.result()

        return results

    def test_step(self, data):
        x, y = data
        student_predictions = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, student_predictions)

        results = {"loss": student_loss}
        for metric in self.metrics_list:
            metric.update_state(y, student_predictions)
            results[metric.name] = metric.result()

        return results

    def call(self, inputs):
        return self.student(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "temperature": self.temperature,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Para la carga, creamos una instancia temporal
        # Los modelos student y teacher se restaurarán desde los pesos guardados
        return cls(student=None, teacher=None, alpha=config.get("alpha", 0.3), temperature=config.get("temperature", 4.0))


def create_data_generators():
    """
    Crear generadores de datos limpios y funcionales
    """
    # Verificar que el dataset existe
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset no encontrado en: {DATASET_PATH}")

    train_dir = os.path.join(DATASET_PATH, "train")
    valid_dir = os.path.join(DATASET_PATH, "valid")
    test_dir = os.path.join(DATASET_PATH, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError("Directorios train/test no encontrados")

    # Verificar si existe directorio de validación separado
    if os.path.exists(valid_dir):
        print("✅ Usando split de validación separado")
        use_validation_split = False
    else:
        print("ℹ️ Usando validación desde train (20%)")
        use_validation_split = True

    # Data augmentation para entrenamiento
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode="nearest",
        validation_split=0.2 if use_validation_split else None,
    )

    # Sin augmentation para validación y test
    eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # Generador de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=INPUT_SIZE[:2],
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="training" if use_validation_split else None,
        shuffle=True,
    )

    # Generador de validación
    if use_validation_split:
        validation_generator = train_datagen.flow_from_directory(
            train_dir, target_size=INPUT_SIZE[:2], batch_size=BATCH_SIZE, class_mode="sparse", subset="validation", shuffle=False
        )
    else:
        validation_generator = eval_datagen.flow_from_directory(
            valid_dir, target_size=INPUT_SIZE[:2], batch_size=BATCH_SIZE, class_mode="sparse", shuffle=False
        )

    # Generador de test
    test_generator = eval_datagen.flow_from_directory(test_dir, target_size=INPUT_SIZE[:2], batch_size=BATCH_SIZE, class_mode="sparse", shuffle=False)

    print(f"✅ Datos cargados:")
    print(f"   - Training: {train_generator.samples} samples")
    print(f"   - Validation: {validation_generator.samples} samples")
    print(f"   - Test: {test_generator.samples} samples")
    print(f"   - Classes: {len(train_generator.class_indices)}")

    return train_generator, validation_generator, test_generator


def create_experiment_dir():
    """
    Crear directorio de experimento con timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"mobilenetv4_clean_{MODEL_VARIANT}_{timestamp}"
    experiment_dir = os.path.join("experiments", experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)
    print(f"📁 Experimento creado: {experiment_dir}")

    return experiment_dir


def save_experiment_config(experiment_dir):
    """
    Guardar configuración del experimento
    """
    config = {
        "model_variant": MODEL_VARIANT,
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "teacher_alpha": TEACHER_ALPHA,
        "teacher_temperature": TEACHER_TEMPERATURE,
        "use_teacher_model": USE_TEACHER_MODEL,
        "dataset_path": DATASET_PATH,
    }

    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"💾 Configuración guardada: {config_path}")


def plot_training_history(history, experiment_dir):
    """
    Generar gráficos de entrenamiento
    """
    print("\n📊 Generando gráficos de entrenamiento...")

    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Entrenamiento MobileNetV4 + Knowledge Distillation", fontsize=16)

    # Accuracy
    axes[0, 0].plot(history.history["accuracy"], label="Training Accuracy", color="blue")
    axes[0, 0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="red")
    axes[0, 0].set_title("Model Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(history.history["loss"], label="Training Loss", color="blue")
    axes[0, 1].plot(history.history["val_loss"], label="Validation Loss", color="red")
    axes[0, 1].set_title("Model Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Student Loss (si está disponible)
    if "student_loss" in history.history:
        axes[1, 0].plot(history.history["student_loss"], label="Student Loss", color="green")
        axes[1, 0].set_title("Student Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Distillation Loss (si está disponible)
    if "distillation_loss" in history.history:
        axes[1, 1].plot(history.history["distillation_loss"], label="Distillation Loss", color="orange")
        axes[1, 1].set_title("Knowledge Distillation Loss")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()

    # Guardar gráfico
    plot_path = os.path.join(experiment_dir, "training_curves_detailed.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Gráficos guardados en: {plot_path}")


def plot_confusion_matrix(cm, class_names, experiment_dir):
    """
    Generar matriz de confusión
    """
    print("\n📊 Generando matriz de confusión...")

    plt.figure(figsize=(12, 10))

    # Crear heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar_kws={"label": "Número de muestras"})

    plt.title("Matriz de Confusión - MobileNetV4 + Knowledge Distillation", fontsize=16, pad=20)
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Verdadero", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Guardar matriz
    cm_path = os.path.join(experiment_dir, "confusion_matrix_real.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Matriz de confusión guardada en: {cm_path}")
    return cm_path


def evaluate_model_complete(model, test_generator, experiment_dir):
    """
    Evaluación completa del modelo - IGUAL QUE DENSENET.PY
    """
    print("\n🔬 Iniciando evaluación completa del modelo...")

    # Obtener nombres de clases
    class_indices = test_generator.class_indices
    class_names = list(class_indices.keys())

    print(f"📋 Clases detectadas: {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")

    # Hacer predicciones
    print("\n🔄 Generando predicciones...")
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    # Obtener etiquetas verdaderas
    true_classes = test_generator.classes

    print(f"✅ Predicciones completadas:")
    print(f"   - Total muestras: {len(true_classes)}")
    print(f"   - Forma predicciones: {predictions.shape}")

    # Calcular accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"📊 Accuracy del modelo: {accuracy:.4f}")

    # Generar classification report
    print("\n📋 Generando classification report...")
    class_report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)

    # Guardar classification report
    report_path = os.path.join(experiment_dir, "classification_report_real.json")
    with open(report_path, "w") as f:
        json.dump(class_report, f, indent=2)
    print(f"✅ Classification report guardado en: {report_path}")

    # Mostrar resumen del classification report
    print("\n📊 MÉTRICAS POR CLASE:")
    print("-" * 60)
    for class_name in class_names:
        if class_name in class_report:
            metrics = class_report[class_name]
            print(f"{class_name:25} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")

    # Métricas macro y weighted
    print("-" * 60)
    macro_avg = class_report["macro avg"]
    weighted_avg = class_report["weighted avg"]
    print(f"{'MACRO AVG':25} | Precision: {macro_avg['precision']:.3f} | Recall: {macro_avg['recall']:.3f} | F1: {macro_avg['f1-score']:.3f}")
    print(
        f"{'WEIGHTED AVG':25} | Precision: {weighted_avg['precision']:.3f} | Recall: {weighted_avg['recall']:.3f} | F1: {weighted_avg['f1-score']:.3f}"
    )

    # Generar matriz de confusión
    print("\n🔄 Calculando matriz de confusión...")
    cm = confusion_matrix(true_classes, predicted_classes)

    # Guardar matriz de confusión como JSON
    cm_data = {"confusion_matrix": cm.tolist(), "class_names": class_names, "accuracy": float(accuracy), "total_samples": int(len(true_classes))}

    cm_json_path = os.path.join(experiment_dir, "confusion_matrix_data.json")
    with open(cm_json_path, "w") as f:
        json.dump(cm_data, f, indent=2)

    # Generar gráfico de matriz de confusión
    cm_plot_path = plot_confusion_matrix(cm, class_names, experiment_dir)

    # Guardar resumen final
    final_metrics = {
        "test_accuracy": float(accuracy),
        "total_samples": int(len(true_classes)),
        "num_classes": len(class_names),
        "macro_precision": float(macro_avg["precision"]),
        "macro_recall": float(macro_avg["recall"]),
        "macro_f1": float(macro_avg["f1-score"]),
        "weighted_precision": float(weighted_avg["precision"]),
        "weighted_recall": float(weighted_avg["recall"]),
        "weighted_f1": float(weighted_avg["f1-score"]),
        "class_names": class_names,
        "files_generated": {"classification_report": report_path, "confusion_matrix_data": cm_json_path, "confusion_matrix_plot": cm_plot_path},
    }

    metrics_path = os.path.join(experiment_dir, "evaluation_metrics_complete.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n✅ Evaluación completa finalizada")
    print(f"📁 Métricas guardadas en: {metrics_path}")

    return final_metrics


def main():
    """
    Función principal - limpia y organizada
    """
    print("🚀 Iniciando entrenamiento MobileNetV4 + Knowledge Distillation")
    print("=" * 70)

    # Configurar entorno
    setup_gpu()

    # Crear directorio de experimento
    experiment_dir = create_experiment_dir()
    save_experiment_config(experiment_dir)

    # Cargar datos
    print("\n📊 Cargando datos...")
    train_gen, val_gen, test_gen = create_data_generators()

    # Crear modelo estudiante
    print(f"\n🏗️ Creando modelo MobileNetV4-{MODEL_VARIANT}...")
    student_model = create_mobilenetv4_model(variant=MODEL_VARIANT, input_shape=INPUT_SIZE, num_classes=NUM_CLASSES)

    print(f"📏 Parámetros del modelo: {student_model.count_params():,}")

    # Cargar modelo profesor
    teacher_model = None
    if USE_TEACHER_MODEL:
        print("\n👨‍🏫 Cargando modelo profesor...")
        teacher_model = load_teacher_model()

    # Crear modelo de Knowledge Distillation
    print("\n🔬 Creando modelo de Knowledge Distillation...")
    kd_model = KnowledgeDistillationModel(student=student_model, teacher=teacher_model, alpha=TEACHER_ALPHA, temperature=TEACHER_TEMPERATURE)

    # Compilar modelo
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    kd_model.compile(optimizer=optimizer, metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

    # Callbacks
    student_model_path = os.path.join(experiment_dir, "student_model.h5")
    callbacks = [
        StudentModelCheckpoint(
            filepath=student_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=SAVE_BEST_ONLY,
            verbose=1,
        ),
        StudentModelRestoreBest(
            filepath=student_model_path,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=False,  # Usamos nuestro callback personalizado
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(experiment_dir, "training_log.csv"), append=True),
    ]

    # Entrenar modelo
    print(f"\n🎯 Iniciando entrenamiento ({EPOCHS} épocas)...")
    print("=" * 70)

    history = kd_model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks, verbose=VERBOSE_TRAINING)

    # EL MODELO STUDENT YA SE GUARDÓ AUTOMÁTICAMENTE CON LOS MEJORES PESOS
    print("\n💾 Modelo student ya guardado automáticamente con mejores pesos")
    student_model_path = os.path.join(experiment_dir, "student_model.h5")
    print(f"✅ Modelo estudiante disponible en: {student_model_path}")

    # Evaluar en test con métricas básicas
    print("\n📊 Evaluando métricas básicas...")
    test_results = kd_model.evaluate(test_gen, verbose=1)  # EVALUACIÓN COMPLETA - IGUAL QUE DENSENET.PY
    print("\n" + "=" * 70)
    print("🔬 EVALUACIÓN COMPLETA DEL MODELO")
    print("=" * 70)

    # Usar el modelo estudiante para evaluación completa
    complete_metrics = evaluate_model_complete(kd_model.student, test_gen, experiment_dir)

    # Generar gráficos de entrenamiento
    plot_training_history(history, experiment_dir)

    # Guardar resultados COMPLETOS
    results = {
        "test_loss": float(test_results[0]),
        "test_accuracy": float(test_results[1]),
        "test_accuracy_complete": float(complete_metrics["test_accuracy"]),
        "model_parameters": int(student_model.count_params()),
        "experiment_dir": experiment_dir,
        "config": {
            "model_variant": MODEL_VARIANT,
            "teacher_temperature": TEACHER_TEMPERATURE,
            "teacher_alpha": TEACHER_ALPHA,
            "epochs": EPOCHS,
            "final_lr": float(kd_model.optimizer.learning_rate.numpy()),
        },
        "complete_metrics": {
            "macro_precision": complete_metrics["macro_precision"],
            "macro_recall": complete_metrics["macro_recall"],
            "macro_f1": complete_metrics["macro_f1"],
            "weighted_precision": complete_metrics["weighted_precision"],
            "weighted_recall": complete_metrics["weighted_recall"],
            "weighted_f1": complete_metrics["weighted_f1"],
            "num_classes": complete_metrics["num_classes"],
            "total_samples": complete_metrics["total_samples"],
        },
    }

    results_path = os.path.join(experiment_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # REPORTE FINAL COMPLETO
    print("\n" + "=" * 70)
    print("🎉 ENTRENAMIENTO Y EVALUACIÓN COMPLETADOS")
    print("=" * 70)
    print(f"🏗️  Modelo: MobileNetV4-{MODEL_VARIANT} + Knowledge Distillation")
    print(f"🧠 Parámetros: {results['model_parameters']:,}")
    print(f"🌡️  Temperatura KD: {TEACHER_TEMPERATURE}")
    print(f"⚖️  Alpha KD: {TEACHER_ALPHA}")
    print("-" * 70)
    print(f"📊 Accuracy (evaluación básica): {results['test_accuracy']:.4f}")
    print(f"📊 Accuracy (evaluación completa): {results['test_accuracy_complete']:.4f}")
    print(f"📉 Pérdida final: {results['test_loss']:.4f}")
    print("-" * 70)
    print("📋 MÉTRICAS COMPLETAS:")
    print(f"   • Precision (macro): {complete_metrics['macro_precision']:.4f}")
    print(f"   • Recall (macro): {complete_metrics['macro_recall']:.4f}")
    print(f"   • F1-Score (macro): {complete_metrics['macro_f1']:.4f}")
    print(f"   • Precision (weighted): {complete_metrics['weighted_precision']:.4f}")
    print(f"   • Recall (weighted): {complete_metrics['weighted_recall']:.4f}")
    print(f"   • F1-Score (weighted): {complete_metrics['weighted_f1']:.4f}")
    print("-" * 70)
    print(f"📁 Resultados guardados en: {experiment_dir}")
    print("📄 Archivos generados:")
    print(f"   • Modelo estudiante (mejores pesos): student_model.h5")
    print(f"   • Matriz de confusión: confusion_matrix_real.png")
    print(f"   • Métricas por clase: classification_report_real.json")
    print(f"   • Gráficos de entrenamiento: training_curves_detailed.png")
    print(f"   • Configuración: config.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
