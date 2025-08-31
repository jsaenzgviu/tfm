#!/usr/bin/env python3
"""
Conversión de MobileNetV4 a TensorFlow Lite para Dispositivos Android
Script para preparar el modelo para despliegue móvil
"""

import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import warnings
import hashlib
import platform
import importlib

warnings.filterwarnings("ignore")

# ==================================================================================
# CONFIGURACIÓN - MODIFICAR ESTAS VARIABLES SEGÚN NECESIDADES
# ==================================================================================

# Modelo a convertir (último entrenamiento: 66.21%) - RUTAS DINÁMICAS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Nombre del experimento (modificar según corresponda)
EXPERIMENT_NAME = "mobilenetv4_fixed_v2_20250827_225128"
EXPERIMENT_DIR = os.path.join(SCRIPT_DIR, "experiments", EXPERIMENT_NAME)

# BEST_MODEL_PATH se resuelve dinámicamente buscando candidatos en el directorio del experimento
# para nuestro caso específico fue student_model.weights_complete.h5 ya que es que contiene todos los pesos
PREFERRED_MODEL_NAMES = [
    "student_model.weights_complete.h5",
]


def _resolve_best_model_path():
    # Si hay una ruta explícita en ambiente, respetarla
    env_path = os.environ.get("BEST_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Buscar dentro del experimento preferidos
    if os.path.isdir(EXPERIMENT_DIR):
        for name in PREFERRED_MODEL_NAMES:
            candidate = os.path.join(EXPERIMENT_DIR, name)
            if os.path.exists(candidate):
                return candidate

        # Si no hay nombres preferidos, elegir el último .h5 por mtime
        candidates = [os.path.join(EXPERIMENT_DIR, f) for f in os.listdir(EXPERIMENT_DIR) if f.endswith((".h5", ".keras", ".pb"))]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]

    # Fallback: devolver None para que el caller lance el error controladamente
    return None


BEST_MODEL_PATH = _resolve_best_model_path()
BACKUP_MODEL_PATH = EXPERIMENT_DIR

# Configuración de TensorFlow Lite
INPUT_SIZE = (224, 224, 3)
NUM_CLASSES = 11

# Opciones de optimización - DESACTIVAR CUANTIZACIÓN AGRESIVA
ENABLE_QUANTIZATION = False  # Desactivar cuantización INT8 para preservar precisión
ENABLE_OPTIMIZATION = True  # Mantener optimizaciones básicas
ENABLE_TF_SELECT = True  # Reintentar con TF Select si la conversión falla
TARGET_ANDROID_API = 21  # Android API mínimo (Lollipop)

# Directorio de salida
OUTPUT_DIR = "mobile_deployment"
TFLITE_MODEL_NAME = "tomato_disease_mobilenetv4.tflite"

# Dataset para representative dataset (cuantización). Relativo al script por defecto.
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset_final")
REPRESENTATIVE_SAMPLES = 100  # Muestras para calibrar cuantización

# ==================================================================================
# FIN DE CONFIGURACIÓN
# ==================================================================================


def setup_environment():
    """Configurar el entorno para conversión"""
    # Crear directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Configurar TensorFlow para eficiencia
    tf.config.optimizer.set_jit(True)

    print("✅ Entorno configurado para conversión móvil")


def load_best_model():
    """Cargar el mejor modelo entrenado desde el experimento con varios fallbacks."""
    print(f"\n🔍 Buscando modelo en: {EXPERIMENT_DIR}")

    if not BEST_MODEL_PATH:
        print(f"❌ No se encontró ningún archivo de modelo en {EXPERIMENT_DIR}")
        # Intentar listar archivos para debugging
        if os.path.isdir(EXPERIMENT_DIR):
            print("Archivos en el directorio de experimento:")
            for f in os.listdir(EXPERIMENT_DIR):
                print(" - ", f)
        raise FileNotFoundError(f"No se encontró modelo en {EXPERIMENT_DIR}")

    print(f"⏳ Intentando cargar modelo: {BEST_MODEL_PATH}")
    # Intento 1: cargar modelo completo
    try:
        model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)
        print("✅ Modelo cargado con tf.keras.models.load_model()")
        print(f"📊 Parámetros del modelo: {model.count_params():,}")
        return model
    except Exception as e_load:
        print(f"⚠️ load_model falló: {e_load}")

    # Intento 2: si el archivo parece contener solo pesos, intentar reconstruir la arquitectura
    # Buscamos un constructor de modelo conocido en el repo
    possible_builders = [
        ("mobilenetv4_clean", "create_mobilenetv4_model"),
        ("mobilenetv4_knowledge_distillation", "create_mobilenetv4_model"),
        ("mobilenetv4_clean", "create_mobilenetv4"),
    ]
    for module_name, fn_name in possible_builders:
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, fn_name):
                print(f"🔧 Intentando reconstruir modelo usando {module_name}.{fn_name}() and cargar pesos")
                builder = getattr(mod, fn_name)
                # Assumimos parámetros por defecto compatibles (img size, num_classes)
                model = builder(num_classes=NUM_CLASSES, input_shape=INPUT_SIZE)
                model.load_weights(BEST_MODEL_PATH)
                print("✅ Modelo reconstruido y pesos cargados exitosamente")
                print(f"📊 Parámetros del modelo: {model.count_params():,}")
                return model
        except Exception as e_builder:
            print(f"⚠️ No se pudo usar {module_name}.{fn_name}: {e_builder}")

    # Si todos los intentos fallan, levantar error con diagnóstico
    raise RuntimeError(f"No se pudo cargar ni reconstruir el modelo desde: {BEST_MODEL_PATH}")


def prepare_representative_dataset():
    """Preparar dataset representativo para cuantización"""
    print("\n📊 Preparando dataset representativo...")

    def representative_data_gen():
        dataset_dir = DATASET_PATH
        test_dir = os.path.join(dataset_dir, "test")

        if not os.path.exists(test_dir):
            print(f"⚠️ Dataset no encontrado en {test_dir}, generando datos sintéticos")
            # Generar datos sintéticos si no hay dataset
            for _ in range(REPRESENTATIVE_SAMPLES):
                data = np.random.random((1, *INPUT_SIZE)).astype(np.float32)
                yield [data]
            return

        # Cargar muestras reales del dataset
        sample_count = 0
        for class_name in os.listdir(test_dir):
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_file in os.listdir(class_dir):
                if sample_count >= REPRESENTATIVE_SAMPLES:
                    return

                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        img_path = os.path.join(class_dir, img_file)

                        # Cargar y procesar imagen
                        img = tf.io.read_file(img_path)
                        img = tf.image.decode_image(img, channels=3)
                        img = tf.image.resize(img, INPUT_SIZE[:2])
                        img = tf.cast(img, tf.float32) / 255.0
                        img = tf.expand_dims(img, 0)

                        yield [img.numpy()]
                        sample_count += 1

                    except Exception as e:
                        print(f"⚠️ Error procesando {img_file}: {e}")
                        continue

    print(f"✅ Dataset representativo preparado ({REPRESENTATIVE_SAMPLES} muestras)")
    return representative_data_gen


def convert_to_tflite(model, representative_dataset_fn=None):
    """Convertir modelo a TensorFlow Lite con optimizaciones y TF Select"""
    print("\n🔄 Convirtiendo modelo a TensorFlow Lite...")

    # Crear convertidor
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Configurar optimizaciones básicas
    if ENABLE_OPTIMIZATION:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Primero intentar conversión solo con TFLITE_BUILTINS
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if ENABLE_OPTIMIZATION:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if ENABLE_QUANTIZATION and representative_dataset_fn:
        print("📏 Aplicando cuantización INT8... (requiere representative_dataset)")
        converter.representative_dataset = representative_dataset_fn
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    else:
        converter.target_spec.supported_types = [tf.float16]

    try:
        print("⏳ Intentando conversión con TFLite BUILTINS...")
        tflite_model = converter.convert()
        print("✅ Conversión con BUILTINS exitosa")
        return tflite_model
    except Exception as e_builtin:
        print(f"⚠️ Conversión con BUILTINS falló: {e_builtin}")

    # Si falla, y ENABLE_TF_SELECT está activado, reintentar con SELECT_TF_OPS
    if ENABLE_TF_SELECT:
        try:
            print("� Reintentando conversión con TF Select (SELECT_TF_OPS)...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            if ENABLE_OPTIMIZATION:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            # permitir custom ops solo en el fallback
            converter.allow_custom_ops = True
            if ENABLE_QUANTIZATION and representative_dataset_fn:
                converter.representative_dataset = representative_dataset_fn
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            else:
                converter.target_spec.supported_types = [tf.float16]

            tflite_model = converter.convert()
            print("✅ Conversión con TF Select exitosa")
            return tflite_model
        except Exception as e_select:
            print(f"❌ Conversión con TF Select también falló: {e_select}")

    # Si llegamos aquí, ninguna estrategia funcionó
    raise RuntimeError("No se pudo convertir el modelo a TFLite con las estrategias disponibles")


def analyze_tflite_model(tflite_model):
    """Analizar el modelo TensorFlow Lite convertido"""
    print("\n🔍 Analizando modelo TensorFlow Lite...")

    # Crear intérprete
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Obtener detalles de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Información del modelo
    model_size_mb = len(tflite_model) / (1024 * 1024)

    analysis = {
        "model_size_mb": round(model_size_mb, 2),
        "input_shape": input_details[0]["shape"].tolist(),
        "input_dtype": str(input_details[0]["dtype"]),
        "output_shape": output_details[0]["shape"].tolist(),
        "output_dtype": str(output_details[0]["dtype"]),
        "quantized": "int8" in str(input_details[0]["dtype"]).lower(),
    }

    print(f"📏 Tamaño del modelo: {analysis['model_size_mb']} MB")
    print(f"📐 Forma de entrada: {analysis['input_shape']}")
    print(f"🎯 Forma de salida: {analysis['output_shape']}")
    print(f"🔢 Cuantizado: {'Sí' if analysis['quantized'] else 'No'}")

    return analysis, interpreter


def test_tflite_inference(interpreter):
    """Probar inferencia con el modelo TensorFlow Lite"""
    print("\n🧪 Probando inferencia TensorFlow Lite...")

    # Obtener detalles
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Crear datos de prueba
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.uint8:
        # Para modelos cuantizados
        test_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
    else:
        # Para modelos float
        test_data = np.random.random(input_shape).astype(input_dtype)

    # Realizar inferencia
    try:
        interpreter.set_tensor(input_details[0]["index"], test_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        print("✅ Inferencia exitosa")
        print(f"🎯 Forma de salida: {output.shape}")
        print(f"📊 Rango de salida: [{output.min():.4f}, {output.max():.4f}]")

        return True
    except Exception as e:
        print(f"❌ Error en inferencia: {e}")
        return False


def save_model_info(analysis, tflite_path):
    """Guardar información del modelo para la aplicación Android"""
    print("\n💾 Guardando información del modelo...")
    # Leer métricas finales desde el directorio del experimento
    metrics_path = os.path.join(EXPERIMENT_DIR, "final_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as mf:
            metrics = json.load(mf)
    else:
        print(f"⚠️ No se encontró final_metrics.json en {metrics_path}, usando valores por defecto")

    # Extraer métricas
    test_accuracy = metrics.get("test_accuracy", None)
    test_loss = metrics.get("test_loss", None)
    macro_f1 = metrics.get("macro_f1", None)
    weighted_f1 = metrics.get("weighted_f1", None)
    model_params = metrics.get("model_params", None)
    experiment_name = metrics.get("experiment_name", EXPERIMENT_NAME)

    # Formatear precisión
    if test_accuracy is not None:
        accuracy_str = f"{test_accuracy * 100:.2f}%"
    else:
        accuracy_str = "N/A"

    # Intentar leer class_names desde class_distribution.json
    class_names = None
    class_dist_path = os.path.join(EXPERIMENT_DIR, "class_distribution.json")
    if os.path.exists(class_dist_path):
        try:
            with open(class_dist_path, "r", encoding="utf-8") as cdf:
                cd = json.load(cdf)
                class_names = cd.get("class_names")
        except Exception as e:
            print(f"⚠️ Error leyendo class_distribution.json: {e}")

    if not class_names:
        # Fallback a lista por defecto (mantener compatibilidad)
        class_names = [
            "bacterial_spot",
            "early_blight",
            "healthy",
            "late_blight",
            "leaf_mold",
            "powdery_mildew",
            "septoria_leaf_spot",
            "spider_mites_two_spotted_spider_mite",
            "target_spot",
            "tomato_mosaic_virus",
            "tomato_yellow_leaf_curl_virus",
        ]

    # Checksum del archivo tflite
    def _sha256(path):
        try:
            h = hashlib.sha256()
            with open(path, "rb") as rf:
                for chunk in iter(lambda: rf.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    tflite_checksum = _sha256(tflite_path) if os.path.exists(tflite_path) else None

    # Artifacts presentes
    artifacts = {}
    for name in ["training_history.png", "confusion_matrix.png", "training_log.csv"]:
        p = os.path.join(EXPERIMENT_DIR, name)
        if os.path.exists(p):
            artifacts[name] = p

    # Información del entorno
    env_info = {
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "platform": platform.platform(),
    }

    # Información del modelo para Android
    model_info = {
        "model_name": "Tomato Disease Detection - MobileNetV4",
        "version": "1.0.0",
        "created_date": datetime.now().isoformat(),
        "accuracy": accuracy_str,
        "model_file": os.path.basename(tflite_path),
        "model_checksum": tflite_checksum,
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
        "preprocessing": {
            "resize": [INPUT_SIZE[0], INPUT_SIZE[1]],
            "normalize": "0-1",
            "channels": "RGB",
        },
        "performance": {
            **analysis,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "model_params": model_params,
            "experiment_name": experiment_name,
        },
        "artifacts": artifacts,
        "environment": env_info,
        "android_requirements": {
            "min_api": TARGET_ANDROID_API,
            "permissions": ["android.permission.CAMERA", "android.permission.READ_EXTERNAL_STORAGE"],
            "libraries": ["tensorflow-lite"],
            "note": "If model uses TF ops not supported by TFLite, enable TF Select in app dependencies",
        },
    }

    # Guardar JSON
    info_path = os.path.join(OUTPUT_DIR, "model_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print(f"✅ Información guardada en: {info_path}")
    return model_info


def create_android_integration_guide():
    """Crear guía de integración para Android"""
    # Nota sobre JDK/Gradle: proyectos modernos requieren Java 17 para Android Gradle Plugin.
    jdk_note = (
        "\n"
        "**Nota importante (JDK/Gradle):** El Android Gradle Plugin moderno requiere Java 17. "
        "Asegúrate de que Android Studio/Gradle use JDK 17.\n"
        "- En Android Studio: File → Settings → Build, Execution, Deployment → Build Tools → Gradle → Gradle JDK → seleccionar JDK 17.\n"
        "- Alternativamente, configura `org.gradle.java.home=/ruta/a/jdk17` en `~/.gradle/gradle.properties` o en el `gradle.properties` del proyecto.\n"
    )

    # TF Select: mostrar la dependencia solo si ENABLE_TF_SELECT está activado
    if ENABLE_TF_SELECT:
        tf_select_entry = "    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1'  // Para TF Select (solo si es necesario)\n"
    else:
        tf_select_entry = (
            "    // Nota: No agregues 'tensorflow-lite-select-tf-ops' salvo que la conversión requiera TF Select. "
            "Incluirlo incrementa significativamente el tamaño del APK.\n"
        )

    guide_content = (
        "# Guía de Integración Android - Tomato Disease Detection\n\n"
        "## 📱 Requisitos del Sistema\n"
        "- Android API 21+ (Android 5.0 Lollipop)\n"
        "- Cámara del dispositivo\n"
        "- Mínimo 2GB RAM\n"
        "- 50MB de almacenamiento libre\n"
        f"{jdk_note}\n"
        "## 📦 Archivos Necesarios\n"
        "- `tomato_disease_mobilenetv4.tflite` - Modelo TensorFlow Lite\n"
        "- `model_info.json` - Metadatos del modelo\n\n"
        "## 🔧 Dependencias de Android Studio\n\n"
        "### build.gradle (Module: app)\n"
        "```gradle\n"
        "dependencies {\n"
        "    implementation 'org.tensorflow:tensorflow-lite:2.16.1'\n"
        f"{tf_select_entry}"
        "    implementation 'org.tensorflow:tensorflow-lite-gpu:2.16.1'\n"
        "    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'\n"
        "    implementation 'androidx.camera:camera-camera2:1.3.0'\n"
        "    implementation 'androidx.camera:camera-lifecycle:1.3.0'\n"
        "    implementation 'androidx.camera:camera-view:1.3.0'\n"
        "}\n"
        "```\n\n"
        "### AndroidManifest.xml\n"
        "```xml\n"
        '<uses-permission android:name="android.permission.CAMERA" />\n'
        '<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />\n'
        '<uses-feature android:name="android.hardware.camera" android:required="true" />\n'
        "```\n\n"
        "## 🚀 Implementación Básica\n\n"
        "(Se incluye ejemplo de carga / preprocesado / inferencia en la guía completa del repo)\n\n"
        "## 📱 Optimizaciones para Producción\n\n"
        "- GPU: usar `GpuDelegate()` en `Interpreter.Options()` si se desea acelerar en dispositivos compatibles.\n"
        "- NNAPI: `options.setUseNNAPI(true)` puede ayudar en dispositivos que soporten NNAPI.\n\n"
        "## 🔍 Testing y Validación\n"
        "1. Probar con imágenes de cada clase de enfermedad\n"
        "2. Verificar rendimiento en diferentes dispositivos\n"
        "3. Medir latencia de inferencia\n"
        "4. Validar precisión vs modelo original\n\n"
        "## 📊 Métricas Esperadas\n"
        "- Precisión: ~95% (ajustar según `mobile_deployment/model_info.json`)\n"
        "- Latencia: <200ms en dispositivos modernos\n"
        "- Tamaño del modelo: ~2-4MB (dependiendo de cuantización)\n"
        "- RAM usage: <100MB durante inferencia\n"
    )

    guide_path = os.path.join(OUTPUT_DIR, "ANDROID_INTEGRATION_GUIDE.md")
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide_content)

    print(f"✅ Guía de integración creada: {guide_path}")


def main():
    """Función principal de conversión"""
    print("🚀 CONVERSIÓN MOBILENETV4 PARA ANDROID")
    print("=" * 50)

    # 1. Configurar entorno
    setup_environment()

    # 2. Cargar modelo
    model = load_best_model()

    # 3. Preparar dataset representativo para cuantización
    representative_dataset_fn = None
    if ENABLE_QUANTIZATION:
        representative_dataset_fn = prepare_representative_dataset()

    # 4. Convertir a TensorFlow Lite
    tflite_model = convert_to_tflite(model, representative_dataset_fn)

    # 5. Analizar modelo convertido
    analysis, interpreter = analyze_tflite_model(tflite_model)

    # 6. Probar inferencia
    inference_success = test_tflite_inference(interpreter)

    # 7. Guardar modelo TFLite
    tflite_path = os.path.join(OUTPUT_DIR, TFLITE_MODEL_NAME)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Modelo TFLite guardado: {tflite_path}")

    # 8. Guardar información del modelo
    save_model_info(analysis, tflite_path)

    # 9. Crear guía de integración
    create_android_integration_guide()

    # 10. Resumen final
    print("\n🎉 CONVERSIÓN COMPLETADA")
    print("=" * 50)
    print(f"📱 Modelo Android: {tflite_path}")
    print(f"📏 Tamaño: {analysis['model_size_mb']} MB")
    print(f"🎯 Cuantizado: {'Sí' if analysis['quantized'] else 'No'}")
    print(f"✅ Inferencia: {'Funcional' if inference_success else 'Error'}")
    print(f"📁 Archivos en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
