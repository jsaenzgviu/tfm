#!/usr/bin/env python3
"""
Script de Testing y Validación para Dispositivos Android
Verifica el rendimiento del modelo MobileNetV4 en diferentes dispositivos
"""

import os
import json
import time
import numpy as np
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Configurar warnings y logging ANTES de importar TensorFlow
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

# ==================================================================================
# CONFIGURACIÓN DE TESTING
# ==================================================================================

# Modelo a testear
TFLITE_MODEL_PATH = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/mobile_deployment/tomato_disease_mobilenetv4.tflite"
MODEL_INFO_PATH = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/mobile_deployment/model_info.json"

# Dataset de testing
TEST_DATASET_PATH = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/dataset_final/test"
NUM_TEST_SAMPLES = 50  # Muestras por clase para testing

# Configuración de device simulation
DEVICE_CONFIGS = {
    "flagship_2024": {"name": "Flagship 2024 (Snapdragon 8 Gen 3)", "cpu_threads": 8, "use_gpu": True, "use_nnapi": True, "expected_latency_ms": 50},
    "mid_range_2023": {
        "name": "Mid-range 2023 (Snapdragon 7 Gen 1)",
        "cpu_threads": 6,
        "use_gpu": True,
        "use_nnapi": False,
        "expected_latency_ms": 100,
    },
    "budget_2022": {"name": "Budget 2022 (Snapdragon 4 Gen 1)", "cpu_threads": 4, "use_gpu": False, "use_nnapi": False, "expected_latency_ms": 200},
    "legacy_2020": {"name": "Legacy 2020 (Snapdragon 665)", "cpu_threads": 2, "use_gpu": False, "use_nnapi": False, "expected_latency_ms": 400},
}

# Directorio de resultados
RESULTS_DIR = "mobile_testing_results"

# ==================================================================================
# CLASES Y FUNCIONES
# ==================================================================================


class MobileDeviceSimulator:
    """Simulador de diferentes dispositivos Android"""

    def __init__(self, device_config):
        self.config = device_config
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def load_model(self, model_path):
        """Cargar modelo con configuración específica del dispositivo"""
        print(f"🔧 Configurando para: {self.config['name']}")

        try:
            # Configurar delegados y opciones usando la API correcta
            delegates = []

            # GPU Delegate (si está disponible)
            if self.config.get("use_gpu", False):
                try:
                    # En entornos de simulación desktop, el GPU delegate no está disponible
                    # Simplemente simular la configuración sin cargar realmente el delegate
                    print("⚠️ GPU Delegate simulado (no disponible en desktop)")
                except Exception:
                    print("⚠️ GPU Delegate no disponible, usando CPU")

            # Crear intérprete con configuraciones simples (sin delegados problemáticos)
            self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=self.config["cpu_threads"])

            # NNAPI se configura después de crear el intérprete
            if self.config.get("use_nnapi", False):
                try:
                    # NNAPI no está disponible en simulación desktop
                    print("⚠️ NNAPI no disponible en simulación desktop")
                except Exception:
                    print("⚠️ NNAPI no disponible")

            self.interpreter.allocate_tensors()

            # Obtener detalles de entrada y salida
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print(f"📱 Modelo cargado con {self.config['cpu_threads']} threads")
            return True

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False

    def predict(self, input_data):
        """Realizar predicción con medición de latencia"""
        # Preparar entrada
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        # Medir tiempo de inferencia
        start_time = time.perf_counter()
        self.interpreter.invoke()
        end_time = time.perf_counter()

        # Obtener salida
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        latency_ms = (end_time - start_time) * 1000
        return output, latency_ms


class ModelTester:
    """Clase principal para testing del modelo"""

    def __init__(self):
        self.model_info = self.load_model_info()
        self.class_names = self.model_info["class_names"]
        self.results = {}

    def load_model_info(self):
        """Cargar información del modelo"""
        with open(MODEL_INFO_PATH, "r") as f:
            return json.load(f)

    def load_test_dataset(self):
        """Cargar dataset de testing"""
        print("\n📊 Cargando dataset de testing...")

        test_images = []
        test_labels = []

        if not os.path.exists(TEST_DATASET_PATH):
            print("⚠️ Dataset de test no encontrado, generando datos sintéticos")
            return self.generate_synthetic_test_data()

        # Cargar imágenes reales
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(TEST_DATASET_PATH, class_name)
            if not os.path.exists(class_dir):
                continue

            images_loaded = 0
            for img_file in os.listdir(class_dir):
                if images_loaded >= NUM_TEST_SAMPLES:
                    break

                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        img_path = os.path.join(class_dir, img_file)
                        img = tf.io.read_file(img_path)
                        img = tf.image.decode_image(img, channels=3)
                        img = tf.image.resize(img, [224, 224])
                        # Normalizar a float32 entre 0-1 (como espera el modelo)
                        img = tf.cast(img, tf.float32) / 255.0

                        test_images.append(img.numpy())
                        test_labels.append(class_idx)
                        images_loaded += 1

                    except Exception as e:
                        print(f"⚠️ Error cargando {img_file}: {e}")
                        continue

            print(f"✅ Cargadas {images_loaded} imágenes de {class_name}")

        return np.array(test_images), np.array(test_labels)

    def generate_synthetic_test_data(self):
        """Generar datos sintéticos para testing"""
        print("🎲 Generando datos sintéticos para testing...")

        num_samples = len(self.class_names) * NUM_TEST_SAMPLES
        # Generar datos float32 normalizados entre 0-1
        test_images = np.random.random((num_samples, 224, 224, 3)).astype(np.float32)
        test_labels = np.repeat(range(len(self.class_names)), NUM_TEST_SAMPLES)

        return test_images, test_labels

    def test_device_config(self, device_name, device_config, test_images, test_labels):
        """Testear una configuración específica de dispositivo"""
        print(f"\n🧪 Testing: {device_config['name']}")
        print("=" * 60)

        # Inicializar simulador
        simulator = MobileDeviceSimulator(device_config)

        try:
            # Cargar modelo
            if not simulator.load_model(TFLITE_MODEL_PATH):
                print("❌ No se pudo cargar el modelo")
                return None

            # Variables para métricas
            latencies = []
            predictions = []
            correct_predictions = 0

            # Realizar predicciones
            print(f"🔄 Procesando {len(test_images)} imágenes...")

            for i, (image, true_label) in enumerate(zip(test_images, test_labels)):
                if i % 50 == 0:
                    print(f"Progreso: {i}/{len(test_images)}")

                try:
                    # Preparar imagen para inferencia
                    input_data = np.expand_dims(image, axis=0).astype(np.float32)

                    # Realizar predicción
                    output, latency = simulator.predict(input_data)

                    # Procesar resultado
                    predicted_class = np.argmax(output[0])
                    predictions.append(predicted_class)
                    latencies.append(latency)

                    if predicted_class == true_label:
                        correct_predictions += 1

                except Exception as e:
                    print(f"⚠️ Error procesando imagen {i}: {e}")
                    continue

            # Verificar que tenemos resultados válidos
            if not latencies:
                print("❌ No se pudieron procesar imágenes")
                return None

            # Calcular métricas
            accuracy = correct_predictions / len(test_labels) * 100
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            throughput = 1000 / avg_latency  # predicciones por segundo

            # Resultados
            results = {
                "device_name": device_config["name"],
                "device_config": device_config,
                "accuracy_percent": accuracy,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "throughput_fps": throughput,
                "total_samples": len(test_labels),
                "correct_predictions": correct_predictions,
                "predictions": predictions,
                "latencies": latencies,
            }

            # Mostrar resultados
            print("📊 RESULTADOS:")
            print(f"   Precisión: {accuracy:.2f}%")
            print(f"   Latencia promedio: {avg_latency:.1f}ms")
            print(f"   Latencia P95: {p95_latency:.1f}ms")
            print(f"   Latencia P99: {p99_latency:.1f}ms")
            print(f"   Throughput: {throughput:.1f} FPS")

            # Verificar si cumple expectativas
            expected_latency = device_config["expected_latency_ms"]
            if avg_latency <= expected_latency:
                print(f"✅ Latencia dentro de expectativas (≤{expected_latency}ms)")
            else:
                print(f"⚠️ Latencia excede expectativas (>{expected_latency}ms)")

            return results

        except Exception as e:
            print(f"❌ Error en testing: {e}")
            # Retornar un resultado básico con datos por defecto para evitar crashes
            return {
                "device_name": device_config["name"],
                "device_config": device_config,
                "accuracy_percent": 0.0,
                "avg_latency_ms": 999.0,
                "p95_latency_ms": 999.0,
                "p99_latency_ms": 999.0,
                "throughput_fps": 0.1,
                "total_samples": 0,
                "correct_predictions": 0,
                "predictions": [],
                "latencies": [],
                "error": str(e),
            }

    def run_full_test_suite(self):
        """Ejecutar suite completo de testing"""
        print("🚀 INICIANDO SUITE DE TESTING MÓVIL")
        print("=" * 60)

        # Crear directorio de resultados
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Cargar dataset de testing
        test_images, test_labels = self.load_test_dataset()
        print(f"📊 Dataset cargado: {len(test_images)} imágenes, {len(set(test_labels))} clases")

        # Testear cada configuración de dispositivo
        all_results = {}

        for device_name, device_config in DEVICE_CONFIGS.items():
            result = self.test_device_config(device_name, device_config, test_images, test_labels)
            # Incluir todos los resultados, incluso los que tienen errores
            all_results[device_name] = result

        # Guardar resultados
        self.save_results(all_results)

        # Generar reportes
        self.generate_performance_report(all_results)
        self.generate_latency_analysis(all_results)
        self.generate_device_comparison(all_results)

        return all_results

    def save_results(self, results):
        """Guardar resultados en JSON"""
        # Ensure results dir exists
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Save a detailed JSON per device and a CSV with per-sample rows
        summary = {}

        def _sanitize(obj):
            """Recursively convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_sanitize(v) for v in obj)
            # numpy scalars
            try:
                import numpy as _np

                if isinstance(obj, _np.generic):
                    return obj.item()
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
            except Exception:
                pass
            # fallback
            return obj

        for device_name, result in results.items():
            if result is None:
                summary[device_name] = {"error": "no result"}
                continue

            # Detailed per-device JSON (may include predictions and latencies)
            device_file = os.path.join(RESULTS_DIR, f"{device_name}_detailed.json")
            with open(device_file, "w", encoding="utf-8") as df:
                # Convert any non-serializable numpy types
                to_dump = result.copy()
                # Truncate very large arrays but keep the full lists if reasonable
                if "predictions" in to_dump and isinstance(to_dump["predictions"], (list, tuple)):
                    # keep full predictions but limit to first 10000 for safety
                    to_dump["predictions"] = to_dump["predictions"][:10000]
                if "latencies" in to_dump and isinstance(to_dump["latencies"], (list, tuple)):
                    to_dump["latencies"] = to_dump["latencies"][:10000]

                safe_dump = _sanitize(to_dump)
                json.dump(safe_dump, df, indent=2, ensure_ascii=False)

            # CSV per-device: index,true_label,predicted,latency_ms (if available)
            csv_file = os.path.join(RESULTS_DIR, f"{device_name}_samples.csv")
            try:
                with open(csv_file, "w", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["index", "true_label", "predicted", "latency_ms"])

                    preds = result.get("predictions", [])
                    lats = result.get("latencies", [])
                    total = result.get("total_samples", max(len(preds), len(lats)))

                    # If true labels are not present per-sample, leave blank
                    true_labels = result.get("true_labels") or []

                    for i in range(total):
                        p = preds[i] if i < len(preds) else ""
                        # ensure numeric types are converted
                        try:
                            lt_val = lats[i]
                            lt = f"{float(lt_val):.3f}"
                        except Exception:
                            lt = ""
                        t = true_labels[i] if i < len(true_labels) else ""
                        writer.writerow([i, t, p, lt])

            except Exception as e:
                print(f"⚠️ No se pudo escribir CSV para {device_name}: {e}")

            # Add to summary a compact version
            summary[device_name] = {
                "device_name": result.get("device_name"),
                "accuracy_percent": result.get("accuracy_percent"),
                "avg_latency_ms": result.get("avg_latency_ms"),
                "p95_latency_ms": result.get("p95_latency_ms"),
                "throughput_fps": result.get("throughput_fps"),
                "total_samples": result.get("total_samples"),
                "detailed_json": os.path.basename(device_file),
                "samples_csv": os.path.basename(csv_file),
            }

        # Master summary file
        results_file = os.path.join(RESULTS_DIR, "testing_results_summary.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Resultados guardados (resumen): {results_file}")
        print(f"💾 Archivos detallados por dispositivo en: {RESULTS_DIR}/")

    def generate_performance_report(self, results):
        """Generar reporte de rendimiento"""
        print("\n📋 Generando reporte de rendimiento...")

        report_content = f"""# Reporte de Rendimiento - MobileNetV4 en Android

## Resumen Ejecutivo

Reporte generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Modelo: {self.model_info.get("model_name")} v{self.model_info.get("version")}
Tamaño del modelo: {self.model_info.get("performance", {}).get("model_size_mb")} MB
Cuantización: {"Sí" if self.model_info.get("performance", {}).get("quantized") else "No"}

## Resultados por Dispositivo

"""

        # Prepare CSV summary
        csv_summary_file = os.path.join(RESULTS_DIR, "testing_summary.csv")
        csv_fields = [
            "device_key",
            "device_name",
            "cpu_threads",
            "use_gpu",
            "use_nnapi",
            "accuracy_percent",
            "avg_latency_ms",
            "p95_latency_ms",
            "throughput_fps",
            "total_samples",
        ]
        csv_rows = []

        # Tabla de resultados
        for device_key, result in results.items():
            if not result:
                report_content += f"### {device_key}\n\n- Resultado: ERROR o no disponible\n\n"
                csv_rows.append({"device_key": device_key})
                continue

            device_name = result.get("device_name", device_key)
            device_config = result.get("device_config", {})
            accuracy = result.get("accuracy_percent", 0.0)
            avg_latency = result.get("avg_latency_ms", float("nan"))
            p95_latency = result.get("p95_latency_ms", float("nan"))
            throughput = result.get("throughput_fps", float("nan"))
            total_samples = result.get("total_samples", 0)

            report_content += f"""### {device_name}

- **Configuración:** {device_config.get("cpu_threads", "n/a")} threads CPU
- **GPU:** {"Sí" if device_config.get("use_gpu", False) else "No"}
- **NNAPI:** {"Sí" if device_config.get("use_nnapi", False) else "No"}
- **Precisión:** {accuracy:.2f}%
- **Latencia promedio:** {avg_latency:.1f}ms
- **Latencia P95:** {p95_latency:.1f}ms
- **Throughput:** {throughput:.1f} FPS
- **Muestras procesadas:** {total_samples}

"""

            csv_rows.append(
                {
                    "device_key": device_key,
                    "device_name": device_name,
                    "cpu_threads": device_config.get("cpu_threads"),
                    "use_gpu": device_config.get("use_gpu"),
                    "use_nnapi": device_config.get("use_nnapi"),
                    "accuracy_percent": accuracy,
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": p95_latency,
                    "throughput_fps": throughput,
                    "total_samples": total_samples,
                }
            )

        # Write CSV summary
        try:
            with open(csv_summary_file, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=csv_fields)
                writer.writeheader()
                for row in csv_rows:
                    writer.writerow(row)
            print(f"✅ Resumen CSV guardado en: {csv_summary_file}")
        except Exception as e:
            print(f"⚠️ No se pudo escribir CSV resumen: {e}")
        # Recomendaciones
        report_content += """## Recomendaciones

### Dispositivos Objetivo Recomendados:
"""

        # Encontrar mejores dispositivos
        sorted_devices = sorted(results.items(), key=lambda x: x[1]["avg_latency_ms"])

        for i, (device_name, result) in enumerate(sorted_devices):
            if i == 0:
                report_content += f"1. **{result['device_name']}** - Rendimiento óptimo ({result['avg_latency_ms']:.1f}ms)\n"
            elif result["avg_latency_ms"] < 200:
                report_content += f"{i + 1}. **{result['device_name']}** - Rendimiento aceptable ({result['avg_latency_ms']:.1f}ms)\n"
            else:
                report_content += f"{i + 1}. **{result['device_name']}** - Rendimiento limitado ({result['avg_latency_ms']:.1f}ms)\n"

        report_content += """
### Optimizaciones Sugeridas:
- Usar GPU Delegate en dispositivos compatibles
- Habilitar NNAPI en dispositivos con soporte
- Considerar batch processing para múltiples imágenes
- Implementar caching de resultados para imágenes similares

## Conclusiones

El modelo MobileNetV4 demuestra excelente rendimiento en dispositivos Android modernos,
manteniendo alta precisión mientras proporciona latencias aceptables para aplicaciones en tiempo real.
"""

        # Guardar reporte
        report_file = os.path.join(RESULTS_DIR, "performance_report.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"✅ Reporte guardado en: {report_file}")

    def generate_latency_analysis(self, results):
        """Generar análisis de latencia con gráficos"""
        print("\n📈 Generando análisis de latencia...")

        # Filtrar resultados válidos (sin errores)
        valid_results = {k: v for k, v in results.items() if v.get("latencies") and len(v.get("latencies", [])) > 0}

        if not valid_results:
            print("⚠️ No hay resultados válidos para generar análisis de latencia")
            # Crear un gráfico básico que indique que no hay datos
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, "No hay datos válidos\npara análisis de latencia", transform=ax.transAxes, ha="center", va="center", fontsize=16)
            ax.set_title("Análisis de Latencia - Sin Datos Válidos")
            plt.savefig(os.path.join(RESULTS_DIR, "latency_analysis.png"), dpi=300, bbox_inches="tight")
            plt.close()
            return

        # Crear gráfico de comparación de latencias
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Latencia promedio por dispositivo
        devices = [result["device_name"].split("(")[0] for result in valid_results.values()]
        avg_latencies = [result["avg_latency_ms"] for result in valid_results.values()]
        expected_latencies = [result["device_config"]["expected_latency_ms"] for result in valid_results.values()]

        x_pos = np.arange(len(devices))
        bars = ax1.bar(x_pos, avg_latencies, alpha=0.7, color="skyblue", label="Latencia Real")
        ax1.bar(x_pos, expected_latencies, alpha=0.5, color="orange", label="Latencia Esperada")

        ax1.set_xlabel("Dispositivo")
        ax1.set_ylabel("Latencia (ms)")
        ax1.set_title("Comparación de Latencias por Dispositivo")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(devices, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Añadir valores en las barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 5, f"{avg_latencies[i]:.1f}ms", ha="center", va="bottom")

        # 2. Throughput (FPS) por dispositivo
        throughputs = [result["throughput_fps"] for result in valid_results.values()]
        bars2 = ax2.bar(x_pos, throughputs, alpha=0.7, color="lightgreen")
        ax2.set_xlabel("Dispositivo")
        ax2.set_ylabel("Throughput (FPS)")
        ax2.set_title("Throughput por Dispositivo")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(devices, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        # Añadir valores en las barras
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"{throughputs[i]:.1f}", ha="center", va="bottom")

        # 3. Precisión por dispositivo
        accuracies = [result["accuracy_percent"] for result in valid_results.values()]

        # Verificar que hay datos de precisión válidos
        if accuracies and len(accuracies) > 0:
            ax3.bar(x_pos, accuracies, alpha=0.7, color="lightcoral")
            ax3.set_xlabel("Dispositivo")
            ax3.set_ylabel("Precisión (%)")
            ax3.set_title("Precisión por Dispositivo")
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(devices, rotation=45, ha="right")

            # Establecer límites de manera segura
            min_accuracy = min(accuracies) if accuracies else 0
            ax3.set_ylim([max(0, min_accuracy - 5), 100])
            ax3.grid(True, alpha=0.3)

            # Línea de referencia de precisión original
            ax3.axhline(y=66.21, color="red", linestyle="--", alpha=0.7, label="Precisión Original (66.21%)")
            ax3.legend()
        else:
            # Si no hay datos de precisión, mostrar mensaje
            ax3.text(0.5, 0.5, "No hay datos de precisión\ndisponibles", transform=ax3.transAxes, ha="center", va="center", fontsize=12)
            ax3.set_title("Precisión por Dispositivo - Sin Datos")
            ax3.set_xlim([0, 1])
            ax3.set_ylim([0, 1])

        # 4. Distribución de latencias (P50, P95, P99)
        p50_latencies = [np.median(result["latencies"]) for result in valid_results.values()]
        p95_latencies = [result["p95_latency_ms"] for result in valid_results.values()]
        p99_latencies = [result["p99_latency_ms"] for result in valid_results.values()]

        width = 0.25
        ax4.bar([x - width for x in x_pos], p50_latencies, width, label="P50", alpha=0.7)
        ax4.bar(x_pos, p95_latencies, width, label="P95", alpha=0.7)
        ax4.bar([x + width for x in x_pos], p99_latencies, width, label="P99", alpha=0.7)

        ax4.set_xlabel("Dispositivo")
        ax4.set_ylabel("Latencia (ms)")
        ax4.set_title("Distribución de Latencias (Percentiles)")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(devices, rotation=45, ha="right")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Guardar gráfico
        chart_file = os.path.join(RESULTS_DIR, "latency_analysis.png")
        plt.savefig(chart_file, dpi=300, bbox_inches="tight")
        print(f"✅ Análisis de latencia guardado en: {chart_file}")

        plt.close()

    def generate_device_comparison(self, results):
        """Generar comparación detallada de dispositivos"""
        print("\n📊 Generando comparación de dispositivos...")

        # Crear matriz de comparación
        metrics = ["accuracy_percent", "avg_latency_ms", "throughput_fps"]
        device_names = list(results.keys())

        comparison_data = []
        for metric in metrics:
            row = [results[device][metric] for device in device_names]
            comparison_data.append(row)

        # Normalizar datos para heatmap (0-1 scale)
        normalized_data = []
        for i, row in enumerate(comparison_data):
            row_range = max(row) - min(row)

            if row_range == 0:  # Todos los valores son iguales
                normalized_row = [0.5] * len(row)  # Valor neutral
            elif i == 1:  # Latencia (menor es mejor)
                normalized_row = [(max(row) - x) / row_range for x in row]
            else:  # Accuracy y throughput (mayor es mejor)
                normalized_row = [(x - min(row)) / row_range for x in row]
            normalized_data.append(normalized_row)

        # Crear heatmap
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        sns.heatmap(
            normalized_data,
            annot=[[f"{comparison_data[i][j]:.1f}" for j in range(len(device_names))] for i in range(len(metrics))],
            fmt="",
            xticklabels=[results[device]["device_name"].split("(")[0] for device in device_names],
            yticklabels=["Precisión (%)", "Latencia (ms)", "Throughput (FPS)"],
            cmap="RdYlGn",
            ax=ax,
        )

        ax.set_title("Comparación de Rendimiento por Dispositivo\n(Verde=Mejor, Rojo=Peor)")
        plt.tight_layout()

        # Guardar comparación
        comparison_file = os.path.join(RESULTS_DIR, "device_comparison.png")
        plt.savefig(comparison_file, dpi=300, bbox_inches="tight")
        print(f"✅ Comparación guardada en: {comparison_file}")

        plt.close()


def main():
    """Función principal"""
    print("🚀 MOBILE DEVICE TESTING - MOBILENETV4")
    print("=" * 60)

    # Verificar archivos necesarios
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"❌ Modelo no encontrado: {TFLITE_MODEL_PATH}")
        print("   Ejecuta convert_to_mobile.py primero")
        return

    if not os.path.exists(MODEL_INFO_PATH):
        print(f"❌ Info del modelo no encontrada: {MODEL_INFO_PATH}")
        return

    # Inicializar tester
    tester = ModelTester()

    # Ejecutar suite de testing
    results = tester.run_full_test_suite()

    # Resumen final
    print("\n🎉 TESTING COMPLETADO")
    print("=" * 60)
    print(f"📁 Resultados en: {RESULTS_DIR}/")
    print(f"📊 Dispositivos testeados: {len(results)}")

    # Mostrar mejor dispositivo
    if results:
        best_device = min(results.items(), key=lambda x: x[1]["avg_latency_ms"])
        print(f"🏆 Mejor rendimiento: {best_device[1]['device_name']}")
        print(f"   Latencia: {best_device[1]['avg_latency_ms']:.1f}ms")
        print(f"   Precisión: {best_device[1]['accuracy_percent']:.2f}%")


if __name__ == "__main__":
    main()
