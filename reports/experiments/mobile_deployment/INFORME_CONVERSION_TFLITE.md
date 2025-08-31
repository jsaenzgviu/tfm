# Informe: Conversión a TensorFlow Lite y preparación Android

Fecha: 2025-08-28

Resumen
-------
- Objetivo: convertir el mejor modelo del experimento `mobilenetv4_fixed_v2_20250827_225128` a TensorFlow Lite y generar los artefactos necesarios para integrar el modelo en la app Android.
- Resultado: conversión exitosa usando TFLite BUILTINS; artefactos generados y validación local de inferencia completada.

Contexto y entradas
--------------------
- Script usado: `convert_to_mobile.py` (función principal `main()` en `tomato_dataset/`).
- Experimento origen: `experiments/mobilenetv4_fixed_v2_20250827_225128`.
- Ficheros de entrada: el mejor modelo encontrado dinámicamente en el directorio del experimento (archivo `.h5` o similar).

Entorno de ejecución
--------------------
- Python: 3.11.13
- TensorFlow: 2.19.0
- Sistema: Linux (ver `mobile_deployment/model_info.json` → `environment.platform`)
- Requisito Android: Android Gradle Plugin moderno requiere JDK 17 (configurado como `org.gradle.java.home=/home/xxxx/android-jdk17` en este entorno)

Parámetros clave del pipeline
-----------------------------
- `EXPERIMENT_NAME`: mobilenetv4_fixed_v2_20250827_225128
- `ENABLE_QUANTIZATION`: False (no se aplicó cuantización INT8)
- `ENABLE_TF_SELECT`: True (solo como fallback; no fue necesario)
- `ENABLE_OPTIMIZATION`: True
- `OUTPUT_DIR`: mobile_deployment

Pasos reproducibles
-------------------
1. Activar el entorno Python del proyecto (ej. `myenvtfm`).
2. Desde `tomato_dataset/` ejecutar:

```bash
./../myenvtfm/bin/python convert_to_mobile.py
```

3. El script hace (resumen): busca el mejor `.h5`, intenta `tf.keras.models.load_model()`; si falla, intenta reconstruir el modelo desde funciones del repo y cargar pesos; convierte a TFLite (BUILTINS primero, TF Select si hace falta); analiza y prueba la inferencia; escribe `model_info.json`, `ANDROID_INTEGRATION_GUIDE.md` y el `.tflite` en `mobile_deployment/`.

Resultados (valores extraídos del `model_info.json` generado)
----------------------------------------------------------
- Modelo TFLite: `mobile_deployment/tomato_disease_mobilenetv4.tflite`
- Tamaño: 1.94 MB
- Checksum SHA256: `d7096a4e71cfc63eadf3990b30705ee7cf8efab44cc28ce5f498304d4fc01ba6`
- Test accuracy (reportado): 96.27% (JSON: `performance.test_accuracy` = 0.9626843333)
- Parámetros del modelo: 1,032,707
- Num clases: 11
- Preprocesado documentado: resize 224x224, normalize 0-1, channels RGB
- Conversión: exitoso con TFLite BUILTINS (no se requirió TF Select)
- Cuantización: no aplicada

Artefactos generados
---------------------
- `mobile_deployment/tomato_disease_mobilenetv4.tflite`
- `mobile_deployment/model_info.json`
- `mobile_deployment/ANDROID_INTEGRATION_GUIDE.md`

(Sugerido) Copiar a los assets de la app para pruebas manuales:

```bash
cp mobile_deployment/tomato_disease_mobilenetv4.tflite ../android_app/app/src/main/assets/
cp mobile_deployment/model_info.json ../android_app/app/src/main/assets/
```

Validación realizada
--------------------
- Se construyó un `tf.lite.Interpreter` en memoria con el contenido del `.tflite` y se ejecutó una inferencia de prueba (datos aleatorios con dtype correcto). Resultado: inferencia exitosa y salida con rango válido.
- Android: `MainActivity.java` ya contiene la lógica para cargar el modelo desde assets y leer `model_info.json` (métodos `initializeModel()` y `loadModelInfo()`); el proyecto compila con Gradle en terminal (JDK 17 configurado en `gradle.properties`).

Limitaciones y riesgos
----------------------
- Incluir `tensorflow-lite-select-tf-ops` en la app incrementa mucho el tamaño del APK; usar solo si la conversión falla con BUILTINS.
- Cuantización INT8 no probada: requiere dataset representativo real y evaluación detallada por clases.
- Medidas de latencia y memoria en dispositivo real pendientes; las cifras en la guía son estimadas.

Recomendaciones y próximos pasos
--------------------------------
1. Ejecutar benchmarks en 2–3 dispositivos reales (flagship, mid-range, budget) y añadir tablas al TFM.
2. Si se necesita reducir tamaño/latencia, probar cuantización (activar `ENABLE_QUANTIZATION=True` y proporcionar `dataset_final/test` representativo), documentar pérdida de precisión.
3. Automatizar el pipeline en CI: script parametrizado por `EXPERIMENT_NAME` que publique `mobile_deployment/` como artefacto.
4. Incluir `mobile_deployment/model_info.json` y `ANDROID_INTEGRATION_GUIDE.md` como apéndice en el TFM.

Bloque cita listo para el TFM
---------------------------
"Se convirtió el mejor modelo del experimento `mobilenetv4_fixed_v2_20250827_225128` a TensorFlow Lite usando `convert_to_mobile.py`. La conversión se realizó con TFLite BUILTINS, produciendo un archivo de 1.94 MB con accuracy test 96.27% y 1,032,707 parámetros. Se generaron `model_info.json` y `ANDROID_INTEGRATION_GUIDE.md` para integrar el modelo en la app Android."

Contacto / reproducibilidad
--------------------------
- Script: `tomato_dataset/convert_to_mobile.py` (contiene las decisiones, flags y fallbacks).
- Experimento original: `experiments/mobilenetv4_fixed_v2_20250827_225128`.
- Entorno: documentar `python -V` y `python -c "import tensorflow as tf; print(tf.__version__)"` en el apéndice del TFM.

---

