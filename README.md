
```markdown
# TFM - Diagnóstico de Enfermedades en Hojas de Tomate con Deep Learning y Despliegue Móvil

## Descripción
Sistema completo para la detección automática de enfermedades en hojas de tomate usando Computer Vision y modelos de Deep Learning, optimizado para despliegue en dispositivos Android.

## Estructura del Proyecto
```
py/                  # Código fuente Python
  preprocessing/     # Scripts de preprocesamiento de datos
  processing/        # Scripts de procesamiento y agrupación de patrones
  train/             # Entrenamiento de modelos (DenseNet, MobileNetV4)
  android_app/       # App Android (Java, assets, res)
  convert_to_mobile.py # Conversión de modelos a TFLite
  inference_metrics.csv # Métricas de inferencia
reports/             # Resultados, experimentos, análisis y documentación
  experiments/       # Resultados de experimentos y pruebas
  mobile_deployment/ # Guía y archivos para despliegue móvil
  mobile_testing_results/ # Resultados de pruebas en dispositivos
  ...
tomato_dataset/      # Scripts y documentación adicional
README.md            # Este archivo
requirements.txt     # Dependencias Python
```

## Instalación
1. Clona el repositorio:
   ```bash
   git clone <URL del repositorio>
   ```
2. Instala el entorno Python:
   ```bash
   python3 -m venv myenvtfm
   source myenvtfm/bin/activate
   pip install -r requirements.txt
   ```
3. (Opcional) Instala Android Studio para compilar la app móvil.

## Ejecución
- **Preprocesamiento:**
  ```bash
  python py/preprocessing/colisiones_uuid.py
  # Ejecuta los scripts necesarios según README_ENTRENAMIENTO.md
  ```
- **Entrenamiento:**
  ```bash
  python py/train/mobilenetv4_clean.py
  python py/train/densenet.py
  ```
- **Conversión a TFLite:**
  ```bash
  python py/convert_to_mobile.py
  ```
- **Despliegue Android:**
  - Sigue la guía en `reports/experiments/mobile_deployment/ANDROID_INTEGRATION_GUIDE.md`

## Reproducibilidad
- Todos los scripts tienen configuración al inicio (ver instrucciones en `.github/instructions/ai_.instructions.md`).
- Los resultados y modelos están en `reports/experiments/`.
- Documentación técnica y científica en `reports/` y `tomato_dataset/`.

## Contacto y Licencia
- Autor: [Tu Nombre]
- Licencia: [Selecciona una licencia, por ejemplo MIT]

---
```
