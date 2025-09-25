
```markdown
# TFM - Desarrollar  un  sistema  para diagnóstico  de  enfermedades en las hojas  de las  plantas  mediante Computer  Vision  y  modelos  de aprendizaje  profundo  para aplicaciones móviles.

Este proyecto forma parte del Trabajo de Fin de Máster (TFM) para el desarrollo de un  sistema  para diagnóstico  de  enfermedades en las hojas  de las  plantas  mediante Computer  Vision  y  modelos  de aprendizaje  profundo  para aplicaciones móviles.

## 📋 Descripción del Proyecto

El sistema permite identificar automáticamente 11 tipos diferentes de enfermedades y estados de salud en plantas de tomate mediante el análisis de imágenes de hojas. Incluye tanto el entrenamiento de modelos de Machine Learning como una aplicación móvil Android para uso en campo.

### Clases Detectadas
- **Bacterial Spot** (Mancha bacteriana)
- **Early Blight** (Tizón temprano)
- **Healthy** (Saludable)
- **Late Blight** (Tizón tardío)
- **Leaf Mold** (Moho de la hoja)
- **Powdery Mildew** (Oídio)
- **Septoria Leaf Spot** (Mancha foliar por Septoria)
- **Spider Mites** (Ácaros araña)
- **Target Spot** (Mancha objetivo)
- **Tomato Mosaic Virus** (Virus del mosaico del tomate)
- **Tomato Yellow Leaf Curl Virus** (Virus del rizado amarillo)

## 🗂️ Estructura del Proyecto

```
├── py/                          # Código Python principal
│   ├── requirements.txt         # Dependencias del proyecto
│   ├── preprocessing/           # Scripts de preprocesamiento de datos
│   ├── processing/              # Scripts de procesamiento avanzado
│   ├── train/                   # Scripts de entrenamiento de modelos
│   └── android_app/             # Aplicación Android
├── tomato_dataset/              # Dataset de imágenes
│   └── dataset_final/           # Dataset final procesado
│       ├── train/               # Conjunto de entrenamiento
│       ├── valid/               # Conjunto de validación
│       └── test/                # Conjunto de prueba
└── reports/                     # Reportes y resultados
    ├── experiments/             # Resultados de experimentos
    ├── inicial_group_analysis/  # Análisis inicial por grupos
    └── normalization/           # Reportes de normalización
```

## 🚀 Configuración del Entorno

### Requisitos del Sistema
- Python 3.8+
- CUDA 11.2+ (para entrenamiento con GPU)
- Android Studio (para la aplicación móvil)
- Al menos 8GB RAM
- GPU con 6GB+ VRAM (recomendado para entrenamiento)

### Instalación
```bash
# Clonar el repositorio
git clone <repository-url>
cd tfm

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r py/requirements.txt
```

## 📊 Dataset

El dataset final contiene **20,664 imágenes** distribuidas como:
- **Entrenamiento**: 13,271 imágenes
- **Validación**: 3,323 imágenes
- **Prueba**: 4,070 imágenes

Las imágenes están balanceadas en el conjunto de prueba (370 imágenes por clase) y desbalanceadas de forma natural en entrenamiento y validación para reflejar la distribución real de las enfermedades.

## 🧠 Scripts de Preprocesamiento

### Fase 1: Normalización y Limpieza (`preprocessing/`)
- `01_colisiones_uuid.py` - Detecta colisiones entre conjuntos de datos
- `02_move_colisiones_uuid.py` - Mueve archivos con colisiones
- `03_same_name.py` - Identifica archivos con nombres duplicados
- `04_uuid_imagenes.py` - Genera UUIDs únicos para imágenes
- `05_revisar_uuid.py` - Revisa la integridad de los UUIDs
- `06_delete_duplicate_uuid_dir.py` - Elimina directorios duplicados
- `07_compare_duplicate_hash.py` - Compara duplicados por hash
- `08_delete_valid_duplicate_hash.py` - Elimina duplicados válidos
- `distribucion_imagenes_clase.py` - Analiza distribución por clase

### Fase 2: Procesamiento Avanzado (`processing/`)
- `101_extract_all_real_patterns.py` - Extrae patrones reales de datos
- `102_hybrid_pattern_grouping.py` - Agrupa patrones híbridos
- `103_move_unique_files.py` - Mueve archivos únicos
- `104_base_file_identifier.py` - Identifica archivos base
- `105_move_base_files.py` - Organiza archivos base
- `106_image_sizes.py` - Analiza tamaños de imagen
- `107_resize_image.py` - Redimensiona imágenes
- `108_create_validation_subset.py` - Crea subconjunto de validación

## 🤖 Modelos de Machine Learning

### Scripts de Entrenamiento (`train/`)

#### Gestión de Datasets
- `201_dataset_manager_clean.py` - Gestor limpio de datasets
- `202_dataset_manager_smart.py` - Gestor inteligente con optimizaciones

#### Modelos Principales
- `203_densenet.py` - **DenseNet121** con transfer learning
- `mobilenetv4_clean.py` - **MobileNetV4** optimizado para móviles

#### Utilidades
- `204_fixed_callback_training_v2.py` - Callbacks avanzados de entrenamiento
- `205_convert_to_mobile.py` - Conversión a TensorFlow Lite

### Arquitecturas Implementadas

#### DenseNet121
- Pre-entrenado en ImageNet
- Fine-tuning completo
- Optimizaciones AdamW
- Early Stopping y ReduceLROnPlateau
- Accuracy objetivo: >95%

#### MobileNetV4 + Knowledge Distillation
- Modelo ligero para dispositivos móviles
- Knowledge Distillation desde DenseNet121
- Optimizado para inferencia rápida
- Conversión automática a TensorFlow Lite

## 📱 Aplicación Android

### Características
- **Detección en tiempo real** de enfermedades
- **Interfaz intuitiva** para agricultores
- **Funcionamiento offline** (modelo embebido)
- **Captura de fotos** desde cámara o galería
- **Resultados con confianza** y recomendaciones

### Tecnologías
- Android SDK 34
- TensorFlow Lite
- Java 17
- Material Design

### Construcción
```bash
cd py/android_app
./gradlew assembleDebug
# APK generado en: app/build/outputs/apk/debug/
```

## 📈 Resultados y Métricas

Los experimentos y resultados se almacenan en `reports/experiments/`:
- **DenseNet121**: Múltiples experimentos con diferentes configuraciones
- **MobileNetV4**: Versiones optimizadas para móviles
- **Métricas**: Accuracy, Precision, Recall, F1-Score por clase
- **Visualizaciones**: Matrices de confusión, curvas de entrenamiento

## 🔧 Uso del Sistema

### Entrenamiento de Modelos
```bash
# DenseNet121
cd py/train
python 203_densenet.py

# MobileNetV4
python mobilenetv4_clean.py

# Conversión a móvil
python 205_convert_to_mobile.py
```

### Preprocesamiento de Datos
```bash
cd py/preprocessing
# Ejecutar scripts en orden numérico
python 01_colisiones_uuid.py
python 02_move_colisiones_uuid.py
# ... continuar secuencialmente
```

## 📋 Pipeline Completo

1. **Preprocesamiento** (`01_*` - `08_*`): Limpieza y normalización
2. **Procesamiento** (`101_*` - `108_*`): Análisis y organización avanzada
3. **Entrenamiento** (`201_*` - `205_*`): Modelos y optimización
4. **Despliegue**: Aplicación Android con TensorFlow Lite

## 🏆 Objetivos del Proyecto

- ✅ **Accuracy >95%** en clasificación de enfermedades
- ✅ **Aplicación móvil funcional** para uso en campo
- ✅ **Procesamiento offline** sin conexión a internet
- ✅ **Pipeline automatizado** de preprocesamiento
- ✅ **Modelos optimizados** para dispositivos móviles

## 📚 Dependencias Principales

- **TensorFlow 2.19.0** - Framework de Deep Learning
- **OpenCV 4.10.0** - Procesamiento de imágenes
- **Scikit-learn 1.7.1** - Métricas y evaluación
- **Matplotlib/Seaborn** - Visualización
- **NumPy/Pandas** - Manipulación de datos

## 👥 Contribuciones

Este proyecto es parte de un TFM académico.

## 📄 Licencia

Proyecto académico - Universidad Internacional de Valencia (VIU)

---

*Desarrollado como parte del Trabajo de Fin de Máster en Big Data y Ciencia de Datos*
