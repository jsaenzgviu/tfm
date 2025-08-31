# Guía de Integración Android - Tomato Disease Detection

## 📱 Requisitos del Sistema
- Android API 21+ (Android 5.0 Lollipop)
- Cámara del dispositivo
- Mínimo 2GB RAM
- 50MB de almacenamiento libre

**Nota importante (JDK/Gradle):** El Android Gradle Plugin moderno requiere Java 17. Asegúrate de que Android Studio/Gradle use JDK 17.
- En Android Studio: File → Settings → Build, Execution, Deployment → Build Tools → Gradle → Gradle JDK → seleccionar JDK 17.
- Alternativamente, configura `org.gradle.java.home=/ruta/a/jdk17` en `~/.gradle/gradle.properties` o en el `gradle.properties` del proyecto.

## 📦 Archivos Necesarios
- `tomato_disease_mobilenetv4.tflite` - Modelo TensorFlow Lite
- `model_info.json` - Metadatos del modelo

## 🔧 Dependencias de Android Studio

### build.gradle (Module: app)
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.16.1'
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1'  // Para TF Select (solo si es necesario)
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.16.1'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'androidx.camera:camera-camera2:1.3.0'
    implementation 'androidx.camera:camera-lifecycle:1.3.0'
    implementation 'androidx.camera:camera-view:1.3.0'
}
```

### AndroidManifest.xml
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-feature android:name="android.hardware.camera" android:required="true" />
```

## 🚀 Implementación Básica

(Se incluye ejemplo de carga / preprocesado / inferencia en la guía completa del repo)

## 📱 Optimizaciones para Producción

- GPU: usar `GpuDelegate()` en `Interpreter.Options()` si se desea acelerar en dispositivos compatibles.
- NNAPI: `options.setUseNNAPI(true)` puede ayudar en dispositivos que soporten NNAPI.

## 🔍 Testing y Validación
1. Probar con imágenes de cada clase de enfermedad
2. Verificar rendimiento en diferentes dispositivos
3. Medir latencia de inferencia
4. Validar precisión vs modelo original

## 📊 Métricas Esperadas
- Precisión: ~95% (ajustar según `mobile_deployment/model_info.json`)
- Latencia: <200ms en dispositivos modernos
- Tamaño del modelo: ~2-4MB (dependiendo de cuantización)
- RAM usage: <100MB durante inferencia
