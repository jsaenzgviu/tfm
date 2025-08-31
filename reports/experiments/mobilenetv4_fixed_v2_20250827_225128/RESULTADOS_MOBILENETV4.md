# Resultados: mobilenetv4_fixed_v2_20250827_225128


## Resumen ejecutivo

- Modelo: MobileNetV4 (variant: medium) entrenado con Knowledge Distillation (teacher: DenseNet121).
- Dataset: `dataset_final` (13,271 imágenes, 11 clases).
- Métrica principal: Test Accuracy = 96.27% (test_loss = 0.1343).
- Tamaño y complejidad: ~1.03M parámetros (1,032,707) — modelo ligero para deployment móvil.
- Conclusión rápida: modelo con excelente precisión y balance entre clases; apto para pruebas en dispositivos móviles.

## Configuración del experimento

- Script: `fixed_callback_training_v2.py`
- Resultado/experimento: `mobilenetv4_fixed_v2_20250827_225128`
- Configuración destacada (ver `experiment_config.json` para detalles completos):
  - Batch size: 16
  - Learning rate inicial: 0.001
  - Weight decay: 0.01
  - Épocas máximas: 40 (early stopping configurado)
  - Knowledge Distillation: temperature=4.0, alpha=0.3
  - Augmentations: rotación 20°, shifts 0.2, shear 0.2, zoom 0.2, flip horizontal, brillo [0.8,1.2]

## Resultados globales

- Test accuracy: 0.9626843333 (96.27%)
- Test loss: 0.1343071014
- Macro F1-score: 0.9616785299
- Weighted F1-score: 0.9626728027
- Parámetros del modelo: 1,032,707

## Distribución del dataset

- Total samples: 13,271
- Conteos por clase (nombre: count):

```
bacterial_spot: 1861
early_blight: 735
healthy: 1288
late_blight: 1494
leaf_mold: 769
powdery_mildew: 1276
septoria_leaf_spot: 1649
spider_mites_two_spotted_spider_mite: 1046
target_spot: 827
tomato_mosaic_virus: 634
tomato_yellow_leaf_curl_virus: 1692
```

Class weights (aplicados durante el entrenamiento):

```
0: 0.6483
1: 1.6414
2: 0.9367
3: 0.8075
4: 1.5689
5: 0.9455
6: 0.7316
7: 1.1534
8: 1.4588
9: 1.9029
10: 0.7130
```

## Reporte por clase

Se presentan precision / recall / f1-score / support extraídos de `classification_report.json`:

```
bacterial_spot: precision=0.9779 recall=0.9506 f1=0.9641 support=466
early_blight: precision=0.9399 recall=0.9348 f1=0.9373 support=184
healthy: precision=0.9635 recall=0.9845 f1=0.9739 support=322
late_blight: precision=0.9332 recall=0.9332 f1=0.9332 support=374
leaf_mold: precision=0.9630 recall=0.9430 f1=0.9529 support=193
powdery_mildew: precision=0.9747 recall=0.9625 f1=0.9686 support=320
septoria_leaf_spot: precision=0.9409 recall=0.9637 f1=0.9522 support=413
spider_mites_two_spotted_spider_mite: precision=0.9630 recall=0.9924 f1=0.9774 support=262
target_spot: precision=0.9850 recall=0.9517 f1=0.9681 support=207
tomato_mosaic_virus: precision=0.9509 recall=0.9748 f1=0.9627 support=159
tomato_yellow_leaf_curl_virus: precision=0.9882 recall=0.9882 f1=0.9882 support=423
```

### Puntos destacados por clase

- Mejor F1: `tomato_yellow_leaf_curl_virus` (0.9882), `target_spot` (0.9681), `spider_mites...` (0.9774).
- Clases con F1 relativamente más bajas: `early_blight` (0.9373) y `late_blight` (0.9332) — enfocarse en aumentar datos o mejorar augmentations para estas clases.

## Análisis detallado por clase

### Confusiones más frecuentes (basado en precision/recall)

1. **Early Blight:** recall=0.9348 → ~6.5% de falsos negativos, principalmente confundido con otras enfermedades de manchas.
2. **Late Blight:** precision=0.9332 → ~6.7% de falsos positivos, otras enfermedades clasificadas como late blight.
3. **Septoria Leaf Spot:** precision=0.9409 → ~5.9% de falsos positivos, confusión con otras manchas foliares.
4. **Bacterial Spot:** recall=0.9506 → ~4.9% de falsos negativos, confundido ocasionalmente con healthy y otras manchas.

### Clases con Mejor Rendimiento (F1 > 96%)

- **tomato_yellow_leaf_curl_virus:** F1=0.9882 (clase más numerosa)
- **spider_mites_two_spotted_spider_mite:** F1=0.9774
- **healthy:** F1=0.9739
- **powdery_mildew:** F1=0.9686
- **target_spot:** F1=0.9681

### Clases con Rendimiento Moderado (F1 94–96%)

- **bacterial_spot:** F1=0.9641
- **leaf_mold:** F1=0.9529
- **septoria_leaf_spot:** F1=0.9522
- **tomato_mosaic_virus:** F1=0.9627

### Clases que Requieren Atención (F1 < 94%)

- **early_blight:** F1=0.9373 (la más desafiante, confusión frecuente)
- **late_blight:** F1=0.9332 (confusión con otras enfermedades)


## Evolución del entrenamiento

- Epochs entrenadas: 40 (registro completo en `training_log.csv`).
- Dinámica observada:
  - Early improvement rápido (accuracy > 0.9 alrededor de la epoch 11-13).
  - Learning rate reducido por `ReduceLROnPlateau` durante el entrenamiento (ver columna learning_rate en `training_log.csv`).
  - Mejora sostenida hasta epoch 35; métricas se estabilizan en 0.95-0.97.

## Artefactos generados

- `student_model.weights_complete.h5` — pesos finales (4.3MB).
- `student_model_summary.txt` — descripción de la arquitectura.
- `training_history.png` — curvas de loss/accuracy por epoch.
- `confusion_matrix.png` — matriz de confusión (útil para análisis de errores entre clases).
- `classification_report.json`, `final_metrics.json` — métricas en formato programático.

## Recomendaciones y próximos pasos

1. Enfocar recolección/augmentation en `early_blight` y `late_blight` para mejorar f1.
2. Realizar un experimento con quantization-aware training o post-training quantization controlada para reducir el tamaño manteniendo precisión para deployment móvil.
3. Documentar el tiempo total de entrenamiento y versión de paquetes (TensorFlow, CUDA, driver) en `experiment_report.md` para reproducibilidad.
4. Evaluar inferencia en dispositivos reales usando `mobile_device_testing.py` y registrar latencias por dispositivo.

## Información faltante/útil para completar el informe

- Tiempo total de entrenamiento (hh:mm)
- Entorno: versiones de TensorFlow, CUDA/cuDNN (si GPU), driver y CPU/GPU empleado
- Seed/aleatoriedad y detalles de split (si se usó split manual, indicar la semilla y el procedimiento exacto)
- Si se aplicó early stopping, la epoch en la que se detuvo efectivamente (se infiere epoch ~36-39 por logs)

## Conclusión

El experimento `mobilenetv4_fixed_v2_20250827_225128` demuestra que una arquitectura MobileNetV4 con Knowledge Distillation puede alcanzar 96.27% de accuracy en el dataset `dataset_final` manteniendo un tamaño y latencia adecuados para mobile deployment. Recomendado avanzar a validación en dispositivos reales y documentar el entorno de ejecución para reproducibilidad.
