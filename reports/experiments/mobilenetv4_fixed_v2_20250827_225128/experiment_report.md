# Reporte del Experimento: mobilenetv4_fixed_v2_20250827_225128

## Configuración del Experimento
- **Modelo**: MobileNetV4-medium con Knowledge Distillation
- **Teacher**: densenet121
- **Dataset**: dataset_final
- **Fecha**: 2025-08-27 23:58:13

## Configuración de Entrenamiento
- **Épocas**: 40
- **Batch Size**: 16
- **Learning Rate**: 0.001
- **Temperature**: 4.0
- **Alpha**: 0.3

## Resultados Finales
- **Test Accuracy**: 0.9627 (96.27%)
- **Test Loss**: 0.1343
- **Parámetros del Modelo**: 1,032,707

## Archivos Generados
- Configuración: `experiment_config.json`
- Métricas: `final_metrics.json`
- Distribución de clases: `class_distribution.json`
- Log de entrenamiento: `training_log.csv`
- Gráficos: `training_history.png`, `confusion_matrix.png`
- Modelo: `student_model.weights_complete.h5`

## Directorio del Experimento
```
/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/experiments/mobilenetv4_fixed_v2_20250827_225128
```
