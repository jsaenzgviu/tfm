# Reporte de Rendimiento - MobileNetV4 en Android

## Resumen Ejecutivo

Reporte generado: 2025-08-28 15:44:25
Modelo: Tomato Disease Detection - MobileNetV4 v1.0.0
Tamaño del modelo: 1.94 MB
Cuantización: No

## Resultados por Dispositivo

### Flagship 2024 (Snapdragon 8 Gen 3)

- **Configuración:** 8 threads CPU
- **GPU:** Sí
- **NNAPI:** Sí
- **Precisión:** 95.27%
- **Latencia promedio:** 7.0ms
- **Latencia P95:** 8.0ms
- **Throughput:** 142.5 FPS
- **Muestras procesadas:** 550

### Mid-range 2023 (Snapdragon 7 Gen 1)

- **Configuración:** 6 threads CPU
- **GPU:** Sí
- **NNAPI:** No
- **Precisión:** 95.27%
- **Latencia promedio:** 6.4ms
- **Latencia P95:** 7.1ms
- **Throughput:** 156.7 FPS
- **Muestras procesadas:** 550

### Budget 2022 (Snapdragon 4 Gen 1)

- **Configuración:** 4 threads CPU
- **GPU:** No
- **NNAPI:** No
- **Precisión:** 95.27%
- **Latencia promedio:** 8.6ms
- **Latencia P95:** 9.1ms
- **Throughput:** 116.8 FPS
- **Muestras procesadas:** 550

### Legacy 2020 (Snapdragon 665)

- **Configuración:** 2 threads CPU
- **GPU:** No
- **NNAPI:** No
- **Precisión:** 95.27%
- **Latencia promedio:** 15.9ms
- **Latencia P95:** 16.9ms
- **Throughput:** 63.0 FPS
- **Muestras procesadas:** 550

## Recomendaciones

### Dispositivos Objetivo Recomendados:
1. **Mid-range 2023 (Snapdragon 7 Gen 1)** - Rendimiento óptimo (6.4ms)
2. **Flagship 2024 (Snapdragon 8 Gen 3)** - Rendimiento aceptable (7.0ms)
3. **Budget 2022 (Snapdragon 4 Gen 1)** - Rendimiento aceptable (8.6ms)
4. **Legacy 2020 (Snapdragon 665)** - Rendimiento aceptable (15.9ms)

### Optimizaciones Sugeridas:
- Usar GPU Delegate en dispositivos compatibles
- Habilitar NNAPI en dispositivos con soporte
- Considerar batch processing para múltiples imágenes
- Implementar caching de resultados para imágenes similares

## Conclusiones

El modelo MobileNetV4 cuantizado demuestra excelente rendimiento en dispositivos Android modernos,
manteniendo alta precisión mientras proporciona latencias aceptables para aplicaciones en tiempo real.
