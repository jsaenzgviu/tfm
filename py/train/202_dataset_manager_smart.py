#!/usr/bin/env python3
"""
Dataset Manager Inteligente - Versión corregida
Soluciona el problema de filtros demasiado agresivos
"""

import os
import json
import cv2
import numpy as np

# ==================================================================================
# CONFIGURACIÓN MEJORADA - MÁS INTELIGENTE Y MENOS AGRESIVA
# ==================================================================================

# Directorios base
SOURCE_DATASET_DIR = "dataset"
TARGET_DATASET_DIR = "dataset_final"
OUTPUT_SEGMENTED_DIR = "dataset_smart"  # Nuevo directorio para versión inteligente

# PARÁMETROS INTELIGENTES - MENOS AGRESIVOS, MÁS ADAPTATIVOS
SMART_PARAMS = {
    # Detección HSV mejorada y más permisiva
    "hsv_ranges": [
        ([20, 20, 20], [95, 255, 255]),  # Verde muy amplio
        ([10, 15, 15], [40, 255, 255]),  # Amarillo-verde permisivo
        ([0, 15, 15], [20, 255, 255]),  # Rojizo muy permisivo
    ],
    # Parámetros LAB más adaptativos
    "lab_threshold": 130,  # Más permisivo (era 127)
    # Fusión menos estricta
    "fusion_threshold": 0.2,  # Más permisivo (era 0.3)
    "weight_hsv": 0.8,  # Dar más peso a HSV
    "weight_lab": 0.2,  # Reducir peso de LAB
    # Morfología MUY conservadora
    "morph_kernel_size": 3,  # Más pequeño (era 5)
    "morph_iterations": 1,  # Menos agresivo (era 3)
    # Filtrado de contornos MUY permisivo
    "min_area_ratio": 0.0001,  # 0.01% en lugar de 0.1%
    "max_area_ratio": 0.95,  # Casi toda la imagen
    "min_compactness": 0.01,  # Muy permisivo (era 0.1)
    # Aplicación de máscara suave
    "background_color": [20, 20, 20],  # Menos agresivo
    "blur_kernel": 3,  # Menos suavizado
    "mask_threshold": 0.1,  # Muy permisivo
    # Mejora de contraste opcional
    "enhance_contrast": False,  # Desactivado para ser más conservador
}

# Configuración de imágenes
TARGET_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# ==================================================================================
# FIN DE CONFIGURACIÓN
# ==================================================================================


class SmartDatasetManager:
    """
    Gestor inteligente que evita la sobre-segmentación
    """

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.source_dir = os.path.join(self.script_dir, SOURCE_DATASET_DIR)
        self.target_dir = os.path.join(self.script_dir, TARGET_DATASET_DIR)
        self.smart_dir = os.path.join(self.script_dir, OUTPUT_SEGMENTED_DIR)

        self.class_names = []
        self._verify_source_dataset()

    def _verify_source_dataset(self):
        """Verificar dataset fuente"""
        source_train = os.path.join(self.source_dir, "train")
        source_test = os.path.join(self.source_dir, "test")

        if not os.path.exists(source_train) or not os.path.exists(source_test):
            raise FileNotFoundError("Dataset fuente no encontrado")

        self.class_names = sorted([d for d in os.listdir(source_train) if os.path.isdir(os.path.join(source_train, d))])

        print("✅ Dataset fuente verificado:")
        print(f"   - Clases: {len(self.class_names)}")

    def create_smart_dataset(self):
        """Crear dataset con segmentación inteligente"""
        print(f"\n🧠 Creando dataset inteligente: {OUTPUT_SEGMENTED_DIR}")
        print("=" * 60)

        source_sets = ["train", "valid", "test"]
        stats = {}

        for set_name in source_sets:
            source_set_dir = os.path.join(self.target_dir, set_name)
            target_set_dir = os.path.join(self.smart_dir, set_name)

            if os.path.exists(source_set_dir):
                stats[set_name] = self._segment_dataset_smart(source_set_dir, target_set_dir, set_name)
            else:
                print(f"⚠️  No encontrado: {source_set_dir}")

        # Guardar estadísticas
        stats_path = os.path.join(self.smart_dir, "smart_segmentation_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n✅ Dataset inteligente creado exitosamente")
        print(f"📊 Estadísticas guardadas en: {stats_path}")

        return stats

    def _segment_dataset_smart(self, source_dir, target_dir, set_name):
        """Segmentar un conjunto con algoritmo inteligente"""
        print(f"\n🧠 Segmentando conjunto: {set_name}")

        os.makedirs(target_dir, exist_ok=True)

        total_processed = 0
        total_success = 0
        total_preserved = 0  # Imágenes que mantuvieron contenido

        for class_name in self.class_names:
            source_class_dir = os.path.join(source_dir, class_name)
            target_class_dir = os.path.join(target_dir, class_name)

            if not os.path.exists(source_class_dir):
                continue

            os.makedirs(target_class_dir, exist_ok=True)

            images = [f for f in os.listdir(source_class_dir) if any(f.lower().endswith(ext.lower()) for ext in VALID_EXTENSIONS)]

            class_processed = 0
            class_success = 0
            class_preserved = 0

            for img_name in images:
                try:
                    source_path = os.path.join(source_class_dir, img_name)
                    target_path = os.path.join(target_class_dir, img_name)

                    # Cargar imagen
                    img = cv2.imread(source_path)
                    if img is None:
                        continue

                    # Redimensionar
                    img_resized = cv2.resize(img, TARGET_SIZE)

                    # SEGMENTACIÓN INTELIGENTE
                    segmented, is_preserved = self._segment_smart(img_resized)

                    # Guardar
                    cv2.imwrite(target_path, segmented)

                    class_success += 1
                    total_success += 1

                    if is_preserved:
                        class_preserved += 1
                        total_preserved += 1

                except Exception as e:
                    print(f"     ❌ Error procesando {img_name}: {e}")

                class_processed += 1
                total_processed += 1

            if class_processed > 0:
                success_rate = (class_success / class_processed) * 100
                preserved_rate = (class_preserved / class_processed) * 100
                print(f"   {class_name}: {class_success}/{class_processed} ({success_rate:.1f}% éxito, {preserved_rate:.1f}% preservado)")

        overall_success_rate = (total_success / total_processed) * 100 if total_processed > 0 else 0
        overall_preserved_rate = (total_preserved / total_processed) * 100 if total_processed > 0 else 0

        print(
            f"✅ {set_name} completado: {total_success}/{total_processed} "
            f"({overall_success_rate:.1f}% éxito, {overall_preserved_rate:.1f}% preservado)"
        )

        return {
            "total_processed": total_processed,
            "total_success": total_success,
            "total_preserved": total_preserved,
            "success_rate": overall_success_rate,
            "preserved_rate": overall_preserved_rate,
        }

    def _segment_smart(self, img):
        """
        Segmentación inteligente que evita la sobre-eliminación
        Retorna: (imagen_segmentada, contenido_preservado)
        """

        # 1. DETECCIÓN PERMISIVA
        mask = self._create_permissive_mask(img)

        # 2. VERIFICACIÓN DE DETECCIÓN MÍNIMA
        detected_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

        # Si la detección es muy baja, usar imagen original
        if detected_ratio < 0.05:  # Menos del 5%
            print(f"      ⚠️  Detección insuficiente ({detected_ratio:.1%}), preservando original")
            return img, False

        # 3. REFINAMIENTO MUY CONSERVADOR
        mask_refined = self._refine_mask_conservative(mask)

        # 4. VERIFICACIÓN POST-REFINAMIENTO
        refined_ratio = np.sum(mask_refined > 0) / (mask_refined.shape[0] * mask_refined.shape[1])

        # Si el refinamiento eliminó demasiado, usar máscara sin refinar
        if refined_ratio < detected_ratio * 0.3:  # Si perdió más del 70%
            print(f"      ⚠️  Refinamiento demasiado agresivo, usando máscara original")
            mask_refined = mask

        # 5. APLICACIÓN INTELIGENTE
        result = self._apply_smart_mask(img, mask_refined)

        # 6. VERIFICACIÓN FINAL
        final_mean = np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
        is_preserved = final_mean > 30  # No completamente negro

        return result, is_preserved

    def _create_permissive_mask(self, img):
        """Crear máscara muy permisiva"""
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Máscara HSV muy permisiva
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for lower, upper in SMART_PARAMS["hsv_ranges"]:
            mask_range = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask_range)

        # Análisis LAB permisivo
        a_channel = lab[:, :, 1]
        _, lab_mask = cv2.threshold(a_channel, SMART_PARAMS["lab_threshold"], 255, cv2.THRESH_BINARY_INV)

        # Fusión con pesos ajustados
        weight_hsv = SMART_PARAMS["weight_hsv"]
        weight_lab = SMART_PARAMS["weight_lab"]

        hsv_norm = combined_mask.astype(np.float32) / 255.0
        lab_norm = lab_mask.astype(np.float32) / 255.0

        fused = weight_hsv * hsv_norm + weight_lab * lab_norm

        # Umbral muy permisivo
        final_mask = (fused > SMART_PARAMS["fusion_threshold"]).astype(np.uint8) * 255

        return final_mask

    def _refine_mask_conservative(self, mask):
        """Refinamiento muy conservador"""

        # Operaciones morfológicas mínimas
        kernel_size = SMART_PARAMS["morph_kernel_size"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Solo opening muy suave para eliminar ruido mínimo
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=SMART_PARAMS["morph_iterations"])

        # Filtrado de contornos MUY permisivo
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return mask

        h, w = mask.shape
        total_area = h * w
        min_area = total_area * SMART_PARAMS["min_area_ratio"]
        max_area = total_area * SMART_PARAMS["max_area_ratio"]
        min_compactness = SMART_PARAMS["min_compactness"]

        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # Verificar compacidad muy permisiva
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter * perimeter)
                    if compactness > min_compactness:
                        valid_contours.append(contour)
                else:
                    valid_contours.append(contour)  # Si no se puede calcular, incluir

        # Si no hay contornos válidos, conservar máscara original
        if not valid_contours:
            print(f"      ⚠️  No hay contornos válidos, conservando máscara original")
            return mask

        # Crear máscara refinada
        refined_mask = np.zeros_like(mask)
        cv2.fillPoly(refined_mask, valid_contours, 255)

        return refined_mask

    def _apply_smart_mask(self, img, mask):
        """Aplicar máscara de forma inteligente"""

        # Suavizar máscara mínimamente
        blur_kernel = SMART_PARAMS["blur_kernel"]
        mask_float = mask.astype(np.float32) / 255.0
        mask_smooth = cv2.GaussianBlur(mask_float, (blur_kernel, blur_kernel), 0.5)

        # Umbral muy permisivo
        mask_threshold = SMART_PARAMS["mask_threshold"]
        mask_binary = (mask_smooth > mask_threshold).astype(np.float32)

        # Imagen base
        result = img.copy().astype(np.float32)

        # Fondo menos agresivo
        background_color = SMART_PARAMS["background_color"]

        # Aplicar máscara suavemente
        for c in range(3):
            result[:, :, c] = result[:, :, c] * mask_binary + background_color[c] * (1 - mask_binary)

        return np.clip(result, 0, 255).astype(np.uint8)


def main():
    """Función principal"""
    print("🧠 Dataset Manager Inteligente - Anti Sobre-Segmentación")
    print("=" * 60)

    try:
        manager = SmartDatasetManager()

        # Crear dataset con segmentación inteligente
        stats = manager.create_smart_dataset()

        print(f"\n📊 RESUMEN FINAL:")
        for set_name, data in stats.items():
            print(f"   {set_name}: {data['total_success']:,} imágenes - {data['preserved_rate']:.1f}% contenido preservado")

        print(f"\n🧠 Dataset inteligente creado en: {OUTPUT_SEGMENTED_DIR}")
        print("   - Segmentación permisiva y adaptativa")
        print("   - Previene eliminación completa de contenido")
        print("   - Refinamiento muy conservador")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
