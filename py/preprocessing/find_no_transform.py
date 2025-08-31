import os
import numpy as np
from matplotlib.image import imread
from scipy.ndimage import zoom
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
import hashlib
import multiprocessing as mp
import concurrent.futures
from threading import Lock
import time
import csv
from datetime import datetime

# Intentar importar bibliotecas para GPU
try:
    import cupy as cp

    # FORZAR DESHABILITACIÓN DE GPU PARA TESTING
    GPU_AVAILABLE = True
    print("🚀 GPU (CuPy) disponible para aceleración")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU no disponible, usando CPU optimizada")

# Configuración de rendimiento
NUM_WORKERS = min(mp.cpu_count(), 10)  # Limitar workers para evitar sobrecarga
BATCH_SIZE = 200  # 🧪 EXPERIMENTO: Batch muy grande
HASH_CACHE = {}  # Cache global para hashes
CACHE_LOCK = Lock()

print("🔧 Configuración de rendimiento:")
print(f"   - Workers: {NUM_WORKERS}")
print(f"   - Batch size: {BATCH_SIZE}")
print(f"   - GPU: {'Sí' if GPU_AVAILABLE else 'No'}")

# Directorio a analizar para transformaciones
target_dir = "/home/xxxx/share/VIU/14tfm/06_actividad/tomato_dataset/test1"

# Extraer el nombre del directorio para el informe
dir_name = os.path.basename(target_dir)


def compute_dhash_fast(img, hash_size=8):
    """Versión optimizada de dhash usando GPU si está disponible."""
    if img.ndim == 3:
        gray = np.mean(img, axis=-1)
    else:
        gray = img

    if GPU_AVAILABLE:
        try:
            # Usar GPU para cálculos
            gray_gpu = cp.asarray(gray)

            # Redimensionar usando GPU
            h, w = gray_gpu.shape
            scale_h = hash_size / h
            scale_w = (hash_size + 1) / w

            # Interpolación bilineal rápida en GPU
            resized = cp.ndimage.zoom(gray_gpu, (scale_h, scale_w), order=1)

            # Calcular diferencias
            diff = resized[:, :-1] > resized[:, 1:]

            # Convertir a hash
            hash_bits = cp.asnumpy(diff.flatten())

        except Exception:
            # Fallback a CPU si hay error en GPU
            h, w = gray.shape
            resized = zoom(gray, (hash_size / h, (hash_size + 1) / w))
            diff = resized[:, :-1] > resized[:, 1:]
            hash_bits = diff.flatten()
    else:
        # Versión CPU optimizada
        h, w = gray.shape
        resized = zoom(gray, (hash_size / h, (hash_size + 1) / w))
        diff = resized[:, :-1] > resized[:, 1:]
        hash_bits = diff.flatten()

    # Convertir a entero de forma eficiente
    hash_val = 0
    for b in hash_bits:
        hash_val = (hash_val << 1) | (1 if b else 0)

    return hash_val


def compute_phash_fast(img, hash_size=8):
    """Versión optimizada de phash usando GPU si está disponible."""
    if img.ndim == 3:
        gray = np.mean(img, axis=-1)
    else:
        gray = img

    if GPU_AVAILABLE:
        try:
            # Usar GPU para DCT
            gray_gpu = cp.asarray(gray)
            h, w = gray_gpu.shape
            resized = cp.ndimage.zoom(gray_gpu, (hash_size / h, hash_size / w), order=1)

            # DCT usando FFT en GPU
            dct = cp.fft.fft2(resized)
            dct_low = dct[: hash_size // 2, : hash_size // 2]

            # Mediana en GPU
            median = cp.median(cp.real(dct_low))

            # Hash binario
            hash_bits = cp.asnumpy(cp.real(dct_low).flatten() > median)

        except Exception:
            # Fallback a CPU
            h, w = gray.shape
            resized = zoom(gray, (hash_size / h, hash_size / w))
            dct = np.fft.fft2(resized)
            dct_low = dct[: hash_size // 2, : hash_size // 2]
            median = np.median(np.real(dct_low))
            hash_bits = np.real(dct_low).flatten() > median
    else:
        # Versión CPU
        h, w = gray.shape
        resized = zoom(gray, (hash_size / h, hash_size / w))
        dct = np.fft.fft2(resized)
        dct_low = dct[: hash_size // 2, : hash_size // 2]
        median = np.median(np.real(dct_low))
        hash_bits = np.real(dct_low).flatten() > median

    # Convertir a entero
    hash_val = 0
    for b in hash_bits:
        hash_val = (hash_val << 1) | (1 if b else 0)

    return hash_val


def compute_file_hash(filepath):
    """Computa hash MD5 del archivo para detectar duplicados exactos."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def hamming_distance(hash1, hash2):
    """Calcula la distancia de Hamming entre dos hashes."""
    return bin(hash1 ^ hash2).count("1")


def normalize_image_fast(img):
    """Versión optimizada de normalización usando GPU si está disponible."""
    if img.ndim == 3:
        gray = np.mean(img, axis=-1)
    else:
        gray = img

    if GPU_AVAILABLE:
        try:
            gray_gpu = cp.asarray(gray, dtype=cp.float32)
            gray_gpu = (gray_gpu - cp.min(gray_gpu)) / (cp.max(gray_gpu) - cp.min(gray_gpu) + 1e-8)
            return cp.asnumpy(gray_gpu)
        except Exception:
            pass

    # CPU version
    gray = gray.astype(np.float32)
    return (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)


def compute_image_stats_fast(img):
    """Versión optimizada de estadísticas usando GPU si está disponible."""
    norm_img = normalize_image_fast(img)

    if GPU_AVAILABLE:
        try:
            img_gpu = cp.asarray(norm_img)
            return {
                "mean": float(cp.mean(img_gpu)),
                "std": float(cp.std(img_gpu)),
                "median": float(cp.median(img_gpu)),
                "min": float(cp.min(img_gpu)),
                "max": float(cp.max(img_gpu)),
            }
        except Exception:
            pass

    # CPU version
    return {
        "mean": np.mean(norm_img),
        "std": np.std(norm_img),
        "median": np.median(norm_img),
        "min": np.min(norm_img),
        "max": np.max(norm_img),
    }


def verify_match_robust_fast(img1, trans_func, img2, debug=False):
    """Verificación robusta optimizada usando GPU para cálculos intensivos."""
    try:
        trans_img1 = trans_func(img1)

        # Normalizar ambas imágenes con GPU
        norm_img1 = normalize_image_fast(trans_img1)
        norm_img2 = normalize_image_fast(img2)

        # Verificar aspect ratio (CPU es suficiente)
        h1, w1 = norm_img1.shape
        h2, w2 = norm_img2.shape
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2

        if abs(aspect_ratio1 - aspect_ratio2) > 0.1:
            if debug:
                print(f"    Aspect ratios muy diferentes: {aspect_ratio1:.3f} vs {aspect_ratio2:.3f}")
            return False

        # Redimensionar para comparar
        if norm_img1.shape != norm_img2.shape:
            norm_img1 = zoom(norm_img1, (h2 / h1, w2 / w1))

        # Usar GPU para cálculos intensivos si está disponible
        if GPU_AVAILABLE:
            try:
                img1_gpu = cp.asarray(norm_img1)
                img2_gpu = cp.asarray(norm_img2)

                # MSE en GPU
                mse = float(cp.mean((img1_gpu - img2_gpu) ** 2))

                # MAD en GPU
                mad = float(cp.mean(cp.abs(img1_gpu - img2_gpu)))

                # Correlación en GPU
                flat1 = img1_gpu.flatten()
                flat2 = img2_gpu.flatten()
                corr = float(cp.corrcoef(flat1, flat2)[0, 1])
                if cp.isnan(corr):
                    corr = -1

            except Exception:
                # Fallback a CPU
                mse = np.mean((norm_img1 - norm_img2) ** 2)
                mad = np.mean(np.abs(norm_img1 - norm_img2))
                corr = np.corrcoef(norm_img1.flatten(), norm_img2.flatten())[0, 1]
                if np.isnan(corr):
                    corr = -1
        else:
            # CPU version
            mse = np.mean((norm_img1 - norm_img2) ** 2)
            mad = np.mean(np.abs(norm_img1 - norm_img2))
            corr = np.corrcoef(norm_img1.flatten(), norm_img2.flatten())[0, 1]
            if np.isnan(corr):
                corr = -1

        # SSIM (mantener en CPU por ahora)
        ssim_val = ssim(norm_img1, norm_img2, data_range=1.0)

        # Estadísticas optimizadas
        stats1 = compute_image_stats_fast(trans_img1)
        stats2 = compute_image_stats_fast(img2)
        stats_diff = abs(stats1["mean"] - stats2["mean"]) + abs(stats1["std"] - stats2["std"])

        if debug:
            print(f"    SSIM: {ssim_val:.4f} (needed: >0.85)")
            print(f"    MSE: {mse:.4f} (needed: <0.015)")
            print(f"    Correlación: {corr:.4f} (needed: >0.9)")
            print(f"    MAD: {mad:.4f} (needed: <0.12)")
            print(f"    Stats diff: {stats_diff:.4f} (needed: <0.12)")

        # Mismos criterios estrictos
        is_similar = ssim_val > 0.85 and mse < 0.015 and corr > 0.9 and mad < 0.12 and stats_diff < 0.12

        return is_similar

    except Exception as e:
        if debug:
            print(f"    Error en verify_match: {e}")
        return False


# Transformaciones básicas
transforms = {
    "identity": lambda x: x,
    "mirror_horizontal": np.fliplr,
    "mirror_vertical": np.flipud,
    "rotate_90": lambda x: np.rot90(x, 1),
    "rotate_180": lambda x: np.rot90(x, 2),
    "rotate_270": lambda x: np.rot90(x, 3),
}


def load_image_safe(filepath):
    """Carga imagen de forma segura."""
    try:
        img = imread(filepath)
        if img is None:
            return None
        return img
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return None


def process_image_hashes(args):
    """Función worker para procesar hashes de una imagen con todas las transformaciones."""
    filepath, filename = args

    # Verificar cache primero
    cache_key = f"{filename}_{os.path.getmtime(filepath)}"
    with CACHE_LOCK:
        if cache_key in HASH_CACHE:
            return HASH_CACHE[cache_key]

    try:
        img = load_image_safe(filepath)
        if img is None:
            return None

        results = []
        for trans_name, trans_func in transforms.items():
            try:
                trans_img = trans_func(img)
                dhash = compute_dhash_fast(trans_img)
                phash = compute_phash_fast(trans_img)
                results.append((filename, trans_name, dhash, phash))
            except Exception as e:
                print(f"Error procesando {filename} con {trans_name}: {e}")
                continue

        # Guardar en cache
        with CACHE_LOCK:
            HASH_CACHE[cache_key] = results

        return results
    except Exception as e:
        print(f"Error procesando {filepath}: {e}")
        return None


def preload_batch_images(file_list, target_directory):
    """Pre-carga todas las imágenes únicas de un batch en memoria."""
    image_cache = {}
    for filename in file_list:
        if filename not in image_cache:
            path = os.path.join(target_directory, filename)
            img = load_image_safe(path)
            if img is not None:
                image_cache[filename] = img
    return image_cache


def process_batch_verification(args):
    """Función worker OPTIMIZADA para verificar un batch de candidatos con pre-carga de imágenes."""
    batch_data, target_directory = args
    results = []

    # 🚀 OPTIMIZACIÓN: Extraer lista de archivos únicos para pre-cargar
    unique_files = set()
    for file1, file2, trans_name, distance in batch_data:
        unique_files.add(file1)
        unique_files.add(file2)

    # 🚀 OPTIMIZACIÓN: Pre-cargar todas las imágenes únicas en memoria
    image_cache = preload_batch_images(unique_files, target_directory)

    for file1, file2, trans_name, distance in batch_data:
        try:
            # 🚀 OPTIMIZACIÓN: Usar cache en lugar de cargar desde disco
            img1 = image_cache.get(file1)
            img2 = image_cache.get(file2)

            if img1 is None or img2 is None:
                continue

            trans_func = transforms[trans_name]
            if verify_match_robust_fast(img1, trans_func, img2, debug=False):
                results.append((file1, file2, trans_name))

        except Exception as e:
            print(f"Error verificando {file1} vs {file2}: {e}")
            continue

    return results


def find_transformations_in_directory():
    """Analiza un directorio para identificar imágenes con transformaciones usando la lógica del código original."""
    start_time = time.time()

    # Lista de imágenes
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    all_files = [f for f in os.listdir(target_dir) if f.lower().endswith(extensions)]

    log_messages = []
    log_messages.append("� ANÁLISIS DE TRANSFORMACIONES INICIADO")
    log_messages.append(f"Directorio: {target_dir}")
    log_messages.append(f"Total de archivos: {len(all_files)}")
    print("� ANÁLISIS DE TRANSFORMACIONES INICIADO")
    print(f"Directorio: {target_dir}")
    print(f"Total de archivos: {len(all_files)}")

    # Función para extraer número de enfermedad (del código original)
    def extract_disease_number(filename):
        import re

        match = re.search(r"_pm(\d+)", filename)
        return match.group(1) if match else None

    # PASO 1: Detectar duplicados exactos (adaptado para un directorio)
    log_messages.append("\n=== PASO 1: Verificando duplicados exactos ===")
    print("\n=== PASO 1: Verificando duplicados exactos ===")

    # Procesar hashes de archivos en paralelo
    def compute_file_hash_pair(x):
        return (x[1], compute_file_hash(x[0]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        file_paths = [(os.path.join(target_dir, f), f) for f in all_files]
        file_hashes = dict(executor.map(compute_file_hash_pair, file_paths))

    # Buscar duplicados exactos dentro del mismo directorio
    exact_matches = []
    processed_files = set()
    hash_to_file = {}

    for filename, file_hash in file_hashes.items():
        if file_hash in hash_to_file:
            # Duplicado exacto encontrado
            original_file = hash_to_file[file_hash]
            exact_matches.append((original_file, filename, "exact_copy"))
            processed_files.add(filename)
            log_msg = f"  Duplicado exacto: {filename} == {original_file}"
            log_messages.append(log_msg)
            print(log_msg)
        else:
            hash_to_file[file_hash] = filename

    exact_time = time.time() - start_time
    log_msg = f"Duplicados exactos encontrados: {len(exact_matches)} (⏱️ {exact_time:.2f}s)"
    log_messages.append(log_msg)
    print(log_msg)

    # PASO 2: Detectar duplicados perceptuales (adaptado)
    step2_start = time.time()
    log_messages.append("\n=== PASO 2: Verificando duplicados perceptuales ===")
    print("\n=== PASO 2: Verificando duplicados perceptuales ===")

    # Procesar hashes perceptuales en paralelo
    log_messages.append("🔄 Calculando hashes perceptuales en paralelo...")
    print("🔄 Calculando hashes perceptuales en paralelo...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        file_args = [(os.path.join(target_dir, f), f) for f in all_files]
        hash_results = list(executor.map(process_image_hashes, file_args))

    # Crear índices optimizados
    dhash_index = defaultdict(list)
    phash_index = defaultdict(list)

    for result in hash_results:
        if result:
            for filename, trans_name, dhash, phash in result:
                dhash_index[dhash].append((filename, trans_name))
                phash_index[phash].append((filename, trans_name))

    index_time = time.time() - step2_start
    log_msg = f"📊 Índices creados en {index_time:.2f}s"
    log_messages.append(log_msg)
    print(log_msg)

    # Buscar candidatos dentro del mismo directorio
    verification_candidates = []

    for i, file1 in enumerate(all_files):
        if i % 100 == 0:  # Progreso cada 100 archivos
            progress_msg = f"🔍 Buscando candidatos: {i}/{len(all_files)}"
            print(progress_msg)

        # Saltar duplicados exactos
        if any(match[1] == file1 for match in exact_matches):
            continue

        path1 = os.path.join(target_dir, file1)
        img1 = load_image_safe(path1)
        if img1 is None:
            continue

        dhash1 = compute_dhash_fast(img1)
        phash1 = compute_phash_fast(img1)

        # Buscar candidatos en dhash
        for hash_val, candidates in dhash_index.items():
            distance = hamming_distance(dhash1, hash_val)
            if distance <= 5:
                for file2, trans_name in candidates:
                    if file1 != file2:  # No comparar consigo mismo
                        verification_candidates.append((file1, file2, trans_name, distance))

        # Buscar candidatos en phash
        for hash_val, candidates in phash_index.items():
            distance = hamming_distance(phash1, hash_val)
            if distance <= 5:
                for file2, trans_name in candidates:
                    if file1 != file2:  # No comparar consigo mismo
                        verification_candidates.append((file1, file2, trans_name, distance))

    candidate_time = time.time() - step2_start - index_time
    log_msg = f"🎯 {len(verification_candidates)} candidatos encontrados en {candidate_time:.2f}s"
    log_messages.append(log_msg)
    print(log_msg)

    # Verificar candidatos en batches paralelos
    perceptual_matches = []
    if verification_candidates:
        log_messages.append("✅ Verificando candidatos en paralelo...")
        print("✅ Verificando candidatos en paralelo...")

        # Dividir en batches
        batches = []
        for i in range(0, len(verification_candidates), BATCH_SIZE):
            batch = verification_candidates[i : i + BATCH_SIZE]
            batches.append((batch, target_dir))

        # Procesar batches en paralelo
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_results = executor.map(process_batch_verification, batches)

            for batch_matches in batch_results:
                for file1, file2, trans_name in batch_matches:
                    # Evitar duplicados
                    if not any(m[0] == file1 and m[1] == file2 for m in perceptual_matches):
                        perceptual_matches.append((file1, file2, trans_name))
                        processed_files.add(file1)
                        processed_files.add(file2)
                        confirm_msg = f"  ✅ MATCH: {file1} ↔ {file2} ({trans_name})"
                        log_messages.append(confirm_msg)
                        print(confirm_msg)

    verification_time = time.time() - step2_start - index_time - candidate_time
    step2_total = time.time() - step2_start
    log_msg = f"🎊 Duplicados perceptuales: {len(perceptual_matches)} (⏱️ {step2_total:.2f}s total)"
    log_messages.append(log_msg)
    print(log_msg)

    # PASO 3: Detectar versiones adicionales (clave del código original)
    step3_start = time.time()
    log_messages.append("\n=== PASO 3: Detectando versiones adicionales por número de enfermedad ===")
    print("\n=== PASO 3: Detectando versiones adicionales por número de enfermedad ===")

    # Crear diccionario de enfermedades procesadas (lógica del código original)
    all_matches = exact_matches + perceptual_matches
    processed_diseases = {}
    for match in all_matches:
        file1, file2, trans = match
        disease_num = extract_disease_number(file1)
        if disease_num:
            if disease_num not in processed_diseases:
                processed_diseases[disease_num] = []
            processed_diseases[disease_num].append((file1, file2, trans))

    additional_versions_found = 0
    for file1 in all_files:
        if file1 in processed_files:
            continue

        disease_num = extract_disease_number(file1)
        if not disease_num:
            continue

        if disease_num in processed_diseases:
            # Hay otros archivos con el mismo número de enfermedad - es una versión adicional
            reference_match = processed_diseases[disease_num][0]
            file2_ref = reference_match[1] if len(reference_match) > 1 else file1

            additional_match = (file1, file2_ref, "additional_version")
            all_matches.append(additional_match)
            processed_files.add(file1)
            additional_versions_found += 1

            additional_msg = f"  📎 Versión adicional: {file1} (pm{disease_num})"
            log_messages.append(additional_msg)
            print(additional_msg)

    step3_time = time.time() - step3_start
    log_msg = f"🔗 Versiones adicionales: {additional_versions_found} (⏱️ {step3_time:.2f}s)"
    log_messages.append(log_msg)
    print(log_msg)

    # PASO 4: Identificar archivos sin transformaciones (aplicando lógica del código original)
    log_messages.append("\n=== PASO 4: Identificando archivos sin transformaciones ===")
    print("\n=== PASO 4: Identificando archivos sin transformaciones ===")

    # Agrupar por número de enfermedad (lógica clave del código original)
    disease_groups = defaultdict(list)
    for file in all_files:
        disease_num = extract_disease_number(file)
        if disease_num:
            disease_groups[disease_num].append(file)
        else:
            # Si no tiene número de enfermedad, considerarlo como grupo individual
            disease_groups[f"no_pm_{file}"] = [file]

    files_without_transformations = []
    transformation_groups = {}

    for disease_num, files_in_group in disease_groups.items():
        if len(files_in_group) == 1:
            # Solo hay un archivo con este número de enfermedad - sin transformaciones
            single_file = files_in_group[0]
            files_without_transformations.append(single_file)
            log_msg = f"  📌 Sin transformaciones: {single_file} (único archivo pm{disease_num})"
            log_messages.append(log_msg)
            print(log_msg)
        else:
            # Hay múltiples archivos con el mismo número - tienen transformaciones
            base_file = files_in_group[0]  # Usar el primero como base
            transformation_groups[base_file] = {
                "related_files": files_in_group[1:],
                "transformations_found": ["identity"],
                "disease_number": disease_num,
            }

    total_time = time.time() - start_time
    log_msg = f"\n🎉 PROCESO COMPLETADO EN {total_time:.2f}s"
    log_msg += f"\n   - Exactos: {exact_time:.2f}s"
    log_msg += f"\n   - Perceptuales: {step2_total:.2f}s"
    log_msg += f"\n   - Versiones: {step3_time:.2f}s"
    log_msg += f"\n   - Archivos con transformaciones: {len(all_files) - len(files_without_transformations)}"
    log_msg += f"\n   - Archivos sin transformaciones: {len(files_without_transformations)}"
    log_messages.append(log_msg)
    print(log_msg)

    return transformation_groups, files_without_transformations, log_messages


# Resto de funciones sin cambios (generate_report, move_duplicates, etc.)
def generate_report(matches, log_messages):
    """Genera el reporte de duplicados."""
    report_file = f"duplicates_report_{dir_name}_optimized.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== REPORTE DE DUPLICADOS OPTIMIZADO ===\n\n")
        f.write(f"TOTAL DE DUPLICADOS ENCONTRADOS: {len(matches)}\n\n")

        # Separar por tipos
        perceptual_matches = [m for m in matches if m[2] not in ["exact_copy", "additional_version"]]
        exact_matches = [m for m in matches if m[2] == "exact_copy"]
        additional_versions = [m for m in matches if m[2] == "additional_version"]

        if exact_matches:
            f.write("DUPLICADOS EXACTOS:\n")
            for file1, file2, trans in exact_matches:
                f.write(f"  {file2} (dir2) es idéntico a {file1} (dir1)\n")
            f.write(f"Total: {len(exact_matches)}\n\n")

        if perceptual_matches:
            f.write("DUPLICADOS CON TRANSFORMACIONES:\n")
            for file1, file2, trans in perceptual_matches:
                f.write(f"  {file2} (dir2) es similar a {file1} (dir1) con transformación: {trans}\n")
            f.write(f"Total: {len(perceptual_matches)}\n\n")

        if additional_versions:
            f.write("VERSIONES ADICIONALES DETECTADAS:\n")
            for file1, file2, trans in additional_versions:
                f.write(f"  {file1} (dir1) es una versión adicional relacionada con {file2} (dir2)\n")
            f.write(f"Total: {len(additional_versions)}\n\n")

        # Agregar logs de procesamiento
        f.write("=== LOGS DE PROCESAMIENTO ===\n\n")
        for msg in log_messages:
            f.write(msg + "\n")

    print(f"📄 Reporte guardado en: {report_file}")
    return report_file


def move_duplicates(matches):
    """Mueve los duplicados a subdirectorios."""
    if not matches:
        print("No hay duplicados para mover.")
        return

    # Crear directorios de destino
    duplicates_dir1 = os.path.join(dir1, "duplicates")
    duplicates_dir2 = os.path.join(dir2, "duplicates")

    os.makedirs(duplicates_dir1, exist_ok=True)
    os.makedirs(duplicates_dir2, exist_ok=True)

    moved_count = 0

    for file1, file2, transformation in matches:
        try:
            # Mover archivo de dir1
            src1 = os.path.join(dir1, file1)
            dst1 = os.path.join(duplicates_dir1, file1)
            if os.path.exists(src1):
                os.rename(src1, dst1)
                moved_count += 1
                print(f"📦 Movido: {file1} → duplicates/")

            # Mover archivo de dir2 (solo si no es additional_version)
            if transformation != "additional_version":
                src2 = os.path.join(dir2, file2)
                dst2 = os.path.join(duplicates_dir2, file2)
                if os.path.exists(src2):
                    os.rename(src2, dst2)
                    moved_count += 1
                    print(f"📦 Movido: {file2} → duplicates/")

        except Exception as e:
            print(f"❌ Error moviendo archivos: {e}")

    print(f"\n🎯 Total de archivos movidos: {moved_count}")


def generate_transformation_report(transformation_groups, files_without_transformations, log_messages):
    """Genera reportes de las transformaciones encontradas"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"duplicates_report_{dir_name}_{timestamp}.txt"
    csv_file = f"transformation_analysis_{timestamp}.csv"

    # Generar reporte de texto
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE DE ANÁLISIS DE TRANSFORMACIONES\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Directorio analizado: {target_dir}\n")
        f.write(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Calcular estadísticas
        total_with_transformations = 0
        for base_file, data in transformation_groups.items():
            group_size = len(data["related_files"]) + 1  # +1 por el archivo base
            total_with_transformations += group_size

        total_files = total_with_transformations + len(files_without_transformations)

        f.write(f"Estadísticas:\n")
        f.write(f"- Total de archivos analizados: {total_files}\n")
        f.write(f"- Grupos de transformaciones: {len(transformation_groups)}\n")
        f.write(f"- Archivos con transformaciones: {total_with_transformations}\n")
        f.write(f"- Archivos sin transformaciones: {len(files_without_transformations)}\n\n")

        # Grupos de transformaciones
        if transformation_groups:
            f.write("GRUPOS DE TRANSFORMACIONES ENCONTRADOS:\n")
            f.write("-" * 50 + "\n")
            for i, (base_file, data) in enumerate(transformation_groups.items(), 1):
                group_files = [base_file] + data["related_files"]
                f.write(f"\nGrupo {i} ({len(group_files)} archivos) - pm{data['disease_number']}:\n")
                for file_path in sorted(group_files):
                    filename = os.path.basename(file_path)
                    f.write(f"  - {filename}\n")

        # Archivos sin transformaciones
        if files_without_transformations:
            f.write("\n\nARCHIVOS SIN TRANSFORMACIONES:\n")
            f.write("-" * 50 + "\n")
            for file_path in sorted(files_without_transformations):
                filename = os.path.basename(file_path)
                f.write(f"  - {filename}\n")

        # Log de detecciones
        if log_messages:
            f.write("\n\nDETALLE DEL PROCESO:\n")
            f.write("-" * 50 + "\n")
            for msg in log_messages:
                f.write(f"{msg}\n")

    # Generar CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "group_id", "has_transformations", "group_size", "disease_number"])

        # Archivos con transformaciones
        for group_id, (base_file, data) in enumerate(transformation_groups.items(), 1):
            group_files = [base_file] + data["related_files"]
            for file_path in group_files:
                filename = os.path.basename(file_path)
                writer.writerow([filename, group_id, "YES", len(group_files), data["disease_number"]])

        # Archivos sin transformaciones
        for file_path in files_without_transformations:
            filename = os.path.basename(file_path)
            writer.writerow([filename, 0, "NO", 1, "unknown"])

    return report_file, csv_file


def move_files_without_transformations(files_without_transformations):
    """Mueve archivos sin transformaciones a un subdirectorio"""
    if not files_without_transformations:
        return 0

    # Crear subdirectorio no_transform
    no_transform_dir = os.path.join(target_dir, "no_transform")
    os.makedirs(no_transform_dir, exist_ok=True)

    moved_count = 0
    for filename in files_without_transformations:
        try:
            # Construir ruta completa del archivo fuente
            source_path = os.path.join(target_dir, filename)
            destination = os.path.join(no_transform_dir, filename)

            # Mover archivo
            import shutil

            shutil.move(source_path, destination)
            moved_count += 1
            print(f"📦 Movido: {filename} → no_transform/")

        except Exception as e:
            print(f"❌ Error moviendo {filename}: {e}")

    print(f"\n🎯 Total de archivos movidos: {moved_count}")
    return moved_count


if __name__ == "__main__":
    print("� Iniciando análisis de transformaciones...")

    try:
        # Ejecutar análisis de transformaciones
        transformation_groups, files_without_transformations, log_messages = find_transformations_in_directory()

        # Generar reportes
        report_file, csv_file = generate_transformation_report(transformation_groups, files_without_transformations, log_messages)

        # Mover archivos sin transformaciones
        if files_without_transformations:
            print(f"\n📋 Se encontraron {len(files_without_transformations)} archivos sin transformaciones.")
            print("🔄 Moviendo archivos sin transformaciones a subdirectorio...")
            move_files_without_transformations(files_without_transformations)
        else:
            print("✅ Todos los archivos tienen al menos una transformación.")

        print(f"\n🎉 Proceso completado.")
        print(f"📄 Reporte detallado: {report_file}")
        print(f"📊 Archivo CSV: {csv_file}")

    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback

        traceback.print_exc()
