package com.tomato.disease;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

/**
 * Actividad principal para detección de enfermedades del tomate
 * Implementa MobileNetV4 con TensorFlow Lite para inferencia local
 */
public class MainActivity extends AppCompatActivity {
    // Ruta del archivo de métricas
    private static final String METRICS_FILE = "inference_metrics.csv";

    private static final String TAG = "TomatoDetection";
    private static final int REQUEST_CAMERA_PERMISSION = 1001;
    private static final int REQUEST_IMAGE_CAPTURE = 1002;
    private static final int REQUEST_IMAGE_GALLERY = 1003;

    // Configuración del modelo
    private static final String MODEL_FILE = "tomato_disease_mobilenetv4.tflite";
    private static final int INPUT_SIZE = 224;
    private static final int NUM_CLASSES = 11;

    // Nombres de las clases (correspondientes al modelo)

    // TextView para mostrar la información del modelo
    private TextView modelInfoText;
    private static final String[] CLASS_NAMES = {
        "Bacterial Spot",
        "Early Blight",
        "Healthy",
        "Late Blight",
        "Leaf Mold",
        "Powdery Mildew",
        "Septoria Leaf Spot",
        "Spider Mites",
        "Target Spot",
        "Tomato Mosaic Virus",
        "Yellow Leaf Curl Virus"
    };

    // Descripciones de las enfermedades
    private static final String[] DISEASE_DESCRIPTIONS = {
        "Mancha bacteriana - Causada por Xanthomonas vesicatoria",
        "Tizón temprano - Causado por Alternaria solani",
        "Planta saludable - Sin signos de enfermedad",
        "Tizón tardío - Causado por Phytophthora infestans",
        "Moho foliar - Causado por Passalora fulva",
        "Mildiu polvoriento - Causado por Leveillula taurica",
        "Mancha foliar de Septoria - Causada por Septoria lycopersici",
        "Araña roja - Tetranychus urticae",
        "Mancha objetivo - Causada por Corynespora cassiicola",
        "Virus del mosaico del tomate - TMV",
        "Virus del rizado amarillo - TYLCV"
    };

    // UI Components
    private ImageView imageView;
    private TextView resultText;
    private TextView confidenceText;
    private TextView descriptionText;
    private Button captureButton;
    private Button galleryButton;
    private Button shareMetricsButton;

    // TensorFlow Lite
    private Interpreter tflite;
    private ByteBuffer inputBuffer;
    private float[][] outputArray;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

    initializeViews();
    setupButtons();
    initializeModel();
    modelInfoText = findViewById(R.id.modelInfoText);
    loadModelInfo();
    requestCameraPermission();
    }
    /**
     * Leer model_info.json desde assets y mostrar la información real en el TextView
     */
    private void loadModelInfo() {
        final String assetName = "model_info.json";
        try (java.io.InputStream is = getAssets().open(assetName);
             java.io.BufferedReader reader = new java.io.BufferedReader(
                     new java.io.InputStreamReader(is, java.nio.charset.StandardCharsets.UTF_8))) {

            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }

            org.json.JSONObject json = new org.json.JSONObject(sb.toString());
            String arquitectura = json.optString("model_name", "MobileNetV4");
            String version = json.optString("version", "1.0.0");
            String precision = json.optString("accuracy", "-");
            org.json.JSONObject perf = json.optJSONObject("performance");
            String parametros = "-";
            String tamano = "-";
            String cuantizacion = "-";
            String clases = "-";
            if (perf != null) {
                int params = perf.optInt("model_params", -1);
                parametros = (params > 0) ? String.format("%,d", params) : "-";
                double tamMb = perf.optDouble("model_size_mb", -1);
                tamano = (tamMb > 0) ? String.format("%.2f MB", tamMb) : "-";
                boolean quant = perf.optBoolean("quantized", false);
                String inputType = perf.optString("input_dtype", "");
                cuantizacion = quant ? "Sí (INT8)" : (inputType.contains("float") ? "No (float32)" : "No");
                org.json.JSONArray classArr = json.optJSONArray("class_names");
                if (classArr != null) {
                    clases = String.valueOf(classArr.length());
                }
            }

            String info = String.format(
                    "• Arquitectura: %s\n" +
                    "• Precisión: %s\n" +
                    "• Parámetros: %s\n" +
                    "• Tamaño del modelo: %s\n" +
                    "• Cuantización: %s\n" +
                    "• Clases detectables: %s\n" +
                    "• Versión: %s",
                    arquitectura,
                    precision,
                    parametros,
                    tamano,
                    cuantizacion,
                    clases,
                    version
            );
            modelInfoText.setText(info);

        } catch (Exception e) {
            Log.e(TAG, "Error leyendo '" + assetName + "' desde assets", e);
            String msg = e.getMessage();
            if (msg == null) msg = e.getClass().getSimpleName();
            modelInfoText.setText("Error cargando información del modelo: " + msg);
        }
    }

    /**
     * Inicializar las vistas de la UI
     */
    private void initializeViews() {
        imageView = findViewById(R.id.imageView);
        resultText = findViewById(R.id.resultText);
        confidenceText = findViewById(R.id.confidenceText);
        descriptionText = findViewById(R.id.descriptionText);
        captureButton = findViewById(R.id.captureButton);
        galleryButton = findViewById(R.id.galleryButton);
    shareMetricsButton = findViewById(R.id.shareMetricsButton);

        // Configurar texto inicial
        resultText.setText("Toma una foto o selecciona una imagen de una hoja de tomate");
        confidenceText.setText("");
        descriptionText.setText("La aplicación analizará la imagen y detectará posibles enfermedades");
    }

    /**
     * Configurar los listeners de los botones
     */
    private void setupButtons() {
        captureButton.setOnClickListener(v -> {
            if (checkCameraPermission()) {
                openCamera();
            } else {
                requestCameraPermission();
            }
        });

        galleryButton.setOnClickListener(v -> openGallery());

    shareMetricsButton.setOnClickListener(v -> shareMetricsFile());
    }

    /**
     * Inicializar el modelo TensorFlow Lite
     */
    private void initializeModel() {
        try {
            // Cargar el modelo desde assets
            MappedByteBuffer tfliteModel = loadModelFile();

            // Configurar opciones del intérprete
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4); // Usar 4 threads para mejor rendimiento

            // Crear intérprete
            tflite = new Interpreter(tfliteModel, options);

            // Preparar buffers de entrada y salida
            prepareBuffers();

            Log.d(TAG, "Modelo TensorFlow Lite cargado exitosamente");
            Toast.makeText(this, "Modelo cargado correctamente", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            Log.e(TAG, "Error cargando modelo: " + e.getMessage(), e);
            Toast.makeText(this, "Error cargando modelo: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    /**
     * Cargar el archivo del modelo desde assets
     */
    private MappedByteBuffer loadModelFile() throws IOException {
        FileInputStream inputStream = new FileInputStream(getAssets().openFd(MODEL_FILE).getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = getAssets().openFd(MODEL_FILE).getStartOffset();
        long declaredLength = getAssets().openFd(MODEL_FILE).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Preparar buffers de entrada y salida para el modelo
     */
    private void prepareBuffers() {
    // Buffer de entrada: [1, 224, 224, 3] con valores float32 normalizados
    inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3); // 4 bytes por float
    inputBuffer.order(ByteOrder.nativeOrder());

    // Array de salida: [1, 11] con probabilidades
    outputArray = new float[1][NUM_CLASSES];
    }

    /**
     * Verificar permisos de cámara
     */
    private boolean checkCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
               == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Solicitar permisos de cámara
     */
    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this,
            new String[]{Manifest.permission.CAMERA},
            REQUEST_CAMERA_PERMISSION);
    }

    /**
     * Abrir cámara para capturar imagen
     */
    private void openCamera() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        } else {
            Toast.makeText(this, "No se encontró aplicación de cámara", Toast.LENGTH_SHORT).show();
        }
    }

    /**
     * Abrir galería para seleccionar imagen
     */
    private void openGallery() {
    Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
    intent.setType("image/*");
    intent.addCategory(Intent.CATEGORY_OPENABLE);
    startActivityForResult(Intent.createChooser(intent, "Selecciona una imagen"), REQUEST_IMAGE_GALLERY);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                         @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permiso de cámara concedido", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Permiso de cámara denegado", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == Activity.RESULT_OK && data != null) {
            Bitmap bitmap = null;

            try {
                if (requestCode == REQUEST_IMAGE_CAPTURE) {
                    // Imagen de cámara
                    Bundle extras = data.getExtras();
                    bitmap = (Bitmap) extras.get("data");

                } else if (requestCode == REQUEST_IMAGE_GALLERY) {
                    // Imagen de galería
                    Uri imageUri = data.getData();
                    bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                }

                if (bitmap != null) {
                    // Mostrar imagen
                    imageView.setImageBitmap(bitmap);

                    // Clasificar imagen
                    classifyImage(bitmap);
                }

            } catch (Exception e) {
                Log.e(TAG, "Error procesando imagen: " + e.getMessage(), e);
                Toast.makeText(this, "Error procesando imagen", Toast.LENGTH_SHORT).show();
            }
        }
    }

    /**
     * Clasificar imagen usando el modelo TensorFlow Lite
     */
    private void classifyImage(Bitmap bitmap) {
        if (tflite == null) {
            Toast.makeText(this, "Modelo no cargado", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            // Preprocesar imagen
            Bitmap preprocessedBitmap = preprocessImage(bitmap);

            // Convertir a buffer de entrada
            fillInputBuffer(preprocessedBitmap);

            // Ejecutar inferencia
            long startTime = System.currentTimeMillis();
            tflite.run(inputBuffer, outputArray);
            long inferenceTime = System.currentTimeMillis() - startTime;

            // Procesar resultados
            int predictedClass = postprocessOutput(outputArray[0]);
            float confidence = outputArray[0][predictedClass] * 100;

            // Mostrar resultados
            displayResults(predictedClass, confidence, inferenceTime);

            Log.d(TAG, String.format("Inferencia completada en %dms, clase: %s, confianza: %.1f%%",
                  inferenceTime, CLASS_NAMES[predictedClass], confidence));

            // Guardar métrica en archivo
            saveInferenceMetric(CLASS_NAMES[predictedClass], confidence, inferenceTime);

        } catch (Exception e) {
            Log.e(TAG, "Error en clasificación: " + e.getMessage(), e);
            Toast.makeText(this, "Error en clasificación: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    /**
     * Preprocesar imagen para el modelo
     */
    private Bitmap preprocessImage(Bitmap bitmap) {
        // Redimensionar a 224x224 (tamaño de entrada del modelo)
        return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
    }

    /**
     * Llenar el buffer de entrada con los datos de la imagen
     */
    private void fillInputBuffer(Bitmap bitmap) {
        inputBuffer.rewind();

        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                // Extraer componentes RGB y normalizar a float32 [0,1]
                float r = ((val >> 16) & 0xFF) / 255.0f;
                float g = ((val >> 8) & 0xFF) / 255.0f;
                float b = (val & 0xFF) / 255.0f;
                inputBuffer.putFloat(r);
                inputBuffer.putFloat(g);
                inputBuffer.putFloat(b);
            }
        }
    }

    /**
     * Procesar salida del modelo para obtener clase predicha
     */
    private int postprocessOutput(float[] probabilities) {
        int maxIndex = 0;
        float maxProb = probabilities[0];

        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /**
     * Mostrar resultados en la UI
     */
    private void displayResults(int predictedClass, float confidence, long inferenceTime) {
        String className = CLASS_NAMES[predictedClass];
        String description = DISEASE_DESCRIPTIONS[predictedClass];

        resultText.setText(String.format("Diagnóstico: %s", className));
        confidenceText.setText(String.format("Confianza: %.1f%% (Inferencia: %dms)",
                                            confidence, inferenceTime));
        descriptionText.setText(description);

        // Cambiar color según el diagnóstico
        if (className.equals("Healthy")) {
            resultText.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
        } else {
            resultText.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        // Liberar recursos
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
    }

    /**
     * Compartir el archivo de métricas por email, drive, etc.
     */
    private void shareMetricsFile() {
        try {
            java.io.File file = new java.io.File(getFilesDir(), METRICS_FILE);
            if (!file.exists()) {
                Toast.makeText(this, "No hay métricas guardadas aún", Toast.LENGTH_SHORT).show();
                return;
            }
            Uri fileUri = androidx.core.content.FileProvider.getUriForFile(
                this,
                getApplicationContext().getPackageName() + ".fileprovider",
                file
            );
            Intent intent = new Intent(Intent.ACTION_SEND);
            intent.setType("text/csv");
            intent.putExtra(Intent.EXTRA_STREAM, fileUri);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            intent.putExtra(Intent.EXTRA_SUBJECT, "Métricas de inferencia Tomato Disease");
            intent.putExtra(Intent.EXTRA_TEXT, "Adjunto archivo de métricas de inferencia generadas por la app.");
            startActivity(Intent.createChooser(intent, "Compartir métricas"));
        } catch (Exception e) {
            Log.e(TAG, "Error compartiendo métricas: " + e.getMessage(), e);
            Toast.makeText(this, "Error compartiendo métricas: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    /**
     * Guardar métrica de inferencia en archivo CSV
     */
    private void saveInferenceMetric(String className, float confidence, long inferenceTime) {
        String line = String.format("%s,%.1f,%d,%d\n", className, confidence, inferenceTime, System.currentTimeMillis());
        try {
            java.io.File file = new java.io.File(getFilesDir(), METRICS_FILE);
            boolean exists = file.exists();
            java.io.FileOutputStream fos = new java.io.FileOutputStream(file, true);
            if (!exists) {
                String header = "Clase,Confianza (%),Latencia (ms),Timestamp\n";
                fos.write(header.getBytes());
            }
            fos.write(line.getBytes());
            fos.close();
        } catch (Exception e) {
            Log.e(TAG, "Error guardando métrica: " + e.getMessage(), e);
        }
    }
}
