package com.application.AIChecker

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.OpenableColumns
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.application.AIChecker.databinding.ActivityMainBinding
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val executor    = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var isModelLoaded = false
    private var selectedPath  : String? = null
    private val tempFiles     = mutableListOf<File>()

    private var ortEnv          : OrtEnvironment? = null
    private var ortSession      : OrtSession?     = null

    // Deep feature ONNX models
    private var resnet50Session     : OrtSession? = null
    private var efficientnetSession : OrtSession? = null
    private var deepModelsLoaded    : Int = 0  // how many deep models loaded successfully

    companion object {
        private const val REQ_PICK_VIDEO = 1001
        private var activeModel = "x.onnx"
        private const val TAG = "AIChecker"

        private val DEEP_FEATURE_MODELS = listOf(
            "resnet50_features.onnx",
            "efficientnet_b0_features.onnx"
        )

        // ImageNet normalization constants
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Number of frames to sample for deep feature extraction
        private const val DEEP_SAMPLE_FRAMES = 10

        // Expected deep feature names in exact order (must match training)
        private val DEEP_FEATURE_NAMES = listOf(
            "deep_feat_mean", "deep_feat_std", "deep_feat_max", "deep_feat_min",
            "deep_temporal_var_mean", "deep_temporal_var_std",
            "deep_l2_norm_mean", "deep_l2_norm_std",
            "deep_similarity_mean", "deep_similarity_std", "deep_sparsity"
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Available model listing
    // ─────────────────────────────────────────────────────────────────────────

    private fun getAvailableModels(): Array<String> {
        return try {
            assets.list("models")
                ?.filter { it.endsWith(".onnx") && !it.endsWith("_features.onnx") }
                ?.sorted()
                ?.toTypedArray()
                ?: arrayOf("x.onnx")
        } catch (e: Exception) {
            arrayOf("x.onnx")
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        ortEnv = OrtEnvironment.getEnvironment()

        setupUI()
        loadModels(activeModel)
    }

    override fun onDestroy() {
        super.onDestroy()
        ortSession?.close()
        resnet50Session?.close()
        efficientnetSession?.close()
        ortEnv?.close()
        tempFiles.forEach { if (it.exists()) it.delete() }
        tempFiles.clear()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_PICK_VIDEO && resultCode == Activity.RESULT_OK)
            data?.data?.let { handlePickedVideo(it) }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  UI Setup
    // ─────────────────────────────────────────────────────────────────────────

    private fun setupUI() {
        setControlsEnabled(false)
        binding.resultCard.visibility = View.GONE
        binding.btnClear.visibility   = View.GONE

        binding.btnSelectFile.setOnClickListener  { pickVideoFile() }
        binding.btnPasteUrl.setOnClickListener    { showUrlDialog() }
        binding.btnAnalyze.setOnClickListener     { selectedPath?.let { analyzeVideo(it) } }
        binding.btnModelPicker.setOnClickListener { showModelPicker() }
        binding.btnClear.setOnClickListener       { clearResult() }
    }

    private fun setControlsEnabled(on: Boolean) {
        binding.btnSelectFile.isEnabled  = on
        binding.btnPasteUrl.isEnabled    = on
        binding.btnModelPicker.isEnabled = on
        binding.btnAnalyze.isEnabled     = on && selectedPath != null
    }

    private fun setBadge(text: String, colorRes: Int) {
        binding.tvBadge.text = text
        binding.tvBadge.setTextColor(ContextCompat.getColor(this, colorRes))
    }

    private fun clearResult() {
        binding.resultCard.visibility = View.GONE
        binding.btnClear.visibility   = View.GONE
        binding.progressBar.progress  = 0
        binding.tvStatus.text         = "Ready - select a file or paste a URL."
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Model Loading
    // ─────────────────────────────────────────────────────────────────────────

    private fun loadModels(modelName: String) {
        isModelLoaded = false
        setControlsEnabled(false)
        binding.tvStatus.text = "Loading model: $modelName ..."
        setBadge("* LOADING", R.color.warning)

        executor.execute {
            try {
                val configFile = copyAsset("config.yaml")

                val scalerName = modelName.removeSuffix(".onnx") + "_scaler.pkl"

                val scalerExists = try {
                    assets.open("models/$scalerName").close()
                    true
                } catch (e: Exception) { false }

                if (!scalerExists) {
                    mainHandler.post {
                        binding.tvStatus.text = "Thieu file: models/$scalerName"
                        setBadge("!! ERROR", R.color.danger)
                    }
                    return@execute
                }

                val scalerFile = copyAsset("models/$scalerName")

                // Copy deep feature ONNX models to filesDir
                for (deepModelFile in DEEP_FEATURE_MODELS) {
                    try {
                        assets.open("models/$deepModelFile").close()
                        copyAsset("models/$deepModelFile")
                        Log.d(TAG, "Copied deep feature model: $deepModelFile")
                    } catch (e: Exception) {
                        Log.w(TAG, "Deep feature model not found in assets: $deepModelFile — skipping")
                    }
                }

                // Load deep feature ONNX sessions in Kotlin
                loadDeepFeatureModels()

                // Load Python detector (traditional features + scaler)
                val result = Python.getInstance()
                    .getModule("detector")
                    .callAttr("load_models",
                        scalerFile.absolutePath,
                        configFile.absolutePath)
                    .toString()

                Log.d(TAG, "load_models result: $result")

                if (result != "OK") {
                    mainHandler.post {
                        binding.tvStatus.text = result
                        setBadge("!! ERROR", R.color.danger)
                    }
                    return@execute
                }

                // Load classifier ONNX model
                val onnxFile = copyAsset("models/$modelName")
                val env      = ortEnv ?: OrtEnvironment.getEnvironment().also { ortEnv = it }
                ortSession?.close()
                ortSession = env.createSession(onnxFile.absolutePath)

                Log.d(TAG, "Classifier loaded. Deep models: $deepModelsLoaded / ${DEEP_FEATURE_MODELS.size}")

                mainHandler.post {
                    isModelLoaded            = true
                    activeModel              = modelName
                    binding.tvModelName.text = modelName
                    val deepStatus = if (deepModelsLoaded > 0)
                        "deep=$deepModelsLoaded ✓"
                    else
                        "deep=0 ⚠ (re-export ONNX needed)"
                    binding.tvStatus.text    = "Ready ($deepStatus) - select a file or paste a URL."
                    setBadge("* READY", R.color.success)
                    setControlsEnabled(true)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Load models failed", e)
                mainHandler.post {
                    binding.tvStatus.text = "Load failed: ${e.message}"
                    setBadge("!! ERROR", R.color.danger)
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Deep Feature Model Loading (Kotlin ONNX sessions)
    // ─────────────────────────────────────────────────────────────────────────

    private fun loadDeepFeatureModels() {
        val env = ortEnv ?: return

        resnet50Session?.close()
        resnet50Session = null
        efficientnetSession?.close()
        efficientnetSession = null
        deepModelsLoaded = 0

        // Try resnet50
        try {
            val f = File(filesDir, "models/resnet50_features.onnx")
            if (f.exists()) {
                resnet50Session = env.createSession(f.absolutePath)
                deepModelsLoaded++
                Log.d(TAG, "✓ Loaded resnet50_features.onnx (Kotlin ONNX)")
            } else {
                Log.w(TAG, "resnet50_features.onnx not found in filesDir")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load resnet50: ${e.message}")
            // Check if it's an IR version issue
            if (e.message?.contains("Unsupported model IR version") == true ||
                e.message?.contains("IR version") == true) {
                Log.e(TAG, "❌ IR version incompatible! Re-export ONNX with export_deep_features_onnx.py")
                Log.e(TAG, "   Then rebuild the app. Error: ${e.message}")
            }
        }

        // Try efficientnet
        try {
            val f = File(filesDir, "models/efficientnet_b0_features.onnx")
            if (f.exists()) {
                efficientnetSession = env.createSession(f.absolutePath)
                deepModelsLoaded++
                Log.d(TAG, "✓ Loaded efficientnet_b0_features.onnx (Kotlin ONNX)")
            } else {
                Log.w(TAG, "efficientnet_b0_features.onnx not found in filesDir")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load efficientnet: ${e.message}")
            if (e.message?.contains("Unsupported model IR version") == true ||
                e.message?.contains("IR version") == true) {
                Log.e(TAG, "❌ IR version incompatible! Re-export ONNX with export_deep_features_onnx.py")
            }
        }

        Log.d(TAG, "Deep models loaded: $deepModelsLoaded / ${DEEP_FEATURE_MODELS.size}")
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Deep Feature Extraction (Kotlin ONNX runtime)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Extract video frames using MediaMetadataRetriever.
     * Returns preprocessed FloatArrays (CHW, ImageNet normalized) for ONNX input.
     */
    private fun extractVideoFrames(videoPath: String): List<FloatArray> {
        val frames = mutableListOf<FloatArray>()
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(videoPath)
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs  = durationStr?.toLongOrNull() ?: return frames
            if (durationMs <= 0) return frames

            for (i in 0 until DEEP_SAMPLE_FRAMES) {
                val fraction = if (DEEP_SAMPLE_FRAMES > 1) i.toDouble() / (DEEP_SAMPLE_FRAMES - 1) else 0.5
                val timeUs   = (fraction * durationMs * 1000L).toLong()
                    .coerceIn(0L, (durationMs - 1) * 1000L)

                try {
                    val bitmap = retriever.getFrameAtTime(
                        timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC
                    )
                    if (bitmap != null) {
                        frames.add(preprocessBitmapForOnnx(bitmap))
                        bitmap.recycle()
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Frame $i extraction failed: ${e.message}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "extractVideoFrames failed: ${e.message}")
        } finally {
            try { retriever.release() } catch (_: Exception) {}
        }
        Log.d(TAG, "Extracted ${frames.size} frames for deep features")
        return frames
    }

    /**
     * Resize bitmap to 256×256, center-crop to 224×224, normalize with ImageNet stats.
     * Returns FloatArray in CHW format (3 × 224 × 224).
     */
    private fun preprocessBitmapForOnnx(src: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(src, 256, 256, true)
        val cropped = Bitmap.createBitmap(resized, 16, 16, 224, 224)  // center crop
        if (resized !== src) resized.recycle()

        val pixels = IntArray(224 * 224)
        cropped.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        cropped.recycle()

        val tensor = FloatArray(3 * 224 * 224)
        for (i in pixels.indices) {
            val p = pixels[i]
            tensor[i]                 = ((p shr 16 and 0xFF) / 255f - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            tensor[224 * 224 + i]     = ((p shr 8  and 0xFF) / 255f - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            tensor[2 * 224 * 224 + i] = ((p         and 0xFF) / 255f - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        }
        return tensor
    }

    /**
     * Run one frame through an ONNX feature-extraction model.
     * Handles both [1, D] and [1, D, 1, 1] output shapes (ResNet vs EfficientNet).
     */
    private fun runDeepModelOnFrame(session: OrtSession, frameData: FloatArray): FloatArray? {
        return try {
            val env = ortEnv ?: return null
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(frameData), longArrayOf(1, 3, 224, 224))
            tensor.use {
                val out = session.run(mapOf(session.inputNames.first() to it))
                out.use { flattenOnnxOutput(out[0].value) }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Deep ONNX inference failed: ${e.message}")
            null
        }
    }

    /** Recursively flatten any nested float array from ONNX output into FloatArray. */
    @Suppress("UNCHECKED_CAST")
    private fun flattenOnnxOutput(value: Any?): FloatArray? {
        return when (value) {
            is FloatArray -> value
            is Array<*>  -> {
                val buf = mutableListOf<Float>()
                flattenInto(value, buf)
                buf.toFloatArray()
            }
            else -> null
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun flattenInto(arr: Array<*>, buf: MutableList<Float>) {
        for (item in arr) {
            when (item) {
                is Float       -> buf.add(item)
                is Double      -> buf.add(item.toFloat())
                is FloatArray  -> item.forEach { buf.add(it) }
                is DoubleArray -> item.forEach { buf.add(it.toFloat()) }
                is Array<*>    -> flattenInto(item, buf)
            }
        }
    }

    /**
     * Compute the 11 canonical deep_* statistics from a list of per-frame feature vectors.
     * Matches exactly what the Python DeepFeatureExtractor computes.
     */
    private fun computeDeepStats(allFrameFeatures: List<FloatArray>): Map<String, Float> {
        val n = allFrameFeatures.size
        if (n == 0) return emptyMap()
        val dim = allFrameFeatures[0].size
        if (dim == 0) return emptyMap()

        // ── Global stats across all elements ──────────────────────────────────
        var sumAll   = 0.0
        var sumSqAll = 0.0
        var maxAll   = Float.NEGATIVE_INFINITY
        var minAll   = Float.POSITIVE_INFINITY
        var total    = 0
        var sparse   = 0

        for (feat in allFrameFeatures) {
            for (v in feat) {
                sumAll   += v
                sumSqAll += v.toDouble() * v
                if (v > maxAll) maxAll = v
                if (v < minAll) minAll = v
                if (abs(v) < 0.01f) sparse++
                total++
            }
        }

        val gMean     = (sumAll / total).toFloat()
        val gVariance = (sumSqAll / total - gMean.toDouble() * gMean).coerceAtLeast(0.0)
        val gStd      = sqrt(gVariance).toFloat()

        // ── Temporal variance: var across time for each feature dimension ─────
        val tempVarArr = DoubleArray(dim)
        for (d in 0 until dim) {
            val colMean = allFrameFeatures.sumOf { it[d].toDouble() } / n
            tempVarArr[d] = allFrameFeatures.sumOf { feat ->
                val diff = feat[d].toDouble() - colMean
                diff * diff
            } / n
        }
        val tvMean   = tempVarArr.average().toFloat()
        val tvMeanSq = tempVarArr.map { it * it }.average()
        val tvStd    = sqrt((tvMeanSq - tvMean.toDouble() * tvMean).coerceAtLeast(0.0)).toFloat()

        // ── L2 norms per frame ────────────────────────────────────────────────
        val l2Arr  = DoubleArray(n) { i -> sqrt(allFrameFeatures[i].sumOf { v -> v.toDouble() * v }) }
        val l2Mean = l2Arr.average().toFloat()
        val l2MSq  = l2Arr.map { it * it }.average()
        val l2Std  = sqrt((l2MSq - l2Mean.toDouble() * l2Mean).coerceAtLeast(0.0)).toFloat()

        // ── Cosine similarity between consecutive frames ───────────────────────
        val sims = mutableListOf<Double>()
        for (i in 0 until n - 1) {
            val a = allFrameFeatures[i]
            val b = allFrameFeatures[i + 1]
            var dot = 0.0; var normA = 0.0; var normB = 0.0
            for (d in 0 until dim) {
                dot  += a[d].toDouble() * b[d]
                normA += a[d].toDouble() * a[d]
                normB += b[d].toDouble() * b[d]
            }
            sims.add(dot / (sqrt(normA) * sqrt(normB) + 1e-10))
        }
        val simMean = if (sims.isEmpty()) 0f else sims.average().toFloat()
        val simStd  = if (sims.size < 2) 0f else {
            val simMSq = sims.map { it * it }.average()
            sqrt((simMSq - simMean.toDouble() * simMean).coerceAtLeast(0.0)).toFloat()
        }

        val sparsity = sparse.toFloat() / total

        return mapOf(
            "deep_feat_mean"         to gMean,
            "deep_feat_std"          to gStd,
            "deep_feat_max"          to maxAll,
            "deep_feat_min"          to minAll,
            "deep_temporal_var_mean" to tvMean,
            "deep_temporal_var_std"  to tvStd,
            "deep_l2_norm_mean"      to l2Mean,
            "deep_l2_norm_std"       to l2Std,
            "deep_similarity_mean"   to simMean,
            "deep_similarity_std"    to simStd,
            "deep_sparsity"          to sparsity
        )
    }

    /**
     * Validate that deep features look reasonable (not all zeros, not NaN/Inf).
     * Returns true if features appear valid.
     */
    private fun validateDeepFeatures(features: Map<String, Double>): Boolean {
        if (features.isEmpty()) return false
        if (features.values.all { it == 0.0 }) {
            Log.w(TAG, "Deep features are ALL ZERO — likely extraction failed")
            return false
        }
        if (features.values.any { it.isNaN() || it.isInfinite() }) {
            Log.w(TAG, "Deep features contain NaN/Inf — invalid")
            return false
        }
        return true
    }

    /**
     * Main entry point: extract the 11 canonical deep_* features using Kotlin ONNX runtime.
     * Returns a JSON string to be passed to Python's extract_features().
     * Returns "{}" if no deep models are loaded (Python will use zeros as fallback).
     */
    private fun extractDeepFeaturesJson(videoPath: String): String {
        val sessions = listOfNotNull(
            resnet50Session?.let    { "resnet50"         to it },
            efficientnetSession?.let { "efficientnet_b0" to it }
        )

        if (sessions.isEmpty()) {
            Log.w(TAG, "No deep feature ONNX models loaded — deep_* features will be 0.0")
            Log.w(TAG, "FIX: Re-export ONNX models using export_deep_features_onnx.py")
            Log.w(TAG, "     Then rebuild the app with the new .onnx files in assets/models/")
            return "{}"
        }

        return try {
            val frames = extractVideoFrames(videoPath)
            if (frames.isEmpty()) {
                Log.w(TAG, "No frames could be extracted — deep_* features will be 0.0")
                return "{}"
            }

            val allModelStats = mutableListOf<Map<String, Float>>()

            for ((modelName, session) in sessions) {
                val frameFeatures = mutableListOf<FloatArray>()
                for (frame in frames) {
                    val feat = runDeepModelOnFrame(session, frame)
                    if (feat != null && feat.isNotEmpty()) frameFeatures.add(feat)
                }
                if (frameFeatures.isNotEmpty()) {
                    val stats = computeDeepStats(frameFeatures)
                    allModelStats.add(stats)
                    Log.d(TAG, "$modelName: ${frameFeatures.size} frames, dim=${frameFeatures[0].size}")
                } else {
                    Log.w(TAG, "$modelName: no frame features extracted")
                }
            }

            if (allModelStats.isEmpty()) return "{}"

            // Average canonical keys across models
            val result = JSONObject()
            for (key in DEEP_FEATURE_NAMES) {
                val vals = allModelStats.mapNotNull { it[key] }
                if (vals.isNotEmpty()) {
                    result.put(key, vals.average())
                } else {
                    result.put(key, 0.0)
                }
            }

            // Validate the result
            val featMap = DEEP_FEATURE_NAMES.associateWith { result.optDouble(it, 0.0) }
            if (!validateDeepFeatures(featMap)) {
                Log.w(TAG, "Deep feature validation failed — sending {} to Python")
                return "{}"
            }

            Log.d(TAG, "Deep features extracted successfully: ${result.length()} features via Kotlin ONNX")
            val mean = String.format("%.4f", result.optDouble("deep_feat_mean", 0.0))
            val std  = String.format("%.4f", result.optDouble("deep_feat_std", 0.0))

            Log.d(TAG, "Sample: mean=$mean, std=$std")
            result.toString()

        } catch (e: Exception) {
            Log.e(TAG, "extractDeepFeaturesJson failed: ${e.message}", e)
            "{}"
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Model Picker
    // ─────────────────────────────────────────────────────────────────────────

    private fun showModelPicker() {
        val available = getAvailableModels()
        if (available.isEmpty()) {
            Toast.makeText(this, "Khong tim thay file .onnx trong assets/models/", Toast.LENGTH_LONG).show()
            return
        }
        val displayNames = available.map { name ->
            if (name == activeModel) "✓ $name (dang dung)" else name
        }.toTypedArray()

        AlertDialog.Builder(this, R.style.DarkDialog)
            .setTitle("Chon Model (${available.size} models)")
            .setItems(displayNames) { _, i ->
                val selected = available[i]
                if (selected != activeModel) loadModels(selected)
                else Toast.makeText(this, "Model nay dang duoc dung", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Huy", null)
            .show()
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  File / URL picking
    // ─────────────────────────────────────────────────────────────────────────

    private fun pickVideoFile() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "video/*"
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        startActivityForResult(Intent.createChooser(intent, "Select Video"), REQ_PICK_VIDEO)
    }

    private fun handlePickedVideo(uri: Uri) {
        val name = getFileName(uri) ?: "video.mp4"
        val dest = File(cacheDir, name)
        contentResolver.openInputStream(uri)?.use { inp ->
            dest.outputStream().use { out -> inp.copyTo(out) }
        }
        tempFiles.add(dest)
        setVideo(dest.absolutePath, name, isTemp = false)
    }

    private fun getFileName(uri: Uri): String? {
        var name: String? = null
        contentResolver.query(uri, null, null, null, null)?.use { c ->
            val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (c.moveToFirst() && i >= 0) name = c.getString(i)
        }
        return name ?: uri.lastPathSegment
    }

    private fun setVideo(path: String, displayName: String, isTemp: Boolean) {
        selectedPath = path
        binding.tvFileName.text = displayName
        binding.tvFileTag.text  = if (isTemp) "[TMP]" else "[FILE]"
        binding.tvFileTag.setTextColor(
            ContextCompat.getColor(this, if (isTemp) R.color.warning else R.color.accent))
        binding.btnAnalyze.isEnabled = isModelLoaded
        setBadge("* LOADED", R.color.accent)
        binding.tvStatus.text = "File ready. Tap RUN ANALYSIS."
    }

    private fun showUrlDialog() {
        val view     = layoutInflater.inflate(R.layout.dialog_url, null)
        val etUrl    = view.findViewById<EditText>(R.id.etUrl)
        val tvStatus = view.findViewById<TextView>(R.id.tvUrlStatus)

        val clip = getSystemService(android.content.ClipboardManager::class.java)
        val cb   = clip?.primaryClip?.getItemAt(0)?.text?.toString() ?: ""
        if (cb.startsWith("http")) etUrl.setText(cb)

        val dialog = AlertDialog.Builder(this, R.style.DarkDialog)
            .setTitle("Paste Video URL")
            .setView(view)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Download & Analyze", null)
            .create()

        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val url = etUrl.text.toString().trim()
                if (!url.startsWith("http")) {
                    tvStatus.text = "Invalid URL - must start with http(s)"
                    tvStatus.setTextColor(ContextCompat.getColor(this, R.color.danger))
                    return@setOnClickListener
                }
                tvStatus.text = "Downloading..."
                tvStatus.setTextColor(ContextCompat.getColor(this, R.color.warning))
                dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = false

                downloadVideo(url,
                    onSuccess = { path, name ->
                        dialog.dismiss()
                        setVideo(path, name, isTemp = true)
                    },
                    onError = { err ->
                        tvStatus.text = err
                        tvStatus.setTextColor(ContextCompat.getColor(this, R.color.danger))
                        dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = true
                    }
                )
            }
        }
        dialog.show()
    }

    private fun downloadVideo(
        url: String,
        onSuccess: (path: String, name: String) -> Unit,
        onError: (String) -> Unit
    ) {
        executor.execute {
            try {
                val raw  = Python.getInstance()
                    .getModule("downloader")
                    .callAttr("download_video", url)
                    .toString()
                val json = JSONObject(raw)
                if (json.has("error")) {
                    mainHandler.post { onError(json.getString("error")) }
                } else {
                    val path = json.getString("path")
                    val name = json.getString("name")
                    tempFiles.add(File(path))
                    mainHandler.post { onSuccess(path, name) }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Download failed", e)
                mainHandler.post { onError("Download failed: ${e.message}") }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Video Analysis (main flow)
    // ─────────────────────────────────────────────────────────────────────────

    private fun analyzeVideo(videoPath: String) {
        setControlsEnabled(false)
        binding.resultCard.visibility = View.GONE
        binding.btnClear.visibility   = View.GONE
        binding.progressBar.progress  = 0
        binding.tvStatus.text         = "Starting analysis..."
        setBadge(">> ANALYZING", R.color.warning)
        animateProgress()

        executor.execute {
            try {
                Log.d(TAG, "Starting analysis for: $videoPath")

                // ── Step 1: Extract deep features via Kotlin ONNX ─────────────
                mainHandler.post { binding.tvStatus.text = "Extracting deep visual features..." }
                val deepFeaturesJson = extractDeepFeaturesJson(videoPath)

                // Log whether we got real features or fallback
                if (deepFeaturesJson == "{}") {
                    Log.w(TAG, "⚠ Deep features unavailable — using zeros (accuracy may be lower)")
                    Log.w(TAG, "  To fix: re-export ONNX with export_deep_features_onnx.py (opset=11)")
                } else {
                    Log.d(TAG, "✓ Deep features OK: ${deepFeaturesJson.take(100)}...")
                }

                // ── Step 2: Python extracts traditional features + merges deep ─
                mainHandler.post { binding.tvStatus.text = "Analyzing forensic features..." }
                val raw = Python.getInstance()
                    .getModule("detector")
                    .callAttr("extract_features", videoPath, deepFeaturesJson)
                    .toString()

                Log.d(TAG, "Python result length: ${raw.length}")

                val json = JSONObject(raw)

                if (json.has("error")) {
                    mainHandler.post {
                        showError(json.getString("error"))
                        setControlsEnabled(true)
                    }
                    return@execute
                }

                val vectorArr = json.getJSONArray("vector")
                val floats    = FloatArray(vectorArr.length()) { vectorArr.getDouble(it).toFloat() }

                Log.d(TAG, "Feature vector length: ${floats.size}")

                // Check debug info
                val debugInfo = json.optJSONObject("debug_info")
                if (debugInfo != null) {
                    val nDeep = debugInfo.optInt("n_deep_features", 0)
                    val nMiss = debugInfo.optInt("n_missing_features", 0)
                    Log.d(TAG, "Deep features used: $nDeep, Missing: $nMiss")
                    if (nDeep == 0) {
                        Log.w(TAG, "⚠ No deep features — predictions may differ from PC app")
                    }
                }

                // ── Step 3: Run LightGBM ONNX classifier ──────────────────────
                mainHandler.post { binding.tvStatus.text = "Running AI classifier..." }
                val (probFake, label) = runOnnxInference(floats)

                Log.d(TAG, "ONNX result - probFake: $probFake, label: $label")

                val finalJson = JSONObject().apply {
                    put("prediction",       if (label == 1) "FAKE" else "REAL")
                    put("probability_fake", probFake)
                    put("confidence",       json.getString("fusion_confidence"))
                    put("artifact_score",   json.getDouble("fusion_artifact"))
                    put("reality_score",    json.getDouble("fusion_reality"))
                    put("explanations",     json.getJSONArray("explanations"))
                }

                mainHandler.post {
                    binding.progressBar.progress = 100
                    showResult(finalJson)
                    setControlsEnabled(true)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Analysis failed", e)
                mainHandler.post {
                    showError("Analysis failed: ${e.message}")
                    setControlsEnabled(true)
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  LightGBM ONNX Classifier Inference
    // ─────────────────────────────────────────────────────────────────────────

    private fun runOnnxInference(floats: FloatArray): Pair<Float, Int> {
        val session = ortSession ?: throw IllegalStateException("ONNX session chua duoc load")
        val env     = ortEnv    ?: OrtEnvironment.getEnvironment()

        val shape  = longArrayOf(1, floats.size.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), shape)

        tensor.use {
            val output = session.run(mapOf(session.inputNames.first() to it))
            output.use {
                val labels = output[0].value as LongArray
                val label  = labels[0].toInt()

                Log.d(TAG, "ONNX raw label: $label")

                val probFake: Float = try {
                    @Suppress("UNCHECKED_CAST")
                    val probList = output[1].value as List<*>
                    if (probList.isNotEmpty()) {
                        @Suppress("UNCHECKED_CAST")
                        val probMap = probList[0] as Map<*, *>
                        probMap[1L] as? Float
                            ?: probMap[1] as? Float
                            ?: probMap["1"] as? Float
                            ?: 0.5f
                    } else 0.5f
                } catch (e1: Exception) {
                    Log.w(TAG, "Failed to parse ONNX output format 1: $e1")
                    try {
                        val probArray = output[1].value as? FloatArray
                        if (probArray != null && probArray.size >= 2) probArray[1] else 0.5f
                    } catch (e2: Exception) {
                        Log.w(TAG, "Failed to parse ONNX output format 2: $e2")
                        if (label == 1) 0.75f else 0.25f
                    }
                }

                Log.d(TAG, "Final probFake: $probFake")
                return Pair(probFake, label)
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  UI Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private fun animateProgress() {
        val steps = listOf(10 to 400L, 35 to 700L, 62 to 900L, 80 to 500L)
        steps.forEachIndexed { idx, (value, delay) ->
            mainHandler.postDelayed({
                if (binding.progressBar.progress < 100)
                    binding.progressBar.progress = value
            }, delay * (idx + 1))
        }
    }

    private fun showResult(json: JSONObject) {
        val prediction = json.getString("prediction")
        val probFake   = json.getDouble("probability_fake")
        val probReal   = 1.0 - probFake
        val confidence = json.getString("confidence")
        val artifact   = json.getDouble("artifact_score")
        val reality    = json.getDouble("reality_score")
        val isFake     = prediction == "FAKE"

        val verdictColor = ContextCompat.getColor(this,
            if (isFake) R.color.danger else R.color.success)
        val bgColor = ContextCompat.getColor(this,
            if (isFake) R.color.danger_dim else R.color.success_dim)

        val deepWarning = if (deepModelsLoaded == 0) " ⚠deep=0" else ""

        binding.resultCard.visibility = View.VISIBLE
        binding.btnClear.visibility   = View.VISIBLE
        binding.tvStatus.text         = "Analysis complete. Model: $activeModel$deepWarning"
        setBadge(
            if (isFake) "!! FAKE" else "OK REAL",
            if (isFake) R.color.danger else R.color.success
        )

        binding.resultBanner.setBackgroundColor(bgColor)
        binding.tvVerdictIcon.text = if (isFake) "!!" else "OK"
        binding.tvVerdictIcon.setTextColor(verdictColor)
        binding.tvVerdictText.text = if (isFake) "AI GENERATED" else "AUTHENTIC VIDEO"
        binding.tvVerdictText.setTextColor(verdictColor)
        binding.tvConfidence.text  = "Confidence: $confidence  |  Model: $activeModel"

        binding.tvProbFakeVal.text    = "${"%.1f".format(probFake * 100)}%"
        binding.tvProbRealVal.text    = "${"%.1f".format(probReal * 100)}%"
        binding.progressFake.progress = (probFake * 100).toInt()
        binding.progressReal.progress = (probReal * 100).toInt()
        binding.progressFake.progressTintList =
            android.content.res.ColorStateList.valueOf(
                ContextCompat.getColor(this, R.color.danger))
        binding.progressReal.progressTintList =
            android.content.res.ColorStateList.valueOf(
                ContextCompat.getColor(this, R.color.success))

        binding.tvArtifactVal.text = "${"%.3f".format(artifact)}"
        binding.tvRealityVal.text  = "${"%.3f".format(reality)}"

        binding.layoutFindings.removeAllViews()
        val explanations = json.optJSONArray("explanations")
        if (explanations != null) {
            for (i in 0 until minOf(explanations.length(), 5)) {
                val tv = TextView(this).apply {
                    text     = "> ${explanations.getString(i)}"
                    textSize = 12f
                    setTextColor(ContextCompat.getColor(context, R.color.text_primary))
                    setPadding(0, 8, 0, 8)
                }
                binding.layoutFindings.addView(tv)
            }
        }

        // Show warning if deep models not loaded
        if (deepModelsLoaded == 0) {
            val warnTv = TextView(this).apply {
                text = "⚠ Deep features unavailable (ONNX IR version issue). " +
                        "Re-export models with export_deep_features_onnx.py for full accuracy."
                textSize = 11f
                setTextColor(ContextCompat.getColor(context, R.color.warning))
                setPadding(0, 12, 0, 0)
            }
            binding.layoutFindings.addView(warnTv)
        }

        binding.scrollView.post {
            binding.scrollView.smoothScrollTo(0, binding.resultCard.top)
        }
    }

    private fun showError(msg: String) {
        binding.tvStatus.text        = msg
        binding.progressBar.progress = 0
        setBadge("!! ERROR", R.color.danger)
        Log.e(TAG, "Error: $msg")
    }

    private fun copyAsset(assetPath: String): File {
        val outFile = File(filesDir, assetPath)
        outFile.parentFile?.mkdirs()
        assets.open(assetPath).use { inp ->
            outFile.outputStream().use { out -> inp.copyTo(out) }
        }
        return outFile
    }
}