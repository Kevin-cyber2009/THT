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

    private var ortEnv           : OrtEnvironment? = null
    private var ortSession       : OrtSession?     = null
    private var resnet50Session  : OrtSession?     = null
    private var efficientnetSess : OrtSession?     = null
    private var deepModelsLoaded : Int = 0

    companion object {
        private const val REQ_PICK_VIDEO = 1001
        private var activeModel = "x.onnx"
        private const val TAG   = "AIChecker"

        private val DEEP_FEATURE_MODELS = listOf(
            "resnet50_features.onnx",
            "efficientnet_b0_features.onnx"
        )

        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD  = floatArrayOf(0.229f, 0.224f, 0.225f)
        private const val DEEP_SAMPLE_FRAMES = 10

        private val DEEP_FEATURE_NAMES = listOf(
            "deep_feat_mean", "deep_feat_std", "deep_feat_max", "deep_feat_min",
            "deep_temporal_var_mean", "deep_temporal_var_std",
            "deep_l2_norm_mean", "deep_l2_norm_std",
            "deep_similarity_mean", "deep_similarity_std", "deep_sparsity"
        )
    }

    // ─────────────────────────────────────────────────────────────────────────

    private fun getAvailableModels(): Array<String> {
        return try {
            assets.list("models")
                ?.filter { it.endsWith(".onnx") && !it.endsWith("_features.onnx") }
                ?.sorted()
                ?.toTypedArray()
                ?: arrayOf("x.onnx")
        } catch (e: Exception) { arrayOf("x.onnx") }
    }

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
        efficientnetSess?.close()
        ortEnv?.close()
        tempFiles.forEach { if (it.exists()) it.delete() }
        tempFiles.clear()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_PICK_VIDEO && resultCode == Activity.RESULT_OK)
            data?.data?.let { handlePickedVideo(it) }
    }

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
        binding.tvStatus.text = "Loading: $modelName ..."
        setBadge("* LOADING", R.color.warning)

        executor.execute {
            try {
                val configFile = copyAsset("config.yaml")
                val scalerName = modelName.removeSuffix(".onnx") + "_scaler.pkl"

                val scalerExists = try {
                    assets.open("models/$scalerName").close(); true
                } catch (e: Exception) { false }

                if (!scalerExists) {
                    mainHandler.post {
                        binding.tvStatus.text = "Missing: models/$scalerName"
                        setBadge("!! ERROR", R.color.danger)
                    }
                    return@execute
                }

                val scalerFile = copyAsset("models/$scalerName")

                // Also copy scaler_params.json if present (preferred for Android)
                val paramsName = modelName.removeSuffix(".onnx") + "_scaler_params.json"
                try {
                    assets.open("models/$paramsName").close()
                    copyAsset("models/$paramsName")
                    Log.d(TAG, "✓ Copied $paramsName (manual scaling mode)")
                } catch (e: Exception) {
                    Log.w(TAG, "No $paramsName found — falling back to sklearn scaler")
                    Log.w(TAG, "Run save_scaler_params.py on PC to fix sklearn version mismatch!")
                }

                // Copy deep feature ONNX models
                for (f in DEEP_FEATURE_MODELS) {
                    try { assets.open("models/$f").close(); copyAsset("models/$f"); Log.d(TAG, "Copied $f") }
                    catch (_: Exception) { Log.w(TAG, "Deep model not in assets: $f") }
                }

                loadDeepFeatureModels()

                val result = Python.getInstance()
                    .getModule("detector")
                    .callAttr("load_models", scalerFile.absolutePath, configFile.absolutePath)
                    .toString()

                Log.d(TAG, "load_models: $result")

                if (result != "OK") {
                    mainHandler.post { binding.tvStatus.text = result; setBadge("!! ERROR", R.color.danger) }
                    return@execute
                }

                val onnxFile = copyAsset("models/$modelName")
                val env      = ortEnv ?: OrtEnvironment.getEnvironment().also { ortEnv = it }
                ortSession?.close()
                ortSession = env.createSession(onnxFile.absolutePath)

                mainHandler.post {
                    isModelLoaded            = true
                    activeModel              = modelName
                    binding.tvModelName.text = modelName
                    val deepStatus = if (deepModelsLoaded > 0) "deep=$deepModelsLoaded ✓" else "deep=0 ⚠"
                    binding.tvStatus.text    = "Ready ($deepStatus) - select a file or URL."
                    setBadge("* READY", R.color.success)
                    setControlsEnabled(true)
                }
            } catch (e: Exception) {
                Log.e(TAG, "loadModels failed", e)
                mainHandler.post { binding.tvStatus.text = "Load failed: ${e.message}"; setBadge("!! ERROR", R.color.danger) }
            }
        }
    }

    private fun loadDeepFeatureModels() {
        val env = ortEnv ?: return
        resnet50Session?.close();  resnet50Session  = null
        efficientnetSess?.close(); efficientnetSess = null
        deepModelsLoaded = 0

        for ((name, setter) in listOf(
            "resnet50_features.onnx"       to { s: OrtSession -> resnet50Session  = s },
            "efficientnet_b0_features.onnx" to { s: OrtSession -> efficientnetSess = s }
        )) {
            val f = File(filesDir, "models/$name")
            if (!f.exists()) { Log.w(TAG, "$name not in filesDir"); continue }
            try {
                setter(env.createSession(f.absolutePath))
                deepModelsLoaded++
                Log.d(TAG, "✓ Loaded $name")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load $name: ${e.message}")
                if (e.message?.contains("IR version", ignoreCase = true) == true) {
                    Log.e(TAG, "→ Re-export ONNX with export_deep_features_onnx.py (opset=11)")
                }
            }
        }
        Log.d(TAG, "Deep models loaded: $deepModelsLoaded / ${DEEP_FEATURE_MODELS.size}")
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Deep Feature Extraction
    // ─────────────────────────────────────────────────────────────────────────

    private fun extractVideoFrames(videoPath: String): List<FloatArray> {
        val frames    = mutableListOf<FloatArray>()
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(videoPath)
            val durationMs = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: return frames
            if (durationMs <= 0) return frames

            for (i in 0 until DEEP_SAMPLE_FRAMES) {
                val frac   = if (DEEP_SAMPLE_FRAMES > 1) i.toDouble() / (DEEP_SAMPLE_FRAMES - 1) else 0.5
                val timeUs = (frac * durationMs * 1000L).toLong().coerceIn(0L, (durationMs - 1) * 1000L)
                try {
                    val bmp = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    if (bmp != null) { frames.add(preprocessBitmap(bmp)); bmp.recycle() }
                } catch (e: Exception) { Log.w(TAG, "Frame $i failed: ${e.message}") }
            }
        } catch (e: Exception) { Log.e(TAG, "extractFrames: ${e.message}") }
        finally { try { retriever.release() } catch (_: Exception) {} }
        Log.d(TAG, "Extracted ${frames.size} frames")
        return frames
    }

    private fun preprocessBitmap(src: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(src, 256, 256, true)
        val cropped = Bitmap.createBitmap(resized, 16, 16, 224, 224)
        if (resized !== src) resized.recycle()
        val pixels = IntArray(224 * 224)
        cropped.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        cropped.recycle()
        val t = FloatArray(3 * 224 * 224)
        for (i in pixels.indices) {
            val p = pixels[i]
            t[i]                 = ((p shr 16 and 0xFF) / 255f - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            t[224 * 224 + i]     = ((p shr 8  and 0xFF) / 255f - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            t[2 * 224 * 224 + i] = ((p         and 0xFF) / 255f - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        }
        return t
    }

    private fun runDeepModel(session: OrtSession, frameData: FloatArray): FloatArray? {
        return try {
            val env = ortEnv ?: return null
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(frameData), longArrayOf(1, 3, 224, 224))
            tensor.use {
                val out = session.run(mapOf(session.inputNames.first() to it))
                out.use { flattenOutput(out[0].value) }
            }
        } catch (e: Exception) { Log.w(TAG, "Deep inference: ${e.message}"); null }
    }

    @Suppress("UNCHECKED_CAST")
    private fun flattenOutput(value: Any?): FloatArray? {
        return when (value) {
            is FloatArray  -> value
            is Array<*>   -> { val buf = mutableListOf<Float>(); flattenInto(value, buf); buf.toFloatArray() }
            else           -> null
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun flattenInto(arr: Array<*>, buf: MutableList<Float>) {
        for (item in arr) when (item) {
            is Float       -> buf.add(item)
            is Double      -> buf.add(item.toFloat())
            is FloatArray  -> item.forEach { buf.add(it) }
            is DoubleArray -> item.forEach { buf.add(it.toFloat()) }
            is Array<*>    -> flattenInto(item, buf)
        }
    }

    private fun computeDeepStats(allFeatures: List<FloatArray>): Map<String, Float> {
        val n = allFeatures.size
        if (n == 0) return emptyMap()
        val dim = allFeatures[0].size
        if (dim == 0) return emptyMap()

        var sumAll = 0.0; var sumSqAll = 0.0
        var maxAll = Float.NEGATIVE_INFINITY; var minAll = Float.POSITIVE_INFINITY
        var total = 0; var sparse = 0

        for (feat in allFeatures) for (v in feat) {
            sumAll += v; sumSqAll += v.toDouble() * v
            if (v > maxAll) maxAll = v; if (v < minAll) minAll = v
            if (abs(v) < 0.01f) sparse++; total++
        }

        val gMean = (sumAll / total).toFloat()
        val gStd  = sqrt((sumSqAll / total - gMean.toDouble() * gMean).coerceAtLeast(0.0)).toFloat()

        val tvArr = DoubleArray(dim) { d ->
            val m = allFeatures.sumOf { it[d].toDouble() } / n
            allFeatures.sumOf { val diff = it[d] - m; diff * diff } / n
        }
        val tvMean = tvArr.average().toFloat()
        val tvStd  = sqrt((tvArr.map { it * it }.average() - tvMean.toDouble() * tvMean).coerceAtLeast(0.0)).toFloat()

        val l2 = DoubleArray(n) { i -> sqrt(allFeatures[i].sumOf { v -> v.toDouble() * v }) }
        val l2Mean = l2.average().toFloat()
        val l2Std  = sqrt((l2.map { it * it }.average() - l2Mean.toDouble() * l2Mean).coerceAtLeast(0.0)).toFloat()

        val sims = (0 until n - 1).map { i ->
            val a = allFeatures[i]; val b = allFeatures[i + 1]
            var dot = 0.0; var nA = 0.0; var nB = 0.0
            for (d in 0 until dim) { dot += a[d] * b[d]; nA += a[d] * a[d]; nB += b[d] * b[d] }
            dot / (sqrt(nA) * sqrt(nB) + 1e-10)
        }
        val simMean = if (sims.isEmpty()) 0f else sims.average().toFloat()
        val simStd  = if (sims.size < 2) 0f else
            sqrt((sims.map { it * it }.average() - simMean.toDouble() * simMean).coerceAtLeast(0.0)).toFloat()

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
            "deep_sparsity"          to sparse.toFloat() / total
        )
    }

    private fun extractDeepFeaturesJson(videoPath: String): String {
        val sessions = listOfNotNull(
            resnet50Session?.let  { "resnet50"         to it },
            efficientnetSess?.let { "efficientnet_b0"  to it }
        )

        if (sessions.isEmpty()) {
            Log.w(TAG, "No deep models loaded — deep features = 0")
            return "{}"
        }

        return try {
            val frames = extractVideoFrames(videoPath)
            if (frames.isEmpty()) { Log.w(TAG, "No frames extracted"); return "{}" }

            val allStats = mutableListOf<Map<String, Float>>()

            for ((modelName, session) in sessions) {
                val frameFeats = frames.mapNotNull { runDeepModel(session, it) }
                    .filter { it.isNotEmpty() }
                if (frameFeats.isNotEmpty()) {
                    val stats = computeDeepStats(frameFeats)
                    allStats.add(stats)
                    Log.d(TAG, "$modelName: ${frameFeats.size} frames, dim=${frameFeats[0].size}, " +
                            "mean=${"%.4f".format(stats["deep_feat_mean"] ?: 0f)}")
                } else {
                    Log.w(TAG, "$modelName: no features extracted")
                }
            }

            if (allStats.isEmpty()) return "{}"

            val result = JSONObject()
            for (key in DEEP_FEATURE_NAMES) {
                val vals = allStats.mapNotNull { it[key] }
                result.put(key, if (vals.isNotEmpty()) vals.average() else 0.0)
            }

            // Quick sanity check
            val meanVal = result.optDouble("deep_feat_mean", 0.0)
            val stdVal  = result.optDouble("deep_feat_std",  0.0)
            Log.d(TAG, "Deep features OK: mean=${"%.4f".format(meanVal)}, std=${"%.4f".format(stdVal)}")

            if (meanVal == 0.0 && stdVal == 0.0) {
                Log.w(TAG, "All deep features are 0 — extraction may have failed")
                return "{}"
            }

            result.toString()

        } catch (e: Exception) {
            Log.e(TAG, "extractDeepFeatures: ${e.message}", e)
            "{}"
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Model Picker
    // ─────────────────────────────────────────────────────────────────────────

    private fun showModelPicker() {
        val available = getAvailableModels()
        if (available.isEmpty()) {
            Toast.makeText(this, "No .onnx in assets/models/", Toast.LENGTH_LONG).show()
            return
        }
        val names = available.map { if (it == activeModel) "✓ $it (active)" else it }.toTypedArray()
        AlertDialog.Builder(this, R.style.DarkDialog)
            .setTitle("Select Model")
            .setItems(names) { _, i ->
                if (available[i] != activeModel) loadModels(available[i])
                else Toast.makeText(this, "Already active", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  File / URL picking
    // ─────────────────────────────────────────────────────────────────────────

    private fun pickVideoFile() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "video/*"; addCategory(Intent.CATEGORY_OPENABLE)
        }
        startActivityForResult(Intent.createChooser(intent, "Select Video"), REQ_PICK_VIDEO)
    }

    private fun handlePickedVideo(uri: Uri) {
        val name = getFileName(uri) ?: "video.mp4"
        val dest = File(cacheDir, name)
        contentResolver.openInputStream(uri)?.use { it.copyTo(dest.outputStream()) }
        tempFiles.add(dest)
        setVideo(dest.absolutePath, name, false)
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
        binding.tvFileTag.setTextColor(ContextCompat.getColor(this,
            if (isTemp) R.color.warning else R.color.accent))
        binding.btnAnalyze.isEnabled = isModelLoaded
        setBadge("* LOADED", R.color.accent)
        binding.tvStatus.text = "Ready. Tap RUN ANALYSIS."
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
                    tvStatus.text = "Invalid URL — must start with https://"
                    tvStatus.setTextColor(ContextCompat.getColor(this, R.color.danger))
                    return@setOnClickListener
                }
                tvStatus.text = "Downloading..."
                tvStatus.setTextColor(ContextCompat.getColor(this, R.color.warning))
                dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = false

                downloadVideo(url,
                    onSuccess = { path, name -> dialog.dismiss(); setVideo(path, name, true) },
                    onError   = { err ->
                        tvStatus.text = err
                        tvStatus.setTextColor(ContextCompat.getColor(this, R.color.danger))
                        dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = true
                    }
                )
            }
        }
        dialog.show()
    }

    private fun downloadVideo(url: String, onSuccess: (String, String) -> Unit, onError: (String) -> Unit) {
        executor.execute {
            try {
                val raw  = Python.getInstance().getModule("downloader")
                    .callAttr("download_video", url).toString()
                val json = JSONObject(raw)
                if (json.has("error")) mainHandler.post { onError(json.getString("error")) }
                else {
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
    //  Analysis
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
                mainHandler.post { binding.tvStatus.text = "Extracting deep features..." }
                val deepJson = extractDeepFeaturesJson(videoPath)
                Log.d(TAG, "Deep JSON length: ${deepJson.length}, empty: ${deepJson == "{}"}")

                mainHandler.post { binding.tvStatus.text = "Analyzing forensic features..." }
                val raw = Python.getInstance().getModule("detector")
                    .callAttr("extract_features", videoPath, deepJson).toString()

                val json = JSONObject(raw)
                if (json.has("error")) {
                    mainHandler.post { showError(json.getString("error")); setControlsEnabled(true) }
                    return@execute
                }

                // Log debug info from Python
                val dbg = json.optJSONObject("debug_info")
                if (dbg != null) {
                    Log.d(TAG, "Python debug: deep_valid=${dbg.optBoolean("deep_features_valid")}, " +
                            "scaled_mean=${"%.4f".format(dbg.optDouble("scaled_mean", 0.0))}, " +
                            "scaled_std=${"%.4f".format(dbg.optDouble("scaled_std", 0.0))}")
                }

                val vectorArr = json.getJSONArray("vector")
                val floats    = FloatArray(vectorArr.length()) { vectorArr.getDouble(it).toFloat() }
                Log.d(TAG, "Feature vector: dim=${floats.size}, " +
                        "mean=${"%.4f".format(floats.average())}")

                mainHandler.post { binding.tvStatus.text = "Running classifier..." }
                val (probFake, label) = runOnnxInference(floats)
                Log.d(TAG, "ONNX: label=$label, probFake=${"%.4f".format(probFake)}")

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
                mainHandler.post { showError("Analysis failed: ${e.message}"); setControlsEnabled(true) }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  ONNX Classifier — FIXED probability parsing
    // ─────────────────────────────────────────────────────────────────────────

    private fun runOnnxInference(floats: FloatArray): Pair<Float, Int> {
        val session = ortSession ?: throw IllegalStateException("ONNX session not loaded")
        val env     = ortEnv    ?: OrtEnvironment.getEnvironment()

        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floats),
            longArrayOf(1, floats.size.toLong()))

        tensor.use {
            val output = session.run(mapOf(session.inputNames.first() to it))
            output.use {

                // ── Parse label ──────────────────────────────────────────────
                val label: Int = when (val rawLabel = output[0].value) {
                    is LongArray  -> rawLabel[0].toInt()
                    is IntArray   -> rawLabel[0]
                    is Array<*>   -> (rawLabel[0] as? Long)?.toInt() ?: (rawLabel[0] as? Int) ?: 0
                    else          -> 0
                }
                Log.d(TAG, "ONNX label raw=${output[0].value?.javaClass?.simpleName}, label=$label")

                // ── Parse probability — handles Float, Double, Number ─────────
                // LightGBM ONNX output[1] is typically List<Map<Long, Float>> or List<Map<Long, Double>>
                val probFake: Float = try {
                    @Suppress("UNCHECKED_CAST")
                    val probList = output[1].value as? List<*>
                    Log.d(TAG, "ONNX prob type: ${output[1].value?.javaClass?.simpleName}")

                    if (probList != null && probList.isNotEmpty()) {
                        @Suppress("UNCHECKED_CAST")
                        val probMap = probList[0] as? Map<*, *>
                        if (probMap != null) {
                            Log.d(TAG, "ONNX probMap keys: ${probMap.keys.map { "${it?.javaClass?.simpleName}:$it" }}")

                            // Try all possible key types for class-1 (FAKE)
                            val rawProb = probMap[1L]      // Long key
                                ?: probMap[1]              // Int key
                                ?: probMap["1"]            // String key
                                ?: probMap[1.0]            // Double key
                                ?: probMap[1.0f]           // Float key

                            Log.d(TAG, "ONNX rawProb=$rawProb (${rawProb?.javaClass?.simpleName})")

                            // Convert to Float regardless of the original type
                            when (rawProb) {
                                is Float  -> rawProb
                                is Double -> rawProb.toFloat()
                                is Number -> rawProb.toFloat()
                                else      -> {
                                    // Last resort: look for the higher-value entry (= FAKE prob)
                                    val values = probMap.values.mapNotNull { v ->
                                        when (v) {
                                            is Float  -> v.toDouble()
                                            is Double -> v
                                            is Number -> v.toDouble()
                                            else      -> null
                                        }
                                    }
                                    Log.d(TAG, "ONNX fallback: probMap values = $values")
                                    if (values.size >= 2) {
                                        // Assume values are [prob_real, prob_fake]
                                        values[1].toFloat()
                                    } else {
                                        if (label == 1) 0.75f else 0.25f
                                    }
                                }
                            }
                        } else null
                    } else null
                } catch (e: Exception) {
                    Log.w(TAG, "Prob parse exception: $e")
                    null
                } ?: run {
                    // Try output[1] as a direct FloatArray
                    try {
                        val arr = output[1].value as? FloatArray
                        Log.d(TAG, "Prob fallback FloatArray: ${arr?.toList()}")
                        if (arr != null && arr.size >= 2) arr[1] else if (label == 1) 0.75f else 0.25f
                    } catch (_: Exception) {
                        Log.w(TAG, "All prob parsing failed — using label-based fallback")
                        if (label == 1) 0.75f else 0.25f
                    }
                }

                Log.d(TAG, "Final: label=$label, probFake=${"%.4f".format(probFake)}")
                return Pair(probFake, label)
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  UI
    // ─────────────────────────────────────────────────────────────────────────

    private fun animateProgress() {
        listOf(10 to 400L, 35 to 700L, 62 to 900L, 80 to 500L).forEachIndexed { idx, (v, d) ->
            mainHandler.postDelayed({
                if (binding.progressBar.progress < 100) binding.progressBar.progress = v
            }, d * (idx + 1))
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

        val vColor = ContextCompat.getColor(this, if (isFake) R.color.danger  else R.color.success)
        val bgColor= ContextCompat.getColor(this, if (isFake) R.color.danger_dim else R.color.success_dim)

        binding.resultCard.visibility = View.VISIBLE
        binding.btnClear.visibility   = View.VISIBLE
        val deepWarn = if (deepModelsLoaded == 0) " ⚠deep=0" else ""
        binding.tvStatus.text = "Done. Model: $activeModel$deepWarn"
        setBadge(if (isFake) "!! FAKE" else "OK REAL", if (isFake) R.color.danger else R.color.success)

        binding.resultBanner.setBackgroundColor(bgColor)
        binding.tvVerdictIcon.text = if (isFake) "!!" else "OK"
        binding.tvVerdictIcon.setTextColor(vColor)
        binding.tvVerdictText.text = if (isFake) "AI GENERATED" else "AUTHENTIC VIDEO"
        binding.tvVerdictText.setTextColor(vColor)
        binding.tvConfidence.text  = "Confidence: $confidence  |  $activeModel"

        binding.tvProbFakeVal.text    = "${"%.1f".format(probFake * 100)}%"
        binding.tvProbRealVal.text    = "${"%.1f".format(probReal * 100)}%"
        binding.progressFake.progress = (probFake * 100).toInt()
        binding.progressReal.progress = (probReal * 100).toInt()
        binding.progressFake.progressTintList =
            android.content.res.ColorStateList.valueOf(ContextCompat.getColor(this, R.color.danger))
        binding.progressReal.progressTintList =
            android.content.res.ColorStateList.valueOf(ContextCompat.getColor(this, R.color.success))

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

        if (deepModelsLoaded == 0) {
            binding.layoutFindings.addView(TextView(this).apply {
                text     = "⚠ Deep features not available. Re-export ONNX with export_deep_features_onnx.py"
                textSize = 11f
                setTextColor(ContextCompat.getColor(context, R.color.warning))
                setPadding(0, 12, 0, 0)
            })
        }

        binding.scrollView.post { binding.scrollView.smoothScrollTo(0, binding.resultCard.top) }
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
        assets.open(assetPath).use { it.copyTo(outFile.outputStream()) }
        return outFile
    }
}