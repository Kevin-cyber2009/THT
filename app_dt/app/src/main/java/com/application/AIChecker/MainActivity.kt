package com.application.AIChecker

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
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

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val executor    = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var isModelLoaded = false
    private var selectedPath  : String? = null
    private val tempFiles     = mutableListOf<File>()

    private var ortEnv     : OrtEnvironment? = null
    private var ortSession : OrtSession?     = null

    companion object {
        private const val REQ_PICK_VIDEO = 1001
        private var activeModel = "x.onnx"
        private const val TAG = "AIChecker"
    }

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
    // Lifecycle
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
    // UI Setup
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
        binding.tvStatus.text         = "Ready — select a file or paste a URL."
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Model Loading
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

                // Call Python load_models
                val result = Python.getInstance()
                    .getModule("detector")
                    .callAttr("load_models",
                        scalerFile.absolutePath,
                        configFile.absolutePath)
                    .toString()

                Log.d(TAG, "load_models → $result")

                if (result != "OK") {
                    mainHandler.post {
                        binding.tvStatus.text = result
                        setBadge("!! ERROR", R.color.danger)
                    }
                    return@execute
                }

                // Load classifier ONNX
                val onnxFile = copyAsset("models/$modelName")
                val env      = ortEnv ?: OrtEnvironment.getEnvironment().also { ortEnv = it }
                ortSession?.close()
                ortSession = env.createSession(onnxFile.absolutePath)

                mainHandler.post {
                    isModelLoaded            = true
                    activeModel              = modelName
                    binding.tvModelName.text = modelName
                    binding.tvStatus.text    = "Ready — select a file or paste a URL."
                    setBadge("* READY", R.color.success)
                    setControlsEnabled(true)
                }
            } catch (e: Exception) {
                Log.e(TAG, "loadModels failed", e)
                mainHandler.post {
                    binding.tvStatus.text = "Load failed: ${e.message}"
                    setBadge("!! ERROR", R.color.danger)
                }
            }
        }
    }

    private fun showModelPicker() {
        val available = getAvailableModels()
        if (available.isEmpty()) {
            Toast.makeText(this, "No .onnx files found in assets/models/", Toast.LENGTH_LONG).show()
            return
        }
        val displayNames = available.map {
            if (it == activeModel) "✓ $it (active)" else it
        }.toTypedArray()
        AlertDialog.Builder(this, R.style.DarkDialog)
            .setTitle("Select Model (${available.size})")
            .setItems(displayNames) { _, i ->
                if (available[i] != activeModel) loadModels(available[i])
                else Toast.makeText(this, "Already active", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // File / URL picking
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
    // Analysis
    // ─────────────────────────────────────────────────────────────────────────

    private fun analyzeVideo(videoPath: String) {
        setControlsEnabled(false)
        binding.resultCard.visibility = View.GONE
        binding.btnClear.visibility   = View.GONE
        binding.progressBar.progress  = 0
        binding.tvStatus.text         = "Analyzing..."
        setBadge(">> ANALYZING", R.color.warning)
        animateProgress()

        executor.execute {
            try {
                Log.d(TAG, "Starting analysis: $videoPath")

                // Python extracts traditional features + neutralises deep features
                val raw = Python.getInstance()
                    .getModule("detector")
                    .callAttr("extract_features", videoPath)
                    .toString()

                Log.d(TAG, "Python result (first 300 chars): ${raw.take(300)}")

                val json = JSONObject(raw)

                if (json.has("error")) {
                    mainHandler.post {
                        showError(json.getString("error"))
                        setControlsEnabled(true)
                    }
                    return@execute
                }

                val vectorArr = json.getJSONArray("vector")
                val floats    = FloatArray(vectorArr.length()) {
                    vectorArr.getDouble(it).toFloat()
                }

                Log.d(TAG, "Feature vector dim=${floats.size}, " +
                        "mean=${"%.4f".format(floats.average())}")

                // Log debug info if present
                json.optJSONObject("debug_info")?.let { dbg ->
                    Log.d(TAG, "Debug: missing=${dbg.optInt("n_missing")}, " +
                            "scaled_mean=${"%.4f".format(dbg.optDouble("scaled_mean"))}, " +
                            "scaled_std=${"%.4f".format(dbg.optDouble("scaled_std"))}")
                }

                val (probFake, label) = runOnnxInference(floats)
                Log.d(TAG, "ONNX → label=$label, probFake=${"%.4f".format(probFake)}")

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
    // ONNX Inference  ←  THE MAIN FIX IS HERE
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Convert any numeric value (Float, Double, Long, Int …) to Float.
     * LightGBM ONNX returns probabilities as Double on most ONNX Runtime
     * versions — the old code cast directly to Float and always got null.
     */
    private fun anyToFloat(value: Any?): Float? = when (value) {
        is Float  -> value
        is Double -> value.toFloat()
        is Long   -> value.toFloat()
        is Int    -> value.toFloat()
        is Number -> value.toFloat()
        else      -> null
    }

    /**
     * Parse P(class=1) from output[1] of an LightGBM ONNX model.
     *
     * Possible formats returned by different ONNX Runtime / opset combos:
     *   A) List<Map<Long, Float>>    – e.g. [{0L→0.23f, 1L→0.77f}]
     *   B) List<Map<Long, Double>>   – same but Double values  ← was broken!
     *   C) FloatArray [p0, p1]
     *   D) Array<FloatArray>         – [[p0, p1]]
     */
    private fun parseProbFake(output: OrtSession.Result): Float {
        return try {
            when (val raw1 = output[1].value) {

                // ── Format A / B: List<Map<*, *>> ────────────────────────────
                is List<*> -> {
                    val map = raw1.firstOrNull() as? Map<*, *>
                    if (map != null) {
                        // Try the class-1 key under every plausible type
                        anyToFloat(map[1L])
                            ?: anyToFloat(map[1])
                            ?: anyToFloat(map["1"])
                            ?: run {
                                // Fallback: sort entries by numeric key, take index 1
                                val sorted = map.entries
                                    .mapNotNull { e ->
                                        val k = (e.key as? Number)?.toDouble()
                                            ?: return@mapNotNull null
                                        val v = anyToFloat(e.value)
                                            ?: return@mapNotNull null
                                        k to v
                                    }
                                    .sortedBy { it.first }
                                Log.d(TAG, "Prob map sorted: $sorted")
                                sorted.getOrNull(1)?.second ?: 0.5f
                            }
                    } else 0.5f
                }

                // ── Format C: FloatArray [p0, p1] ────────────────────────────
                is FloatArray -> {
                    Log.d(TAG, "Prob FloatArray: ${raw1.toList()}")
                    if (raw1.size >= 2) raw1[1] else 0.5f
                }

                // ── Format D: Array<FloatArray> ──────────────────────────────
                is Array<*> -> {
                    val inner = raw1.firstOrNull()
                    when (inner) {
                        is FloatArray  -> if (inner.size >= 2) inner[1] else 0.5f
                        is DoubleArray -> if (inner.size >= 2) inner[1].toFloat() else 0.5f
                        else           -> 0.5f
                    }
                }

                else -> {
                    Log.w(TAG, "Unknown prob output type: ${raw1?.javaClass?.name}")
                    0.5f
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "parseProbFake exception: ${e.message}")
            // IMPORTANT: do NOT use 0.75f here — that was the source of the bias!
            0.5f
        }
    }

    private fun runOnnxInference(floats: FloatArray): Pair<Float, Int> {
        val session = ortSession
            ?: throw IllegalStateException("ONNX session not loaded")
        val env = ortEnv ?: OrtEnvironment.getEnvironment()

        val tensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(floats),
            longArrayOf(1, floats.size.toLong())
        )

        tensor.use {
            val output = session.run(mapOf(session.inputNames.first() to it))
            output.use {

                // ── Parse label (output[0]) ───────────────────────────────────
                val label: Int = try {
                    when (val rawLabel = output[0].value) {
                        is LongArray  -> rawLabel[0].toInt()
                        is IntArray   -> rawLabel[0]
                        is Array<*>   -> {
                            val first = rawLabel.firstOrNull()
                            when (first) {
                                is Long -> first.toInt()
                                is Int  -> first
                                else    -> first.toString().toLongOrNull()?.toInt() ?: 0
                            }
                        }
                        else -> {
                            Log.w(TAG, "Unknown label type: ${output[0].value?.javaClass?.name}")
                            0
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Label parse error: ${e.message}")
                    0
                }

                Log.d(TAG, "Raw label=$label (type=${output[0].value?.javaClass?.simpleName})")

                // ── Parse probability (output[1]) ─────────────────────────────
                val probFake = parseProbFake(output)

                return Pair(probFake, label)
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UI helpers
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

        binding.resultCard.visibility = View.VISIBLE
        binding.btnClear.visibility   = View.VISIBLE
        binding.tvStatus.text         = "Analysis complete. Model: $activeModel"
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