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

    // ── Companion object duy nhất ─────────────────────────────────────────────
    companion object {
        private const val REQ_PICK_VIDEO = 1001
        private var activeModel = "x.onnx"
    }

    // ── Quét tất cả .onnx trong assets/models/ ────────────────────────────────
    private fun getAvailableModels(): Array<String> {
        return try {
            assets.list("models")
                ?.filter { it.endsWith(".onnx") }
                ?.sorted()
                ?.toTypedArray()
                ?: arrayOf("x.onnx")
        } catch (e: Exception) {
            arrayOf("x.onnx")
        }
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────────

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

    // ── UI Setup ───────────────────────────────────────────────────────────────

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

    // ── Load Model ─────────────────────────────────────────────────────────────

    private fun loadModels(modelName: String) {
        isModelLoaded = false
        setControlsEnabled(false)
        binding.tvStatus.text = "Loading model: $modelName ..."
        setBadge("* LOADING", R.color.warning)

        executor.execute {
            try {
                val configFile = copyAsset("config.yaml")

                // Scaler pkl: abc.onnx → abc_scaler.pkl
                val scalerName = modelName.removeSuffix(".onnx") + "_scaler.pkl"

                // Kiem tra scaler ton tai
                val scalerExists = try {
                    assets.open("models/$scalerName").close()
                    true
                } catch (e: Exception) {
                    false
                }

                if (!scalerExists) {
                    mainHandler.post {
                        binding.tvStatus.text = "Thiếu file: models/$scalerName"
                        setBadge("!! ERROR", R.color.danger)
                    }
                    return@execute
                }

                val scalerFile = copyAsset("models/$scalerName")

                // Python: load config + scaler + extractor
                val result = Python.getInstance()
                    .getModule("detector")
                    .callAttr("load_models",
                        scalerFile.absolutePath,
                        configFile.absolutePath)
                    .toString()

                if (result != "OK") {
                    mainHandler.post {
                        binding.tvStatus.text = result
                        setBadge("!! ERROR", R.color.danger)
                    }
                    return@execute
                }

                // Kotlin: load ONNX session
                val onnxFile = copyAsset("models/$modelName")
                val env      = ortEnv ?: OrtEnvironment.getEnvironment().also { ortEnv = it }
                ortSession?.close()
                ortSession = env.createSession(onnxFile.absolutePath)

                mainHandler.post {
                    isModelLoaded            = true
                    activeModel              = modelName
                    binding.tvModelName.text = modelName
                    binding.tvStatus.text    = "Model ready - select a file or paste a URL."
                    setBadge("* READY", R.color.success)
                    setControlsEnabled(true)
                }
            } catch (e: Exception) {
                mainHandler.post {
                    binding.tvStatus.text = "Load failed: ${e.message}"
                    setBadge("!! ERROR", R.color.danger)
                }
            }
        }
    }

    // ── Model Picker — quét động từ assets ────────────────────────────────────

    private fun showModelPicker() {
        val available = getAvailableModels()

        if (available.isEmpty()) {
            Toast.makeText(this, "Không tìm thấy file .onnx trong assets/models/", Toast.LENGTH_LONG).show()
            return
        }

        // Đánh dấu model đang dùng
        val displayNames = available.map { name ->
            if (name == activeModel) "✓ $name (đang dùng)" else name
        }.toTypedArray()

        AlertDialog.Builder(this, R.style.DarkDialog)
            .setTitle("Chọn Model (${available.size} models)")
            .setItems(displayNames) { _, i ->
                val selected = available[i]
                if (selected != activeModel) {
                    loadModels(selected)
                } else {
                    Toast.makeText(this, "Model này đang được dùng", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Huỷ", null)
            .show()
    }

    // ── Pick Video File ────────────────────────────────────────────────────────

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

    // ── URL Dialog ─────────────────────────────────────────────────────────────

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
                mainHandler.post { onError("Download failed: ${e.message}") }
            }
        }
    }

    // ── Analyze ────────────────────────────────────────────────────────────────

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
                val raw  = Python.getInstance()
                    .getModule("detector")
                    .callAttr("extract_features", videoPath)
                    .toString()
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
                val (probFake, label) = runOnnxInference(floats)

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
                mainHandler.post {
                    showError("Analysis failed: ${e.message}")
                    setControlsEnabled(true)
                }
            }
        }
    }

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

                val probFake: Float = try {
                    val onnxMap = output[1] as ai.onnxruntime.OnnxMap
                    val map     = onnxMap.value
                    (map[1L] as? Float) ?: (map[1] as? Float) ?: 0.5f
                } catch (e1: Exception) {
                    try {
                        val list    = output[1].value as List<*>
                        val onnxMap = list[0] as ai.onnxruntime.OnnxMap
                        val map     = onnxMap.value
                        (map[1L] as? Float) ?: (map[1] as? Float) ?: 0.5f
                    } catch (e2: Exception) {
                        if (label == 1) 0.75f else 0.25f
                    }
                }

                return Pair(probFake, label)
            }
        }
    }

    private fun animateProgress() {
        val steps = listOf(10 to 400L, 35 to 700L, 62 to 900L, 80 to 500L)
        steps.forEachIndexed { idx, (value, delay) ->
            mainHandler.postDelayed({
                if (binding.progressBar.progress < 100)
                    binding.progressBar.progress = value
            }, delay * (idx + 1))
        }
    }

    // ── Show Result ────────────────────────────────────────────────────────────

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
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private fun copyAsset(assetPath: String): File {
        val outFile = File(filesDir, assetPath)
        outFile.parentFile?.mkdirs()
        assets.open(assetPath).use { inp ->
            outFile.outputStream().use { out -> inp.copyTo(out) }
        }
        return outFile
    }
}