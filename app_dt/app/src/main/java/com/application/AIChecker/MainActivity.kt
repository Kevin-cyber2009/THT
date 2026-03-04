package com.application.AIChecker

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
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val executor    = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var isModelLoaded = false
    private var selectedPath  : String? = null
    private val tempFiles     = mutableListOf<File>()

    companion object {
        private const val REQ_PICK_VIDEO = 1001
        // Them ten .pkl vao day neu co nhieu model
        private val MODELS      = listOf("alpha.pkl", "beta.pkl")
        private var activeModel = MODELS[0]
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!Python.isStarted()) Python.start(AndroidPlatform(this))

        setupUI()
        loadModels(activeModel)
    }

    override fun onDestroy() {
        super.onDestroy()
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
                val modelFile  = copyAsset("models/$modelName")

                val result = Python.getInstance()
                    .getModule("detector")
                    .callAttr("load_models",
                        modelFile.absolutePath,
                        configFile.absolutePath)
                    .toString()

                mainHandler.post {
                    if (result == "OK") {
                        isModelLoaded        = true
                        activeModel          = modelName
                        binding.tvModelName.text = modelName
                        binding.tvStatus.text    = "Model ready - select a file or paste a URL."
                        setBadge("* READY", R.color.success)
                        setControlsEnabled(true)
                    } else {
                        binding.tvStatus.text = result
                        setBadge("!! ERROR", R.color.danger)
                    }
                }
            } catch (e: Exception) {
                val msg = "Load failed: ${e.message}"
                mainHandler.post {
                    binding.tvStatus.text = msg
                    setBadge("!! ERROR", R.color.danger)
                }
            }
        }
    }

    // ── Model Picker ───────────────────────────────────────────────────────────

    private fun showModelPicker() {
        val available = MODELS.filter { name ->
            runCatching { assets.open("models/$name").close() }.isSuccess
        }.toTypedArray()

        if (available.isEmpty()) {
            Toast.makeText(this, "No models found in assets/models/", Toast.LENGTH_SHORT).show()
            return
        }

        AlertDialog.Builder(this, R.style.DarkDialog)
            .setTitle("Select Model")
            .setItems(available) { _, i ->
                if (available[i] != activeModel) loadModels(available[i])
            }
            .setNegativeButton("Cancel", null)
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

        // Auto-paste clipboard neu la URL
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
                val raw = Python.getInstance()
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
                val raw = Python.getInstance()
                    .getModule("detector")
                    .callAttr("analyze_video", videoPath)
                    .toString()
                val json = JSONObject(raw)

                mainHandler.post {
                    binding.progressBar.progress = 100
                    if (json.has("error")) showError(json.getString("error"))
                    else showResult(json)
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
        binding.tvStatus.text         = "Analysis complete."
        setBadge(
            if (isFake) "!! FAKE" else "OK REAL",
            if (isFake) R.color.danger else R.color.success
        )

        // Banner
        binding.resultBanner.setBackgroundColor(bgColor)
        binding.tvVerdictIcon.text = if (isFake) "!!" else "OK"
        binding.tvVerdictIcon.setTextColor(verdictColor)
        binding.tvVerdictText.text = if (isFake) "DEEPFAKE DETECTED" else "AUTHENTIC VIDEO"
        binding.tvVerdictText.setTextColor(verdictColor)
        binding.tvConfidence.text  = "Confidence: $confidence"

        // Probability bars
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

        // Scores
        binding.tvArtifactVal.text = "${"%.3f".format(artifact)}"
        binding.tvRealityVal.text  = "${"%.3f".format(reality)}"

        // Key findings - inject TextViews dynamically
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

        // Auto-scroll to result
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
        if (!outFile.exists()) {
            assets.open(assetPath).use { inp ->
                outFile.outputStream().use { out -> inp.copyTo(out) }
            }
        }
        return outFile
    }
}