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
import com.application.AIChecker.databinding.ActivityMainBinding
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import kotlin.math.abs

class MainActivity : AppCompatActivity() {

    data class ScalerParams(
        val featureNames: List<String>,
        val mean: FloatArray,
        val scale: FloatArray,
    )

    private lateinit var binding: ActivityMainBinding
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var isModelLoaded = false
    private var selectedPath: String? = null
    private val tempFiles = mutableListOf<File>()

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val modelSessions = mutableMapOf<String, OrtSession>()
    private val deepSessions = mutableMapOf<String, OrtSession>()
    private val scalerParamsCache = mutableMapOf<String, ScalerParams>()

    companion object {
        private const val REQ_PICK_VIDEO = 1001
        private const val TAG = "AIChecker"
        private var activeModel = "onestar.onnx"
        private var activeScanMode = "accurate"
        private var activeProfile = "balanced"
    }

    private fun getAvailableModels(): Array<String> {
        return try {
            assets.list("models")
                ?.filter { it.endsWith(".onnx") && !it.endsWith("_features.onnx") }
                ?.sorted()
                ?.toTypedArray()
                ?: arrayOf("onestar.onnx")
        } catch (_: Exception) {
            arrayOf("onestar.onnx")
        }
    }

    private fun loadPreferredModel(): String {
        return try {
            assets.open("models/benchmark_best_model.json").use { input ->
                val text = input.bufferedReader().readText()
                val best = JSONObject(text).optString("best_model", "").trim()
                if (best.isNotEmpty()) "$best.onnx" else "onestar.onnx"
            }
        } catch (_: Exception) {
            "onestar.onnx"
        }
    }

    private fun getEnsembleModels(): List<String> {
        return getAvailableModels().filter { modelName ->
            try {
                assets.open("models/${modelName.removeSuffix(".onnx")}_scaler_params.json").close()
                true
            } catch (_: Exception) {
                false
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        ortEnv = OrtEnvironment.getEnvironment()
        activeModel = loadPreferredModel()

        setupUI()
        loadModels(activeModel)
        maybeShowWelcome()
    }

    override fun onDestroy() {
        super.onDestroy()
        ortSession?.close()
        modelSessions.values.forEach { it.close() }
        modelSessions.clear()
        deepSessions.values.forEach { it.close() }
        deepSessions.clear()
        ortEnv?.close()
        tempFiles.forEach { if (it.exists()) it.delete() }
        tempFiles.clear()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_PICK_VIDEO && resultCode == Activity.RESULT_OK) {
            data?.data?.let { handlePickedVideo(it) }
        }
    }

    private fun setupUI() {
        setControlsEnabled(false)
        binding.resultCard.visibility = View.GONE
        binding.btnClear.visibility = View.GONE
        binding.tvScanModeName.text = scanModeLabel(activeScanMode)
        binding.tvProfileName.text = profileLabel(activeProfile)
        updateRecommendationNote()

        binding.btnSelectFile.setOnClickListener { pickVideoFile() }
        binding.btnPasteUrl.setOnClickListener { showUrlDialog() }
        binding.btnAnalyze.setOnClickListener { selectedPath?.let { analyzeVideo(it) } }
        binding.btnModelPicker.setOnClickListener { showModelPicker() }
        binding.btnProfilePicker.setOnClickListener { showProfilePicker() }
        binding.btnScanModePicker.setOnClickListener { showScanModePicker() }
        binding.btnClear.setOnClickListener { clearResult() }
    }

    private fun setControlsEnabled(enabled: Boolean) {
        binding.btnSelectFile.isEnabled = enabled
        binding.btnPasteUrl.isEnabled = enabled
        binding.btnModelPicker.isEnabled = enabled
        binding.btnProfilePicker.isEnabled = enabled
        binding.btnScanModePicker.isEnabled = enabled
        binding.btnAnalyze.isEnabled = enabled && selectedPath != null
    }

    private fun setBadge(text: String, colorRes: Int) {
        binding.tvBadge.text = text
        binding.tvBadge.setTextColor(ContextCompat.getColor(this, colorRes))
    }

    private fun clearResult() {
        binding.resultCard.visibility = View.GONE
        binding.btnClear.visibility = View.GONE
        binding.progressBar.progress = 0
        binding.tvStatus.text = "Sẵn sàng - hãy chọn tệp hoặc dán URL."
    }

    private fun loadModels(modelName: String) {
        isModelLoaded = false
        setControlsEnabled(false)
        binding.tvStatus.text = "Đang tải mô hình: $modelName..."
        setBadge("ĐANG TẢI", R.color.warning)

        executor.execute {
            try {
                val configFile = copyAsset("config.yaml")
                val scalerName = modelName.removeSuffix(".onnx") + "_scaler.pkl"
                val scalerExists = try {
                    assets.open("models/$scalerName").close()
                    true
                } catch (_: Exception) {
                    false
                }

                if (!scalerExists) {
                    mainHandler.post {
                        binding.tvStatus.text = "Thiếu tệp: models/$scalerName"
                        setBadge("LỖI", R.color.danger)
                    }
                    return@execute
                }

                val scalerFile = copyAsset("models/$scalerName")
                copyDeepFeatureAssets()
                val result = Python.getInstance()
                    .getModule("detector")
                    .callAttr("load_models", scalerFile.absolutePath, configFile.absolutePath)
                    .toString()

                Log.d(TAG, "load_models -> $result")

                if (result != "OK") {
                    mainHandler.post {
                        binding.tvStatus.text = result
                        setBadge("LỖI", R.color.danger)
                    }
                    return@execute
                }

                val onnxFile = copyAsset("models/$modelName")
                val env = ortEnv ?: OrtEnvironment.getEnvironment().also { ortEnv = it }
                ortSession?.close()
                ortSession = env.createSession(onnxFile.absolutePath)
                modelSessions[modelName]?.close()
                modelSessions[modelName] = ortSession!!

                mainHandler.post {
                    isModelLoaded = true
                    activeModel = modelName
                    binding.tvModelName.text = modelName
                    binding.tvStatus.text = "Sẵn sàng - hãy chọn tệp hoặc dán URL."
                    setBadge("SẴN SÀNG", R.color.success)
                    setControlsEnabled(true)
                }
            } catch (e: Exception) {
                Log.e(TAG, "loadModels failed", e)
                mainHandler.post {
                    binding.tvStatus.text = "Tải mô hình thất bại: ${e.message}"
                    setBadge("LỖI", R.color.danger)
                }
            }
        }
    }

    private fun showModelPicker() {
        val available = getAvailableModels()
        if (available.isEmpty()) {
            Toast.makeText(this, "Không tìm thấy tệp .onnx trong assets/models/", Toast.LENGTH_LONG).show()
            return
        }

        val displayNames = available.map {
            if (it == activeModel) "[đang dùng] $it" else it
        }.toTypedArray()

        AlertDialog.Builder(this, R.style.DarkDialog)
            .setTitle("Chọn mô hình (${available.size})")
            .setItems(displayNames) { _, index ->
                if (available[index] != activeModel) {
                    loadModels(available[index])
                } else {
                    Toast.makeText(this, "Mô hình này đang được dùng", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Hủy", null)
            .show()
    }

    private fun showScanModePicker() {
        val modes = arrayOf("Quét nhanh", "Quét chính xác")
        val values = arrayOf("quick", "accurate")
        val currentIndex = values.indexOf(activeScanMode).coerceAtLeast(0)

        AlertDialog.Builder(this, R.style.DarkDialog)
            .setTitle("Chọn chế độ quét")
            .setSingleChoiceItems(modes, currentIndex) { dialog, which ->
                activeScanMode = values[which]
                binding.tvScanModeName.text = scanModeLabel(activeScanMode)
                binding.tvStatus.text = if (activeScanMode == "quick") {
                    "Đã chọn Quét nhanh - cho kết quả nhanh hơn với ít khung hình hơn."
                } else {
                    "Đã chọn Quét chính xác - dùng nhiều khung hình hơn để ổn định hơn."
                }
                dialog.dismiss()
            }
            .setNegativeButton("Hủy", null)
            .show()
    }

    private fun showProfilePicker() {
        val labels = arrayOf("Cân bằng", "Ưu tiên độ chính xác", "Ưu tiên tốc độ")
        val values = arrayOf("balanced", "accuracy", "speed")
        val currentIndex = values.indexOf(activeProfile).coerceAtLeast(0)

        AlertDialog.Builder(this, R.style.DarkDialog)
            .setTitle("Chọn hồ sơ sử dụng")
            .setSingleChoiceItems(labels, currentIndex) { dialog, which ->
                activeProfile = values[which]
                binding.tvProfileName.text = profileLabel(activeProfile)
                applyProfileRecommendation()
                dialog.dismiss()
            }
            .setNegativeButton("Hủy", null)
            .show()
    }

    private fun maybeShowWelcome() {
        val prefs = getSharedPreferences("aichecker_prefs", MODE_PRIVATE)
        if (prefs.getBoolean("welcome_seen", false)) {
            return
        }

        AlertDialog.Builder(this, R.style.DarkDialog)
            .setTitle("Chào mừng đến với AIChecker")
            .setMessage(
                "Ứng dụng này chạy hoàn toàn ngoại tuyến để kiểm tra video AI.\n\n" +
                    "Gợi ý sử dụng:\n" +
                    "- Ưu tiên video rõ và đủ ánh sáng\n" +
                    "- Dùng hồ sơ Ưu tiên độ chính xác cho các ca quan trọng\n" +
                    "- Nếu ứng dụng báo chất lượng thấp, hãy thử video rõ hơn"
            )
            .setPositiveButton("Bắt đầu") { _, _ ->
                prefs.edit().putBoolean("welcome_seen", true).apply()
            }
            .setCancelable(false)
            .show()
    }

    private fun pickVideoFile() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "video/*"
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        startActivityForResult(Intent.createChooser(intent, "Chọn video"), REQ_PICK_VIDEO)
    }

    private fun handlePickedVideo(uri: Uri) {
        val name = getFileName(uri) ?: "video.mp4"
        val dest = File(cacheDir, name)
        contentResolver.openInputStream(uri)?.use { input ->
            dest.outputStream().use { output -> input.copyTo(output) }
        }
        tempFiles.add(dest)
        setVideo(dest.absolutePath, name, isTemp = false)
    }

    private fun getFileName(uri: Uri): String? {
        var name: String? = null
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (cursor.moveToFirst() && index >= 0) {
                name = cursor.getString(index)
            }
        }
        return name ?: uri.lastPathSegment
    }

    private fun setVideo(path: String, displayName: String, isTemp: Boolean) {
        selectedPath = path
        binding.tvFileName.text = displayName
        binding.tvFileTag.text = if (isTemp) "[TẠM]" else "[TỆP]"
        binding.tvFileTag.setTextColor(
            ContextCompat.getColor(this, if (isTemp) R.color.warning else R.color.accent)
        )
        binding.btnAnalyze.isEnabled = isModelLoaded
        setBadge("ĐÃ TẢI", R.color.accent)
        binding.tvStatus.text = "Sẵn sàng. Hãy bấm PHÂN TÍCH."
        applyProfileRecommendation()
    }

    private fun showUrlDialog() {
        val view = layoutInflater.inflate(R.layout.dialog_url, null)
        val etUrl = view.findViewById<EditText>(R.id.etUrl)
        val tvStatus = view.findViewById<TextView>(R.id.tvUrlStatus)

        val clipboard = getSystemService(android.content.ClipboardManager::class.java)
        val clipText = clipboard?.primaryClip?.getItemAt(0)?.text?.toString() ?: ""
        if (clipText.startsWith("http")) {
            etUrl.setText(clipText)
        }

        val dialog = AlertDialog.Builder(this, R.style.DarkDialog)
            .setTitle("Dán URL video")
            .setView(view)
            .setNegativeButton("Hủy", null)
            .setPositiveButton("Tải xuống và phân tích", null)
            .create()

        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val url = etUrl.text.toString().trim()
                if (!url.startsWith("http")) {
                    tvStatus.text = "URL không hợp lệ - phải bắt đầu bằng http:// hoặc https://"
                    tvStatus.setTextColor(ContextCompat.getColor(this, R.color.danger))
                    return@setOnClickListener
                }

                tvStatus.text = "Đang tải video..."
                tvStatus.setTextColor(ContextCompat.getColor(this, R.color.warning))
                dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = false

                downloadVideo(
                    url,
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
        onError: (String) -> Unit,
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
                Log.e(TAG, "Download failed", e)
                mainHandler.post { onError("Tải video thất bại: ${e.message}") }
            }
        }
    }

    private fun analyzeVideo(videoPath: String) {
        setControlsEnabled(false)
        binding.resultCard.visibility = View.GONE
        binding.btnClear.visibility = View.GONE
        binding.progressBar.progress = 0
        binding.tvStatus.text = "Đang phân tích với ${scanModeLabel(activeScanMode)}..."
        setBadge("ĐANG PHÂN TÍCH", R.color.warning)
        animateProgress()

        executor.execute {
            try {
                Log.d(TAG, "Starting analysis: $videoPath")
                val deepPlan = Python.getInstance()
                    .getModule("detector")
                    .callAttr("get_deep_sample_plan", videoPath, activeScanMode)
                    .toString()
                val deepTimesUs = JSONObject(deepPlan).optJSONArray("times_us")?.let { arr ->
                    List(arr.length()) { arr.getLong(it) }
                } ?: emptyList()
                val deepJson = extractDeepFeatures(videoPath, deepTimesUs)

                val raw = Python.getInstance()
                    .getModule("detector")
                    .callAttr("extract_features", videoPath, deepJson.toString(), activeScanMode)
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
                val floats = FloatArray(vectorArr.length()) { index ->
                    vectorArr.getDouble(index).toFloat()
                }

                Log.d(TAG, "Feature vector dim=${floats.size}, mean=${"%.4f".format(floats.average())}")
                json.optJSONObject("debug_info")?.let { dbg ->
                    Log.d(TAG, "Debug: models=${dbg.optInt("onnx_models_loaded")}, " +
                            "scaled_mean=${"%.4f".format(dbg.optDouble("scaled_mean"))}, " +
                            "scaled_std=${"%.4f".format(dbg.optDouble("scaled_std"))}")
                }

                val modelProb = json.optDouble("model_probability", 0.5).toFloat()
                val fusionProb = json.optDouble("fusion_probability", 0.5).toFloat()
                val finalProb = json.optDouble("final_probability", 0.5).toFloat()
                val inputQuality = json.optDouble("input_quality_score", 0.75).toFloat()
                val modelUsed = json.optBoolean("model_used", false)
                val modelPrediction = json.optString("model_prediction", "REAL")
                val ensembleDetails = json.optJSONArray("ensemble_details") ?: JSONArray()
                val onnxModels = json.optJSONArray("onnx_models") ?: JSONArray()
                val deepValid = json.optBoolean("deep_features_valid", false)
                val deepCount = json.optInt("deep_features_count", 0)

                val prediction = if (finalProb >= 0.5) "FAKE" else "REAL"
                val label = if (modelPrediction == "FAKE") 1 else 0

                val confidence = deriveConfidence(
                    blendedProb = finalProb,
                    modelProb = modelProb,
                    fusionProb = fusionProb,
                    inputQuality = inputQuality,
                )

                val qualityFlags = json.optJSONArray("quality_flags") ?: JSONArray()
                val verdict = determineCustomerVerdict(
                    blendedProb = finalProb,
                    confidence = confidence,
                    inputQuality = inputQuality,
                )
                val reasonPoints = buildReasonPoints(
                    artifact = json.getDouble("fusion_artifact"),
                    reality = json.getDouble("fusion_reality"),
                    stress = json.optDouble("fusion_stress", 0.5),
                    modelProb = modelProb.toDouble(),
                    fusionProb = fusionProb.toDouble(),
                    inputQuality = inputQuality.toDouble(),
                    qualityFlags = qualityFlags,
                )
                val reasonSummary = buildReasonSummary(verdict.second, reasonPoints)

                Log.d(TAG, "Final result: model=${"%.4f".format(modelProb)}, " +
                        "fusion=${"%.4f".format(fusionProb)}, " +
                        "final=${"%.4f".format(finalProb)}, " +
                        "prediction=$prediction")

                val finalJson = JSONObject().apply {
                    put("prediction", prediction)
                    put("probability_fake", finalProb.toDouble())
                    put("confidence", confidence)
                    put("artifact_score", json.getDouble("fusion_artifact"))
                    put("reality_score", json.getDouble("fusion_reality"))
                    put("stress_score", json.optDouble("fusion_stress", 0.5))
                    put("model_probability_fake", modelProb.toDouble())
                    put("model_prediction", modelPrediction)
                    put("model_ensemble", ensembleDetails)
                    put("fusion_probability", fusionProb.toDouble())
                    put("fusion_prediction", json.optString("fusion_prediction", "UNKNOWN"))
                    put("input_quality_score", inputQuality.toDouble())
                    put("input_quality_label", json.optString("input_quality_label", "MEDIUM"))
                    put("quality_flags", qualityFlags)
                    put("customer_verdict", verdict.first)
                    put("verdict_headline", verdict.second)
                    put("reason_summary", reasonSummary)
                    put("reason_points", JSONArray(reasonPoints))
                    put("scan_mode", activeScanMode)
                    put("explanations", json.optJSONArray("explanations") ?: JSONArray())
                    put("python_model_used", modelUsed)
                    put("onnx_models_used", onnxModels)
                    put("deep_features_valid", deepValid)
                    put("deep_features_count", deepCount)
                }

                mainHandler.post {
                    binding.progressBar.progress = 100
                    showResult(finalJson)
                    setControlsEnabled(true)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Analysis failed", e)
                mainHandler.post {
                    showError("Phân tích thất bại: ${e.message}")
                    setControlsEnabled(true)
                }
            }
        }
    }

    private fun anyToFloat(value: Any?): Float? = when (value) {
        is Float -> value
        is Double -> value.toFloat()
        is Long -> value.toFloat()
        is Int -> value.toFloat()
        is Number -> value.toFloat()
        else -> null
    }

    private fun getOrCreateModelSession(modelName: String): OrtSession {
        modelSessions[modelName]?.let { return it }
        val env = ortEnv ?: OrtEnvironment.getEnvironment().also { ortEnv = it }
        val modelFile = copyAsset("models/$modelName")
        val session = env.createSession(modelFile.absolutePath)
        modelSessions[modelName] = session
        return session
    }

    private fun loadScalerParams(modelName: String): ScalerParams {
        scalerParamsCache[modelName]?.let { return it }
        val stem = modelName.removeSuffix(".onnx")
        val text = assets.open("models/${stem}_scaler_params.json").use { input ->
            input.bufferedReader().readText()
        }
        val json = JSONObject(text)
        val featureNamesJson = json.getJSONArray("feature_names")
        val meanJson = json.getJSONArray("mean_")
        val scaleJson = json.getJSONArray("scale_")
        val featureNames = List(featureNamesJson.length()) { idx -> featureNamesJson.getString(idx) }
        val mean = FloatArray(meanJson.length()) { idx -> meanJson.getDouble(idx).toFloat() }
        val scale = FloatArray(scaleJson.length()) { idx ->
            val value = scaleJson.getDouble(idx).toFloat()
            if (value == 0f) 1f else value
        }
        return ScalerParams(featureNames, mean, scale).also { scalerParamsCache[modelName] = it }
    }

    private fun scaleFeatures(rawFeatures: JSONObject, scalerParams: ScalerParams): FloatArray {
        val scaled = FloatArray(scalerParams.featureNames.size)
        scalerParams.featureNames.forEachIndexed { idx, featureName ->
            val rawValue = rawFeatures.optDouble(featureName, 0.0).toFloat()
            scaled[idx] = (rawValue - scalerParams.mean[idx]) / scalerParams.scale[idx]
        }
        return scaled
    }

    private fun parseProbabilityMap(map: Map<*, *>): Float? {
        anyToFloat(map[1L])?.let { return it }
        anyToFloat(map[1])?.let { return it }
        anyToFloat(map[1.0])?.let { return it }
        anyToFloat(map["1"])?.let { return it }
        anyToFloat(map["fake"])?.let { return it }
        anyToFloat(map["FAKE"])?.let { return it }

        val numericPairs = map.entries.mapNotNull { entry ->
            val key = (entry.key as? Number)?.toDouble() ?: entry.key?.toString()?.toDoubleOrNull() ?: return@mapNotNull null
            val value = anyToFloat(entry.value) ?: return@mapNotNull null
            key to value
        }.sortedBy { it.first }
        if (numericPairs.size >= 2) {
            return numericPairs[1].second
        }

        val values = map.values.mapNotNull { anyToFloat(it) }
        return when {
            values.size >= 2 -> values[1]
            values.size == 1 -> values[0]
            else -> null
        }
    }

    private fun parseProbFake(output: OrtSession.Result): Float {
        return try {
            when (val raw1 = output[1].value) {
                is List<*> -> {
                    val map = raw1.firstOrNull() as? Map<*, *>
                    when {
                        map != null -> parseProbabilityMap(map) ?: 0.5f
                        raw1.firstOrNull() is FloatArray -> {
                            val arr = raw1.firstOrNull() as FloatArray
                            if (arr.size >= 2) arr[1] else arr.firstOrNull() ?: 0.5f
                        }
                        raw1.firstOrNull() is DoubleArray -> {
                            val arr = raw1.firstOrNull() as DoubleArray
                            if (arr.size >= 2) arr[1].toFloat() else arr.firstOrNull()?.toFloat() ?: 0.5f
                        }
                        else -> raw1.mapNotNull { anyToFloat(it) }.let { values ->
                            when {
                                values.size >= 2 -> values[1]
                                values.size == 1 -> values[0]
                                else -> 0.5f
                            }
                        }
                    }
                }

                is FloatArray -> {
                    Log.d(TAG, "Prob FloatArray: ${raw1.toList()}")
                    if (raw1.size >= 2) raw1[1] else raw1.firstOrNull() ?: 0.5f
                }

                is Array<*> -> {
                    when (val inner = raw1.firstOrNull()) {
                        is FloatArray -> if (inner.size >= 2) inner[1] else 0.5f
                        is DoubleArray -> if (inner.size >= 2) inner[1].toFloat() else 0.5f
                        is Map<*, *> -> parseProbabilityMap(inner) ?: 0.5f
                        is Number -> anyToFloat(inner) ?: 0.5f
                        else -> 0.5f
                    }
                }

                is Map<*, *> -> parseProbabilityMap(raw1) ?: 0.5f
                is DoubleArray -> if (raw1.size >= 2) raw1[1].toFloat() else raw1.firstOrNull()?.toFloat() ?: 0.5f
                is Number -> raw1.toFloat()

                else -> {
                    Log.w(TAG, "Unknown prob output type: ${raw1?.javaClass?.name}")
                    0.5f
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "parseProbFake exception: ${e.message}")
            0.5f
        }
    }

    private fun runOnnxInference(
        floats: FloatArray,
        session: OrtSession = (ortSession ?: throw IllegalStateException("ONNX session not loaded")),
    ): Pair<Float, Int> {
        val env = ortEnv ?: OrtEnvironment.getEnvironment()

        val tensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(floats),
            longArrayOf(1, floats.size.toLong()),
        )

        tensor.use {
            val output = session.run(mapOf(session.inputNames.first() to it))
            output.use {
                val label: Int = try {
                    when (val rawLabel = output[0].value) {
                        is LongArray -> rawLabel[0].toInt()
                        is IntArray -> rawLabel[0]
                        is FloatArray -> rawLabel[0].toInt()
                        is DoubleArray -> rawLabel[0].toInt()
                        is Array<*> -> {
                            when (val first = rawLabel.firstOrNull()) {
                                is Long -> first.toInt()
                                is Int -> first
                                is Float -> first.toInt()
                                is Double -> first.toInt()
                                else -> first.toString().toLongOrNull()?.toInt() ?: 0
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
                val probFake = parseProbFake(output)
                return Pair(probFake, label)
            }
        }
    }
    
    private fun runOnnxInferenceWithScaledVector(
        floats: FloatArray,
    ): Triple<Float, Int, JSONArray> {
        return try {
            val session = ortSession ?: throw IllegalStateException("ONNX session not loaded")
            val env = ortEnv ?: OrtEnvironment.getEnvironment()
            
            val tensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(floats),
                longArrayOf(1, floats.size.toLong()),
            )
            
            tensor.use {
                val output = session.run(mapOf(session.inputNames.first() to it))
                output.use {
                    val label: Int = try {
                        when (val rawLabel = output[0].value) {
                            is LongArray -> rawLabel[0].toInt()
                            is IntArray -> rawLabel[0]
                            is FloatArray -> rawLabel[0].toInt()
                            is DoubleArray -> rawLabel[0].toInt()
                            is Array<*> -> {
                                when (val first = rawLabel.firstOrNull()) {
                                    is Long -> first.toInt()
                                    is Int -> first
                                    is Float -> first.toInt()
                                    is Double -> first.toInt()
                                    else -> first.toString().toLongOrNull()?.toInt() ?: 0
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
                    
                    Log.d(TAG, "Scaled inference - label=$label (type=${output[0].value?.javaClass?.simpleName})")
                    val probFake = parseProbFake(output)
                    Triple(probFake, label, JSONArray())
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "runOnnxInferenceWithScaledVector failed: ${e.message}")
            Triple(0.5f, 0, JSONArray())
        }
    }

    private fun runEnsembleInference(rawFeatures: JSONObject): Triple<Float, Int, JSONArray> {
        val models = getEnsembleModels().ifEmpty { listOf(activeModel) }
        val primaryWeight = if (models.size <= 1) 1.0f else 0.4f
        val secondaryWeight = if (models.size <= 1) 0.0f else 0.6f / (models.size - 1)

        var weightedProb = 0.0f
        var weightedLabel = 0.0f
        val details = JSONArray()

        for (modelName in models) {
            val session = getOrCreateModelSession(modelName)
            val scalerParams = loadScalerParams(modelName)
            val scaledVector = scaleFeatures(rawFeatures, scalerParams)
            val (probFake, label) = runOnnxInference(scaledVector, session)
            val weight = if (modelName == activeModel) primaryWeight else secondaryWeight
            weightedProb += probFake * weight
            weightedLabel += label * weight
            details.put(
                JSONObject().apply {
                    put("model", modelName)
                    put("probability_fake", probFake.toDouble())
                    put("prediction", if (label == 1) "FAKE" else "REAL")
                    put("weight", weight.toDouble())
                },
            )
        }

        return Triple(weightedProb.coerceIn(0f, 1f), if (weightedLabel >= 0.5f) 1 else 0, details)
    }

    private fun getOrCreateDeepSession(modelName: String): OrtSession {
        deepSessions[modelName]?.let { return it }
        val env = ortEnv ?: OrtEnvironment.getEnvironment().also { ortEnv = it }
        val modelFile = copyAsset("models/$modelName")
        val session = env.createSession(modelFile.absolutePath)
        deepSessions[modelName] = session
        return session
    }

    private fun extractDeepFeatures(videoPath: String, sampleTimesUs: List<Long>): JSONObject {
        return try {
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(videoPath)
            val frames = mutableListOf<Bitmap>()
            for (positionUs in sampleTimesUs) {
                retriever.getFrameAtTime(positionUs, MediaMetadataRetriever.OPTION_CLOSEST)?.let { frames.add(it) }
            }
            retriever.release()

            if (frames.isEmpty()) {
                JSONObject()
            } else {
                val perModel = listOf(
                    "resnet50_features.onnx",
                    "efficientnet_b0_features.onnx",
                ).mapNotNull { modelName ->
                    try {
                        val session = getOrCreateDeepSession(modelName)
                        computeDeepStats(frames, session)
                    } catch (e: Exception) {
                        Log.w(TAG, "Deep model failed: $modelName (${e.message})")
                        null
                    }
                }

                val merged = JSONObject()
                val keys = listOf(
                    "deep_feat_mean", "deep_feat_std", "deep_feat_max", "deep_feat_min",
                    "deep_temporal_var_mean", "deep_temporal_var_std",
                    "deep_l2_norm_mean", "deep_l2_norm_std",
                    "deep_similarity_mean", "deep_similarity_std", "deep_sparsity",
                )
                for (key in keys) {
                    val values = perModel.mapNotNull { if (it.has(key)) it.optDouble(key) else null }
                    if (values.isNotEmpty()) {
                        merged.put(key, values.average())
                    }
                }
                merged
            }
        } catch (e: Exception) {
            Log.w(TAG, "extractDeepFeatures failed: ${e.message}")
            JSONObject()
        }
    }

    private fun computeDeepStats(bitmaps: List<Bitmap>, session: OrtSession): JSONObject {
        val vectors = mutableListOf<FloatArray>()
        for (bitmap in bitmaps) {
            val input = preprocessBitmap(bitmap)
            val tensor = OnnxTensor.createTensor(
                ortEnv ?: OrtEnvironment.getEnvironment(),
                FloatBuffer.wrap(input),
                longArrayOf(1, 3, 224, 224),
            )
            tensor.use {
                val output = session.run(mapOf(session.inputNames.first() to it))
                output.use {
                    val raw = output[0].value
                    val vector = when (raw) {
                        is FloatArray -> raw
                        is Array<*> -> (raw.firstOrNull() as? FloatArray) ?: FloatArray(0)
                        else -> FloatArray(0)
                    }
                    if (vector.isNotEmpty()) {
                        vectors.add(vector)
                    }
                }
            }
        }

        if (vectors.isEmpty()) {
            return JSONObject()
        }

        val flatValues = vectors.flatMap { it.asList() }
        val temporalVars = FloatArray(vectors[0].size) { idx ->
            val mean = vectors.map { it[idx].toDouble() }.average()
            vectors.map { val d = it[idx] - mean.toFloat(); d * d }.average().toFloat()
        }
        val l2Norms = vectors.map { vector -> kotlin.math.sqrt(vector.fold(0.0) { acc, v -> acc + v * v }.toFloat()) }
        val similarities = mutableListOf<Float>()
        for (i in 0 until vectors.size - 1) {
            val a = vectors[i]
            val b = vectors[i + 1]
            var dot = 0.0
            var na = 0.0
            var nb = 0.0
            for (j in a.indices) {
                dot += a[j] * b[j]
                na += a[j] * a[j]
                nb += b[j] * b[j]
            }
            similarities.add((dot / (kotlin.math.sqrt(na) * kotlin.math.sqrt(nb) + 1e-10)).toFloat())
        }

        return JSONObject().apply {
            put("deep_feat_mean", flatValues.average())
            put("deep_feat_std", stdOf(flatValues.map { it.toDouble() }))
            put("deep_feat_max", flatValues.maxOrNull()?.toDouble() ?: 0.0)
            put("deep_feat_min", flatValues.minOrNull()?.toDouble() ?: 0.0)
            put("deep_temporal_var_mean", temporalVars.map { it.toDouble() }.average())
            put("deep_temporal_var_std", stdOf(temporalVars.map { it.toDouble() }))
            put("deep_l2_norm_mean", l2Norms.map { it.toDouble() }.average())
            put("deep_l2_norm_std", stdOf(l2Norms.map { it.toDouble() }))
            put("deep_similarity_mean", if (similarities.isEmpty()) 0.0 else similarities.map { it.toDouble() }.average())
            put("deep_similarity_std", if (similarities.isEmpty()) 0.0 else stdOf(similarities.map { it.toDouble() }))
            put("deep_sparsity", flatValues.count { kotlin.math.abs(it) < 0.01f }.toDouble() / flatValues.size.toDouble())
        }
    }

    private fun preprocessBitmap(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
        val left = (256 - 224) / 2
        val top = (256 - 224) / 2
        val cropped = Bitmap.createBitmap(resized, left, top, 224, 224)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val pixels = IntArray(224 * 224)
        cropped.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        val chw = FloatArray(3 * 224 * 224)
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = pixels[y * 224 + x]
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f
                val idx = y * 224 + x
                chw[idx] = (r - mean[0]) / std[0]
                chw[224 * 224 + idx] = (g - mean[1]) / std[1]
                chw[2 * 224 * 224 + idx] = (b - mean[2]) / std[2]
            }
        }
        return chw
    }

    private fun stdOf(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val mean = values.average()
        return kotlin.math.sqrt(values.map { (it - mean) * (it - mean) }.average())
    }

    private fun blendProbabilities(modelProb: Float, fusionProb: Float, inputQuality: Float): Float {
        val safeQuality = inputQuality.coerceIn(0f, 1f)
        val fusionWeight = (0.18f + (1f - safeQuality) * 0.22f).coerceIn(0.18f, 0.40f)
        val blended = modelProb * (1f - fusionWeight) + fusionProb * fusionWeight
        return blended.coerceIn(0f, 1f)
    }

    private fun deriveConfidence(
        blendedProb: Float,
        modelProb: Float,
        fusionProb: Float,
        inputQuality: Float,
    ): String {
        val certainty = abs(blendedProb - 0.5f) * 2f
        val agreement = 1f - abs(modelProb - fusionProb).coerceIn(0f, 1f)
        val score = (0.45f * certainty + 0.35f * agreement + 0.20f * inputQuality)
            .coerceIn(0f, 1f)

        return when {
            score >= 0.78f -> "HIGH"
            score >= 0.55f -> "MEDIUM"
            else -> "LOW"
        }
    }

    private fun scanModeLabel(scanMode: String): String {
        return if (scanMode == "quick") "Quét nhanh" else "Quét chính xác"
    }

    private fun profileLabel(profile: String): String {
        return when (profile) {
            "accuracy" -> "Ưu tiên độ chính xác"
            "speed" -> "Ưu tiên tốc độ"
            else -> "Cân bằng"
        }
    }

    private fun suggestScanModeForVideo(videoPath: String): Pair<String, String> {
        return try {
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(videoPath)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            retriever.release()

            when {
                durationMs in 1..3999 -> "accurate" to "Video ngắn, nên ưu tiên Quét chính xác để giữ độ ổn định."
                durationMs > 90000 || (width * height >= 1920 * 1080) -> "quick" to "Video dài hoặc độ phân giải cao, Quét nhanh sẽ nhẹ hơn."
                else -> "accurate" to "Quét chính xác được gợi ý cho video mục tiêu."
            }
        } catch (_: Exception) {
            "accurate" to "Quét chính xác được giữ làm mặc định."
        }
    }

    private fun updateRecommendationNote(extra: String? = null) {
        val base = when (activeProfile) {
            "accuracy" -> "Gợi ý hiện tại: $activeModel + Quét chính xác. Hồ sơ này ưu tiên độ chính xác."
            "speed" -> "Gợi ý hiện tại: $activeModel + Quét nhanh. Hồ sơ này ưu tiên tốc độ."
            else -> "Gợi ý hiện tại: $activeModel + ${scanModeLabel(activeScanMode)}. Hồ sơ này cân bằng tốc độ và độ chính xác."
        }
        binding.tvRecommendationNote.text = if (extra.isNullOrBlank()) base else "$base $extra"
    }

    private fun applyProfileRecommendation() {
        val (suggestedMode, reason) = selectedPath?.let { suggestScanModeForVideo(it) }
            ?: ("accurate" to "Quét chính xác được giữ làm mặc định.")

        activeScanMode = when (activeProfile) {
            "accuracy" -> "accurate"
            "speed" -> "quick"
            else -> suggestedMode
        }
        binding.tvScanModeName.text = scanModeLabel(activeScanMode)
        updateRecommendationNote(reason)
    }

    private fun determineCustomerVerdict(
        blendedProb: Float,
        confidence: String,
        inputQuality: Float,
    ): Pair<String, String> {
        return when {
            inputQuality < 0.48f -> "INSUFFICIENT_QUALITY" to "Không đủ chất lượng để kết luận"
            (blendedProb in 0.42f..0.58f) || confidence == "LOW" -> "UNCERTAIN" to "Cần kiểm tra thêm"
            blendedProb >= 0.5f -> "LIKELY_FAKE" to "Có khả năng là video AI"
            else -> "LIKELY_REAL" to "Có khả năng là video thật"
        }
    }

    private fun buildReasonPoints(
        artifact: Double,
        reality: Double,
        stress: Double,
        modelProb: Double,
        fusionProb: Double,
        inputQuality: Double,
        qualityFlags: JSONArray,
    ): List<String> {
        val points = mutableListOf<String>()

        if (inputQuality < 0.48 && qualityFlags.length() > 0) {
            points.add("video đầu vào chưa đủ rõ hoặc quá ngắn")
        }
        if (artifact >= 0.62) {
            points.add("tín hiệu artifact và texture bất thường khá rõ")
        }
        if (stress >= 0.58) {
            points.add("chuyển động giữa các khung hình kém ổn định")
        }
        if (modelProb >= 0.68 && fusionProb >= 0.58) {
            points.add("cả model và fusion đều nghiêng về video AI")
        }
        if (reality >= 0.62) {
            points.add("chỉ số reality vẫn giữ được độ tự nhiên khá tốt")
        }
        if (modelProb <= 0.35 && fusionProb <= 0.42) {
            points.add("cả model và fusion đều nghiêng về video thật")
        }
        if (points.isEmpty() && qualityFlags.length() > 0) {
            points.add("cần cân nhắc chất lượng nguồn video trước khi kết luận")
        }
        if (points.isEmpty()) {
            points.add("các tín hiệu hiện tại chưa quá cực đoan")
        }

        return points.take(3)
    }

    private fun buildReasonSummary(headline: String, reasonPoints: List<String>): String {
        return if (reasonPoints.isEmpty()) headline else "$headline, vi ${reasonPoints.first()}."
    }

    private fun animateProgress() {
        val steps = listOf(10 to 400L, 35 to 700L, 62 to 900L, 80 to 500L)
        steps.forEachIndexed { idx, (value, delay) ->
            mainHandler.postDelayed({
                if (binding.progressBar.progress < 100) {
                    binding.progressBar.progress = value
                }
            }, delay * (idx + 1))
        }
    }

    private fun showResult(json: JSONObject) {
        val prediction = json.getString("prediction")
        val probFake = json.getDouble("probability_fake")
        val probReal = 1.0 - probFake
        val confidence = json.getString("confidence")
        val artifact = json.getDouble("artifact_score")
        val reality = json.getDouble("reality_score")
        val stress = json.optDouble("stress_score", 0.5)
        val modelProb = json.optDouble("model_probability_fake", probFake)
        val fusionProb = json.optDouble("fusion_probability", probFake)
        val inputQualityLabel = json.optString("input_quality_label", "MEDIUM")
        val inputQualityScore = json.optDouble("input_quality_score", 0.75)
        val customerVerdict = json.optString("customer_verdict", prediction)
        val headline = json.optString("verdict_headline", prediction)
        val reasonSummary = json.optString("reason_summary", headline)
        val reasonPoints = json.optJSONArray("reason_points") ?: JSONArray()
        val ensembleDetails = json.optJSONArray("model_ensemble")

        val verdictColor = when (customerVerdict) {
            "LIKELY_FAKE" -> ContextCompat.getColor(this, R.color.danger)
            "LIKELY_REAL" -> ContextCompat.getColor(this, R.color.success)
            "INSUFFICIENT_QUALITY" -> ContextCompat.getColor(this, R.color.warning)
            else -> ContextCompat.getColor(this, R.color.accent2)
        }
        val bgColor = when (customerVerdict) {
            "LIKELY_FAKE" -> ContextCompat.getColor(this, R.color.danger_dim)
            "LIKELY_REAL" -> ContextCompat.getColor(this, R.color.success_dim)
            "INSUFFICIENT_QUALITY" -> ContextCompat.getColor(this, R.color.warning_dim)
            else -> ContextCompat.getColor(this, R.color.accent_dim)
        }

        binding.resultCard.visibility = View.VISIBLE
        binding.btnClear.visibility = View.VISIBLE
        binding.tvStatus.text = "Phân tích hoàn tất. Mô hình: $activeModel | ${scanModeLabel(activeScanMode)}"
        when (customerVerdict) {
            "LIKELY_FAKE" -> setBadge("NGHI AI", R.color.danger)
            "LIKELY_REAL" -> setBadge("CÓ VẺ THẬT", R.color.success)
            "INSUFFICIENT_QUALITY" -> setBadge("CHẤT LƯỢNG THẤP", R.color.warning)
            else -> setBadge("CẦN XEM THÊM", R.color.accent2)
        }

        binding.resultBanner.setBackgroundColor(bgColor)
        binding.tvVerdictIcon.text = when (customerVerdict) {
            "LIKELY_FAKE" -> "AI"
            "LIKELY_REAL" -> "OK"
            "INSUFFICIENT_QUALITY" -> "--"
            else -> "?"
        }
        binding.tvVerdictIcon.setTextColor(verdictColor)
        binding.tvVerdictText.text = when (customerVerdict) {
            "LIKELY_FAKE" -> "CÓ DẤU HIỆU VIDEO AI"
            "LIKELY_REAL" -> "CÓ KHẢ NĂNG LÀ VIDEO THẬT"
            "INSUFFICIENT_QUALITY" -> "CHẤT LƯỢNG VIDEO QUÁ THẤP"
            else -> "NÊN KIỂM TRA THÊM"
        }
        binding.tvVerdictText.setTextColor(verdictColor)
        val ensembleInfo = ensembleDetails?.length()?.takeIf { it > 1 }?.let { "  |  Ensemble: $it model" } ?: ""
        binding.tvConfidence.text = "Độ tin cậy: ${confidenceLabel(confidence)}  |  Đầu vào: ${qualityLabelVi(inputQualityLabel)}  |  Mô hình: $activeModel$ensembleInfo  |  ${scanModeLabel(activeScanMode)}"
        binding.tvSummaryHeadline.text = headline
        binding.tvSummaryReason.text = reasonSummary

        binding.tvProbFakeVal.text = "${"%.1f".format(probFake * 100)}%"
        binding.tvProbRealVal.text = "${"%.1f".format(probReal * 100)}%"
        binding.progressFake.progress = (probFake * 100).toInt()
        binding.progressReal.progress = (probReal * 100).toInt()
        binding.progressFake.progressTintList =
            android.content.res.ColorStateList.valueOf(ContextCompat.getColor(this, R.color.danger))
        binding.progressReal.progressTintList =
            android.content.res.ColorStateList.valueOf(ContextCompat.getColor(this, R.color.success))

        binding.tvArtifactVal.text = "${"%.3f".format(artifact)}"
        binding.tvRealityVal.text = "${"%.3f".format(reality)}"

        binding.layoutFindings.removeAllViews()
        for (i in 0 until reasonPoints.length()) {
            addFinding("- ${reasonPoints.getString(i)}")
        }
        addFinding("- Điểm kết hợp cuối: ${"%.1f".format(probFake * 100)}% nghi AI")
        addFinding("- Điểm mô hình ONNX: ${"%.1f".format(modelProb * 100)}% nghi AI")
        addFinding("- Điểm fusion quy tắc: ${"%.1f".format(fusionProb * 100)}% nghi AI")
        addFinding("- Độ ổn định: ${"%.3f".format(stress)} | Chất lượng đầu vào: ${qualityLabelVi(inputQualityLabel)} (${String.format("%.2f", inputQualityScore)})")
        if (ensembleDetails != null && ensembleDetails.length() > 1) {
            for (i in 0 until ensembleDetails.length()) {
                val item = ensembleDetails.getJSONObject(i)
                addFinding("- ${item.getString("model")}: ${"%.1f".format(item.getDouble("probability_fake") * 100)}% nghi AI")
            }
        }

        val qualityFlags = json.optJSONArray("quality_flags")
        if (qualityFlags != null && qualityFlags.length() > 0) {
            for (i in 0 until qualityFlags.length()) {
                addFinding("- Cảnh báo đầu vào: ${qualityFlags.getString(i)}")
            }
        }

        val explanations = json.optJSONArray("explanations")
        if (explanations != null) {
            for (i in 0 until minOf(explanations.length(), 6)) {
                addFinding("- ${explanations.getString(i)}")
            }
        }

        binding.scrollView.post {
            binding.scrollView.smoothScrollTo(0, binding.resultCard.top)
        }
    }

    private fun addFinding(text: String) {
        val tv = TextView(this).apply {
            this.text = text
            textSize = 12f
            setTextColor(ContextCompat.getColor(context, R.color.text_primary))
            setPadding(0, 8, 0, 8)
        }
        binding.layoutFindings.addView(tv)
    }

    private fun showError(msg: String) {
        binding.tvStatus.text = msg
        binding.progressBar.progress = 0
        setBadge("LỖI", R.color.danger)
        Log.e(TAG, "Error: $msg")
    }

    private fun confidenceLabel(confidence: String): String {
        return when (confidence.uppercase()) {
            "HIGH" -> "Cao"
            "MEDIUM" -> "Trung bình"
            "LOW" -> "Thấp"
            else -> confidence
        }
    }

    private fun qualityLabelVi(label: String): String {
        return when (label.uppercase()) {
            "HIGH" -> "Cao"
            "MEDIUM" -> "Trung bình"
            "LOW" -> "Thấp"
            else -> label
        }
    }

    private fun copyAsset(assetPath: String): File {
        val outFile = File(filesDir, assetPath)
        outFile.parentFile?.mkdirs()
        assets.open(assetPath).use { input ->
            outFile.outputStream().use { output -> input.copyTo(output) }
        }
        return outFile
    }

    private fun copyDeepFeatureAssets() {
        val deepAssets = listOf(
            "models/resnet50_features.onnx",
            "models/resnet50_features.onnx.data",
            "models/efficientnet_b0_features.onnx",
            "models/efficientnet_b0_features.onnx.data",
        )
        deepAssets.forEach { assetPath ->
            try {
                copyAsset(assetPath)
            } catch (e: Exception) {
                Log.w(TAG, "Deep asset not copied: $assetPath (${e.message})")
            }
        }
    }
}
