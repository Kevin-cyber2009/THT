package com.application.AIChecker

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.*
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var titleText: TextView
    private lateinit var selectButton: Button
    private lateinit var statusText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var resultCard: CardView
    private lateinit var resultText: TextView
    private lateinit var confidenceText: TextView
    private lateinit var probabilityText: TextView

    private val PICK_VIDEO = 1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize views
        titleText = findViewById(R.id.titleText)
        selectButton = findViewById(R.id.selectButton)
        statusText = findViewById(R.id.statusText)
        progressBar = findViewById(R.id.progressBar)
        resultCard = findViewById(R.id.resultCard)
        resultText = findViewById(R.id.resultText)
        confidenceText = findViewById(R.id.confidenceText)
        probabilityText = findViewById(R.id.probabilityText)

        // Initialize Python
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        // Initialize detector
        initializeDetector()

        // Setup button
        selectButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "video/*"
            startActivityForResult(intent, PICK_VIDEO)
        }
    }

    private fun initializeDetector() {
        statusText.text = "⏳ Initializing AI model..."
        selectButton.isEnabled = false

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val py = Python.getInstance()
                val module = py.getModule("detector")
                module.callAttr("initialize")

                withContext(Dispatchers.Main) {
                    statusText.text = "✅ Ready! Select a video to analyze"
                    selectButton.isEnabled = true
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    statusText.text = "❌ Error: ${e.message}"
                    Toast.makeText(this@MainActivity,
                        "Initialization failed: ${e.message}",
                        Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == PICK_VIDEO && resultCode == Activity.RESULT_OK) {
            data?.data?.let { uri ->
                analyzeVideo(uri)
            }
        }
    }

    private fun analyzeVideo(uri: Uri) {
        // Show progress
        resultCard.visibility = View.GONE
        progressBar.visibility = View.VISIBLE
        statusText.text = "🔍 Analyzing video..."
        selectButton.isEnabled = false

        CoroutineScope(Dispatchers.IO).launch {
            try {
                // Copy video to cache
                val tempFile = File(cacheDir, "temp_video.mp4")
                contentResolver.openInputStream(uri)?.use { input ->
                    tempFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }

                // Call Python detector
                val py = Python.getInstance()
                val module = py.getModule("detector")
                val result = module.callAttr("analyze", tempFile.absolutePath)

                // Parse results
                val prediction = result["prediction"]?.toString() ?: "UNKNOWN"
                val probFake = result["probability_fake"]?.toFloat() ?: 0f
                val probReal = 1f - probFake
                val confidence = result["confidence"]?.toString() ?: "MEDIUM"

                withContext(Dispatchers.Main) {
                    // Hide progress
                    progressBar.visibility = View.GONE
                    statusText.text = "✅ Analysis complete!"
                    selectButton.isEnabled = true

                    // Show result
                    resultCard.visibility = View.VISIBLE

                    if (prediction == "FAKE") {
                        resultText.text = "🚨 DEEPFAKE DETECTED"
                        resultText.setTextColor(getColor(android.R.color.holo_red_dark))
                    } else {
                        resultText.text = "✅ AUTHENTIC"
                        resultText.setTextColor(getColor(android.R.color.holo_green_dark))
                    }

                    confidenceText.text = "Confidence: $confidence"
                    probabilityText.text = String.format(
                        "Fake: %.1f%% | Real: %.1f%%",
                        probFake * 100,
                        probReal * 100
                    )
                }

                // Cleanup
                tempFile.delete()

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    statusText.text = "❌ Error: ${e.message}"
                    selectButton.isEnabled = true
                    Toast.makeText(this@MainActivity,
                        "Analysis failed: ${e.message}",
                        Toast.LENGTH_LONG).show()
                }
            }
        }
    }
}