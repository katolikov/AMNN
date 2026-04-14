package com.mnn.benchmarkapp.ui.vlm

import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.mnn.benchmarkapp.data.GeneratedConfig
import com.mnn.benchmarkapp.data.InferenceResult
import com.mnn.benchmarkapp.jni.BenchmarkSession
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.io.File

class VlmViewModel : ViewModel() {

    var vlmPrompt by mutableStateOf(GeneratedConfig.DEFAULT_VLM_PROMPT)

    private val _outputText = MutableStateFlow("")
    val outputText: StateFlow<String> = _outputText

    private val _lastResult = MutableStateFlow<InferenceResult?>(null)
    val lastResult: StateFlow<InferenceResult?> = _lastResult

    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing

    private val _capturedImagePath = MutableStateFlow<String?>(null)
    val capturedImagePath: StateFlow<String?> = _capturedImagePath

    /** True when a gallery/captured image has been set and file exists. */
    private val _hasValidImage = MutableStateFlow(false)
    val hasValidImage: StateFlow<Boolean> = _hasValidImage

    /** Error message for gallery attachment failures. */
    private val _imageError = MutableStateFlow<String?>(null)
    val imageError: StateFlow<String?> = _imageError

    private var analyzeJob: Job? = null

    fun setCapturedImage(path: String) {
        val file = File(path)
        if (file.exists() && file.length() > 0) {
            _capturedImagePath.value = path
            _hasValidImage.value = true
            _imageError.value = null
            Log.d("VlmVM", "Captured image set: $path (${file.length()} bytes)")
        } else {
            Log.e("VlmVM", "Captured image not found or empty: $path")
            _imageError.value = "Failed to capture image"
            _hasValidImage.value = false
        }
    }

    fun setGalleryImage(path: String?) {
        if (path == null) {
            _imageError.value = "Failed to load gallery image"
            _hasValidImage.value = false
            return
        }
        val file = File(path)
        if (file.exists() && file.length() > 0) {
            _capturedImagePath.value = path
            _hasValidImage.value = true
            _imageError.value = null
            Log.d("VlmVM", "Gallery image set: $path (${file.length()} bytes)")
        } else {
            Log.e("VlmVM", "Gallery image not found or empty: $path")
            _imageError.value = "Failed to load gallery image"
            _hasValidImage.value = false
        }
    }

    fun clearImageError() {
        _imageError.value = null
    }

    fun analyzeImage(session: BenchmarkSession) {
        val imagePath = _capturedImagePath.value ?: return
        if (!session.isModelLoaded) return

        // Validate image file exists before sending to native
        val imageFile = File(imagePath)
        if (!imageFile.exists() || imageFile.length() == 0L) {
            Log.e("VlmVM", "Image file missing or empty: $imagePath")
            _imageError.value = "Image file not found"
            return
        }

        _isAnalyzing.value = true
        _outputText.value = ""
        _lastResult.value = null

        // Reset conversation history before each analysis to prevent
        // context contamination from previous image analyses
        session.resetHistory()

        analyzeJob = viewModelScope.launch {
            // Collect streaming output
            val textJob = launch {
                session.streamingText.collect { text ->
                    _outputText.value = text
                }
            }

            val result = session.generateWithImage(
                prompt = vlmPrompt,
                imagePath = imagePath
            )
            textJob.cancel()

            _outputText.value = result.outputText
            _lastResult.value = result
            _isAnalyzing.value = false
        }
    }
}
