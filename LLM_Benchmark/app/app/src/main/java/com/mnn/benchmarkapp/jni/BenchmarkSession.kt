package com.mnn.benchmarkapp.jni

import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.mnn.benchmarkapp.data.InferenceResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext

/**
 * Lifecycle-aware wrapper around the native MNN inference session.
 * Manages model loading, inference execution, and resource cleanup.
 */
class BenchmarkSession {

    private var nativePtr: Long = 0L

    var modelName by mutableStateOf("No model")
        private set

    var isModelLoaded by mutableStateOf(false)
        private set

    var isModelLoading by mutableStateOf(false)
        private set

    var isGenerating by mutableStateOf(false)
        private set

    private val _streamingText = MutableStateFlow("")
    val streamingText: StateFlow<String> = _streamingText

    /**
     * Load a model from the given directory path on a background thread.
     */
    suspend fun loadModel(modelDir: String, configJson: String = "{}") {
        isModelLoading = true
        withContext(Dispatchers.IO) {
            // Release previous session
            releaseInternal()

            try {
                nativePtr = NativeBridge.initNative(modelDir, configJson)
                isModelLoaded = nativePtr != 0L
            } catch (e: Exception) {
                Log.e("BenchSession", "Model load failed: ${e.message}")
                nativePtr = 0L
                isModelLoaded = false
            }
            modelName = if (isModelLoaded) {
                modelDir.substringAfterLast("/")
            } else {
                "Load failed"
            }
        }
        isModelLoading = false
    }

    /**
     * Run text-only LLM inference with streaming output.
     * Uses StringBuilder for thread-safe serial token accumulation
     * to avoid StateFlow conflation losing intermediate tokens.
     */
    suspend fun generate(prompt: String): InferenceResult = withContext(Dispatchers.IO) {
        if (!isModelLoaded) return@withContext InferenceResult.empty()

        isGenerating = true
        _streamingText.value = ""
        val builder = StringBuilder()

        val metrics = NativeBridge.generateNative(nativePtr, prompt, object : ProgressListener {
            override fun onProgress(token: String?): Boolean {
                if (token != null) {
                    builder.append(token)
                    _streamingText.value = stripThinkingTags(builder.toString())
                }
                return true // continue generation
            }
        })

        val finalText = stripThinkingTags(builder.toString())
        isGenerating = false
        InferenceResult.fromMetrics(metrics, finalText)
    }

    /**
     * Run VLM inference with an image.
     * Uses StringBuilder for thread-safe serial token accumulation.
     */
    suspend fun generateWithImage(prompt: String, imagePath: String): InferenceResult =
        withContext(Dispatchers.IO) {
            if (!isModelLoaded) return@withContext InferenceResult.empty()

            isGenerating = true
            _streamingText.value = ""
            val builder = StringBuilder()

            val metrics = NativeBridge.generateWithImageNative(
                nativePtr, prompt, imagePath, object : ProgressListener {
                    override fun onProgress(token: String?): Boolean {
                        if (token != null) {
                            builder.append(token)
                            _streamingText.value = stripThinkingTags(builder.toString())
                        }
                        return true
                    }
                }
            )

            val finalText = stripThinkingTags(builder.toString())
            isGenerating = false
            InferenceResult.fromMetrics(metrics, finalText)
        }

    /** Set system prompt for chat context. */
    fun setSystemPrompt(prompt: String) {
        if (nativePtr != 0L) {
            NativeBridge.setSystemPromptNative(nativePtr, prompt)
        }
    }

    /** Reset conversation history. */
    fun resetHistory() {
        if (nativePtr != 0L) {
            NativeBridge.resetNative(nativePtr)
        }
    }

    /** Release native resources. */
    fun release() {
        releaseInternal()
    }

    private fun releaseInternal() {
        if (nativePtr != 0L) {
            NativeBridge.releaseNative(nativePtr)
            nativePtr = 0L
            isModelLoaded = false
            modelName = "No model"
        }
    }

    companion object {
        /**
         * Strip model thinking tags from output.
         * Qwen3 and similar models emit <think>...</think> blocks
         * that should not be shown to the user.
         */
        fun stripThinkingTags(text: String): String {
            // Remove <think>...</think> blocks (including partial ones during streaming)
            var result = text
            // Complete thinking blocks
            result = result.replace(Regex("<think>[\\s\\S]*?</think>\\s*"), "")
            // Partial opening tag at the end (streaming)
            result = result.replace(Regex("<think>[\\s\\S]*$"), "")
            // Strip <eop> stop token if leaked
            result = result.replace("<eop>", "")
            return result.trimStart()
        }
    }
}
