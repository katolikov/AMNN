package com.mnn.benchmarkapp.jni

/**
 * JNI bridge to the MNN inference engine.
 * Loads libmnnbench.so which links against libMNN.so, libllm.so, libMNN_Express.so.
 */
object NativeBridge {

    init {
        System.loadLibrary("mnnbench")
    }

    /**
     * Initialize a native session with model directory and optional config JSON.
     * @return native pointer to the session, or 0 on failure.
     */
    external fun initNative(modelDir: String, configJson: String): Long

    /**
     * Run text-only inference with streaming progress callback.
     * @return HashMap with timing metrics (prompt_len, decode_len, prefill_time, decode_time).
     */
    external fun generateNative(
        sessionPtr: Long,
        prompt: String,
        progressListener: ProgressListener
    ): HashMap<String, Any>

    /**
     * Run VLM inference with an image file path.
     * @return HashMap with timing metrics including vision_time.
     */
    external fun generateWithImageNative(
        sessionPtr: Long,
        prompt: String,
        imagePath: String,
        progressListener: ProgressListener
    ): HashMap<String, Any>

    /** Release the native session and free resources. */
    external fun releaseNative(sessionPtr: Long)

    /** Reset conversation history. */
    external fun resetNative(sessionPtr: Long)

    /** Set system prompt for chat context. */
    external fun setSystemPromptNative(sessionPtr: Long, prompt: String)
}

/**
 * Progress callback interface for streaming token generation.
 * Implement onProgress to receive tokens as they are generated.
 */
interface ProgressListener {
    /**
     * Called for each generated token.
     * @param token The generated token text, or null when generation ends.
     * @return true to continue, false to stop generation.
     */
    fun onProgress(token: String?): Boolean
}
