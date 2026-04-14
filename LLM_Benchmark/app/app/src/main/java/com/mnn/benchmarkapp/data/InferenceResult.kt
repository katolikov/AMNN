package com.mnn.benchmarkapp.data

data class InferenceResult(
    val promptLen: Long,
    val decodeLen: Long,
    val visionTimeUs: Long,
    val prefillTimeUs: Long,
    val decodeTimeUs: Long,
    val outputText: String,
) {
    val prefillToksPerSec: Double
        get() = if (prefillTimeUs > 0) promptLen * 1_000_000.0 / prefillTimeUs else 0.0

    val decodeToksPerSec: Double
        get() = if (decodeTimeUs > 0) decodeLen * 1_000_000.0 / decodeTimeUs else 0.0

    val ttftMs: Double
        get() = (prefillTimeUs + visionTimeUs) / 1000.0

    val totalTimeMs: Double
        get() = (prefillTimeUs + decodeTimeUs + visionTimeUs) / 1000.0

    companion object {
        fun empty() = InferenceResult(0, 0, 0, 0, 0, "")

        fun fromMetrics(metrics: HashMap<String, Any>, text: String): InferenceResult {
            fun getLong(key: String): Long = (metrics[key] as? Long) ?: 0L
            return InferenceResult(
                promptLen = getLong("prompt_len"),
                decodeLen = getLong("decode_len"),
                visionTimeUs = getLong("vision_time"),
                prefillTimeUs = getLong("prefill_time"),
                decodeTimeUs = getLong("decode_time"),
                outputText = text
            )
        }
    }
}
