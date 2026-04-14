package com.mnn.benchmarkapp.download

import kotlinx.serialization.Serializable

@Serializable
data class ModelInfo(
    val id: String,
    val name: String,
    val sizeGb: Double,
    val type: String, // "llm" or "vlm"
    val sources: Map<String, String>, // source name -> repo id
    val description: String = "",
)

enum class DownloadSource(val displayName: String) {
    HUGGING_FACE("HuggingFace"),
    MODEL_SCOPE("ModelScope"),
    MODELERS("Modelers"),
}

/**
 * Built-in model catalog. Matches MNN Chat's model_market.json entries.
 */
object ModelRegistry {

    val models = listOf(
        ModelInfo(
            id = "SmolVLM2-2.2B-Instruct-MNN",
            name = "SmolVLM2 2.2B",
            sizeGb = 2.2,
            type = "vlm",
            sources = mapOf(
                "HuggingFace" to "taobao-mnn/SmolVLM2-2.2B-Instruct-MNN",
                "ModelScope" to "MNN/SmolVLM2-2.2B-Instruct-MNN",
                "Modelers" to "MNN/SmolVLM2-2.2B-Instruct-MNN",
            ),
            description = "Vision-Language model, 2.2B parameters"
        ),
        ModelInfo(
            id = "SmolVLM2-500M-Video-Instruct-MNN",
            name = "SmolVLM2 500M",
            sizeGb = 0.5,
            type = "vlm",
            sources = mapOf(
                "HuggingFace" to "taobao-mnn/SmolVLM2-500M-Video-Instruct-MNN",
                "ModelScope" to "MNN/SmolVLM2-500M-Video-Instruct-MNN",
                "Modelers" to "MNN/SmolVLM2-500M-Video-Instruct-MNN",
            ),
            description = "Lightweight VLM, 500M parameters"
        ),
        ModelInfo(
            id = "Qwen2.5-1.5B-Instruct-MNN",
            name = "Qwen2.5 1.5B",
            sizeGb = 1.6,
            type = "llm",
            sources = mapOf(
                "HuggingFace" to "taobao-mnn/Qwen2.5-1.5B-Instruct-MNN",
                "ModelScope" to "MNN/Qwen2.5-1.5B-Instruct-MNN",
            ),
            description = "Text-only LLM, 1.5B parameters"
        ),
        ModelInfo(
            id = "Qwen3-0.6B-MNN",
            name = "Qwen3 0.6B",
            sizeGb = 0.6,
            type = "llm",
            sources = mapOf(
                "HuggingFace" to "taobao-mnn/Qwen3-0.6B-MNN",
                "ModelScope" to "MNN/Qwen3-0.6B-MNN",
            ),
            description = "Compact LLM, 0.6B parameters"
        ),
    )

    fun findById(id: String): ModelInfo? = models.find { it.id == id }
}
