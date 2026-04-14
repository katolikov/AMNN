package com.mnn.benchmarkapp.ui.llm

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.mnn.benchmarkapp.data.InferenceResult
import com.mnn.benchmarkapp.jni.BenchmarkSession
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class ChatMessage(
    val role: String, // "user" or "assistant"
    val content: String,
    val metrics: InferenceResult? = null,
    val isStreaming: Boolean = false,
)

class LlmViewModel : ViewModel() {

    var systemPrompt by mutableStateOf("You are a helpful assistant.")

    val messages = mutableStateListOf<ChatMessage>()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating

    private var generateJob: Job? = null

    fun sendMessage(prompt: String, session: BenchmarkSession) {
        if (prompt.isBlank() || !session.isModelLoaded) return

        messages.add(ChatMessage(role = "user", content = prompt))
        messages.add(ChatMessage(role = "assistant", content = "", isStreaming = true))

        _isGenerating.value = true
        val assistantIndex = messages.lastIndex

        generateJob = viewModelScope.launch {
            // Pass system prompt to native layer before generating
            session.setSystemPrompt(systemPrompt)

            // Collect streaming text
            val textJob = launch {
                session.streamingText.collect { text ->
                    if (assistantIndex < messages.size) {
                        messages[assistantIndex] = messages[assistantIndex].copy(content = text)
                    }
                }
            }

            val result = session.generate(prompt)
            textJob.cancel()

            if (assistantIndex < messages.size) {
                messages[assistantIndex] = ChatMessage(
                    role = "assistant",
                    content = result.outputText,
                    metrics = result,
                    isStreaming = false
                )
            }
            _isGenerating.value = false
        }
    }

    fun clearChat(session: BenchmarkSession) {
        generateJob?.cancel()
        messages.clear()
        session.resetHistory()
        _isGenerating.value = false
    }
}
