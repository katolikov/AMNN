package com.mnn.benchmarkapp.ui.llm

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.DeleteOutline
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mnn.benchmarkapp.jni.BenchmarkSession
import com.mnn.benchmarkapp.ui.components.AiRunButton
import com.mnn.benchmarkapp.ui.components.TemperatureChip
import com.mnn.benchmarkapp.ui.theme.DarkCard
import com.mnn.benchmarkapp.ui.theme.DarkSurface
import com.mnn.benchmarkapp.ui.theme.ModelBubble
import com.mnn.benchmarkapp.ui.theme.PureBlack
import com.mnn.benchmarkapp.ui.theme.PureWhite
import com.mnn.benchmarkapp.ui.theme.SubtleGray
import com.mnn.benchmarkapp.ui.theme.UserBubble

@Composable
fun LlmScreen(
    session: BenchmarkSession,
    viewModel: LlmViewModel = viewModel(),
) {
    val isGenerating by viewModel.isGenerating.collectAsState()
    val listState = rememberLazyListState()

    // Prompt edit dialog state
    var showPromptDialog by remember { mutableStateOf(false) }
    var editingPrompt by remember { mutableStateOf(viewModel.systemPrompt) }

    // Auto-scroll to bottom
    LaunchedEffect(viewModel.messages.size) {
        if (viewModel.messages.isNotEmpty()) {
            listState.animateScrollToItem(viewModel.messages.lastIndex)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(PureBlack)
    ) {
        // Header
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "Chat",
                style = MaterialTheme.typography.headlineLarge,
                color = PureWhite
            )
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                TemperatureChip()

                // Prompt edit button
                val promptButtonScale by animateFloatAsState(
                    targetValue = if (showPromptDialog) 0.85f else 1f,
                    animationSpec = spring(stiffness = Spring.StiffnessMedium),
                    label = "llm_prompt_btn_scale"
                )
                IconButton(
                    onClick = {
                        editingPrompt = viewModel.systemPrompt
                        showPromptDialog = true
                    },
                    modifier = Modifier
                        .size(36.dp)
                        .scale(promptButtonScale)
                        .clip(CircleShape)
                        .background(DarkSurface),
                    colors = IconButtonDefaults.iconButtonColors(
                        containerColor = Color.Transparent
                    )
                ) {
                    Icon(
                        Icons.Outlined.Edit,
                        contentDescription = "Edit prompt",
                        tint = PureWhite,
                        modifier = Modifier.size(18.dp)
                    )
                }

                IconButton(onClick = { viewModel.clearChat(session) }) {
                    Icon(
                        Icons.Outlined.DeleteOutline,
                        contentDescription = "Clear chat",
                        tint = SubtleGray
                    )
                }
            }
        }

        // Prompt edit dialog
        if (showPromptDialog) {
            AlertDialog(
                onDismissRequest = { showPromptDialog = false },
                containerColor = DarkSurface,
                title = {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text("System Prompt", color = PureWhite)
                        IconButton(
                            onClick = { showPromptDialog = false },
                            modifier = Modifier.size(32.dp)
                        ) {
                            Icon(
                                Icons.Outlined.Close,
                                contentDescription = "Close",
                                tint = SubtleGray,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                },
                text = {
                    TextField(
                        value = editingPrompt,
                        onValueChange = { editingPrompt = it },
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp)),
                        placeholder = { Text("You are a helpful assistant...", color = SubtleGray) },
                        colors = TextFieldDefaults.colors(
                            focusedContainerColor = PureBlack,
                            unfocusedContainerColor = PureBlack,
                            focusedIndicatorColor = Color.Transparent,
                            unfocusedIndicatorColor = Color.Transparent,
                            focusedTextColor = PureWhite,
                            unfocusedTextColor = PureWhite,
                            cursorColor = PureWhite,
                        ),
                        textStyle = MaterialTheme.typography.bodyMedium,
                        maxLines = 6,
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                        keyboardActions = KeyboardActions(
                            onDone = {
                                viewModel.systemPrompt = editingPrompt
                                showPromptDialog = false
                            }
                        ),
                    )
                },
                confirmButton = {
                    TextButton(onClick = {
                        viewModel.systemPrompt = editingPrompt
                        showPromptDialog = false
                    }) {
                        Text("Save", color = PureWhite)
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showPromptDialog = false }) {
                        Text("Cancel", color = SubtleGray)
                    }
                },
            )
        }

        // Message list
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            state = listState,
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(viewModel.messages) { message ->
                AnimatedVisibility(
                    visible = true,
                    enter = fadeIn() + slideInVertically { it / 3 } + scaleIn(initialScale = 0.95f),
                ) {
                    ChatBubble(message)
                }
            }

            if (viewModel.messages.isEmpty()) {
                item {
                    Box(
                        modifier = Modifier.fillParentMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = if (session.isModelLoaded) "Send a message to start"
                            else "Load a model to begin",
                            style = MaterialTheme.typography.bodyLarge,
                            color = SubtleGray
                        )
                    }
                }
            }
        }

        // Input field with AI send button
        ChatInput(
            enabled = session.isModelLoaded && !isGenerating,
            isModelLoading = session.isModelLoading,
            isModelReady = session.isModelLoaded,
            isGenerating = isGenerating,
            onSend = { viewModel.sendMessage(it, session) }
        )
    }
}

@Composable
private fun ChatBubble(message: ChatMessage) {
    val isUser = message.role == "user"

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 300.dp)
                .clip(
                    RoundedCornerShape(
                        topStart = 20.dp,
                        topEnd = 20.dp,
                        bottomStart = if (isUser) 20.dp else 4.dp,
                        bottomEnd = if (isUser) 4.dp else 20.dp
                    )
                )
                .background(if (isUser) UserBubble else ModelBubble)
                .padding(14.dp)
        ) {
            if (message.content.isEmpty() && message.isStreaming) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp,
                    color = SubtleGray
                )
            } else {
                Text(
                    text = message.content,
                    style = MaterialTheme.typography.bodyLarge,
                    color = PureWhite
                )
            }
        }

        // Inline metrics after assistant messages
        message.metrics?.let { metrics ->
            Spacer(Modifier.height(4.dp))
            Text(
                text = "%.1f tok/s decode | %.0f ms TTFT".format(
                    metrics.decodeToksPerSec, metrics.ttftMs
                ),
                style = MaterialTheme.typography.bodySmall,
                color = SubtleGray,
                modifier = Modifier.padding(horizontal = 4.dp)
            )
        }
    }
}

@Composable
private fun ChatInput(
    enabled: Boolean,
    isModelLoading: Boolean,
    isModelReady: Boolean,
    isGenerating: Boolean,
    onSend: (String) -> Unit,
) {
    var text by remember { mutableStateOf("") }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .imePadding()
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        TextField(
            value = text,
            onValueChange = { text = it },
            modifier = Modifier
                .weight(1f)
                .clip(RoundedCornerShape(28.dp)),
            placeholder = {
                Text("Message...", color = SubtleGray)
            },
            enabled = enabled,
            colors = TextFieldDefaults.colors(
                focusedContainerColor = DarkSurface,
                unfocusedContainerColor = DarkSurface,
                disabledContainerColor = DarkCard,
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent,
                disabledIndicatorColor = Color.Transparent,
                focusedTextColor = PureWhite,
                unfocusedTextColor = PureWhite,
                cursorColor = PureWhite
            ),
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
            keyboardActions = KeyboardActions(
                onSend = {
                    if (text.isNotBlank()) {
                        onSend(text)
                        text = ""
                    }
                }
            ),
            singleLine = true
        )

        // AI sparkle send button
        AiRunButton(
            isModelLoading = isModelLoading,
            isModelReady = isModelReady && text.isNotBlank(),
            isGenerating = isGenerating,
            onClick = {
                if (text.isNotBlank()) {
                    onSend(text)
                    text = ""
                }
            },
            modifier = Modifier.size(48.dp)
        )
    }
}
