package com.mnn.benchmarkapp.ui.vlm

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.ImeAction
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mnn.benchmarkapp.jni.BenchmarkSession
import com.mnn.benchmarkapp.ui.components.AiRunButton
import com.mnn.benchmarkapp.ui.components.MetricsCard
import com.mnn.benchmarkapp.ui.components.TemperatureChip
import com.mnn.benchmarkapp.ui.theme.DarkSurface
import com.mnn.benchmarkapp.ui.theme.OverlayBackground
import com.mnn.benchmarkapp.ui.theme.PureBlack
import com.mnn.benchmarkapp.ui.theme.PureWhite
import com.mnn.benchmarkapp.ui.theme.SubtleGray
import java.io.File

@Composable
fun VlmScreen(
    session: BenchmarkSession,
    viewModel: VlmViewModel = viewModel(),
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val outputText by viewModel.outputText.collectAsState()
    val lastResult by viewModel.lastResult.collectAsState()
    val isAnalyzing by viewModel.isAnalyzing.collectAsState()
    val capturedPath by viewModel.capturedImagePath.collectAsState()

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasCameraPermission = granted }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    val imageError by viewModel.imageError.collectAsState()
    val hasValidImage by viewModel.hasValidImage.collectAsState()

    // Gallery picker with proper error handling
    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        if (uri == null) {
            // User cancelled the picker
            return@rememberLauncherForActivityResult
        }
        try {
            val inputStream = context.contentResolver.openInputStream(uri)
            if (inputStream == null) {
                android.util.Log.e("VlmScreen", "openInputStream returned null for $uri")
                viewModel.setGalleryImage(null)
                return@rememberLauncherForActivityResult
            }
            val cacheFile = File(context.cacheDir, "gallery_image_${System.currentTimeMillis()}.jpg")
            inputStream.use { input ->
                cacheFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            if (cacheFile.exists() && cacheFile.length() > 0) {
                viewModel.setGalleryImage(cacheFile.absolutePath)
            } else {
                android.util.Log.e("VlmScreen", "Gallery image copy failed: ${cacheFile.absolutePath}")
                viewModel.setGalleryImage(null)
            }
        } catch (e: Exception) {
            android.util.Log.e("VlmScreen", "Gallery image error: ${e.message}", e)
            viewModel.setGalleryImage(null)
        }
    }

    val imageCapture = remember { ImageCapture.Builder().build() }

    // Prompt edit dialog state
    var showPromptDialog by remember { mutableStateOf(false) }
    var editingPrompt by remember { mutableStateOf(viewModel.vlmPrompt) }

    Box(modifier = Modifier.fillMaxSize().background(PureBlack)) {

        // Camera preview
        if (hasCameraPermission) {
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).also { previewView ->
                        val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                        cameraProviderFuture.addListener({
                            val cameraProvider = cameraProviderFuture.get()
                            val preview = Preview.Builder().build().also {
                                it.surfaceProvider = previewView.surfaceProvider
                            }
                            try {
                                cameraProvider.unbindAll()
                                cameraProvider.bindToLifecycle(
                                    lifecycleOwner,
                                    CameraSelector.DEFAULT_BACK_CAMERA,
                                    preview,
                                    imageCapture
                                )
                            } catch (_: Exception) {}
                        }, ContextCompat.getMainExecutor(ctx))
                    }
                },
                modifier = Modifier.fillMaxSize()
            )
        } else {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Text("Camera permission required", color = SubtleGray)
            }
        }

        // Top-right: Prompt edit icon button
        val promptButtonScale by animateFloatAsState(
            targetValue = if (showPromptDialog) 0.85f else 1f,
            animationSpec = spring(stiffness = Spring.StiffnessMedium),
            label = "prompt_btn_scale"
        )
        IconButton(
            onClick = {
                editingPrompt = viewModel.vlmPrompt
                showPromptDialog = true
            },
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 16.dp, end = 16.dp)
                .size(44.dp)
                .scale(promptButtonScale)
                .clip(CircleShape)
                .background(DarkSurface.copy(alpha = 0.8f)),
            colors = IconButtonDefaults.iconButtonColors(
                containerColor = Color.Transparent
            )
        ) {
            Icon(
                Icons.Outlined.Edit,
                contentDescription = "Edit prompt",
                tint = PureWhite,
                modifier = Modifier.size(22.dp)
            )
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
                        Text("VLM Prompt", color = PureWhite)
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
                        placeholder = { Text("Describe this image...", color = SubtleGray) },
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
                        maxLines = 4,
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                        keyboardActions = KeyboardActions(
                            onDone = {
                                viewModel.vlmPrompt = editingPrompt
                                showPromptDialog = false
                            }
                        ),
                    )
                },
                confirmButton = {
                    TextButton(onClick = {
                        viewModel.vlmPrompt = editingPrompt
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

        // Bottom section with animations
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .navigationBarsPadding()
        ) {
            // Temperature chip
            TemperatureChip(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
            )

            // Image loaded indicator
            AnimatedVisibility(
                visible = hasValidImage && capturedPath != null,
                enter = fadeIn() + scaleIn(initialScale = 0.8f),
                exit = fadeOut() + scaleOut(targetScale = 0.8f),
            ) {
                Row(
                    modifier = Modifier
                        .padding(horizontal = 16.dp, vertical = 2.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color(0xFF1B5E20).copy(alpha = 0.8f))
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "\u2713 Image ready",
                        style = MaterialTheme.typography.labelSmall,
                        color = PureWhite,
                        maxLines = 1,
                    )
                }
            }

            // Image error indicator
            AnimatedVisibility(
                visible = imageError != null,
                enter = fadeIn() + slideInVertically { it / 2 },
                exit = fadeOut() + slideOutVertically { it / 2 },
            ) {
                Row(
                    modifier = Modifier
                        .padding(horizontal = 16.dp, vertical = 2.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color(0xFFB71C1C).copy(alpha = 0.8f))
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = imageError ?: "",
                        style = MaterialTheme.typography.labelSmall,
                        color = PureWhite,
                        maxLines = 1,
                    )
                }
            }

            // Current prompt preview chip
            AnimatedVisibility(
                visible = true,
                enter = fadeIn() + slideInVertically { it / 2 },
            ) {
                Row(
                    modifier = Modifier
                        .padding(horizontal = 16.dp, vertical = 2.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(DarkSurface.copy(alpha = 0.7f))
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        Icons.Outlined.Edit,
                        contentDescription = null,
                        tint = SubtleGray,
                        modifier = Modifier.size(14.dp)
                    )
                    Spacer(Modifier.width(6.dp))
                    Text(
                        text = viewModel.vlmPrompt,
                        style = MaterialTheme.typography.labelSmall,
                        color = SubtleGray,
                        maxLines = 1,
                    )
                }
            }

            // Model output overlay
            AnimatedVisibility(
                visible = outputText.isNotEmpty(),
                enter = fadeIn() + slideInVertically { it },
                exit = fadeOut() + slideOutVertically { it },
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .background(OverlayBackground)
                        .padding(16.dp)
                ) {
                    Text(
                        text = outputText,
                        style = MaterialTheme.typography.bodyMedium,
                        color = PureWhite,
                        maxLines = 6
                    )
                }
            }
            if (outputText.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
            }

            // Metrics card
            AnimatedVisibility(
                visible = lastResult != null,
                enter = fadeIn() + scaleIn(initialScale = 0.9f),
                exit = fadeOut() + scaleOut(targetScale = 0.9f),
            ) {
                lastResult?.let { result ->
                    MetricsCard(
                        result = result,
                        modifier = Modifier.padding(horizontal = 16.dp)
                    )
                }
            }
            if (lastResult != null) {
                Spacer(Modifier.height(8.dp))
            }

            Spacer(Modifier.height(4.dp))

            // Control row
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 16.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Gallery button (left)
                IconButton(
                    onClick = {
                        galleryLauncher.launch(
                            androidx.activity.result.PickVisualMediaRequest(
                                ActivityResultContracts.PickVisualMedia.ImageOnly
                            )
                        )
                    },
                    colors = IconButtonDefaults.iconButtonColors(
                        containerColor = DarkSurface
                    ),
                    modifier = Modifier.size(56.dp).clip(CircleShape)
                ) {
                    Icon(
                        Icons.Outlined.Image,
                        contentDescription = "Gallery",
                        tint = PureWhite,
                        modifier = Modifier.size(24.dp)
                    )
                }

                // Capture button (center, large) with press animation
                var capturePressed by remember { mutableStateOf(false) }
                val captureScale by animateFloatAsState(
                    targetValue = if (capturePressed) 0.85f else 1f,
                    animationSpec = spring(
                        dampingRatio = Spring.DampingRatioMediumBouncy,
                        stiffness = Spring.StiffnessHigh
                    ),
                    label = "capture_scale",
                    finishedListener = { capturePressed = false }
                )
                IconButton(
                    onClick = {
                        capturePressed = true
                        val file = File(context.cacheDir, "capture_${System.currentTimeMillis()}.jpg")
                        val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
                        imageCapture.takePicture(
                            outputOptions,
                            ContextCompat.getMainExecutor(context),
                            object : ImageCapture.OnImageSavedCallback {
                                override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                                    viewModel.setCapturedImage(file.absolutePath)
                                }
                                override fun onError(exception: ImageCaptureException) {}
                            }
                        )
                    },
                    modifier = Modifier
                        .size(72.dp)
                        .scale(captureScale)
                        .clip(CircleShape)
                        .background(PureWhite),
                    colors = IconButtonDefaults.iconButtonColors(
                        containerColor = PureWhite
                    )
                ) {
                    Icon(
                        Icons.Outlined.CameraAlt,
                        contentDescription = "Capture",
                        tint = PureBlack,
                        modifier = Modifier.size(32.dp)
                    )
                }

                // Analyze button (right) — AI sparkle style
                AiRunButton(
                    isModelLoading = session.isModelLoading,
                    isModelReady = session.isModelLoaded && capturedPath != null,
                    isGenerating = isAnalyzing,
                    onClick = { viewModel.analyzeImage(session) },
                    modifier = Modifier.size(56.dp)
                )
            }
        }
    }
}
