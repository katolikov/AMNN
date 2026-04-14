package com.mnn.benchmarkapp.download

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.outlined.Smartphone
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import com.mnn.benchmarkapp.data.GeneratedConfig
import com.mnn.benchmarkapp.ui.theme.*
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelDownloadSheet(
    downloader: ModelDownloader,
    onModelSelected: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val progress by downloader.progress.collectAsState()

    var selectedSource by remember { mutableStateOf(DownloadSource.HUGGING_FACE) }

    // Read device_path from app_config.json if available
    val configModelPath = remember {
        try {
            val jsonStr = context.assets.open("app_config.json")
                .bufferedReader().use { it.readText() }
            val json = Json.parseToJsonElement(jsonStr).jsonObject
            json["model"]?.jsonObject?.get("device_path")?.jsonPrimitive?.content
                ?.takeIf { it.isNotBlank() }
        } catch (_: Exception) {
            null
        }
    }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = DarkSurface,
        shape = RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp)
                .padding(bottom = 32.dp)
        ) {
            Text(
                text = "Models",
                style = MaterialTheme.typography.headlineLarge,
                color = PureWhite,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            // Load from device path (manual entry)
            DevicePathSection(
                configPath = configModelPath,
                onModelSelected = { path ->
                    onModelSelected(path)
                    onDismiss()
                }
            )

            Spacer(Modifier.height(16.dp))

            // Source selector
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(DarkCard)
                    .padding(4.dp),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                DownloadSource.entries.forEach { source ->
                    val isSelected = selectedSource == source
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .clip(RoundedCornerShape(12.dp))
                            .background(if (isSelected) PureWhite else DarkCard)
                            .clickable { selectedSource = source }
                            .padding(vertical = 8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = source.displayName,
                            style = MaterialTheme.typography.labelSmall,
                            color = if (isSelected) PureBlack else SubtleGray,
                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                        )
                    }
                }
            }

            Spacer(Modifier.height(16.dp))

            // Download progress bar
            if (progress.state == DownloadState.DOWNLOADING || progress.state == DownloadState.LISTING_FILES) {
                DownloadProgressCard(progress)
                Spacer(Modifier.height(16.dp))
            }

            if (progress.state == DownloadState.FAILED) {
                Text(
                    text = "Failed: ${progress.error}",
                    style = MaterialTheme.typography.bodySmall,
                    color = PureWhite.copy(alpha = 0.7f),
                    modifier = Modifier.padding(bottom = 8.dp)
                )
            }

            // Model list
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.heightIn(max = 400.dp)
            ) {
                items(ModelRegistry.models) { model ->
                    val isDownloaded = downloader.isModelDownloaded(model.id)
                    val isDownloading = progress.state == DownloadState.DOWNLOADING ||
                            progress.state == DownloadState.LISTING_FILES
                    val hasSource = model.sources.containsKey(selectedSource.displayName)

                    ModelCard(
                        model = model,
                        isDownloaded = isDownloaded,
                        isDownloading = isDownloading,
                        hasSource = hasSource,
                        onDownload = {
                            downloader.reset()
                            scope.launch {
                                downloader.download(model, selectedSource)
                            }
                        },
                        onSelect = {
                            val modelPath = downloader.getModelDir(model.id).absolutePath
                            onModelSelected(modelPath)
                            onDismiss()
                        }
                    )
                }
            }
        }
    }
}

/**
 * Section for loading a model from a device path (pushed via adb or set in config).
 */
@Composable
private fun DevicePathSection(
    configPath: String?,
    onModelSelected: (String) -> Unit,
) {
    var devicePath by remember { mutableStateOf(configPath ?: GeneratedConfig.DEFAULT_MODEL_PATH) }
    var pathError by remember { mutableStateOf<String?>(null) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(DarkCard)
            .padding(16.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Icon(
                Icons.Outlined.Smartphone,
                contentDescription = null,
                tint = SubtleGray,
                modifier = Modifier.size(18.dp)
            )
            Text(
                text = "Load from device",
                style = MaterialTheme.typography.titleMedium,
                color = PureWhite,
            )
        }
        Spacer(Modifier.height(8.dp))
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            TextField(
                value = devicePath,
                onValueChange = {
                    devicePath = it
                    pathError = null
                },
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(12.dp)),
                placeholder = { Text("/data/local/tmp/...", color = SubtleGray) },
                singleLine = true,
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = PureBlack,
                    unfocusedContainerColor = PureBlack,
                    focusedIndicatorColor = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent,
                    focusedTextColor = PureWhite,
                    unfocusedTextColor = PureWhite,
                    cursorColor = PureWhite,
                ),
                textStyle = MaterialTheme.typography.bodySmall,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(
                    onDone = {
                        if (devicePath.isNotBlank()) {
                            onModelSelected(devicePath.trim())
                        }
                    }
                )
            )
            Button(
                onClick = {
                    if (devicePath.isNotBlank()) {
                        onModelSelected(devicePath.trim())
                    } else {
                        pathError = "Enter a path"
                    }
                },
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MediumGray),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
            ) {
                Text("Load", color = PureWhite, style = MaterialTheme.typography.labelLarge)
            }
        }
        if (pathError != null) {
            Spacer(Modifier.height(4.dp))
            Text(
                text = pathError!!,
                style = MaterialTheme.typography.bodySmall,
                color = PureWhite.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
private fun ModelCard(
    model: ModelInfo,
    isDownloaded: Boolean,
    isDownloading: Boolean,
    hasSource: Boolean,
    onDownload: () -> Unit,
    onSelect: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(if (isDownloaded) DarkCard else PureBlack)
            .clickable(enabled = isDownloaded) { onSelect() }
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = model.name,
                style = MaterialTheme.typography.titleMedium,
                color = PureWhite
            )
            Text(
                text = "${model.type.uppercase()} | ${model.sizeGb} GB",
                style = MaterialTheme.typography.bodySmall,
                color = SubtleGray
            )
            if (model.description.isNotEmpty()) {
                Text(
                    text = model.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = LightGray
                )
            }
        }

        Spacer(Modifier.width(12.dp))

        if (isDownloaded) {
            IconButton(
                onClick = onSelect,
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(PureWhite)
            ) {
                Icon(Icons.Filled.Check, "Select", tint = PureBlack, modifier = Modifier.size(20.dp))
            }
        } else if (hasSource) {
            IconButton(
                onClick = onDownload,
                enabled = !isDownloading,
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(if (!isDownloading) MediumGray else DarkCard)
            ) {
                Icon(
                    Icons.Filled.CloudDownload, "Download",
                    tint = if (!isDownloading) PureWhite else SubtleGray,
                    modifier = Modifier.size(20.dp)
                )
            }
        } else {
            Text("N/A", style = MaterialTheme.typography.bodySmall, color = SubtleGray)
        }
    }
}

@Composable
private fun DownloadProgressCard(progress: DownloadProgress) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(DarkCard)
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = when (progress.state) {
                    DownloadState.LISTING_FILES -> "Listing files..."
                    DownloadState.DOWNLOADING -> "Downloading ${progress.currentFile}"
                    else -> ""
                },
                style = MaterialTheme.typography.bodySmall,
                color = LightGray,
                maxLines = 1,
                modifier = Modifier.weight(1f)
            )
            if (progress.state == DownloadState.DOWNLOADING) {
                Text(
                    text = "${progress.downloadedMb} / ${progress.totalMb} MB",
                    style = MaterialTheme.typography.bodySmall,
                    color = SubtleGray
                )
            }
        }
        Spacer(Modifier.height(8.dp))
        LinearProgressIndicator(
            progress = { if (progress.state == DownloadState.LISTING_FILES) 0f else progress.progressPercent },
            modifier = Modifier
                .fillMaxWidth()
                .height(4.dp)
                .clip(RoundedCornerShape(2.dp)),
            color = PureWhite,
            trackColor = MediumGray,
        )
        if (progress.totalFiles > 0) {
            Spacer(Modifier.height(4.dp))
            Text(
                text = "File ${progress.fileIndex} of ${progress.totalFiles}",
                style = MaterialTheme.typography.labelSmall,
                color = SubtleGray
            )
        }
    }
}
