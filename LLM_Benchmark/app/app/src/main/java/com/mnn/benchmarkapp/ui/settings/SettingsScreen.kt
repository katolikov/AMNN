package com.mnn.benchmarkapp.ui.settings

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mnn.benchmarkapp.ui.theme.DarkCard
import com.mnn.benchmarkapp.ui.theme.DarkSurface
import com.mnn.benchmarkapp.ui.theme.LightGray
import com.mnn.benchmarkapp.ui.theme.MediumGray
import com.mnn.benchmarkapp.ui.theme.PureBlack
import com.mnn.benchmarkapp.ui.theme.PureWhite
import com.mnn.benchmarkapp.ui.theme.SubtleGray

@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    viewModel: SettingsViewModel = viewModel(),
) {
    val context = LocalContext.current

    LaunchedEffect(Unit) {
        viewModel.loadConfig(context)
    }

    val isGpu = viewModel.isGpuBackend()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(PureBlack)
    ) {
        // Header
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onBack) {
                Icon(
                    Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back",
                    tint = PureWhite
                )
            }
            Text(
                text = "Settings",
                style = MaterialTheme.typography.headlineLarge,
                color = PureWhite,
                modifier = Modifier.padding(start = 8.dp)
            )
        }

        // Filter params by backend-dependent visibility
        val visibleParams = KNOWN_PARAMS
            .filter { viewModel.parameters.containsKey(it.key) }
            .filter { param ->
                when (param.visibleWhen) {
                    ParamVisibility.ALWAYS -> true
                    ParamVisibility.CPU_ONLY -> !isGpu
                    ParamVisibility.GPU_ONLY -> isGpu
                }
            }

        val grouped = visibleParams.groupBy { it.category }

        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            grouped.forEach { (category, params) ->
                item {
                    Text(
                        text = category,
                        style = MaterialTheme.typography.titleMedium,
                        color = SubtleGray,
                        modifier = Modifier.padding(top = 16.dp, bottom = 4.dp)
                    )
                }

                items(params) { param ->
                    val value = viewModel.parameters[param.key] ?: ""
                    SettingItem(
                        meta = param,
                        value = value,
                        onValueChange = { viewModel.updateParameter(param.key, it) }
                    )
                }
            }

            // Show unknown parameters as text
            val knownKeys = KNOWN_PARAMS.map { it.key }.toSet() + "enable_op_profile" + "hints" + "gpu_mode" + "session_mode"
            val unknownParams = viewModel.parameters.filter { it.key !in knownKeys }
            if (unknownParams.isNotEmpty()) {
                item {
                    Text(
                        text = "Other",
                        style = MaterialTheme.typography.titleMedium,
                        color = SubtleGray,
                        modifier = Modifier.padding(top = 16.dp, bottom = 4.dp)
                    )
                }
                items(unknownParams.entries.toList()) { (key, value) ->
                    SettingItemText(key, value)
                }
            }

            item { Spacer(Modifier.height(32.dp)) }
        }
    }
}

@Composable
private fun SettingItem(
    meta: ParamMeta,
    value: String,
    onValueChange: (String) -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(DarkSurface)
            .padding(16.dp)
    ) {
        Text(
            text = meta.label,
            style = MaterialTheme.typography.bodyLarge,
            color = PureWhite
        )
        Spacer(Modifier.height(8.dp))

        when (meta.type) {
            ParamType.DROPDOWN -> {
                var expanded by remember { mutableStateOf(false) }
                OutlinedButton(
                    onClick = { expanded = true },
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(value, color = LightGray)
                }
                DropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false },
                    modifier = Modifier.background(DarkCard)
                ) {
                    meta.options?.forEach { option ->
                        DropdownMenuItem(
                            text = { Text(option, color = PureWhite) },
                            onClick = {
                                onValueChange(option)
                                expanded = false
                            }
                        )
                    }
                }
            }

            ParamType.SLIDER_INT -> {
                val range = meta.range ?: 0f..100f
                val floatVal = value.toFloatOrNull() ?: range.start
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Slider(
                        value = floatVal.coerceIn(range),
                        onValueChange = { onValueChange(it.toInt().toString()) },
                        valueRange = range,
                        steps = (range.endInclusive - range.start).toInt() - 1,
                        modifier = Modifier.weight(1f),
                        colors = SliderDefaults.colors(
                            thumbColor = PureWhite,
                            activeTrackColor = PureWhite,
                            inactiveTrackColor = MediumGray
                        )
                    )
                    Text(
                        text = floatVal.toInt().toString(),
                        style = MaterialTheme.typography.titleMedium,
                        color = LightGray,
                        modifier = Modifier.padding(start = 12.dp)
                    )
                }
            }

            ParamType.SLIDER_FLOAT -> {
                val range = meta.range ?: 0f..1f
                val floatVal = value.toFloatOrNull() ?: range.start
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Slider(
                        value = floatVal.coerceIn(range),
                        onValueChange = { onValueChange("%.2f".format(it)) },
                        valueRange = range,
                        modifier = Modifier.weight(1f),
                        colors = SliderDefaults.colors(
                            thumbColor = PureWhite,
                            activeTrackColor = PureWhite,
                            inactiveTrackColor = MediumGray
                        )
                    )
                    Text(
                        text = "%.2f".format(floatVal),
                        style = MaterialTheme.typography.titleMedium,
                        color = LightGray,
                        modifier = Modifier.padding(start = 12.dp)
                    )
                }
            }

            ParamType.TOGGLE -> {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = if (value == "true") "Enabled" else "Disabled",
                        style = MaterialTheme.typography.bodyMedium,
                        color = LightGray
                    )
                    Switch(
                        checked = value == "true",
                        onCheckedChange = { onValueChange(it.toString()) },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = PureBlack,
                            checkedTrackColor = PureWhite,
                            uncheckedThumbColor = SubtleGray,
                            uncheckedTrackColor = MediumGray
                        )
                    )
                }
            }

            ParamType.TEXT -> {
                Text(
                    text = value,
                    style = MaterialTheme.typography.bodyMedium,
                    color = LightGray
                )
            }

            ParamType.MULTI_SELECT -> {
                MultiSelectChips(
                    options = meta.options ?: emptyList(),
                    selected = value.split(",").map { it.trim() }.filter { it.isNotEmpty() }.toSet(),
                    onSelectionChanged = { newSet ->
                        onValueChange(newSet.joinToString(","))
                    }
                )
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun MultiSelectChips(
    options: List<String>,
    selected: Set<String>,
    onSelectionChanged: (Set<String>) -> Unit,
) {
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        options.forEach { option ->
            val isSelected = option in selected
            FilterChip(
                selected = isSelected,
                onClick = {
                    val newSet = if (isSelected) selected - option else selected + option
                    onSelectionChanged(newSet)
                },
                label = {
                    Text(
                        text = option.removePrefix("MNN_GPU_").removePrefix("Session_").removePrefix("Module_"),
                        style = MaterialTheme.typography.labelSmall
                    )
                },
                colors = FilterChipDefaults.filterChipColors(
                    containerColor = DarkCard,
                    labelColor = SubtleGray,
                    selectedContainerColor = PureWhite,
                    selectedLabelColor = PureBlack
                )
            )
        }
    }
}

@Composable
private fun SettingItemText(key: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(DarkSurface)
            .padding(16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = key,
            style = MaterialTheme.typography.bodyLarge,
            color = PureWhite
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            color = LightGray
        )
    }
}
