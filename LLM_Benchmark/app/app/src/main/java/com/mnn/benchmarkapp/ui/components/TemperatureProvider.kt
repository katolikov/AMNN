package com.mnn.benchmarkapp.ui.components

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.mnn.benchmarkapp.ui.theme.OverlayBackground
import com.mnn.benchmarkapp.ui.theme.PureWhite
import com.mnn.benchmarkapp.ui.theme.SubtleGray
import kotlinx.coroutines.delay
import java.io.File

/**
 * Reads battery temperature via BatteryManager and GPU temperature via sysfs thermal zones.
 */
object TemperatureReader {

    fun readBatteryTemp(context: Context): Float {
        val intentFilter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        @Suppress("DEPRECATION")
        val batteryStatus = context.registerReceiver(null, intentFilter)
        val raw = batteryStatus?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        return raw / 10.0f
    }

    fun readGpuTemp(): Float? {
        // Search sysfs thermal zones for GPU-related zone
        val thermalDir = File("/sys/class/thermal/")
        if (!thermalDir.exists()) return null

        val zones = thermalDir.listFiles { f -> f.name.startsWith("thermal_zone") } ?: return null
        for (zone in zones) {
            try {
                val typeFile = File(zone, "type")
                val tempFile = File(zone, "temp")
                if (!typeFile.exists() || !tempFile.exists()) continue
                val type = typeFile.readText().trim().lowercase()
                if (type.contains("gpu") || type.contains("g3d") || type.contains("mali")) {
                    val raw = tempFile.readText().trim().toLongOrNull() ?: continue
                    // Temps >1000 are in millidegrees, otherwise already in degrees
                    return if (raw > 1000) raw / 1000.0f else raw.toFloat()
                }
            } catch (_: Exception) {
                continue
            }
        }
        return null
    }
}

/**
 * Compact temperature chip showing battery and optional GPU temperatures.
 */
@Composable
fun TemperatureChip(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var batteryTemp by remember { mutableStateOf("--") }
    var gpuTemp by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        while (true) {
            batteryTemp = "%.1f\u00B0C".format(TemperatureReader.readBatteryTemp(context))
            val gpu = TemperatureReader.readGpuTemp()
            gpuTemp = gpu?.let { "%.0f\u00B0C".format(it) }
            delay(3000L)
        }
    }

    Row(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .background(OverlayBackground)
            .padding(horizontal = 10.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "\uD83D\uDD0B $batteryTemp",
            style = MaterialTheme.typography.labelSmall,
            color = PureWhite
        )
        gpuTemp?.let {
            Text(
                text = "GPU $it",
                style = MaterialTheme.typography.labelSmall,
                color = SubtleGray
            )
        }
    }
}
