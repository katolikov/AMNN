package com.mnn.benchmarkapp.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import com.mnn.benchmarkapp.data.InferenceResult
import com.mnn.benchmarkapp.ui.theme.DarkSurface
import com.mnn.benchmarkapp.ui.theme.LightGray
import com.mnn.benchmarkapp.ui.theme.PureWhite
import com.mnn.benchmarkapp.ui.theme.SubtleGray

@Composable
fun MetricsCard(
    result: InferenceResult,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .background(DarkSurface)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Performance",
            style = MaterialTheme.typography.titleMedium,
            color = PureWhite
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            MetricItem("Prefill", "%.1f tok/s".format(result.prefillToksPerSec))
            MetricItem("Decode", "%.1f tok/s".format(result.decodeToksPerSec))
            MetricItem("TTFT", "%.1f ms".format(result.ttftMs))
        }

        if (result.visionTimeUs > 0) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                MetricItem("Vision", "%.1f ms".format(result.visionTimeUs / 1000.0))
                MetricItem("Total", "%.1f ms".format(result.totalTimeMs))
                MetricItem("Tokens", "${result.decodeLen}")
            }
        }
    }
}

@Composable
private fun MetricItem(label: String, value: String) {
    Column {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = SubtleGray
        )
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            color = LightGray
        )
    }
}
