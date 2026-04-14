package com.mnn.benchmarkapp.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.FolderOpen
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.mnn.benchmarkapp.ui.navigation.Routes
import com.mnn.benchmarkapp.ui.theme.DarkCard
import com.mnn.benchmarkapp.ui.theme.DarkSurface
import com.mnn.benchmarkapp.ui.theme.PureBlack
import com.mnn.benchmarkapp.ui.theme.PureWhite
import com.mnn.benchmarkapp.ui.theme.SubtleGray

// Samsung Galaxy AI-style blue/purple gradient
private val AiGradientStart = Color(0xFF6B8AFF)
private val AiGradientEnd = Color(0xFFB06BFF)

@Composable
fun TopNavBar(
    currentMode: String,
    modelName: String,
    isModelLoading: Boolean,
    isModelReady: Boolean,
    onModeChanged: (String) -> Unit,
    onModelSelect: () -> Unit,
    onSettingsClick: () -> Unit,
) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .statusBarsPadding(),
        color = PureBlack,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            // Left: Model selector chip
            Row(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(DarkSurface)
                    .clickable { onModelSelect() }
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    Icons.Outlined.FolderOpen,
                    contentDescription = "Select model",
                    tint = PureWhite,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(Modifier.width(6.dp))
                Text(
                    text = modelName,
                    style = MaterialTheme.typography.labelLarge,
                    color = PureWhite,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.width(80.dp)
                )
            }

            // Center: Mode toggle
            ModeToggle(
                currentMode = currentMode,
                onModeChanged = onModeChanged
            )

            // Right: Settings gear with press animation
            val settingsScale by animateFloatAsState(
                targetValue = 1f,
                animationSpec = spring(stiffness = Spring.StiffnessMedium),
                label = "settings_scale"
            )
            IconButton(
                onClick = onSettingsClick,
                modifier = Modifier.scale(settingsScale)
            ) {
                Icon(
                    Icons.Outlined.Settings,
                    contentDescription = "Settings",
                    tint = PureWhite
                )
            }
        }
    }
}

/**
 * Samsung Galaxy AI-style run button with sparkle icon.
 * Shows a rotating gradient ring while model is loading;
 * glows when model is ready.
 */
@Composable
fun AiRunButton(
    isModelLoading: Boolean,
    isModelReady: Boolean,
    isGenerating: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val enabled = isModelReady && !isModelLoading && !isGenerating

    val infiniteTransition = rememberInfiniteTransition(label = "ai_glow")
    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.5f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "glow_alpha"
    )

    val gradientBrush = Brush.linearGradient(
        colors = listOf(AiGradientStart, AiGradientEnd),
        start = Offset.Zero,
        end = Offset(100f, 100f)
    )

    Box(
        modifier = modifier
            .size(56.dp)
            .clip(CircleShape)
            .then(
                if (isModelLoading) {
                    Modifier.background(DarkCard)
                } else if (enabled) {
                    Modifier
                        .drawBehind {
                            drawCircle(
                                brush = gradientBrush,
                                radius = size.minDimension / 2,
                                alpha = glowAlpha
                            )
                        }
                } else {
                    Modifier.background(DarkCard)
                }
            )
            .clickable(enabled = enabled) { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        when {
            isModelLoading -> {
                CircularProgressIndicator(
                    modifier = Modifier.size(28.dp),
                    strokeWidth = 2.5.dp,
                    color = AiGradientStart,
                    trackColor = DarkSurface,
                )
            }
            isGenerating -> {
                CircularProgressIndicator(
                    modifier = Modifier.size(28.dp),
                    strokeWidth = 2.5.dp,
                    color = AiGradientEnd,
                    trackColor = DarkSurface,
                )
            }
            else -> {
                // Sparkle / star icon drawn with Canvas for Samsung AI look
                SparkleIcon(
                    tint = if (enabled) PureWhite else SubtleGray,
                    modifier = Modifier.size(28.dp)
                )
            }
        }
    }
}

/**
 * A four-pointed sparkle icon (Samsung Galaxy AI style).
 */
@Composable
private fun SparkleIcon(
    tint: Color,
    modifier: Modifier = Modifier,
) {
    androidx.compose.foundation.Canvas(modifier = modifier) {
        val cx = size.width / 2
        val cy = size.height / 2
        val r = size.minDimension / 2
        val inner = r * 0.25f

        // Four-pointed star path
        val path = androidx.compose.ui.graphics.Path().apply {
            // Top point
            moveTo(cx, cy - r)
            // Right curve
            quadraticBezierTo(cx + inner, cy - inner, cx + r, cy)
            // Bottom curve
            quadraticBezierTo(cx + inner, cy + inner, cx, cy + r)
            // Left curve
            quadraticBezierTo(cx - inner, cy + inner, cx - r, cy)
            // Back to top
            quadraticBezierTo(cx - inner, cy - inner, cx, cy - r)
            close()
        }
        drawPath(
            path = path,
            color = tint,
        )

        // Small accent sparkle (top-right)
        val sr = r * 0.2f
        val scx = cx + r * 0.55f
        val scy = cy - r * 0.55f
        val si = sr * 0.2f
        val smallPath = androidx.compose.ui.graphics.Path().apply {
            moveTo(scx, scy - sr)
            quadraticBezierTo(scx + si, scy - si, scx + sr, scy)
            quadraticBezierTo(scx + si, scy + si, scx, scy + sr)
            quadraticBezierTo(scx - si, scy + si, scx - sr, scy)
            quadraticBezierTo(scx - si, scy - si, scx, scy - sr)
            close()
        }
        drawPath(
            path = smallPath,
            color = tint,
        )
    }
}

@Composable
private fun ModeToggle(
    currentMode: String,
    onModeChanged: (String) -> Unit,
) {
    val modes = listOf(Routes.VLM to "VLM", Routes.LLM to "LLM")

    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(24.dp))
            .background(DarkSurface)
            .padding(4.dp)
    ) {
        modes.forEach { (route, label) ->
            val isSelected = currentMode == route
            val bgColor by animateColorAsState(
                targetValue = if (isSelected) PureWhite else DarkSurface,
                animationSpec = tween(250),
                label = "mode_bg_$route"
            )
            val textColor by animateColorAsState(
                targetValue = if (isSelected) PureBlack else PureWhite,
                animationSpec = tween(250),
                label = "mode_text_$route"
            )
            val scaleAnim by animateFloatAsState(
                targetValue = if (isSelected) 1f else 0.95f,
                animationSpec = spring(stiffness = Spring.StiffnessMedium),
                label = "mode_scale_$route"
            )
            Box(
                modifier = Modifier
                    .scale(scaleAnim)
                    .clip(RoundedCornerShape(20.dp))
                    .background(bgColor)
                    .clickable { onModeChanged(route) }
                    .padding(horizontal = 20.dp, vertical = 8.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = label,
                    style = MaterialTheme.typography.labelLarge,
                    color = textColor
                )
            }
        }
    }
}
