package com.mnn.benchmarkapp.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.unit.dp

private val BlackWhiteColorScheme = darkColorScheme(
    primary = PureWhite,
    onPrimary = PureBlack,
    primaryContainer = DarkCard,
    onPrimaryContainer = PureWhite,
    secondary = LightGray,
    onSecondary = PureBlack,
    secondaryContainer = DarkCard,
    onSecondaryContainer = LightGray,
    tertiary = SubtleGray,
    background = PureBlack,
    onBackground = PureWhite,
    surface = PureBlack,
    onSurface = PureWhite,
    surfaceVariant = DarkSurface,
    onSurfaceVariant = LightGray,
    outline = MediumGray,
    outlineVariant = DarkCard,
    inverseSurface = PureWhite,
    inverseOnSurface = PureBlack,
)

private val OneUIShapes = Shapes(
    small = RoundedCornerShape(16.dp),
    medium = RoundedCornerShape(24.dp),
    large = RoundedCornerShape(28.dp),
    extraLarge = RoundedCornerShape(32.dp)
)

@Composable
fun BenchmarkTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = BlackWhiteColorScheme,
        typography = BenchmarkTypography,
        shapes = OneUIShapes,
        content = content
    )
}
