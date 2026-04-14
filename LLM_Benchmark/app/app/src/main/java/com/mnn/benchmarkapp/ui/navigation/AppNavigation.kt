package com.mnn.benchmarkapp.ui.navigation

import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.slideOutVertically
import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.mnn.benchmarkapp.jni.BenchmarkSession
import com.mnn.benchmarkapp.ui.llm.LlmScreen
import com.mnn.benchmarkapp.ui.settings.SettingsScreen
import com.mnn.benchmarkapp.ui.vlm.VlmScreen

object Routes {
    const val VLM = "vlm"
    const val LLM = "llm"
    const val SETTINGS = "settings"
}

// Shared animation specs
private const val TRANSITION_DURATION = 350

private fun slideEnterFromRight(): EnterTransition =
    slideInHorizontally(
        initialOffsetX = { fullWidth -> fullWidth / 4 },
        animationSpec = spring(stiffness = Spring.StiffnessMediumLow)
    ) + fadeIn(animationSpec = tween(TRANSITION_DURATION))

private fun slideExitToLeft(): ExitTransition =
    slideOutHorizontally(
        targetOffsetX = { fullWidth -> -fullWidth / 4 },
        animationSpec = spring(stiffness = Spring.StiffnessMediumLow)
    ) + fadeOut(animationSpec = tween(TRANSITION_DURATION / 2))

private fun slideEnterFromLeft(): EnterTransition =
    slideInHorizontally(
        initialOffsetX = { fullWidth -> -fullWidth / 4 },
        animationSpec = spring(stiffness = Spring.StiffnessMediumLow)
    ) + fadeIn(animationSpec = tween(TRANSITION_DURATION))

private fun slideExitToRight(): ExitTransition =
    slideOutHorizontally(
        targetOffsetX = { fullWidth -> fullWidth / 4 },
        animationSpec = spring(stiffness = Spring.StiffnessMediumLow)
    ) + fadeOut(animationSpec = tween(TRANSITION_DURATION / 2))

private fun settingsEnter(): EnterTransition =
    slideInVertically(
        initialOffsetY = { fullHeight -> fullHeight / 3 },
        animationSpec = spring(stiffness = Spring.StiffnessMediumLow)
    ) + fadeIn(animationSpec = tween(TRANSITION_DURATION)) +
    scaleIn(
        initialScale = 0.92f,
        animationSpec = tween(TRANSITION_DURATION)
    )

private fun settingsExit(): ExitTransition =
    slideOutVertically(
        targetOffsetY = { fullHeight -> fullHeight / 3 },
        animationSpec = spring(stiffness = Spring.StiffnessMediumLow)
    ) + fadeOut(animationSpec = tween(TRANSITION_DURATION / 2)) +
    scaleOut(
        targetScale = 0.92f,
        animationSpec = tween(TRANSITION_DURATION / 2)
    )

@Composable
fun AppNavigation(
    navController: NavHostController,
    session: BenchmarkSession,
) {
    NavHost(
        navController = navController,
        startDestination = Routes.VLM,
        enterTransition = { fadeIn(animationSpec = tween(TRANSITION_DURATION)) },
        exitTransition = { fadeOut(animationSpec = tween(TRANSITION_DURATION / 2)) },
    ) {
        composable(
            route = Routes.VLM,
            enterTransition = { slideEnterFromLeft() },
            exitTransition = { slideExitToLeft() },
            popEnterTransition = { slideEnterFromLeft() },
            popExitTransition = { slideExitToRight() },
        ) {
            VlmScreen(session = session)
        }
        composable(
            route = Routes.LLM,
            enterTransition = { slideEnterFromRight() },
            exitTransition = { slideExitToRight() },
            popEnterTransition = { slideEnterFromLeft() },
            popExitTransition = { slideExitToRight() },
        ) {
            LlmScreen(session = session)
        }
        composable(
            route = Routes.SETTINGS,
            enterTransition = { settingsEnter() },
            exitTransition = { settingsExit() },
            popEnterTransition = { fadeIn(tween(TRANSITION_DURATION)) },
            popExitTransition = { settingsExit() },
        ) {
            SettingsScreen(
                onBack = { navController.popBackStack() }
            )
        }
    }
}
