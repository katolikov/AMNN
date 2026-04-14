package com.mnn.benchmarkapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.mnn.benchmarkapp.download.ModelDownloadSheet
import com.mnn.benchmarkapp.download.ModelDownloader
import com.mnn.benchmarkapp.jni.BenchmarkSession
import com.mnn.benchmarkapp.ui.components.TopNavBar
import com.mnn.benchmarkapp.ui.settings.SettingsViewModel
import com.mnn.benchmarkapp.ui.navigation.AppNavigation
import com.mnn.benchmarkapp.ui.navigation.Routes
import com.mnn.benchmarkapp.ui.theme.BenchmarkTheme
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private val session = BenchmarkSession()
    private val settingsViewModel: SettingsViewModel by viewModels()
    private lateinit var downloader: ModelDownloader

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        downloader = ModelDownloader(this)

        setContent {
            BenchmarkTheme {
                val context = LocalContext.current
                val navController = rememberNavController()
                val scope = rememberCoroutineScope()
                var currentMode by remember { mutableStateOf(Routes.VLM) }
                var showDownloadSheet by remember { mutableStateOf(false) }

                // Track whether the current route is Settings to hide TopNavBar
                val navBackStackEntry by navController.currentBackStackEntryAsState()
                val isOnSettingsScreen = navBackStackEntry?.destination?.route == Routes.SETTINGS

                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    topBar = {
                        if (!isOnSettingsScreen) {
                            TopNavBar(
                                currentMode = currentMode,
                                modelName = session.modelName,
                                isModelLoading = session.isModelLoading,
                                isModelReady = session.isModelLoaded,
                                onModeChanged = { mode ->
                                    currentMode = mode
                                    navController.navigate(mode) {
                                        popUpTo(Routes.VLM) { inclusive = false }
                                        launchSingleTop = true
                                    }
                                },
                                onModelSelect = { showDownloadSheet = true },
                                onSettingsClick = {
                                    navController.navigate(Routes.SETTINGS) {
                                        launchSingleTop = true
                                    }
                                }
                            )
                        }
                    }
                ) { innerPadding ->
                    Column(modifier = Modifier.padding(innerPadding)) {
                        AppNavigation(
                            navController = navController,
                            session = session,
                        )
                    }
                }

                // Model download/select bottom sheet
                if (showDownloadSheet) {
                    ModelDownloadSheet(
                        downloader = downloader,
                        onModelSelected = { modelPath ->
                            scope.launch {
                                // Include app cache dir as tmp_path for model cache
                                settingsViewModel.updateParameter(
                                    "tmp_path", context.cacheDir.absolutePath
                                )
                                val configJson = settingsViewModel.toConfigJson()
                                session.loadModel(modelPath, configJson)
                            }
                        },
                        onDismiss = { showDownloadSheet = false }
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        session.release()
    }
}
