package com.mnn.benchmarkapp.download

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.*
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

private const val TAG = "ModelDownloader"

data class DownloadProgress(
    val state: DownloadState = DownloadState.IDLE,
    val totalBytes: Long = 0L,
    val downloadedBytes: Long = 0L,
    val currentFile: String = "",
    val fileIndex: Int = 0,
    val totalFiles: Int = 0,
    val error: String? = null,
) {
    val progressPercent: Float
        get() = if (totalBytes > 0) downloadedBytes.toFloat() / totalBytes else 0f

    val downloadedMb: String get() = "%.1f".format(downloadedBytes / 1_048_576.0)
    val totalMb: String get() = if (totalBytes > 0) "%.1f".format(totalBytes / 1_048_576.0) else "?"
}

enum class DownloadState {
    IDLE, LISTING_FILES, DOWNLOADING, FINISHED, FAILED
}

/**
 * Multi-source model downloader supporting HuggingFace, ModelScope, and Modelers.
 */
class ModelDownloader(private val context: Context) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    private val _progress = MutableStateFlow(DownloadProgress())
    val progress: StateFlow<DownloadProgress> = _progress

    val modelsDir: File
        get() = File(context.getExternalFilesDir(null), "models")

    fun getModelDir(modelId: String): File = File(modelsDir, modelId)

    fun isModelDownloaded(modelId: String): Boolean {
        val dir = getModelDir(modelId)
        return dir.exists() && File(dir, "config.json").exists()
    }

    /**
     * Download a model from the best available source.
     * Tries sources in order: HuggingFace -> ModelScope -> Modelers
     */
    suspend fun download(model: ModelInfo, preferredSource: DownloadSource? = null) {
        val sourceOrder = if (preferredSource != null) {
            listOf(preferredSource) + DownloadSource.entries.filter { it != preferredSource }
        } else {
            listOf(DownloadSource.HUGGING_FACE, DownloadSource.MODEL_SCOPE, DownloadSource.MODELERS)
        }

        var lastError: Exception? = null
        for (source in sourceOrder) {
            val repoId = model.sources[source.displayName] ?: continue
            try {
                Log.d(TAG, "Trying ${source.displayName}: $repoId")
                when (source) {
                    DownloadSource.HUGGING_FACE -> downloadFromHuggingFace(model.id, repoId)
                    DownloadSource.MODEL_SCOPE -> downloadFromModelScope(model.id, repoId)
                    DownloadSource.MODELERS -> downloadFromModelers(model.id, repoId)
                }
                return // Success
            } catch (e: Exception) {
                Log.w(TAG, "${source.displayName} failed: ${e.message}")
                lastError = e
                // Clean up partial download
                getModelDir(model.id).deleteRecursively()
            }
        }

        _progress.value = _progress.value.copy(
            state = DownloadState.FAILED,
            error = "All sources failed: ${lastError?.message}"
        )
    }

    // ── HuggingFace ────────────────────────────────────────────────

    private suspend fun downloadFromHuggingFace(modelId: String, repoId: String) =
        withContext(Dispatchers.IO) {
            _progress.value = DownloadProgress(state = DownloadState.LISTING_FILES)

            // List files via HF API
            val apiUrl = "https://huggingface.co/api/models/$repoId"
            val files = listHfFiles(apiUrl)
            Log.d(TAG, "HF: ${files.size} files to download")

            val outputDir = getModelDir(modelId)
            outputDir.mkdirs()

            val totalSize = files.sumOf { it.second }
            var downloaded = 0L

            _progress.value = DownloadProgress(
                state = DownloadState.DOWNLOADING,
                totalBytes = totalSize,
                totalFiles = files.size
            )

            files.forEachIndexed { index, (path, size) ->
                val fileUrl = "https://huggingface.co/$repoId/resolve/main/$path"
                val outFile = File(outputDir, path)
                outFile.parentFile?.mkdirs()

                _progress.value = _progress.value.copy(
                    currentFile = path,
                    fileIndex = index + 1
                )

                downloadFile(fileUrl, outFile) { bytesRead ->
                    downloaded += bytesRead
                    _progress.value = _progress.value.copy(downloadedBytes = downloaded)
                }
            }

            _progress.value = _progress.value.copy(state = DownloadState.FINISHED)
        }

    private fun listHfFiles(apiUrl: String): List<Pair<String, Long>> {
        val request = Request.Builder().url(apiUrl).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) throw Exception("HF API error: ${response.code}")

        val body = response.body?.string() ?: throw Exception("Empty HF API response")
        val json = Json.parseToJsonElement(body).jsonObject
        val siblings = json["siblings"]?.jsonArray ?: return emptyList()

        return siblings.mapNotNull { elem ->
            val obj = elem.jsonObject
            val rfilename = obj["rfilename"]?.jsonPrimitive?.content ?: return@mapNotNull null
            // Skip hidden files and README
            if (rfilename.startsWith(".") || rfilename == "README.md") return@mapNotNull null
            val size = obj["size"]?.jsonPrimitive?.longOrNull ?: 0L
            rfilename to size
        }
    }

    // ── ModelScope ─────────────────────────────────────────────────

    private suspend fun downloadFromModelScope(modelId: String, repoId: String) =
        withContext(Dispatchers.IO) {
            _progress.value = DownloadProgress(state = DownloadState.LISTING_FILES)

            // List files via ModelScope API
            val apiUrl = "https://modelscope.cn/api/v1/models/$repoId/repo/tree?Recursive=true"
            val files = listMsFiles(apiUrl)
            Log.d(TAG, "MS: ${files.size} files to download")

            val outputDir = getModelDir(modelId)
            outputDir.mkdirs()

            val totalSize = files.sumOf { it.second }
            var downloaded = 0L

            _progress.value = DownloadProgress(
                state = DownloadState.DOWNLOADING,
                totalBytes = totalSize,
                totalFiles = files.size
            )

            files.forEachIndexed { index, (path, size) ->
                val fileUrl = "https://modelscope.cn/api/v1/models/$repoId/repo?Revision=master&FilePath=$path"
                val outFile = File(outputDir, path)
                outFile.parentFile?.mkdirs()

                _progress.value = _progress.value.copy(
                    currentFile = path,
                    fileIndex = index + 1
                )

                downloadFile(fileUrl, outFile) { bytesRead ->
                    downloaded += bytesRead
                    _progress.value = _progress.value.copy(downloadedBytes = downloaded)
                }
            }

            _progress.value = _progress.value.copy(state = DownloadState.FINISHED)
        }

    private fun listMsFiles(apiUrl: String): List<Pair<String, Long>> {
        val request = Request.Builder().url(apiUrl).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) throw Exception("ModelScope API error: ${response.code}")

        val body = response.body?.string() ?: throw Exception("Empty MS API response")
        val json = Json.parseToJsonElement(body).jsonObject
        val data = json["Data"]?.jsonObject ?: return emptyList()
        val files = data["Files"]?.jsonArray ?: return emptyList()

        return files.mapNotNull { elem ->
            val obj = elem.jsonObject
            val path = obj["Path"]?.jsonPrimitive?.content ?: return@mapNotNull null
            val type = obj["Type"]?.jsonPrimitive?.content ?: "file"
            if (type != "file") return@mapNotNull null
            if (path.startsWith(".") || path == "README.md") return@mapNotNull null
            val size = obj["Size"]?.jsonPrimitive?.longOrNull ?: 0L
            path to size
        }
    }

    // ── Modelers ───────────────────────────────────────────────────

    private suspend fun downloadFromModelers(modelId: String, repoId: String) =
        withContext(Dispatchers.IO) {
            _progress.value = DownloadProgress(state = DownloadState.LISTING_FILES)

            // List files via Modelers API
            val apiUrl = "https://modelers.cn/coderepo/web/v1/file/$repoId/main/tree/"
            val files = listMlFiles(apiUrl, repoId)
            Log.d(TAG, "ML: ${files.size} files to download")

            val outputDir = getModelDir(modelId)
            outputDir.mkdirs()

            val totalSize = files.sumOf { it.second }
            var downloaded = 0L

            _progress.value = DownloadProgress(
                state = DownloadState.DOWNLOADING,
                totalBytes = totalSize,
                totalFiles = files.size
            )

            files.forEachIndexed { index, (path, size) ->
                val fileUrl = "https://modelers.cn/coderepo/web/v1/file/$repoId/main/media/$path"
                val outFile = File(outputDir, path)
                outFile.parentFile?.mkdirs()

                _progress.value = _progress.value.copy(
                    currentFile = path,
                    fileIndex = index + 1
                )

                downloadFile(fileUrl, outFile) { bytesRead ->
                    downloaded += bytesRead
                    _progress.value = _progress.value.copy(downloadedBytes = downloaded)
                }
            }

            _progress.value = _progress.value.copy(state = DownloadState.FINISHED)
        }

    private fun listMlFiles(apiUrl: String, repoId: String): List<Pair<String, Long>> {
        val request = Request.Builder().url(apiUrl).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) throw Exception("Modelers API error: ${response.code}")

        val body = response.body?.string() ?: throw Exception("Empty ML API response")
        val json = Json.parseToJsonElement(body).jsonObject
        val data = json["data"]?.jsonArray ?: return emptyList()

        return data.mapNotNull { elem ->
            val obj = elem.jsonObject
            val name = obj["name"]?.jsonPrimitive?.content ?: return@mapNotNull null
            val type = obj["type"]?.jsonPrimitive?.content ?: "file"
            if (type != "file") return@mapNotNull null
            if (name.startsWith(".") || name == "README.md") return@mapNotNull null
            val size = obj["size"]?.jsonPrimitive?.longOrNull ?: 0L
            name to size
        }
    }

    // ── File Download ──────────────────────────────────────────────

    private fun downloadFile(url: String, outFile: File, onBytesRead: (Long) -> Unit) {
        val request = Request.Builder().url(url).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) throw Exception("Download failed ($url): ${response.code}")

        val body = response.body ?: throw Exception("Empty response body")
        val input = body.byteStream()
        val buffer = ByteArray(8192)

        FileOutputStream(outFile).use { output ->
            var bytesRead: Int
            while (input.read(buffer).also { bytesRead = it } != -1) {
                output.write(buffer, 0, bytesRead)
                onBytesRead(bytesRead.toLong())
            }
        }
    }

    fun reset() {
        _progress.value = DownloadProgress()
    }
}
