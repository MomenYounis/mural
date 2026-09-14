package chat.mural.network

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioDeviceCallback
import android.media.AudioDeviceInfo
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Base64
import android.util.Log
import chat.mural.R
import java.util.UUID
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import kotlinx.coroutines.DisposableHandle
import kotlinx.coroutines.Job
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.put
import kotlinx.serialization.json.add
import kotlinx.serialization.json.putJsonArray
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import java.util.concurrent.TimeUnit
import kotlin.math.sqrt

class LiveTransport(
    context: Context,
    private val scope: CoroutineScope,
) {
    var onEvent: ((JsonObject) -> Unit)? = null
    var onFailure: ((String) -> Unit)? = null
    var onLevels: ((Double, Double) -> Unit)? = null

    private val applicationContext = context.applicationContext
    private val audioManager = applicationContext.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val generation = AtomicLong(0)
    private val lock = Any()
    private val retiredAttempts = ArrayDeque<Attempt>()
    private val audioScope = CoroutineScope(SupervisorJob() + AUDIO_DISPATCHER)
    private val credentialStore = CredentialStore(applicationContext)

    @Volatile private var activeAttempt: Attempt? = null
    @Volatile private var startedState = false
    @Volatile private var mutedState = false

    val started: Boolean get() = startedState
    val isMuted: Boolean get() = mutedState

    suspend fun connect(
        api: LiveSessionProvider,
        instructions: String,
        history: JsonArray = JsonArray(emptyList()),
        language: String? = null,
    ) = withContext(AUDIO_DISPATCHER) {
        val attemptGeneration = detachAttempt()
        drainRetiredAttempts()
        emitZeroLevels(attemptGeneration)
        if (applicationContext.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            throw microphoneException()
        }

        val attempt = Attempt(
            id = attemptGeneration,
            ownership = LiveSessionOwnership(audioScope),
            previousAudioMode = audioManager.mode,
            previousSpeakerphone = if (Build.VERSION.SDK_INT < 31) legacySpeakerphoneState() else false,
        )
        // Keep lease for cleanup; for hosted it will be used to close server session
        // For Gemini personal, lease may be null
        synchronized(lock) {
            if (generation.get() != attemptGeneration) throw CancellationException("Voice connection superseded")
            activeAttempt = attempt
            startedState = false
            mutedState = false
        }

        try {
            configureAudio(attempt)
            requireCurrent(attempt)

            // Resolve API key: prefer CredentialStore; Hosted path may not have key but we still try store first
            val apiKey = credentialStore.read()
                ?: run {
                    Log.e("MuralLive", "No Gemini API key found in CredentialStore")
                    throw APIClient.APIException.MissingKey
                }
            Log.i("MuralLive", "Gemini Live connect: key prefix=" + apiKey.take(6) + " len=" + apiKey.length + " instructions len=" + instructions.length)

            // Build Gemini Live WebSocket URL

            // Build Gemini Live WebSocket URL
            // Gemini API keys now may be AIza... or AQ... (new format). Always send as ?key= and x-goog-api-key
            val wsUrl = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key=$apiKey"
            Log.i("MuralLive", "Connecting to Gemini Live: " + wsUrl.take(95) + " keyLen=" + apiKey.length + " prefix=" + apiKey.take(8))
            val request = Request.Builder().url(wsUrl).header("x-goog-api-key", apiKey).build()
            val client = OkHttpClient.Builder()
                .readTimeout(0, TimeUnit.MILLISECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .connectTimeout(15, TimeUnit.SECONDS)
                .build()
            attempt.okHttpClient = client

            // Create AudioTrack for playback (24kHz mono PCM16)
            val track = createAudioTrack()
            attempt.audioTrack = track
            try { track.play() } catch (_: Exception) { throw connectionException() }

            // Create AudioRecord for capture (16kHz mono PCM16)
            val record = createAudioRecord()
            attempt.audioRecord = record
            try { record.startRecording() } catch (_: Exception) { throw microphoneException() }

            // Setup WebSocket listener
            val openDeferred = CompletableDeferred<Unit>()
            val listener = geminiListener(attempt, openDeferred)
            val ws = client.newWebSocket(request, listener)
            attempt.webSocket = ws

            // Wait for websocket open (max 10s)
            withTimeout(10_000) { openDeferred.await() }
            requireCurrent(attempt)

            // Send setup message
            val setupJson = buildGeminiSetup(instructions, history, language)
            Log.i("MuralLive", "Sending setup: " + setupJson.toString().take(600))
            if (!ws.send(setupJson.toString())) {
                Log.e("MuralLive", "Failed to send setup")
                throw connectionException()
            }
            Log.i("MuralLive", "Setup sent, awaiting setupComplete...")

            // Wait for setupComplete with timeout
            withTimeout(READY_TIMEOUT_MILLISECONDS) { attempt.started.await() }
            requireCurrent(attempt)

            // Start audio pipelines
            startCapture(attempt)
            startPlayback(attempt)
            startMetering(attempt)

            attempt.scopeCompletion = scope.coroutineContext[Job]?.invokeOnCompletion {
                audioScope.launch { cleanupIfCurrent(attempt) }
            }

            // Adopt any hosted lease if provider is Hosted: try to create via provider for closing semantics
            // For personal Gemini, we don't have SDP flow, so just create dummy lease handling.
            // If api is HostedAPIClient, attempt to keep its lease for later close.
            // We try to invoke createLiveSession only for Hosted path with dummy SDP? Hosted requires valid SDP.
            // For Gemini live personal, lease is null.

        } catch (_: TimeoutCancellationException) {
            cleanupIfCurrent(attempt)
            throw timeoutException()
        } catch (error: CancellationException) {
            cleanupIfCurrent(attempt)
            throw error
        } catch (error: Throwable) {
            cleanupIfCurrent(attempt)
            // Map MissingKey to specific exception for UI
            if (error is APIClient.APIException.MissingKey) throw error
            throw error
        }
    }

    /** True means accepted for delivery; every native operation runs on the audio worker. */
    fun send(event: JsonObject): Boolean {
        val attempt = activeAttempt ?: return false
        if (!isCurrent(attempt) || attempt.webSocket == null) return false
        audioScope.launch {
            if (isCurrent(attempt) && !sendNow(attempt, event) && !attempt.closing.get()) {
                fail(attempt, applicationContext.getString(R.string.error_transport_channel_closed))
            }
        }
        return true
    }

    private fun sendNow(attempt: Attempt, event: JsonObject): Boolean {
        if (!isCurrent(attempt)) return false
        val ws = attempt.webSocket ?: return false
        return try {
            val type = event["type"]?.jsonPrimitive?.contentOrNull ?: return false
            val content = event["content"]?.jsonPrimitive?.contentOrNull ?: ""
            val jsonToSend = when {
                type.startsWith("session.") && type.endsWith(".append") -> {
                    // Map to Gemini client_content
                    buildJsonObject {
                        put("clientContent", buildJsonObject {
                            put("turns", buildJsonArray {
                                add(buildJsonObject {
                                    put("role", "user")
                                    put("parts", buildJsonArray { add(buildJsonObject { put("text", content.take(1000)) }) })
                                })
                            })
                            put("turnComplete", true)
                        })
                    }
                }
                type == "session.input_audio.mute" -> {
                    // Mute is handled locally; optionally notify server
                    return true
                }
                type == "session.input_audio.unmute" -> return true
                type == "session.close" -> {
                    // Send close signal; Gemini live expects no specific close JSON, just close websocket
                    attempt.closing.set(true)
                    ws.close(1000, "client close")
                    return true
                }
                else -> {
                    // Generic fallback: send as realtime text input
                    buildJsonObject {
                        put("realtimeInput", buildJsonObject {
                            put("text", content.take(1000))
                        })
                    }
                }
            }
            ws.send(jsonToSend.toString())
        } catch (_: Exception) { false }
    }

    fun mute(muted: Boolean) {
        val attempt = activeAttempt ?: return
        mutedState = muted
        audioScope.launch {
            if (!isCurrent(attempt)) return@launch
            // Pause capture via flag; no need to send to server except optional notification
            // For Gemini, muting just stops sending audio chunks
        }
    }

    fun close() {
        val attempt = activeAttempt ?: return
        attempt.closing.set(true)
        attempt.ownership.close()
        mutedState = true
        audioScope.launch {
            if (!isCurrent(attempt)) return@launch
            try {
                val ws = attempt.webSocket
                // Notify server of interruption if needed
                ws?.send(buildJsonObject { put("clientContent", buildJsonObject { put("turnComplete", true) }) }.toString())
                ws?.close(1000, "close")
            } catch (_: Exception) { }
        }
    }

    fun disconnect() {
        val detached = detachAttempt()
        audioScope.launch { drainRetiredAttempts() }
        emitZeroLevels(detached)
    }

    private fun detachAttempt(): Long = synchronized(lock) {
        activeAttempt?.let(retiredAttempts::addLast)
        activeAttempt = null
        startedState = false
        mutedState = false
        generation.incrementAndGet()
    }

    private fun drainRetiredAttempts() {
        while (true) {
            val retired = synchronized(lock) { retiredAttempts.removeFirstOrNull() } ?: break
            cleanup(retired)
        }
    }

    private fun buildGeminiSetup(instructions: String, history: JsonArray, language: String?): JsonObject {
        // Build history text prefix
        val historyText = StringBuilder()
        for (elem in history) {
            val obj = elem as? JsonObject ?: continue
            val role = obj["role"]?.jsonPrimitive?.contentOrNull ?: "user"
            val contentArr = obj["content"] as? JsonArray ?: continue
            for (c in contentArr) {
                val p = c as? JsonObject ?: continue
                val t = p["text"]?.jsonPrimitive?.contentOrNull ?: continue
                historyText.append("$role: $t\n")
            }
        }
        val fullInstructions = if (historyText.isNotEmpty()) {
            "Conversation history:\n$historyText\n\nCurrent instructions: $instructions"
        } else instructions

        // Use Gemini 2.5 Flash Native Audio Preview as default live model
        return buildJsonObject {
            put("setup", buildJsonObject {
                put("model", "models/gemini-2.5-flash-native-audio-preview-09-2025")
                put("generationConfig", buildJsonObject {
                    put("responseModalities", buildJsonArray { add(JsonPrimitive("AUDIO")) })
                    put("speechConfig", buildJsonObject {
                        put("voiceConfig", buildJsonObject {
                            put("prebuiltVoiceConfig", buildJsonObject { put("voiceName", "Aoede") })
                        })
                    })
                })
                put("systemInstruction", buildJsonObject {
                    put("parts", buildJsonArray { add(buildJsonObject { put("text", fullInstructions.take(12000)) }) })
                })
                // Request transcriptions for both directions
                put("inputAudioTranscription", buildJsonObject {})
                put("outputAudioTranscription", buildJsonObject {})
                // Enable VAD
                put("realtimeInputConfig", buildJsonObject {
                    put("automaticActivityDetection", buildJsonObject {
                        put("disabled", false)
                        put("startOfSpeechSensitivity", "START_SENSITIVITY_HIGH")
                        put("endOfSpeechSensitivity", "END_SENSITIVITY_HIGH")
                        put("prefixPaddingMs", 20)
                        put("silenceDurationMs", 100)
                    })
                })
            })
        }
    }

    private fun createAudioRecord(): AudioRecord {
        val sampleRate = 16000
        val channel = AudioFormat.CHANNEL_IN_MONO
        val encoding = AudioFormat.ENCODING_PCM_16BIT
        val minBuf = AudioRecord.getMinBufferSize(sampleRate, channel, encoding).coerceAtLeast(2048)
        // Use VOICE_COMMUNICATION for echo cancellation
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.VOICE_COMMUNICATION)
                .setAudioFormat(AudioFormat.Builder().setSampleRate(sampleRate).setChannelMask(channel).setEncoding(encoding).build())
                .setBufferSizeInBytes(minBuf * 2)
                .build()
        } else {
            @Suppress("DEPRECATION")
            AudioRecord(MediaRecorder.AudioSource.VOICE_COMMUNICATION, sampleRate, channel, encoding, minBuf * 2)
        }
    }

    private fun createAudioTrack(): AudioTrack {
        val sampleRate = 24000
        val channel = AudioFormat.CHANNEL_OUT_MONO
        val encoding = AudioFormat.ENCODING_PCM_16BIT
        val minBuf = AudioTrack.getMinBufferSize(sampleRate, channel, encoding).coerceAtLeast(4096)
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            AudioTrack.Builder()
                .setAudioAttributes(attrs)
                .setAudioFormat(AudioFormat.Builder().setSampleRate(sampleRate).setChannelMask(channel).setEncoding(encoding).build())
                .setBufferSizeInBytes(minBuf * 4)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
        } else {
            @Suppress("DEPRECATION")
            AudioTrack(attrs, AudioFormat.Builder().setSampleRate(sampleRate).setChannelMask(channel).setEncoding(encoding).build(), minBuf * 4, AudioTrack.MODE_STREAM, AudioManager.AUDIO_SESSION_ID_GENERATE)
        }
    }

    private fun startCapture(attempt: Attempt) {
        attempt.captureJob?.cancel()
        attempt.captureJob = audioScope.launch {
            val record = attempt.audioRecord ?: return@launch
            val bufferSize = 640 // 20ms @ 16kHz 16-bit mono
            val buffer = ByteArray(bufferSize)
            // Chunk send interval control
            while (isActive && isCurrent(attempt) && !attempt.closing.get()) {
                if (mutedState) {
                    delay(40)
                    // consume to avoid overflow but discard
                    try { record.read(buffer, 0, buffer.size) } catch (_: Exception) {}
                    continue
                }
                val read = try { record.read(buffer, 0, buffer.size) } catch (_: Exception) { -1 }
                if (read <= 0) {
                    delay(20)
                    continue
                }
                // Compute input level RMS
                var sum = 0.0
                for (i in 0 until read step 2) {
                    if (i + 1 >= read) break
                    val sample = ((buffer[i + 1].toInt() shl 8) or (buffer[i].toInt() and 0xFF)).toShort().toInt()
                    sum += (sample.toDouble() / 32768.0) * (sample.toDouble() / 32768.0)
                }
                val rms = if (read > 0) sqrt(sum / (read / 2)) else 0.0
                attempt.lastInputLevel = attempt.lastInputLevel * 0.35 + rms.coerceIn(0.0, 1.0) * 0.65

                // Send chunk
                val chunk = buffer.copyOf(read)
                val b64 = Base64.encodeToString(chunk, Base64.NO_WRAP)
                val msg = buildJsonObject {
                    put("realtimeInput", buildJsonObject {
                        put("audio", buildJsonObject {
                            put("mimeType", "audio/pcm;rate=16000")
                            put("data", b64)
                        })
                    })
                }
                try {
                    attempt.webSocket?.send(msg.toString())
                } catch (_: Exception) {
                    // will trigger fail via websocket onFailure
                }
                // ~20ms cadence, but read already took time
                // Optionally delay a bit to avoid busy loop if read is fast
                // No extra delay; next read will block appropriately
            }
        }
    }

    private fun startPlayback(attempt: Attempt) {
        attempt.playbackJob?.cancel()
        attempt.playbackJob = audioScope.launch {
            while (isActive && isCurrent(attempt)) {
                val chunk = attempt.playbackQueue.poll()
                if (chunk == null) {
                    delay(10)
                    continue
                }
                try {
                    val track = attempt.audioTrack ?: continue
                    // Compute output level
                    var sum = 0.0
                    for (i in chunk.indices step 2) {
                        if (i + 1 >= chunk.size) break
                        val sample = ((chunk[i + 1].toInt() shl 8) or (chunk[i].toInt() and 0xFF)).toShort().toInt()
                        sum += (sample.toDouble() / 32768.0) * (sample.toDouble() / 32768.0)
                    }
                    val rms = if (chunk.size > 0) sqrt(sum / (chunk.size / 2)) else 0.0
                    attempt.lastOutputLevel = attempt.lastOutputLevel * 0.35 + rms.coerceIn(0.0, 1.0) * 0.65

                    var offset = 0
                    while (offset < chunk.size && isCurrent(attempt)) {
                        val written = track.write(chunk, offset, chunk.size - offset)
                        if (written <= 0) {
                            delay(10)
                            break
                        }
                        offset += written
                    }
                } catch (_: Exception) {
                    delay(20)
                }
            }
        }
    }

    private fun geminiListener(attempt: Attempt, openDeferred: CompletableDeferred<Unit>) = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            Log.i("MuralLive", "WebSocket onOpen: code=" + response.code + " hs=" + response.headers)
            if (!isCurrent(attempt)) return
            attempt.channelOpen.set(true)
            if (!openDeferred.isCompleted) openDeferred.complete(Unit)
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            Log.i("MuralLive", "WS onMessage: " + text.take(1200))
            if (!isCurrent(attempt)) return
            if (text.length > MAX_EVENT_BYTES) {
                fail(attempt, applicationContext.getString(R.string.error_transport_invalid_event))
                return
            }
            val obj = try { JSON.parseToJsonElement(text).jsonObject } catch (e: Exception) {
                Log.e("MuralLive", "Failed to parse WS message", e)
                return
            }
            handleGeminiMessage(attempt, obj, text)
        }

        override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
            // Binary messages not expected for Gemini; ignore
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            Log.e("MuralLive", "WebSocket onFailure: t=" + t.message + " response=" + response + " code=" + response?.code, t)
            if (!isCurrent(attempt)) {
                if (!openDeferred.isCompleted) openDeferred.completeExceptionally(t)
                return
            }
            if (!openDeferred.isCompleted) openDeferred.completeExceptionally(t)
            val msg = when (response?.code) {
                401 -> applicationContext.getString(R.string.error_http_401)
                403 -> applicationContext.getString(R.string.error_http_403_404)
                429 -> applicationContext.getString(R.string.error_http_429)
                else -> {
                    // Try to read body for Gemini error details
                    val body = try { response?.body?.string() } catch (_: Exception) { null }
                    if (body != null && body.contains("API_KEY_INVALID", ignoreCase = true)) {
                        applicationContext.getString(R.string.error_http_401)
                    } else if (body != null && body.contains("MODEL_NOT_FOUND", ignoreCase = true)) {
                        applicationContext.getString(R.string.error_transport_connection) + " (model)"
                    } else applicationContext.getString(R.string.error_transport_network_lost)
                }
            }
            fail(attempt, msg)
        }

        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            Log.i("MuralLive", "WebSocket onClosing: code=" + code + " reason=" + reason)
            webSocket.close(code, reason)
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            Log.i("MuralLive", "WebSocket onClosed: code=" + code + " reason=" + reason + " closing=" + attempt.closing.get())
            if (!isCurrent(attempt)) return
            if (!attempt.closing.get()) {
                val msg = when (code) {
                    1008 -> applicationContext.getString(R.string.error_http_401) + " (1008)"
                    1002, 1003 -> applicationContext.getString(R.string.error_transport_invalid_event) + " (" + code + ")"
                    else -> applicationContext.getString(R.string.error_transport_channel_closed) + " (" + code + ":" + reason.take(100) + ")"
                }
                fail(attempt, msg)
            }
        }
    }

    private fun handleGeminiMessage(attempt: Attempt, obj: JsonObject, raw: String) {
        // setupComplete
        if ("setupComplete" in obj) {
            if (!attempt.started.isCompleted) {
                attempt.channelOpen.set(true)
                startedState = true
                attempt.started.complete(Unit)
                emitEvent(attempt, buildJsonObject {
                    put("type", "session.started")
                    put("session", buildJsonObject { put("id", UUID.randomUUID().toString()) })
                })
                // Also emit mural.session.created for compatibility
                emitEvent(attempt, buildJsonObject {
                    put("type", "mural.session.created")
                    put("session", buildJsonObject { put("id", UUID.randomUUID().toString()) })
                })
            }
            return
        }

        // serverContent
        val serverContent = obj["serverContent"] as? JsonObject
        if (serverContent != null) {
            // Check for interruption
            val interrupted = (serverContent["interrupted"] as? JsonPrimitive)?.contentOrNull == "true" ||
                    serverContent["interrupted"] == JsonPrimitive(true)
            if (interrupted) {
                attempt.playbackQueue.clear()
                try { attempt.audioTrack?.pause(); attempt.audioTrack?.flush(); attempt.audioTrack?.play() } catch (_: Exception) {}
            }

            // Model turn with audio + text
            val modelTurn = serverContent["modelTurn"] as? JsonObject
            if (modelTurn != null) {
                val parts = modelTurn["parts"] as? JsonArray ?: JsonArray(emptyList())
                for (p in parts) {
                    val part = p as? JsonObject ?: continue
                    // Inline audio
                    val inline = part["inlineData"] as? JsonObject
                    if (inline != null) {
                        val mime = inline["mimeType"]?.jsonPrimitive?.contentOrNull ?: inline["mimeType"]?.jsonPrimitive?.contentOrNull ?: ""
                        val data = inline["data"]?.jsonPrimitive?.contentOrNull
                        if (data != null && mime.contains("audio/pcm")) {
                            try {
                                val pcm = Base64.decode(data, Base64.DEFAULT)
                                // Expect 24kHz PCM16 mono; queue for playback
                                if (pcm.isNotEmpty() && pcm.size <= 100_000) {
                                    attempt.playbackQueue.add(pcm)
                                }
                            } catch (_: Exception) {}
                        }
                    }
                    // Text part
                    val text = part["text"]?.jsonPrimitive?.contentOrNull
                    if (!text.isNullOrEmpty()) {
                        emitTranscript(attempt, text, isUser = false)
                    }
                }
            }

            // Input transcription
            val inputTrans = serverContent["inputTranscription"] as? JsonObject
            if (inputTrans != null) {
                val text = inputTrans["text"]?.jsonPrimitive?.contentOrNull
                if (!text.isNullOrEmpty()) {
                    emitTranscript(attempt, text, isUser = true)
                }
            }
            // Output transcription
            val outputTrans = serverContent["outputTranscription"] as? JsonObject
            if (outputTrans != null) {
                val text = outputTrans["text"]?.jsonPrimitive?.contentOrNull
                if (!text.isNullOrEmpty()) {
                    emitTranscript(attempt, text, isUser = false)
                }
            }

            // Turn complete / usage
            val turnComplete = (serverContent["turnComplete"] as? JsonPrimitive)?.let { it.contentOrNull == "true" || it.booleanOrNull == true } ?: false
            val usage = serverContent["usageMetadata"] as? JsonObject ?: obj["usageMetadata"] as? JsonObject
            if (usage != null || turnComplete) {
                val seconds = attempt.playbackQueue.size * 0.02 // approximate
                emitEvent(attempt, buildJsonObject {
                    put("type", if (turnComplete) "session.closed" else "session.usage.updated")
                    put("usage", buildJsonObject { put("seconds", seconds) })
                })
                if (turnComplete) {
                    // Do not auto-finish here; MuralViewModel handles session.closed via finish(true)
                }
            }

            // Ensure ongoing activity timestamps updated
            lastActivityUpdate()
        }

        // Tool call (delegation) - map to session.delegation.created
        val toolCall = obj["toolCall"] as? JsonObject
        if (toolCall != null) {
            val functionCalls = toolCall["functionCalls"] as? JsonArray
            if (functionCalls != null) {
                for (fc in functionCalls) {
                    val f = fc as? JsonObject ?: continue
                    val name = f["name"]?.jsonPrimitive?.contentOrNull ?: continue
                    val id = f["id"]?.jsonPrimitive?.contentOrNull ?: UUID.randomUUID().toString()
                    if (name == "lookup" || name == "search" || name.contains("client")) {
                        emitEvent(attempt, buildJsonObject {
                            put("type", "session.delegation.created")
                            put("delegation", buildJsonObject { put("target", "client"); put("id", id) })
                        })
                    }
                }
            }
        }

        // Fallback: if message is directly an error
        if ("error" in obj) {
            val err = obj["error"] as? kotlinx.serialization.json.JsonObject
            val msg = err?.get("message")?.let { (it as? kotlinx.serialization.json.JsonPrimitive)?.contentOrNull } ?: ""
            val errorText = when {
                msg.contains("API_KEY_INVALID", ignoreCase = true) -> applicationContext.getString(R.string.error_http_401)
                msg.contains("MODEL_NOT_FOUND", ignoreCase = true) -> applicationContext.getString(R.string.error_transport_connection) + " (model not found)"
                else -> msg.takeIf { it.isNotBlank() } ?: applicationContext.getString(R.string.error_transport_channel_closed)
            }
            // Surface as failure if setup not completed
            if (!attempt.started.isCompleted) {
                fail(attempt, errorText)
            } else {
                emitEvent(attempt, buildJsonObject { put("type", "error") })
            }
        }
    }

    private fun emitTranscript(attempt: Attempt, text: String, isUser: Boolean) {
        if (text.length > 50000) return
        val now = (System.currentTimeMillis() - attempt.startMs).toInt().coerceAtLeast(0)
        val eventType = if (isUser) "session.input_transcript.delta" else "session.output_transcript.delta"
        emitEvent(attempt, buildJsonObject {
            put("type", eventType)
            put("delta", text.take(1000))
            put("start_ms", now)
            put("end_ms", now + text.length * 20) // estimate
            put("event_id", UUID.randomUUID().toString())
        })
    }

    private fun lastActivityUpdate() {
        // Update last activity via levels? Handled via emitLevels
    }

    private fun startMetering(attempt: Attempt) {
        attempt.meterJob?.cancel()
        attempt.meterJob = audioScope.launch {
            var lastInput = 0.0
            var lastOutput = 0.0
            while (isActive && isCurrent(attempt)) {
                lastInput = attempt.lastInputLevel
                lastOutput = attempt.lastOutputLevel
                // Apply smoothing already in capture/playback; just emit
                emitLevels(lastInput, lastOutput, attempt)
                delay(METER_INTERVAL_MILLISECONDS)
            }
        }
    }

    private fun configureAudio(attempt: Attempt) {
        if (attempt.previousAudioMode != AudioManager.MODE_NORMAL || audioManager.mode != AudioManager.MODE_NORMAL ||
            (Build.VERSION.SDK_INT < 31 && LegacyCommunicationAudioRoute.hasExistingSco(applicationContext, audioManager))) {
            throw audioFocusException()
        }
        val attributes = voiceAudioAttributes()
        lateinit var focusRequest: AudioFocusRequest
        val listener = AudioManager.OnAudioFocusChangeListener { change ->
            if (change == AudioManager.AUDIOFOCUS_LOSS ||
                change == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT ||
                change == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK
            ) {
                attempt.focusLost.set(true)
                fail(attempt, applicationContext.getString(R.string.error_transport_audio_interrupted))
            }
        }
        focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
            .setAudioAttributes(attributes)
            .setAcceptsDelayedFocusGain(false)
            .setOnAudioFocusChangeListener(listener, Handler(Looper.getMainLooper()))
            .build()
        attempt.focusRequest = focusRequest
        if (audioManager.requestAudioFocus(focusRequest) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
            throw audioFocusException()
        }
        attempt.ownsAudioFocus = true
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        attempt.ownsAudioMode = true
        if (Build.VERSION.SDK_INT < 31) {
            val legacyRoute = LegacyCommunicationAudioRoute(applicationContext, audioManager, audioScope,
                attempt.previousSpeakerphone, onFailure = { audioFailure(attempt) })
            attempt.legacyAudioRoute = legacyRoute
            legacyRoute.start()
        }
        routeCommunicationAudio(attempt)
        val callback = object : AudioDeviceCallback() {
            override fun onAudioDevicesAdded(addedDevices: Array<out AudioDeviceInfo>) = reroute()
            override fun onAudioDevicesRemoved(removedDevices: Array<out AudioDeviceInfo>) = reroute()
            private fun reroute() {
                audioScope.launch {
                    if (isCurrent(attempt)) {
                        try { routeCommunicationAudio(attempt) }
                        catch (_: Exception) { audioFailure(attempt) }
                    }
                }
            }
        }
        attempt.deviceCallback = callback
        audioManager.registerAudioDeviceCallback(callback, Handler(Looper.getMainLooper()))
    }

    private fun fail(attempt: Attempt, message: String) {
        if (!isCurrent(attempt) || attempt.closing.get() || !attempt.failureReported.compareAndSet(false, true)) return
        audioScope.launch {
            val failureGeneration = cleanupIfCurrent(attempt) ?: return@launch
            scope.launch {
                if (generation.get() == failureGeneration && activeAttempt == null) onFailure?.invoke(message)
            }
        }
    }

    private fun audioFailure(attempt: Attempt) {
        fail(attempt, applicationContext.getString(R.string.error_transport_audio_stopped))
    }

    private fun voiceAudioAttributes() = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()

    private fun routeCommunicationAudio(attempt: Attempt) {
        if (Build.VERSION.SDK_INT >= 31) {
            val current = audioManager.communicationDevice
            val selected = selectCommunicationDevice(current, audioManager.availableCommunicationDevices,
                sameDevice = { left, right -> left.id == right.id }) { it.type }
            if (selected != null && current?.id != selected.id && audioManager.setCommunicationDevice(selected)) {
                attempt.ownsCommunicationRoute = true
            }
        } else attempt.legacyAudioRoute?.devicesChanged()
    }

    private fun releaseAudioRoute(attempt: Attempt) {
        try {
            if (Build.VERSION.SDK_INT >= 31) {
                if (attempt.ownsCommunicationRoute) audioManager.clearCommunicationDevice()
            } else attempt.legacyAudioRoute?.close(restoreSpeakerphone = !attempt.focusLost.get())
        } catch (_: Exception) { }
        attempt.ownsCommunicationRoute = false
        attempt.legacyAudioRoute = null
    }

    @Suppress("DEPRECATION")
    private fun legacySpeakerphoneState(): Boolean = audioManager.isSpeakerphoneOn

    private fun cleanupIfCurrent(attempt: Attempt): Long? {
        val cleanupGeneration = synchronized(lock) {
            if (activeAttempt !== attempt) null
            else {
                val nextGeneration = generation.incrementAndGet()
                retiredAttempts.addLast(attempt)
                activeAttempt = null
                startedState = false
                mutedState = false
                nextGeneration
            }
        }
        if (cleanupGeneration != null) {
            drainRetiredAttempts()
            emitZeroLevels(cleanupGeneration)
        }
        return cleanupGeneration
    }

    private fun cleanup(attempt: Attempt?) {
        if (attempt == null || !attempt.cleaned.compareAndSet(false, true)) return
        attempt.ownership.close()
        attempt.closing.set(true)
        attempt.channelOpen.set(false)
        attempt.networkRecovery?.connected()
        attempt.networkRecovery = null
        try { attempt.deviceCallback?.let { audioManager.unregisterAudioDeviceCallback(it) } } catch (_: Exception) { }
        attempt.deviceCallback = null
        attempt.scopeCompletion?.dispose()
        attempt.scopeCompletion = null
        attempt.meterJob?.cancel()
        attempt.meterJob = null
        attempt.captureJob?.cancel()
        attempt.captureJob = null
        attempt.playbackJob?.cancel()
        attempt.playbackJob = null
        try { attempt.audioRecord?.stop() } catch (_: Exception) {}
        try { attempt.audioRecord?.release() } catch (_: Exception) {}
        attempt.audioRecord = null
        try { attempt.audioTrack?.pause(); attempt.audioTrack?.flush(); attempt.audioTrack?.stop() } catch (_: Exception) {}
        try { attempt.audioTrack?.release() } catch (_: Exception) {}
        attempt.audioTrack = null
        attempt.playbackQueue.clear()
        try { attempt.webSocket?.cancel() } catch (_: Exception) {}
        try { attempt.webSocket?.close(1000, "cleanup") } catch (_: Exception) {}
        attempt.webSocket = null
        try { attempt.okHttpClient?.dispatcher?.executorService?.shutdown() } catch (_: Exception) {}
        attempt.okHttpClient = null
        attempt.started.cancel()
        releaseAudioRoute(attempt)
        if (attempt.ownsAudioMode) {
            try { audioManager.mode = AudioManager.MODE_NORMAL } catch (_: Exception) { }
            attempt.ownsAudioMode = false
        }
        if (attempt.ownsAudioFocus) {
            try { attempt.focusRequest?.let { audioManager.abandonAudioFocusRequest(it) } } catch (_: Exception) { }
            attempt.ownsAudioFocus = false
        }
    }

    private fun emitEvent(attempt: Attempt, event: JsonObject) {
        val type = event.string("type") ?: return
        if (!event.isSafeForCoordinator(type)) return
        scope.launch { if (isCurrent(attempt)) onEvent?.invoke(event) }
    }

    private fun emitLevels(input: Double, output: Double, attempt: Attempt? = null) {
        scope.launch {
            if (attempt == null || isCurrent(attempt)) onLevels?.invoke(input, output)
        }
    }

    private fun emitZeroLevels(expectedGeneration: Long) {
        scope.launch {
            if (generation.get() == expectedGeneration) onLevels?.invoke(0.0, 0.0)
        }
    }

    private fun isCurrent(attempt: Attempt): Boolean =
        activeAttempt === attempt && generation.get() == attempt.id && !attempt.cleaned.get()

    private fun requireCurrent(attempt: Attempt) {
        if (!isCurrent(attempt)) throw CancellationException("Voice connection superseded")
    }

    private class Attempt(
        val id: Long,
        val ownership: LiveSessionOwnership,
        val previousAudioMode: Int,
        val previousSpeakerphone: Boolean,
    ) {
        var networkRecovery: VoiceConnectionRecovery? = null
        var deviceCallback: AudioDeviceCallback? = null
        var focusRequest: AudioFocusRequest? = null
        var ownsAudioFocus = false
        var ownsAudioMode = false
        var ownsCommunicationRoute = false
        var legacyAudioRoute: LegacyCommunicationAudioRoute? = null
        val focusLost = AtomicBoolean(false)
        var webSocket: WebSocket? = null
        var okHttpClient: OkHttpClient? = null
        var audioRecord: AudioRecord? = null
        var audioTrack: AudioTrack? = null
        var captureJob: Job? = null
        var playbackJob: Job? = null
        var meterJob: Job? = null
        var scopeCompletion: DisposableHandle? = null
        val started = CompletableDeferred<Unit>()
        val channelOpen = AtomicBoolean(false)
        val closing = AtomicBoolean(false)
        val failureReported = AtomicBoolean(false)
        val cleaned = AtomicBoolean(false)
        val playbackQueue = ConcurrentLinkedQueue<ByteArray>()
        var lastInputLevel: Double = 0.0
        var lastOutputLevel: Double = 0.0
        val startMs: Long = System.currentTimeMillis()
    }

    sealed class TransportException(message: String) : Exception(message) {
        class Microphone(message: String) : TransportException(message)
        class Connection(message: String) : TransportException(message)
        class Timeout(message: String) : TransportException(message)
        class AudioFocus(message: String) : TransportException(message)
    }

    private fun microphoneException() = TransportException.Microphone(applicationContext.getString(R.string.error_transport_microphone))
    private fun connectionException() = TransportException.Connection(applicationContext.getString(R.string.error_transport_connection))
    private fun timeoutException() = TransportException.Timeout(applicationContext.getString(R.string.error_transport_timeout))
    private fun audioFocusException() = TransportException.AudioFocus(applicationContext.getString(R.string.error_transport_audio_focus))

    companion object {
        private val AUDIO_DISPATCHER = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "mural-audio-control").apply { isDaemon = true }
        }.asCoroutineDispatcher()
        private const val READY_TIMEOUT_MILLISECONDS = 20_000L
        private const val METER_INTERVAL_MILLISECONDS = 100L
        private const val MAX_EVENT_BYTES = 524_288
        private val JSON = Json { ignoreUnknownKeys = true }
    }
}

private fun JsonObject.string(key: String): String? =
    (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.contentOrNull

private fun JsonObject.isSafeForCoordinator(type: String): Boolean = when (type) {
    "mural.session.created", "session.started" ->
        (this["session"] as? JsonObject)?.get("id").isAbsentOrString()
    "session.input_transcript.delta", "session.output_transcript.delta" ->
        this["delta"].isAbsentOrPrimitive() &&
            this["start_ms"].isAbsentOrPrimitive() &&
            this["end_ms"].isAbsentOrPrimitive() &&
            this["event_id"].isAbsentOrPrimitive()
    "session.delegation.created" -> (this["delegation"] as? JsonObject)?.let {
        it["target"].isAbsentOrPrimitive() && it["id"].isAbsentOrPrimitive()
    } ?: true
    "session.usage.updated", "session.closed" -> (this["usage"] as? JsonObject)?.let {
        it["seconds"].isAbsentOrPrimitive()
    } ?: true
    else -> true
}

private fun kotlinx.serialization.json.JsonElement?.isAbsentOrPrimitive(): Boolean =
    this == null || this is JsonPrimitive

private fun kotlinx.serialization.json.JsonElement?.isAbsentOrString(): Boolean =
    this == null || (this as? JsonPrimitive)?.isString == true

internal class VoiceConnectionRecovery(
    private val scope: CoroutineScope,
    private val timeoutMillis: Long = 8_000,
    private val onLost: () -> Unit,
) {
    private var timer: Job? = null
    fun disconnected() {
        if (timer != null) return
        timer = scope.launch {
            delay(timeoutMillis)
            onLost()
        }
    }
    fun connected() {
        timer?.cancel()
        timer = null
    }
}

private fun JsonPrimitive.booleanOrNull(): Boolean? = try {
    when (content) {
        "true" -> true
        "false" -> false
        else -> null
    }
} catch (_: Exception) { null }
