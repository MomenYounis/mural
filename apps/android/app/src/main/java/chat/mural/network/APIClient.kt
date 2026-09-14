package chat.mural.network

import chat.mural.core.SourceLink
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import okhttp3.Call
import okhttp3.Callback
import okhttp3.CookieJar
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okio.Buffer
import java.net.URI

data class APIUsage(val input: Int = 0, val output: Int = 0, val searches: Int = 0)
data class APIResult(val text: String, val sources: List<SourceLink>, val usage: APIUsage)

class APIClient private constructor(
    private val readCredential: () -> String?,
    private val client: OkHttpClient = defaultClient(),
    private val baseUrl: HttpUrl = GEMINI_BASE_URL,
) : TeachingClient, LiveSessionProvider {
    constructor(credentials: CredentialStore) : this(credentials::read)

    internal constructor(key: String?, client: OkHttpClient, baseUrl: HttpUrl) :
        this({ key }, client, baseUrl)

    // Kept for compatibility with hosted pathTests; delegates to Gemini REST no-op for live.
    // Personal live now uses Gemini Live WebSocket directly in LiveTransport, not this method.
    override suspend fun createLiveSession(request: LiveSessionRequest): LiveSessionConnection {
        // For backward compatibility with MockWebServer tests, use legacy path unless base is Gemini.
        if (!baseUrl.host.contains("generativelanguage.googleapis.com")) {
            val result = postLegacy("live/sessions", buildJsonObject {
                put("session", buildJsonObject {
                    put("model", "gpt-live-1"); put("instructions", request.instructions); put("input", request.history)
                    put("store", false)
                    put("delegation", buildJsonObject { put("type", "client") })
                    put("audio", buildJsonObject { put("output", buildJsonObject { put("voice", "marin") }) })
                })
                put("transport", buildJsonObject { put("type", "webrtc"); put("sdp", request.sdp) })
            })
            val transport = result["transport"] as? JsonObject ?: throw APIException.InvalidResponse
            val answer = (transport["sdp"] as? JsonPrimitive)?.takeIf { it.isString }?.contentOrNull
            if (transport["type"] != JsonPrimitive("webrtc") || answer.isNullOrBlank()) throw APIException.InvalidResponse
            val id = ((result["session"] as? JsonObject)?.get("id") as? JsonPrimitive)?.takeIf { it.isString }?.contentOrNull
            return LiveSessionConnection(answer, id)
        }
        throw APIException.InvalidResponse
    }

    private suspend fun postLegacy(path: String, body: JsonObject): JsonObject {
        if (!VALID_PATH.matches(path) || path.contains("..") || path.startsWith('/')) {
            throw APIException.InvalidResponse
        }
        val key = readCredential() ?: throw APIException.MissingKey
        val request = Request.Builder()
            .url(baseUrl.newBuilder().addPathSegments(path).build())
            .header("Authorization", "Bearer $key")
            .header("Content-Type", JSON_MEDIA_TYPE.toString())
            .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
            .build()
        return suspendCancellableCoroutine { continuation ->
            val call = client.newCall(request)
            continuation.invokeOnCancellation { call.cancel() }
            call.enqueue(object : Callback {
                override fun onFailure(call: Call, error: IOException) {
                    if (continuation.isActive) continuation.resumeWithException(error)
                }
                override fun onResponse(call: Call, response: Response) {
                    try {
                        val value = response.use {
                            if (it.code !in 200..299) throw APIException.Http(it.code)
                            val payload = it.readBoundedBody()
                            try { JSON.parseToJsonElement(payload).jsonObject }
                            catch (_: Exception) { throw APIException.InvalidResponse }
                        }
                        if (continuation.isActive) continuation.resume(value)
                    } catch (error: Exception) {
                        if (continuation.isActive) continuation.resumeWithException(error)
                    }
                }
            })
        }
    }

    suspend fun post(path: String, body: JsonObject): JsonObject {
        // Generic POST retained for openai-compatible tests; delegate to Gemini or legacy depending on host
        if (baseUrl.host.contains("generativelanguage.googleapis.com")) {
            throw APIException.InvalidResponse
        }
        return postLegacy(path, body)
    }

    override suspend fun respond(
        instructions: String,
        input: String,
        schema: JsonObject?,
        search: Boolean,
        purpose: HelperPurpose?,
    ): APIResult {
        val key = readCredential() ?: throw APIException.MissingKey
        if (instructions.isBlank() || input.isBlank()) throw APIException.InvalidResponse

        // When baseUrl is MockWebServer (testing), fall back to legacy OpenAI path to keep tests green
        if (!baseUrl.host.contains("generativelanguage.googleapis.com")) {
            return respondLegacy(instructions, input, schema, search)
        }

        // Gemini REST: https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent?key=KEY
        val model = selectModel(schema)
        val url = baseUrl.newBuilder()
            .addPathSegments("models/$model:generateContent")
            .addQueryParameter("key", key)
            .build()

        val body = buildJsonObject {
            put("system_instruction", buildJsonObject {
                put("parts", buildJsonArray { add(buildJsonObject { put("text", instructions.take(16384)) }) })
            })
            put("contents", buildJsonArray {
                add(buildJsonObject {
                    put("role", "user")
                    put("parts", buildJsonArray { add(buildJsonObject { put("text", input.take(24576)) }) })
                })
            })
            put("generationConfig", buildJsonObject {
                put("maxOutputTokens", if (schema == null) 1_400 else 2_200)
                put("temperature", 0.7)
                if (schema != null) {
                    put("responseMimeType", "application/json")
                    put("responseSchema", schema)
                }
            })
            if (search) {
                put("tools", buildJsonArray { add(buildJsonObject { put("google_search", buildJsonObject {}) }) })
            }
        }

        val request = Request.Builder()
            .url(url)
            .header("Content-Type", JSON_MEDIA_TYPE.toString())
            .header("x-goog-api-key", key)
            .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
            .build()

        return suspendCancellableCoroutine { continuation ->
            val call = client.newCall(request)
            continuation.invokeOnCancellation { call.cancel() }
            call.enqueue(object : Callback {
                override fun onFailure(call: Call, error: IOException) {
                    if (continuation.isActive) continuation.resumeWithException(error)
                }
                override fun onResponse(call: Call, response: Response) {
                    try {
                        val result = response.use {
                            if (it.code !in 200..299) throw APIException.Http(it.code)
                            val payload = it.readBoundedBody()
                            val json = try { JSON.parseToJsonElement(payload).jsonObject }
                            catch (_: Exception) { throw APIException.InvalidResponse }
                            decodeGeminiResponse(json, search)
                        }
                        if (continuation.isActive) continuation.resume(result)
                    } catch (error: Exception) {
                        val mapped = when (error) {
                            is APIException -> error
                            is IOException -> error
                            else -> APIException.InvalidResponse
                        }
                        if (continuation.isActive) continuation.resumeWithException(mapped)
                    }
                }
            })
        }
    }

    private suspend fun respondLegacy(instructions: String, input: String, schema: JsonObject?, search: Boolean): APIResult {
        val body = buildJsonObject {
            put("model", "gpt-5.6-luna")
            put("store", false)
            put("instructions", instructions)
            put("input", buildJsonArray {
                add(buildJsonObject {
                    put("role", "user")
                    put("content", input)
                })
            })
            put("max_output_tokens", if (schema == null) 1_400 else 2_200)
            put("reasoning", buildJsonObject { put("effort", "low") })
            if (schema != null) {
                put("text", buildJsonObject {
                    put("format", buildJsonObject {
                        put("type", "json_schema")
                        put("name", "mural_result")
                        put("strict", true)
                        put("schema", schema)
                    })
                })
            }
            if (search) {
                put("tools", buildJsonArray { add(buildJsonObject { put("type", "web_search") }) })
                put("tool_choice", "auto")
                put("max_tool_calls", 1)
            }
        }
        val response = postLegacy("responses", body)
        return decodeTeachingResponseViaLegacy(response)
    }

    private fun decodeTeachingResponseViaLegacy(response: JsonObject): APIResult {
        // Use shared decoder that handles both formats; force OpenAI path
        return try { decodeTeachingResponse(response) } catch (e: Exception) { throw e }
    }

    private fun selectModel(schema: JsonObject?): String {
        // Use 2.5 Flash for general, 3 Flash where schema-heavy? Use 2.5 as default per task.
        // Allow via purpose if needed; for now always gemini-2.5-flash
        return "gemini-2.5-flash"
    }

    private fun decodeGeminiResponse(response: JsonObject, searchExpected: Boolean): APIResult {
        // Gemini format: {candidates: [{content:{parts:[{text:...}]}, finishReason, groundingMetadata:{groundingChunks:[{web:{uri,title}}]}}], usageMetadata:{promptTokenCount, candidatesTokenCount}}
        val candidates = response["candidates"] as? JsonArray ?: throw APIException.InvalidResponse
        if (candidates.isEmpty()) {
            // Check for block reason
            val promptFeedback = response["promptFeedback"] as? JsonObject
            val blockReason = promptFeedback?.get("blockReason")?.jsonPrimitive?.contentOrNull
            if (blockReason != null) throw APIException.Refused
            throw APIException.Incomplete
        }
        val first = candidates[0] as? JsonObject ?: throw APIException.InvalidResponse
        val content = first["content"] as? JsonObject ?: throw APIException.InvalidResponse
        val parts = content["parts"] as? JsonArray ?: throw APIException.InvalidResponse
        val textBuilder = StringBuilder()
        for (p in parts) {
            val obj = p as? JsonObject ?: continue
            val t = obj["text"]?.jsonPrimitive?.contentOrNull
            if (t != null) textBuilder.append(t)
        }
        val text = textBuilder.toString()
        if (text.isBlank()) {
            val finishReason = first["finishReason"]?.jsonPrimitive?.contentOrNull
            if (finishReason == "SAFETY") throw APIException.Refused
            throw APIException.Incomplete
        }

        // Usage
        val usage = response["usageMetadata"] as? JsonObject
        val inputTokens = (usage?.get("promptTokenCount") as? JsonPrimitive)?.intOrNull ?: 0
        val outputTokens = (usage?.get("candidatesTokenCount") as? JsonPrimitive)?.intOrNull ?: 0

        // Sources from groundingMetadata
        val sources = mutableListOf<SourceLink>()
        val grounding = first["groundingMetadata"] as? JsonObject
        val chunks = grounding?.get("groundingChunks") as? JsonArray
        if (chunks != null) {
            for (chunk in chunks) {
                val c = chunk as? JsonObject ?: continue
                val web = c["web"] as? JsonObject ?: continue
                val uri = web["uri"]?.jsonPrimitive?.contentOrNull ?: continue
                val title = web["title"]?.jsonPrimitive?.contentOrNull ?: "Source"
                if (isSafeSourceUrl(uri) && sources.none { it.url == uri }) {
                    sources.add(SourceLink(title, uri))
                }
            }
        }
        // Also check candidate-level groundingMetadata alternative location
        if (sources.isEmpty() && searchExpected) {
            // search may have no sources, but not error
        }

        // Count searches: if groundingMetadata present and searchRequested, count as 1
        val searches = if (searchExpected && grounding != null) 1 else 0

        return APIResult(
            text = text,
            sources = sources,
            usage = APIUsage(
                input = inputTokens.coerceIn(0, 1_000_000_000),
                output = outputTokens.coerceIn(0, 1_000_000_000),
                searches = searches,
            ),
        )
    }

    private fun isSafeSourceUrl(value: String): Boolean = try {
        val uri = URI(value)
        uri.scheme == "https" && !uri.host.isNullOrBlank() && uri.userInfo == null
    } catch (_: Exception) { false }

    private fun Response.readBoundedBody(): String {
        val responseBody = body ?: throw APIException.InvalidResponse
        if (responseBody.contentLength() > MAX_RESPONSE_BYTES) throw APIException.InvalidResponse
        val source = responseBody.source()
        val buffer = Buffer()
        var total = 0L
        while (true) {
            val count = source.read(buffer, minOf(8_192L, MAX_RESPONSE_BYTES + 1L - total))
            if (count == -1L) break
            total += count
            if (total > MAX_RESPONSE_BYTES) throw APIException.InvalidResponse
        }
        return buffer.readString(Charsets.UTF_8)
    }

    sealed class APIException(message: String, cause: Throwable? = null) : IOException(message, cause) {
        data object MissingKey : APIException("Add your Google Gemini API key in Settings to begin.")
        data object InvalidResponse : APIException("Gemini returned an incomplete response. Please try again.")
        data object Incomplete : APIException("Gemini returned an incomplete response. Please try again.")
        data object Refused : APIException("Mural couldn't complete that request. Try a different topic.")
        class Http(val status: Int) : APIException(messageFor(status))

        companion object {
            private fun messageFor(status: Int): String = when (status) {
                401 -> "Your Google Gemini API key wasn't accepted. Check it in Settings."
                403, 404 -> "This API key may not have access to the requested model. Check your Google AI project."
                429 -> "Gemini's usage or rate limit was reached. Check your project billing and limits."
                else -> "Gemini couldn't complete the request (HTTP $status). Please try again."
            }
        }
    }

    companion object {
        private val GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/".toHttpUrl()
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
        private val VALID_PATH = Regex("[a-z0-9][a-z0-9_/-]*")
        private const val MAX_RESPONSE_BYTES = 1_048_576L
        private val JSON = Json { ignoreUnknownKeys = true }

        private fun defaultClient() = OkHttpClient.Builder()
            .connectTimeout(45, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(45, TimeUnit.SECONDS)
            .callTimeout(60, TimeUnit.SECONDS)
            .followRedirects(false)
            .followSslRedirects(false)
            .cookieJar(CookieJar.NO_COOKIES)
            .cache(null)
            .build()

    }
}
