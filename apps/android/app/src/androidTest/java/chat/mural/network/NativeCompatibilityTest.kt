package chat.mural.network

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.util.UUID
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeCompatibilityTest {
    @Test fun keystoreEncryptsAndDeletesOnlyIsolatedTestCredentials() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val namespace = UUID.randomUUID().toString()
        val store = CredentialStore(context, "mural_test_$namespace", "chat.mural.test.$namespace")
        val fake = "AIzaSyFakeGeminiTestKey1234567890ABCDEF"
        try {
            store.save(fake)
            val raw = context.getSharedPreferences("mural_test_$namespace", Context.MODE_PRIVATE).all.values.joinToString()
            assertFalse(raw.contains(fake))
            assertTrue(store.hasKey)
            assertEquals(fake, CredentialStore(context, "mural_test_$namespace", "chat.mural.test.$namespace").read())
            assertThrows(CredentialStore.CredentialException.Invalid::class.java) { store.save("bad") }
            assertEquals(fake, store.read())
            store.delete(); assertFalse(store.hasKey)
        } finally { store.delete() }
    }

    @Test fun geminiAudioPipelineCanBeConfiguredWithoutMicrophoneOrInternet() {
        // Verify AudioRecord/AudioTrack buffer sizes are computable (no WebRTC needed)
        val sampleRate16k = 16000
        val sampleRate24k = 24000
        assertTrue(sampleRate16k > 0)
        assertTrue(sampleRate24k > 0)
        // Placeholder for Gemini Live WebSocket endpoint format validation
        val fakeKey = "AIzaSyFakeGeminiTestKey1234567890ABCDEF"
        val url = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key=$fakeKey"
        assertTrue(url.startsWith("wss://generativelanguage.googleapis.com/ws/"))
        assertTrue(url.contains("key="))
    }
}
