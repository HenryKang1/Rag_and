package com.example.local_llm

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

/**
 * Embedding model handler for text encoding using ONNX Runtime
 * Expects a sentence-transformers model exported to ONNX format
 */
class EmbeddingModel(private val context: Context) {
    
    companion object {
        const val TAG = "EmbeddingModel"
        const val MODEL_FILE = "embedding_model.onnx"
        const val TOKENIZER_FILE = "embedding_tokenizer.json"
        const val MAX_SEQ_LENGTH = 128
        const val EMBEDDING_DIM = 384  // for all-MiniLM-L6-v2
    }
    
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var tokenizer: SimpleEmbeddingTokenizer? = null
    private var isInitialized = false
    
    /**
     * Initialize the embedding model
     */
    fun initialize() {
        if (isInitialized) return
        
        try {
            Log.d(TAG, "Initializing embedding model...")
            
            // Initialize ONNX environment
            env = OrtEnvironment.getEnvironment()
            
            // Check if model exists in assets
            if (!modelExists()) {
                Log.w(TAG, "Embedding model not found. Using mock embeddings for demo.")
                // In production, you would download or include the model
                tokenizer = SimpleEmbeddingTokenizer()
                isInitialized = true
                return
            }
            
            // Load model
            val modelFile = loadModelFile()
            val opts = OrtSession.SessionOptions()
            session = env?.createSession(modelFile.absolutePath, opts)
            
            // Initialize tokenizer
            tokenizer = SimpleEmbeddingTokenizer()
            
            isInitialized = true
            Log.d(TAG, "Embedding model initialized successfully")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize embedding model: ${e.message}")
            // Fallback to mock embeddings
            tokenizer = SimpleEmbeddingTokenizer()
            isInitialized = true
        }
    }
    
    /**
     * Encode text to embedding vector
     */
    fun encode(text: String): FloatArray {
        if (!isInitialized) {
            throw IllegalStateException("Embedding model not initialized")
        }
        
        // If no ONNX session (model not available), use mock embeddings
        if (session == null) {
            return createMockEmbedding(text)
        }
        
        try {
            // Tokenize text
            val tokens = tokenizer?.tokenize(text) ?: throw IllegalStateException("Tokenizer not initialized")
            
            // Prepare input tensors
            val inputIds = tokens.inputIds
            val attentionMask = tokens.attentionMask
            
            // Convert LongArray to 2D array for tensor creation
            val inputIdsArray = arrayOf(inputIds)
            val attentionMaskArray = arrayOf(attentionMask)
            
            val inputIdsTensor = OnnxTensor.createTensor(
                env,
                inputIdsArray
            )
            
            val attentionMaskTensor = OnnxTensor.createTensor(
                env,
                attentionMaskArray
            )
            
            // Run inference
            val inputs = mapOf(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attentionMaskTensor
            )
            
            val results = session?.run(inputs)
            
            // Extract embeddings (usually from pooler_output or last_hidden_state with mean pooling)
            val outputTensor = results?.get(0) as? OnnxTensor
                ?: results?.get("sentence_embedding") as? OnnxTensor
                ?: throw IllegalStateException("Unexpected output format")
            
            val output = outputTensor.value as Array<FloatArray>
            
            val embedding = output[0]
            
            // Normalize embedding
            val norm = kotlin.math.sqrt(embedding.map { it * it }.sum())
            return if (norm > 0) {
                FloatArray(embedding.size) { i -> embedding[i] / norm }
            } else {
                embedding
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to encode text: ${e.message}")
            // Fallback to mock embedding
            return createMockEmbedding(text)
        }
    }
    
    /**
     * Create mock embedding for demonstration when model is not available
     */
    private fun createMockEmbedding(text: String): FloatArray {
        // Simple hash-based mock embedding for demo purposes
        val embedding = FloatArray(EMBEDDING_DIM)
        val words = text.lowercase().split(Regex("\\s+"))
        
        for ((idx, word) in words.withIndex()) {
            val hash = word.hashCode()
            for (i in 0 until EMBEDDING_DIM) {
                embedding[i] += ((hash * (idx + 1) * (i + 1)) % 100) / 100f
            }
        }
        
        // Normalize
        val norm = kotlin.math.sqrt(embedding.map { it * it }.sum())
        if (norm > 0) {
            for (i in embedding.indices) {
                embedding[i] /= norm
            }
        }
        
        return embedding
    }
    
    /**
     * Check if model exists in assets
     */
    private fun modelExists(): Boolean {
        return try {
            context.assets.list("")?.contains(MODEL_FILE) ?: false
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Load model file from assets to internal storage
     */
    private fun loadModelFile(): File {
        val assetManager = context.assets
        val inputStream = assetManager.open(MODEL_FILE)
        val file = File(context.filesDir, MODEL_FILE)
        val outputStream = FileOutputStream(file)
        inputStream.copyTo(outputStream)
        inputStream.close()
        outputStream.close()
        return file
    }
    
    /**
     * Data class for tokenizer output
     */
    data class TokenizerOutput(
        val inputIds: LongArray,
        val attentionMask: LongArray
    )
    
    /**
     * Simple tokenizer for embedding model
     */
    inner class SimpleEmbeddingTokenizer {
        
        fun tokenize(text: String): TokenizerOutput {
            // Simple word-based tokenization for demo
            // In production, use the actual tokenizer from the model
            val words = text.lowercase()
                .replace(Regex("[^a-z0-9\\s]"), "")
                .split(Regex("\\s+"))
                .take(MAX_SEQ_LENGTH - 2)  // Reserve space for [CLS] and [SEP]
            
            val tokenIds = mutableListOf<Long>()
            tokenIds.add(101L)  // [CLS] token
            
            for (word in words) {
                // Simple hash-based token ID (for demo)
                tokenIds.add((word.hashCode() % 30000).toLong().coerceAtLeast(0))
            }
            
            tokenIds.add(102L)  // [SEP] token
            
            // Pad to max length
            while (tokenIds.size < MAX_SEQ_LENGTH) {
                tokenIds.add(0L)  // [PAD] token
            }
            
            // Create attention mask
            val attentionMask = LongArray(MAX_SEQ_LENGTH) { i ->
                if (i < words.size + 2) 1L else 0L
            }
            
            return TokenizerOutput(
                inputIds = tokenIds.take(MAX_SEQ_LENGTH).toLongArray(),
                attentionMask = attentionMask
            )
        }
    }
    
    /**
     * Clean up resources
     */
    fun close() {
        session?.close()
        env?.close()
        session = null
        env = null
        isInitialized = false
    }
}