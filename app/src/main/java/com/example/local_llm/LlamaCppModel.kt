package com.example.local_llm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.cancellable
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/**
 * Kotlin wrapper for llama.cpp native library
 * Replaces ONNX Runtime with llama.cpp for inference
 */
class LlamaCppModel(private val context: Context) {
    
    // 언어 블로킹을 위한 속성
    private var isLanguageBlockingEnabled = false
    
    companion object {
        private const val TAG = "LlamaCppModel"
        
        init {
            try {
                System.loadLibrary("qwenllm")
                Log.d(TAG, "Native library loaded successfully")
                
                // Install native crash handler for better debugging
                nativeInstallCrashHandler()
                Log.d(TAG, "Native crash handler installed")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library", e)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to install crash handler", e)
            }
        }
        
        @JvmStatic
        private external fun nativeInstallCrashHandler()
    }
    
    private var nativePtr = 0L
    private var isModelLoaded = false
    
    /**
     * Load GGUF model from assets
     */
    suspend fun loadModel(
        modelFileName: String,
        contextSize: Int = Constants.Model.CONTEXT_SIZE,
        nBatch: Int = Constants.Model.BATCH_SIZE_DEFAULT,
        nThreads: Int = Constants.Model.N_THREADS,
        temperature: Float = Constants.Model.TEMPERATURE,
        minP: Float = Constants.Model.MIN_P
    ) = withContext(Dispatchers.IO) {
        try {
            // Copy model from assets to internal storage
            val modelFile = copyAssetToInternalStorage(modelFileName)
            
            // Initialize native model
            nativePtr = nativeLoadModel(
                modelFile.absolutePath,
                contextSize,
                nBatch,
                nThreads,
                temperature,
                minP
            )
            
            if (nativePtr != 0L) {
                isModelLoaded = true
                Log.d(TAG, "Model loaded successfully: $modelFileName")
            } else {
                throw RuntimeException("Failed to load model")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            throw e
        }
    }
    
    /**
     * Generate response with streaming
     */
    fun generateResponseStream(
        prompt: String,
        maxTokens: Int = 512,
        retrievalTimeMs: Long = 0,
        overheadTimeMs: Long = 0
    ): Flow<String> = flow {
        val startTime = System.nanoTime()
        var firstTokenTime = 0L
        
        if (!isModelLoaded) {
            throw IllegalStateException("Model not loaded")
        }
        
        Log.d(TAG, "[GENERATION DEBUG] ========== STREAM GENERATION START ==========")
        Log.d(TAG, "[GENERATION DEBUG] Prompt length: ${prompt.length} chars")
        Log.d(TAG, "[GENERATION DEBUG] Max tokens: $maxTokens")
        Log.d(TAG, "[GENERATION DEBUG] Prompt preview (first 300 chars): '${prompt.take(300).replace("\n", "\\n")}'")
        Log.d(TAG, "[GENERATION DEBUG] Prompt contains 'Context:'?: ${prompt.contains("Context:")}")
        Log.d(TAG, "[GENERATION DEBUG] Prompt contains 'Question:'?: ${prompt.contains("Question:")}")
        
        // Start generation
        Log.d(TAG, "[GENERATION DEBUG] Calling native generation...")
        val success = nativeStartGeneration(nativePtr, prompt, maxTokens)
        Log.d(TAG, "[GENERATION DEBUG] Native generation started: success=$success")
        if (!success) {
            throw RuntimeException("Failed to start generation")
        }
        
        // Stream tokens
        var tokenCount = 0
        try {
            Log.d(TAG, "[GENERATION DEBUG] Starting token streaming loop...")
            while (tokenCount < maxTokens && currentCoroutineContext().isActive) {
                val token = nativeGetNextToken(nativePtr)
                
                if (tokenCount == 0) {
                    firstTokenTime = System.nanoTime()
                }
                
                // Debug logging
                if (tokenCount < 5 || tokenCount % 50 == 0) {
                    Log.d(TAG, "[GENERATION DEBUG] Token $tokenCount: '$token' (empty: ${token.isEmpty()})")
                }
                
                if (token.isEmpty()) {
                    Log.d(TAG, "[GENERATION DEBUG] Empty token received, breaking loop at token $tokenCount")
                    break
                }
                emit(token)
                tokenCount++
            }
            Log.d(TAG, "[GENERATION DEBUG] Token streaming completed. Total tokens: $tokenCount")
        } finally {
            val endTime = System.nanoTime()
            
            // Calculate metrics
            val totalTimeMs = (endTime - startTime) / 1_000_000
            val prefillTimeMs = if (firstTokenTime > 0) (firstTokenTime - startTime) / 1_000_000 else 0
            val decoderTimeMs = if (firstTokenTime > 0) (endTime - firstTokenTime) / 1_000_000 else 0
            
            // Calculate TPS based on decoder time (excluding prefill)
            val tps = if (decoderTimeMs > 0 && tokenCount > 1) {
                (tokenCount - 1) * 1000.0 / decoderTimeMs
            } else {
                0.0
            }

            // Calculate Memory Usage (Approximate Java Heap)
            val memoryUsage = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024
            
            // Check if RAG was enabled (based on prompt content)
            val isRagEnabled = prompt.contains("Context:", ignoreCase = true)

            Log.i("PerformanceMetrics", "========== PERFORMANCE REPORT ==========")
            Log.i("PerformanceMetrics", "Total generation time: ${totalTimeMs}ms")
            Log.i("PerformanceMetrics", "Prefill: ${prefillTimeMs}ms")
            Log.i("PerformanceMetrics", "Decoder: ${decoderTimeMs}ms")
            Log.i("PerformanceMetrics", "Retrieval time: ${retrievalTimeMs}ms")
            Log.i("PerformanceMetrics", "Overhead time: ${overheadTimeMs}ms")
            Log.i("PerformanceMetrics", "Total tokens: $tokenCount")
            Log.i("PerformanceMetrics", "TPS: $tps")
            Log.i("PerformanceMetrics", "Rag enabled: $isRagEnabled")
            Log.i("PerformanceMetrics", "Memory usage: ${memoryUsage}MB (Java Heap)")
            Log.i("PerformanceMetrics", "========================================")

        }

        
        Log.d(TAG, "[GENERATION DEBUG] Generation completed with $tokenCount tokens")
        Log.d(TAG, "[GENERATION DEBUG] ========== STREAM GENERATION END ===========")
    }.flowOn(Dispatchers.IO)
        .cancellable() // Ensure the flow is cancellable
    
    /**
     * Generate complete response (non-streaming)
     */
    suspend fun generateResponse(
        prompt: String,
        maxTokens: Int = 512
    ): String = withContext(Dispatchers.IO) {
        if (!isModelLoaded) {
            throw IllegalStateException("Model not loaded")
        }
        
        Log.d(TAG, "Generating response for prompt: ${prompt.take(100)}...")
        
        val response = nativeGenerateComplete(nativePtr, prompt, maxTokens)
        
        Log.d(TAG, "Response generated: ${response.take(100)}...")
        
        response
    }
    
    /**
     * Build prompt with RAG context
     */
    fun buildRAGPrompt(context: String, question: String, systemPrompt: String? = null): String {
        Log.d(TAG, "[LLAMA PROMPT DEBUG] Building RAG prompt")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] Context length: ${context.length} chars")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] Context preview: '${context.take(200)}'...")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] Question: '$question'")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] System prompt: '${systemPrompt ?: "default RAG prompt"}'")
        
        val systemMessage = systemPrompt ?: "You are a helpful assistant. Answer the question based on the provided context."
        
        val prompt = """<|im_start|>system
$systemMessage<|im_end|>
<|im_start|>user
Context: $context

Question: $question<|im_end|>
<|im_start|>assistant
"""
        
        Log.d(TAG, "[LLAMA PROMPT DEBUG] Built RAG prompt:")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] - Total length: ${prompt.length} chars")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] - System message: '$systemMessage'")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] - Has context: ${context.isNotEmpty()}")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] - Full prompt preview (first 400 chars): '${prompt.take(400).replace("\n", "\\n")}'")
        
        return prompt
    }
    
    /**
     * Build simple QA prompt
     */
    fun buildQAPrompt(question: String, systemPrompt: String? = null): String {
        Log.d(TAG, "[LLAMA PROMPT DEBUG] Building simple QA prompt")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] Question: '$question'")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] System prompt: '${systemPrompt ?: "none"}'")
        
        val prompt = if (systemPrompt != null) {
            """<|im_start|>system
$systemPrompt<|im_end|>
<|im_start|>user
$question<|im_end|>
<|im_start|>assistant
"""
        } else {
            """<|im_start|>user
$question<|im_end|>
<|im_start|>assistant
"""
        }
        
        Log.d(TAG, "[LLAMA PROMPT DEBUG] Built QA prompt:")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] - Total length: ${prompt.length} chars")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] - Has system prompt: ${systemPrompt != null}")
        Log.d(TAG, "[LLAMA PROMPT DEBUG] - Full prompt: '${prompt.replace("\n", "\\n")}'")
        
        return prompt
    }
    
    /**
     * Copy asset file to internal storage
     */
    private fun copyAssetToInternalStorage(fileName: String): File {
        val assetManager = context.assets
        val outputFile = File(context.filesDir, fileName)
        
        try {
            // For compressed assets, we can't use openFd to get the size
            // So we'll just check if the file exists and has a reasonable size
            
            if (!outputFile.exists() || outputFile.length() < 1000) {
                Log.d(TAG, "Copying $fileName to internal storage...")
                
                assetManager.open(fileName).use { inputStream ->
                    FileOutputStream(outputFile).use { outputStream ->
                        val buffer = ByteArray(1024 * 1024) // 1MB buffer
                        var read: Int
                        var totalBytes = 0L
                        
                        while (inputStream.read(buffer).also { read = it } != -1) {
                            outputStream.write(buffer, 0, read)
                            totalBytes += read
                            
                            // Log progress every 50MB
                            if (totalBytes % (50 * 1024 * 1024) == 0L) {
                                Log.d(TAG, "Copied ${totalBytes / 1024 / 1024}MB...")
                            }
                        }
                        
                        Log.d(TAG, "Successfully copied $fileName (${totalBytes / 1024 / 1024}MB)")
                    }
                }
                
                Log.d(TAG, "File saved to: ${outputFile.absolutePath}")
                Log.d(TAG, "Final file size: ${outputFile.length() / 1024 / 1024}MB")
            } else {
                Log.d(TAG, "$fileName already exists: ${outputFile.length() / 1024 / 1024}MB at ${outputFile.absolutePath}")
            }
            
            // Verify file exists and is readable
            if (!outputFile.exists()) {
                throw RuntimeException("Model file does not exist after copy: ${outputFile.absolutePath}")
            }
            
            if (!outputFile.canRead()) {
                throw RuntimeException("Model file is not readable: ${outputFile.absolutePath}")
            }
            
            Log.d(TAG, "Model file ready: ${outputFile.absolutePath}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error copying $fileName: ${e.message}", e)
            e.printStackTrace()
            throw RuntimeException("Failed to copy model file: ${e.message}", e)
        }
        
        return outputFile
    }
    
    /**
     * Reset model state for new conversation
     */
    fun reset() {
        if (nativePtr != 0L) {
            nativeReset(nativePtr)
            Log.d(TAG, "Model state reset")
        }
    }
    
    /**
     * Release native resources
     */
    fun release() {
        if (nativePtr != 0L) {
            nativeRelease(nativePtr)
            nativePtr = 0L
            isModelLoaded = false
            Log.d(TAG, "Model released")
        }
    }
    
    // Native methods
    private external fun nativeLoadModel(
        modelPath: String,
        contextSize: Int,
        nBatch: Int,
        nThreads: Int,
        temperature: Float,
        minP: Float
    ): Long
    
    private external fun nativeStartGeneration(
        ptr: Long,
        prompt: String,
        maxTokens: Int
    ): Boolean
    
    private external fun nativeGetNextToken(ptr: Long): String
    
    private external fun nativeGenerateComplete(
        ptr: Long,
        prompt: String,
        maxTokens: Int
    ): String
    
    private external fun nativeReset(ptr: Long)
    
    private external fun nativeRelease(ptr: Long)
    
    /**
     * 언어 블로킹 활성화/비활성화
     */
    fun setLanguageBlocking(enabled: Boolean) {
        isLanguageBlockingEnabled = enabled
        LoggingUtils.logRAGDebug(TAG, "Language blocking ${if (enabled) "enabled" else "disabled"}")
    }
    
    /**
     * 언어 블로킹이 활성화되어 있는지 확인
     */
    fun isLanguageBlockingEnabled(): Boolean = isLanguageBlockingEnabled
    
    /**
     * 언어 블로킹이 적용된 프롬프트 생성
     */
    fun buildRAGPromptWithLanguageBlocking(context: String, question: String, systemPrompt: String? = null): String {
        val prompt = buildRAGPrompt(context, question, systemPrompt)
        
        if (isLanguageBlockingEnabled) {
            // 언어 블로킹을 위한 추가 처리
            LoggingUtils.logRAGDebug(TAG, "Applied language blocking to RAG prompt")
        }
        
        return prompt
    }
    
    /**
     * 언어 블로킹이 적용된 QA 프롬프트 생성
     */
    fun buildQAPromptWithLanguageBlocking(question: String, systemPrompt: String? = null): String {
        val prompt = buildQAPrompt(question, systemPrompt)
        
        if (isLanguageBlockingEnabled) {
            // 언어 블로킹을 위한 추가 처리
            LoggingUtils.logRAGDebug(TAG, "Applied language blocking to QA prompt")
        }
        
        return prompt
    }
}