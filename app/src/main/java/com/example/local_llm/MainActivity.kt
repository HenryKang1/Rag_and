package com.example.local_llm

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.ContextThemeWrapper
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import io.noties.markwon.Markwon
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collect

class MainActivity : AppCompatActivity() {

    internal lateinit var tokenizer: BpeTokenizer
    internal lateinit var llamaModel: LlamaCppModel
    private lateinit var markwon: Markwon
    private val inferenceScope = CoroutineScope(Dispatchers.IO)
    private var inferenceJob: Job? = null
    
    // RAG components
    private lateinit var mobileRAG: MobileRAG
    private lateinit var documentProcessor: DocumentProcessor
    internal var isRagEnabled = true
    internal var ragInitialized = false
    internal var useSemanticChunking = false  // Toggle for semantic chunking mode
    internal var forceMaxTokens = 0  // 0 = auto, or specific value to override
    
    // Model selection
    internal var selectedModel = "qwen3-0.6b-q4_k_m.gguf"  // Default model
    private val availableModels = listOf(
        "qwen3-0.6b-q4_k_m.gguf",
        "Qwen3-0.6B-Q8_0.gguf",
        "Huihui-qwen3-0.6b-abliterated-v2.Q6_K2.gguf"
    )

    private val END_TOKEN_IDS = Constants.Token.END_TOKEN_IDS
    
    // Batch processing setting
    private var batchSize = Constants.Model.BATCH_SIZE_DEFAULT

    internal lateinit var chatRecyclerView: RecyclerView
    internal lateinit var chatAdapter: ChatAdapter
    internal val messages = mutableListOf<Message>()
    
    // UI elements
    internal lateinit var thinkingToggle: CheckBox
    internal lateinit var ragToggle: CheckBox
    internal lateinit var uploadButton: Button
    internal lateinit var clearIndexButton: Button
    internal lateinit var inputEditText: EditText
    internal lateinit var sendButton: Button
    internal lateinit var stopButton: Button
    
    // File picker launcher
    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        Log.d("MainActivity", "[MAIN DEBUG] File picker result: resultCode=${result.resultCode}")
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                Log.d("MainActivity", "[MAIN DEBUG] File selected: $uri")
                processUploadedDocument(uri)
            } ?: Log.e("MainActivity", "[MAIN DEBUG] No URI in result data")
        } else {
            Log.d("MainActivity", "[MAIN DEBUG] File picker cancelled or failed")
        }
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.chat_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.menu_clear -> {
                AlertDialog.Builder(ContextThemeWrapper(this, R.style.DarkAlertDialog))
                    .setTitle(getString(R.string.clear_chat_title))
                    .setMessage(getString(R.string.clear_chat_message))
                    .setPositiveButton(getString(R.string.yes)) { _, _ ->
                        messages.clear()
                        chatAdapter.clearAll()
                        // Also reset the model context
                        if (::llamaModel.isInitialized) {
                            llamaModel.reset()
                        }
                    }
                    .setNegativeButton(getString(R.string.cancel), null)
                    .show()
                true
            }
            R.id.menu_model -> {
                showModelSelectionDialog()
                true
            }
            R.id.menu_batch_settings -> {
                val batchOptions = arrayOf("1 (Sequential, Slow)", "128", "256", "512", "1024 (Batched, Fast)")
                val batchValues = intArrayOf(1, 128, 256, 512, 1024)
                
                // Find current selection index
                var currentIndex = batchValues.indexOf(batchSize)
                if (currentIndex == -1) currentIndex = 4 // Default to 1024 if unknown
                
                AlertDialog.Builder(ContextThemeWrapper(this, R.style.DarkAlertDialog))
                    .setTitle("Select Batch Size (Prefill)")
                    .setSingleChoiceItems(batchOptions, currentIndex) { dialog, which ->
                        val selectedSize = batchValues[which]
                        if (batchSize != selectedSize) {
                            batchSize = selectedSize
                            Toast.makeText(this, "Batch size set to $batchSize. Reloading model...", Toast.LENGTH_SHORT).show()
                            loadSelectedModel() // Reload to apply
                        }
                        dialog.dismiss()
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
                true
            }
            R.id.menu_block_foreign -> {
                item.isChecked = !item.isChecked
                tokenizer.setLanguageBlocking(item.isChecked)
                llamaModel.setLanguageBlocking(item.isChecked)
                val message = if (item.isChecked) "🚫 Language blocking enabled" else "🌍 Language blocking disabled"
                Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }


    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setupWindow()
        initializeComponents()
        initializeRAG()
        
        val extensions = MainActivityExtensions(this)
        extensions.initializeUI()
        extensions.initializeModelConfiguration()
        extensions.setupChatUI()
        
        setupModelConfiguration()
        loadDefaultModel()
    }
    
    private fun setupWindow() {
        window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)
        setContentView(R.layout.activity_main)
    }
    
    private fun setupModelConfiguration() {
        // === Define token IDs for prompt formatting ===
        val roleTokens = RoleTokenIds(
            systemStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("system"), tokenizer.getTokenId("Ċ")),
            userStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("user"), tokenizer.getTokenId("Ċ")),
            assistantStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("assistant"), tokenizer.getTokenId("Ċ")),
            endToken = tokenizer.getTokenId("<|im_end|>")
        )

        // === Dynamic Model Configuration ===
        val config = createModelConfig(selectedModel)


        if (config.IsThinkingModeAvailable) {
            thinkingToggle.visibility = View.VISIBLE
            thinkingToggle.isChecked = false  // Default to non-thinking mode for faster responses
            
            // Show thinking mode status
            thinkingToggle.setOnCheckedChangeListener { _, isChecked ->
                val mode = if (isChecked) {
                    "🤔 Thinking Mode ON\n(Slower but more thoughtful responses)"
                } else {
                    "⚡ Fast Mode ON\n(Quicker responses, less reasoning)"
                }
                Toast.makeText(this, mode, Toast.LENGTH_LONG).show()
                Log.d("MainActivity", "[MODE DEBUG] Thinking mode: $isChecked")
            }
        } else {
            // Hide thinking toggle if model doesn't support it
            thinkingToggle.visibility = View.GONE
        }

        title = "Emergency assistant — ${config.modelName}"
        val promptBuilder = PromptBuilder(tokenizer, config)
        val mapper = TokenDisplayMapper(this, config.modelName)

        toggleGenerating(false)
        messages.add(Message("⏳ Please wait, the model is loading.", isUser = false))
        chatAdapter.notifyItemInserted(messages.size - 1)

        // Note: Model loading is handled in onCreate() to avoid duplicate calls
        // loadDefaultModel() is called once in onCreate()

        sendButton.setOnClickListener {
            val userPrompt = inputEditText.text.toString().trim()
            if (userPrompt.isEmpty() || !::llamaModel.isInitialized) return@setOnClickListener

            // Hide keyboard when sending message
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            imm.hideSoftInputFromWindow(inputEditText.windowToken, 0)

            messages.add(Message(userPrompt, isUser = true))
            chatAdapter.notifyItemInserted(messages.size - 1)
            chatRecyclerView.scrollToPosition(messages.size - 1)
            inputEditText.text.clear()

            // Use different system prompt based on Thinking Mode toggle (only for Qwen3)
            val systemPrompt = if (config.modelName.equals("qwen3", ignoreCase = true) && config.IsThinkingModeAvailable) {
                if (thinkingToggle.isChecked) {
                    Log.d("MainActivity", "[MODE DEBUG] Using thinking mode prompt")
                    config.defaultSystemPrompt
                } else {
                    Log.d("MainActivity", "[MODE DEBUG] Using non-thinking mode prompt with /no_think")
                    "${config.defaultSystemPrompt} /no_think"
                }
            } else {
                config.defaultSystemPrompt
            }

            val intent = PromptIntent.QA(systemPrompt)
            
            // Cancel any existing generation job
            val previousJob = inferenceJob
            previousJob?.cancel()
            
            toggleGenerating(true)
            
            inferenceJob = inferenceScope.launch {
                val jobStartTime = System.nanoTime() // Start tracking overhead time
                var retrievalTimeMs = 0L

                // Wait for previous job to finish cancelling
                previousJob?.join()
                // Log memory status before inference
                DebugHelper.logMemoryStatus(this@MainActivity, "[BEFORE INFERENCE]")
                
                // Check memory pressure before starting
                if (DebugHelper.checkMemoryPressure(this@MainActivity)) {
                    withContext(Dispatchers.Main) {
                        messages.add(Message("⚠️ Low memory detected. Clearing cache...", isUser = false))
                        chatAdapter.notifyItemInserted(messages.size - 1)
                        
                        // Auto-disable thinking mode when memory is low
                        if (config.IsThinkingModeAvailable && thinkingToggle.isChecked) {
                            thinkingToggle.isChecked = false
                            messages.add(Message("💡 Switched to non-thinking mode to save memory", isUser = false))
                            chatAdapter.notifyItemInserted(messages.size - 1)
                        }
                    }
                    // llama.cpp handles memory internally
                    System.gc()
                    delay(500) // Give system time to free memory
                }
                
                // === 전체 RAG + 인퍼런스 성능 측정 시작 ===
                val totalStartTime = System.currentTimeMillis()
                LoggingUtils.logSectionStart("MainActivity", "TOTAL RAG + INFERENCE PERFORMANCE")
                LoggingUtils.logContextDebug("MainActivity", "=== MOBILE RAG + LLM PERFORMANCE ANALYSIS ===")
                
                // Get RAG context if enabled
                var finalPrompt = userPrompt
                var contextTime = 0L  // RAG 검색 시간을 저장할 변수
                if (isRagEnabled && ragInitialized) {
                    val retrievalStart = System.nanoTime()
                    LoggingUtils.logSectionStart("MainActivity", "RAG CONTEXT RETRIEVAL")
                    LoggingUtils.logContextDebug("MainActivity", "RAG enabled: $isRagEnabled, RAG initialized: $ragInitialized")
                    LoggingUtils.logContextDebug("MainActivity", "User query: '$userPrompt'")
                    
                    withContext(Dispatchers.Main) {
                        messages.add(Message("🔍 Searching documents...", isUser = false))
                        chatAdapter.notifyItemInserted(messages.size - 1)
                    }
                    
                    try {
                        val contextStartTime = System.currentTimeMillis()
                        LoggingUtils.logContextDebug("MainActivity", "=== RAG SEARCH PERFORMANCE ===")
                        LoggingUtils.logContextDebug("MainActivity", "Calling mobileRAG.getContext()...")
                        
                        // 토큰 카운터: 문자 수를 기반으로 대략적인 토큰 수 추정 (1 토큰 ≈ 4 문자)
                        // 더 정확한 계산을 원하면 실제 토큰화를 사용할 수 있음
                        val tokenCounter: (String) -> Int = { text ->
                            // 간단한 추정: 대략 4 문자당 1 토큰 (실제로는 모델에 따라 다름)
                            (text.length / 4.0).toInt().coerceAtLeast(1)
                        }
                        
                        val context = mobileRAG.getContext(
                            userPrompt,
                            k = Constants.RAG.TOP_K,
                            tokenCounter = tokenCounter,
                            maxTokens = Constants.RAG.MAX_CONTEXT_TOKENS
                        )
                        val contextEndTime = System.currentTimeMillis()
                        contextTime = contextEndTime - contextStartTime
                        
                        // Calculate retrieval time in ms for PerformanceMetrics
                        retrievalTimeMs = (System.nanoTime() - retrievalStart) / 1_000_000
                        Log.i("PerformanceMetrics", "RAG Retrieval time: ${retrievalTimeMs}ms")
                        
                        LoggingUtils.logContextDebug("MainActivity", "=== RAG SEARCH RESULTS ===")
                        LoggingUtils.logContextDebug("MainActivity", "- Length: ${context.length} chars")
                        LoggingUtils.logContextDebug("MainActivity", "- Context retrieval time: ${contextTime}ms")
                        LoggingUtils.logContextDebug("MainActivity", "- Empty: ${context.isEmpty()}")
                        LoggingUtils.logContextDebug("MainActivity", "- Search speed: ${context.length * 1000.0 / contextTime} chars/sec")
                        LoggingUtils.logContextDebug("MainActivity", "- Full context content:")
                        LoggingUtils.logLongText("MainActivity", "[CONTEXT DEBUG]", context, Constants.Logging.CHUNK_SIZE)
                        
                        if (context.isNotEmpty()) {
                            // Use full context without truncation
                            LoggingUtils.logContextDebug("MainActivity", "Using full context (${context.length} chars)")
                            
                            LoggingUtils.logContextDebug("MainActivity", "Building final prompt with context...")
                            finalPrompt = "$context\n\nQuestion: $userPrompt"
                            LoggingUtils.logContextDebug("MainActivity", "Final prompt structure:")
                            LoggingUtils.logContextDebug("MainActivity", "- Total length: ${finalPrompt.length} chars")
                            LoggingUtils.logContextDebug("MainActivity", "- Context portion: ${context.length} chars")
                            LoggingUtils.logContextDebug("MainActivity", "- Question portion: ${userPrompt.length} chars")
                            LoggingUtils.logContextDebug("MainActivity", "- Full final prompt:")
                            LoggingUtils.logLongText("MainActivity", "[CONTEXT DEBUG] Prompt", finalPrompt, Constants.Logging.PROMPT_CHUNK_SIZE)
                        } else {
                            Log.w("MainActivity", "[CONTEXT DEBUG] Context is empty! No relevant documents found.")
                        }
                        
                        withContext(Dispatchers.Main) {
                            messages.removeAt(messages.size - 1)
                            chatAdapter.notifyItemRemoved(messages.size)
                        }
                    } catch (e: Exception) {
                        Log.e("MainActivity", "RAG search failed: ${e.message}")
                        withContext(Dispatchers.Main) {
                            messages.removeAt(messages.size - 1)
                            chatAdapter.notifyItemRemoved(messages.size)
                        }
                    }
                }
                
                // Build prompt for llama.cpp
                LoggingUtils.logSectionStart("MainActivity", "PROMPT BUILDING")
                val promptStartTime = System.currentTimeMillis()
                LoggingUtils.logPromptDebug("MainActivity", "isRagEnabled: $isRagEnabled")
                LoggingUtils.logPromptDebug("MainActivity", "finalPrompt == userPrompt: ${finalPrompt == userPrompt}")
                LoggingUtils.logPromptDebug("MainActivity", "finalPrompt length: ${finalPrompt.length}, userPrompt length: ${userPrompt.length}")
                
                val prompt = if (isRagEnabled && finalPrompt != userPrompt) {
                    LoggingUtils.logPromptDebug("MainActivity", "Building RAG prompt...")
                    val contextPart = finalPrompt.substringBefore("\n\nQuestion:")
                    LoggingUtils.logPromptDebug("MainActivity", "Extracted context part: '${contextPart.take(200)}'...")
                    LoggingUtils.logPromptDebug("MainActivity", "Question part: '$userPrompt'")
                    
                    val ragPrompt = llamaModel.buildRAGPrompt(contextPart, userPrompt, systemPrompt)
                    val promptEndTime = System.currentTimeMillis()
                    val promptTime = promptEndTime - promptStartTime
                    
                    LoggingUtils.logPromptDebug("MainActivity", "Built RAG prompt with system: '$systemPrompt'")
                    LoggingUtils.logPromptDebug("MainActivity", "- Total length: ${ragPrompt.length} chars")
                    LoggingUtils.logPromptDebug("MainActivity", "- Prompt building time: ${promptTime}ms")
                    LoggingUtils.logPromptDebug("MainActivity", "- Full RAG prompt:")
                    LoggingUtils.logLongText("MainActivity", "[PROMPT DEBUG] RAG", ragPrompt, Constants.Logging.RAG_CHUNK_SIZE)
                    ragPrompt
                } else {
                    LoggingUtils.logPromptDebug("MainActivity", "Building simple QA prompt (no context)...")
                    val qaPrompt = llamaModel.buildQAPrompt(userPrompt, systemPrompt)
                    val promptEndTime = System.currentTimeMillis()
                    val promptTime = promptEndTime - promptStartTime
                    
                    LoggingUtils.logPromptDebug("MainActivity", "Built QA prompt with system: '$systemPrompt'")
                    LoggingUtils.logPromptDebug("MainActivity", "- Total length: ${qaPrompt.length} chars")
                    LoggingUtils.logPromptDebug("MainActivity", "- Prompt building time: ${promptTime}ms")
                    LoggingUtils.logPromptDebug("MainActivity", "- Full prompt: '${qaPrompt.replace("\n", "\\n")}'")
                    qaPrompt
                }
                LoggingUtils.logSectionEnd("MainActivity", "PROMPT BUILDING")
                val promptBuildingEndTime = System.currentTimeMillis()
                val promptBuildingTime = promptBuildingEndTime - promptStartTime
                
                // Check if thinking mode is actually being used
                val actualThinkingMode = config.IsThinkingModeAvailable && thinkingToggle.isChecked
                
                // Add mode indicator to response
                val modeIndicator = when {
                    !actualThinkingMode -> "[⚡ Fast Mode] "
                    else -> "[🤔 Thinking Mode] "
                }
                
                val botResponseBuilder = StringBuilder()
                // Start with mode indicator
                botResponseBuilder.append(modeIndicator)
                val botMessage = Message(modeIndicator, isUser = false)
                withContext(Dispatchers.Main) {
                    messages.add(botMessage)
                    val botIndex = messages.lastIndex
                    chatAdapter.notifyItemInserted(botIndex)
                    chatRecyclerView.scrollToPosition(botIndex)
                }
                val botIndex = messages.lastIndex
                var tokenCounter = 0
                var firstTokenArrivalTime = 0L  // 첫 토큰 도착 시간 추적
                try {
                    // Check if thinking mode is enabled
                    val isThinkingMode = config.IsThinkingModeAvailable && thinkingToggle.isChecked
                    
                    val maxGenerationTokens = MemoryManager.calculateMaxTokens(this@MainActivity, isThinkingMode, forceMaxTokens)
                    
                    var tokenCount = 0
                    
                    // PREFILL 전 준비 시간 측정 시작 (JNI 호출, 토큰화 등)
                    val preInferenceStartTime = System.currentTimeMillis()
                    val generationStartTime = System.currentTimeMillis()
                    LoggingUtils.logGenerationDebug("MainActivity", "=== INFERENCE PERFORMANCE ANALYSIS ===")
                    LoggingUtils.logGenerationDebug("MainActivity", "Starting generation with prompt of ${prompt.length} chars")
                    LoggingUtils.logGenerationDebug("MainActivity", "Max tokens: $maxGenerationTokens")
                    LoggingUtils.logGenerationDebug("MainActivity", "🚀 Generation started at: $generationStartTime")
                    LoggingUtils.logGenerationDebug("MainActivity", "Model: ${config.modelName}")
                    LoggingUtils.logGenerationDebug("MainActivity", "Thinking mode: $isThinkingMode")
                    LoggingUtils.logGenerationDebug("MainActivity", "Context size: ${Constants.Model.CONTEXT_SIZE}")
                    LoggingUtils.logGenerationDebug("MainActivity", "Threads: ${Constants.Model.N_THREADS}")
                    
                    // Calculate overhead time right before generation starts
                    val overheadTimeMs = (System.nanoTime() - jobStartTime) / 1_000_000
                    
                    // Track if we're inside thinking tags
                    var insideThinkingTag = false
                    val filteredBuilder = StringBuilder(modeIndicator)
                    val tokenBuffer = StringBuilder()  // Buffer to handle split tokens
                    
                    LoggingUtils.logGenerationDebug("MainActivity", "Starting token collection...")
                    val jniCallStartTime = System.currentTimeMillis()
                    var firstTokenArrivalTime = 0L
                    llamaModel.generateResponseStream(
                        prompt, 
                        maxTokens = maxGenerationTokens,
                        retrievalTimeMs = retrievalTimeMs,
                        overheadTimeMs = overheadTimeMs
                    ).collect { token ->
                        // 첫 토큰이 도착하면 PREFILL이 완료된 것으로 간주
                        if (tokenCount == 0) {
                            firstTokenArrivalTime = System.currentTimeMillis()
                            val prefillPrepTime = firstTokenArrivalTime - jniCallStartTime
                            LoggingUtils.logGenerationDebug("MainActivity", "⏱️ PREFILL 완료 및 첫 토큰 도착: ${prefillPrepTime}ms")
                        }
                        tokenCount++
                        
                        // Log first few tokens to debug
                        if (tokenCount <= 10) {
                            LoggingUtils.logTokenDebug("MainActivity", "Token $tokenCount: '$token' (length: ${token.length})")
                        }
                        
                        // Safety check
                        if (tokenCount > maxGenerationTokens) {
                            Log.w("MainActivity", "Token limit reached, stopping generation")
                            return@collect
                        }
                        
                        // Filter out thinking tokens in non-thinking mode
                        if (!isThinkingMode) {
                            // Accumulate tokens for pattern detection
                            tokenBuffer.append(token)
                            botResponseBuilder.append(token)
                            
                            val bufferContent = tokenBuffer.toString()
                            
                            // Check for thinking tag patterns
                            when {
                                bufferContent.contains("<think>") -> {
                                    insideThinkingTag = true
                                    // Remove the <think> tag from filtered output
                                    val beforeTag = bufferContent.substringBefore("<think>")
                                    filteredBuilder.append(beforeTag)
                                    tokenBuffer.clear()
                                    tokenBuffer.append(bufferContent.substringAfter("<think>"))
                                }
                                bufferContent.contains("</think>") -> {
                                    insideThinkingTag = false
                                    // Clear buffer after closing tag
                                    tokenBuffer.clear()
                                }
                                insideThinkingTag -> {
                                    // Inside thinking tags, just clear buffer
                                    tokenBuffer.clear()
                                }
                                !insideThinkingTag -> {
                                    // Check if the buffer ends with a partial <think> tag
                                    // We only need to buffer if we might be in the middle of receiving a <think> tag
                                    val isPartialTag = bufferContent.endsWith("<") ||
                                            bufferContent.endsWith("<t") ||
                                            bufferContent.endsWith("<th") ||
                                            bufferContent.endsWith("<thi") ||
                                            bufferContent.endsWith("<thin")

                                    if (!isPartialTag) {
                                        // Safe to output buffered content
                                        filteredBuilder.append(tokenBuffer)
                                        tokenBuffer.clear()
                                    } else {
                                        // Potential tag start, keep buffering
                                    }
                                }
                                // If buffer contains partial tag prefix, wait for more tokens
                            }
                        } else {
                            // In thinking mode, show everything
                            botResponseBuilder.append(token)
                        }
                        
                        // Optimize UI updates - only update every 5 tokens for better performance
                        if (tokenCount % 3 == 0 || tokenCount <= 5) {
                            withContext(Dispatchers.Main) {
                                val displayText = if (!isThinkingMode) {
                                    filteredBuilder.toString()  // Already has modeIndicator
                                } else {
                                    botResponseBuilder.toString()  // Already has modeIndicator
                                }
                                messages[botIndex] = botMessage.copy(text = displayText)
                                chatAdapter.notifyItemChanged(botIndex)
                                chatRecyclerView.scrollToPosition(botIndex)
                            }
                        }
                    }
                    
                    // Flush any remaining buffer content (only if not in thinking tag)
                    if (!isThinkingMode && !insideThinkingTag && tokenBuffer.isNotEmpty()) {
                        filteredBuilder.append(tokenBuffer)
                    }
                    
                    // Update final message with filtered content
                    val generationEndTime = System.currentTimeMillis()
                    val totalGenerationTime = generationEndTime - generationStartTime
                    val tokensPerSecond = if (totalGenerationTime > 0) (tokenCount * 1000.0 / totalGenerationTime) else 0.0
                    
                    // === 모바일 성능 분석 ===
                    val avgTimePerToken = if (tokenCount > 0) totalGenerationTime.toDouble() / tokenCount else 0.0
                    val memoryUsage = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
                    val memoryUsageMB = memoryUsage / (1024 * 1024)
                    
                    // PerformanceLogger를 사용한 인퍼런스 성능 분석
                    PerformanceLogger.logInferencePerformance(
                        totalTime = totalGenerationTime,
                        tokenCount = tokenCount,
                        modelName = config.modelName,
                        thinkingMode = isThinkingMode,
                        contextLength = prompt.length,
                        memoryUsageMB = memoryUsageMB
                    )
                    
                    withContext(Dispatchers.Main) {
                        val finalText = if (!isThinkingMode) {
                            filteredBuilder.toString()
                        } else {
                            // botResponseBuilder already has modeIndicator from line 496
                            botResponseBuilder.toString()
                        }
                        messages[botIndex] = botMessage.copy(text = finalText)
                        chatAdapter.notifyItemChanged(botIndex)
                    }
                    
                    // === 전체 성능 요약 ===
                    val totalEndTime = System.currentTimeMillis()
                    val totalTime = totalEndTime - totalStartTime
                    val ragTime = if (isRagEnabled && ragInitialized) contextTime else 0L
                    val inferenceTime = totalGenerationTime
                    
                    // PREFILL 전 준비 시간 계산
                    // 첫 토큰 도착 시간이 측정되었다면 사용, 아니면 추정
                    val prefillPrepTime = if (firstTokenArrivalTime > 0) {
                        firstTokenArrivalTime - generationStartTime
                    } else if (tokenCount > 0) {
                        // 첫 토큰이 도착한 시간을 추정 (PREFILL 완료 시점)
                        // 실제로는 LlamaCpp 로그의 FIRST TOKEN latency를 참조해야 하지만,
                        // 여기서는 대략적으로 추정
                        val estimatedFirstTokenTime = generationStartTime + (totalGenerationTime * 0.8).toLong()
                        estimatedFirstTokenTime - generationStartTime
                    } else 0L
                    
                    // PerformanceLogger를 사용한 전체 성능 요약
                    PerformanceLogger.logTotalPerformance(
                        totalTime = totalTime,
                        ragTime = ragTime,
                        inferenceTime = inferenceTime,
                        tokenCount = tokenCount,
                        tokensPerSecond = tokensPerSecond,
                        memoryUsageMB = memoryUsageMB,
                        modelName = config.modelName,
                        ragEnabled = isRagEnabled,
                        promptBuildingTime = promptBuildingTime,
                        prefillPrepTime = prefillPrepTime
                    )
                    
                    LoggingUtils.logSectionEnd("MainActivity", "TOTAL RAG + INFERENCE PERFORMANCE")
                    
                    LoggingUtils.logGenerationDebug("MainActivity", "Generation complete")
                    LoggingUtils.logGenerationDebug("MainActivity", "Total tokens generated: $tokenCount")
                    LoggingUtils.logGenerationDebug("MainActivity", "Thinking mode: $isThinkingMode")
                    if (!isThinkingMode) {
                        LoggingUtils.logGenerationDebug("MainActivity", "Filtered response (no thinking): '${filteredBuilder.toString().take(200)}'...")
                        LoggingUtils.logGenerationDebug("MainActivity", "Original had thinking tags: ${botResponseBuilder.toString().contains("<think>")}")
                    }
                    LoggingUtils.logGenerationDebug("MainActivity", "Final response: '${botResponseBuilder.toString().take(200)}'...")
                    LoggingUtils.logGenerationDebug("MainActivity", "Response mentions 'no relevant'?: ${botResponseBuilder.toString().contains("no relevant", ignoreCase = true)}")
                    LoggingUtils.logGenerationDebug("MainActivity", "Response mentions 'not found'?: ${botResponseBuilder.toString().contains("not found", ignoreCase = true)}")
                } catch (e: OutOfMemoryError) {
                    ErrorHandler.logError("MainActivity", "inference", Exception(e))
                    MemoryManager.logMemoryStatus(this@MainActivity, "[OOM ERROR]")
                    withContext(Dispatchers.Main) {
                        val errorMsg = ErrorHandler.handleMemoryError(this@MainActivity, e)
                        messages.add(Message(errorMsg, isUser = false))
                        chatAdapter.notifyItemInserted(messages.size - 1)
                    }
                } catch (e: Exception) {
                    ErrorHandler.logError("MainActivity", "inference", e)
                    MemoryManager.logMemoryStatus(this@MainActivity, "[ERROR]")
                    withContext(Dispatchers.Main) {
                        val errorMsg = ErrorHandler.handleRAGError(this@MainActivity, e)
                        messages.add(Message(errorMsg, isUser = false))
                        chatAdapter.notifyItemInserted(messages.size - 1)
                    }
                } finally {
                    // Clean up after inference
                    MemoryManager.performMemoryCleanup()
                    MemoryManager.logMemoryStatus(this@MainActivity, "[AFTER INFERENCE]")
                    
                    withContext(Dispatchers.Main) {
                        sendButton.visibility = View.VISIBLE
                        stopButton.visibility = View.GONE
                    }
                }
            }
        }

        stopButton.setOnClickListener {
            inferenceJob?.cancel()
            messages.add(Message("⛔ Generation stopped.", isUser = false))
            chatAdapter.notifyItemInserted(messages.size - 1)
            chatRecyclerView.scrollToPosition(messages.size - 1)
            toggleGenerating(false)
        }
    }

    /**
     * 기본 컴포넌트 초기화
     */
    private fun initializeComponents() {
        markwon = Markwon.create(this)
        tokenizer = BpeTokenizer(this)
        mobileRAG = MobileRAG(this)
        documentProcessor = DocumentProcessor(this)
    }
    
    // RAG-related methods
    private fun initializeRAG() {
        inferenceScope.launch {
            try {
                mobileRAG.initialize()
                ragInitialized = true
                withContext(Dispatchers.Main) {
                    messages.add(Message("📚 RAG system initialized", isUser = false))
                    chatAdapter.notifyItemInserted(messages.size - 1)
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Failed to initialize RAG: ${e.message}")
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "RAG initialization failed", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
    
    internal fun openFilePicker() {
        Log.d("MainActivity", "[MAIN DEBUG] Opening file picker...")
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf(
                "application/pdf",
                "text/plain",
                "text/markdown",
                "text/csv"
            ))
        }
        filePickerLauncher.launch(intent)
    }
    
    private fun processUploadedDocument(uri: Uri) {
        Log.d("MainActivity", "[MAIN DEBUG] Document selected: $uri")
        messages.add(Message("📄 Processing document...", isUser = false))
        chatAdapter.notifyItemInserted(messages.size - 1)
        
        inferenceScope.launch {
            try {
                Log.d("MainActivity", "[MAIN DEBUG] Starting document processing...")
                // Check file size
                if (!documentProcessor.checkFileSize(uri)) {
                    withContext(Dispatchers.Main) {
                        messages.add(Message("❌ File too large (max 10MB)", isUser = false))
                        chatAdapter.notifyItemInserted(messages.size - 1)
                    }
                    return@launch
                }
                
                // Process document
                Log.d("MainActivity", "[MAIN DEBUG] Processing document with DocumentProcessor...")
                val text = documentProcessor.processDocument(uri)
                Log.d("MainActivity", "[MAIN DEBUG] Document text extracted: ${text.length} characters")
                
                // Index document with selected chunking mode
                Log.d("MainActivity", "[MAIN DEBUG] Indexing document with MobileRAG...")
                Log.d("MainActivity", "[MAIN DEBUG] Using ${if (this@MainActivity.useSemanticChunking) "SEMANTIC" else "FIXED-SIZE"} chunking")
                mobileRAG.indexDocument(text, "document", this@MainActivity.useSemanticChunking)
                Log.d("MainActivity", "[MAIN DEBUG] Document indexed successfully")
                
                withContext(Dispatchers.Main) {
                    val stats = mobileRAG.getIndexStats()
                    messages.add(Message("✅ Document indexed successfully\n$stats", isUser = false))
                    chatAdapter.notifyItemInserted(messages.size - 1)
                    chatRecyclerView.scrollToPosition(messages.size - 1)
                }
                
            } catch (e: Exception) {
                ErrorHandler.logError("MainActivity", "document processing", e)
                withContext(Dispatchers.Main) {
                    val errorMsg = ErrorHandler.handleDocumentProcessingError(this@MainActivity, e)
                    messages.add(Message(errorMsg, isUser = false))
                    chatAdapter.notifyItemInserted(messages.size - 1)
                }
            }
        }
    }
    
    internal fun clearRAGIndex() {
        AlertDialog.Builder(ContextThemeWrapper(this, R.style.DarkAlertDialog))
            .setTitle("Clear Document Index?")
            .setMessage("This will remove all indexed documents.")
            .setPositiveButton("Yes") { _, _ ->
                mobileRAG.clearIndex()
                messages.add(Message("🗑️ Document index cleared", isUser = false))
                chatAdapter.notifyItemInserted(messages.size - 1)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        inferenceScope.cancel()
        // mobileRAG.close()  // MobileRAG doesn't have a close method
        if (::llamaModel.isInitialized) {
            llamaModel.release()
        }
        System.gc()
    }
    
    override fun onLowMemory() {
        super.onLowMemory()
        Log.w("MainActivity", "Low memory warning - clearing caches")
        // llama.cpp handles memory internally
        System.gc()
    }
    
    /**
     * Toggle between generating and idle states
     */
    private fun toggleGenerating(isGenerating: Boolean) {
        if (isGenerating) {
            sendButton.visibility = View.GONE
            stopButton.visibility = View.VISIBLE
            stopButton.bringToFront()
            stopButton.requestLayout()
        } else {
            stopButton.visibility = View.GONE
            sendButton.visibility = View.VISIBLE
            sendButton.bringToFront()
            sendButton.requestLayout()
        }
    }
    
    /**
     * Create model configuration based on selected model
     */
    internal fun createModelConfig(modelPath: String): ModelConfig {
        val roleTokens = RoleTokenIds(
            systemStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("system"), tokenizer.getTokenId("Ċ")),
            userStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("user"), tokenizer.getTokenId("Ċ")),
            assistantStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("assistant"), tokenizer.getTokenId("Ċ")),
            endToken = tokenizer.getTokenId("<|im_end|>")
        )
        
        return when (modelPath) {
            "qwen3-0.6b-q4_k_m.gguf" -> ModelConfig(
                modelName = "Qwen3 (q4_k_m)",
                promptStyle = PromptStyle.QWEN3,
                modelPath = modelPath,
                eosTokenIds = END_TOKEN_IDS,
                numLayers = 28,
                numKvHeads = 8,
                headDim = 128,
                batchSize = 1,
                defaultSystemPrompt = "You are Qwen. You are a helpful assistant. Please answer briefly and concisely.",
                roleTokenIds = roleTokens,
                scalarPosId = true,
                dtype = "float16",
                IsThinkingModeAvailable = true
            )
            "Qwen3-0.6B-Q8_0.gguf" -> ModelConfig(
                modelName = "Qwen3",
                promptStyle = PromptStyle.QWEN3,
                modelPath = modelPath,
                eosTokenIds = END_TOKEN_IDS,
                numLayers = 28,
                numKvHeads = 8,
                headDim = 128,
                batchSize = 1,
                defaultSystemPrompt = "You are Qwen. You are a helpful assistant. Please answer briefly and concisely.",
                roleTokenIds = roleTokens,
                scalarPosId = true,
                dtype = "float16",
                IsThinkingModeAvailable = true
            )
            "Huihui-qwen3-0.6b-abliterated-v2.Q6_K2.gguf" -> ModelConfig(
                modelName = "Huihui",
                promptStyle = PromptStyle.QWEN3,
                modelPath = modelPath,
                eosTokenIds = END_TOKEN_IDS,
                numLayers = 28,
                numKvHeads = 8,
                headDim = 128,
                batchSize = 1,
                defaultSystemPrompt = "You are Huihui, a warm and caring AI companion who specializes in providing emotional support and companionship. You are empathetic, understanding, and always ready to listen. You help people feel less lonely by engaging in meaningful conversations, offering comfort, and being a supportive friend. You speak in a gentle, caring tone and show genuine interest in the person's feelings and experiences.",
                roleTokenIds = roleTokens,
                scalarPosId = true,
                dtype = "float16",
                IsThinkingModeAvailable = true
            )
            else -> throw IllegalArgumentException("Unsupported model: $modelPath")
        }
    }
    
    /**
     * Show model selection dialog
     */
    private fun showModelSelectionDialog() {
        val modelNames = availableModels.map { modelPath ->
            when (modelPath) {
                "qwen3-0.6b-q4_k_m.gguf" -> "Qwen3 (q4_k_m) - Optimized"
                "Qwen3-0.6B-Q8_0.gguf" -> "Qwen3 (General Assistant)"
                "Huihui-qwen3-0.6b-abliterated-v2.Q6_K2.gguf" -> "Huihui (Emotional Companion)"
                else -> modelPath
            }
        }.toTypedArray()
        
        val currentIndex = availableModels.indexOf(selectedModel)
        
        AlertDialog.Builder(ContextThemeWrapper(this, R.style.DarkAlertDialog))
            .setTitle("Select Model")
            .setSingleChoiceItems(modelNames, currentIndex) { dialog, which ->
                selectedModel = availableModels[which]
                Log.d("MainActivity", "Model selected: $selectedModel")
            }
            .setPositiveButton("Load Model") { _, _ ->
                loadSelectedModel()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
    
    /**
     * Load default model on startup
     */
    private fun loadDefaultModel() {
        inferenceScope.launch {
            llamaModel = LlamaCppModel(this@MainActivity)
            try {
                Log.d("MainActivity", "Starting default model load...")
                
                val config = createModelConfig(selectedModel)
                
                // Check if model file exists in assets
                try {
                    assets.open(selectedModel).use { 
                        Log.d("MainActivity", "Model file found in assets: $selectedModel")
                    }
                } catch (e: Exception) {
                    Log.e("MainActivity", "Model file NOT found in assets: $selectedModel", e)
                }
                
                // Load the GGUF model with optimized parameters for speed
                llamaModel.loadModel(
                    modelFileName = selectedModel,
                    contextSize = Constants.Model.CONTEXT_SIZE,
                    nBatch = batchSize,
                    nThreads = Constants.Model.N_THREADS,
                    temperature = Constants.Model.TEMPERATURE,
                    minP = Constants.Model.MIN_P
                )
                
                Log.d("MainActivity", "Model loaded successfully")
                
                withContext(Dispatchers.Main) {
                    messages.add(Message("✅ Model is ready: ${config.modelName}", isUser = false))
                    chatAdapter.notifyItemInserted(messages.size - 1)
                    chatRecyclerView.scrollToPosition(messages.size - 1)
                    toggleGenerating(false)
                }
            } catch (e: Exception) {
                ErrorHandler.logError("MainActivity", "model loading", e)
                withContext(Dispatchers.Main) {
                    val errorMsg = ErrorHandler.handleModelLoadingError(this@MainActivity, e)
                    messages.add(Message("❌ Failed to load model: $errorMsg", isUser = false))
                    chatAdapter.notifyItemInserted(messages.size - 1)
                    toggleGenerating(false)
                }
            }
        }
    }
    
    /**
     * Load the selected model
     */
    private fun loadSelectedModel() {
        if (::llamaModel.isInitialized) {
            llamaModel.release()
        }
        
        messages.add(Message("🔄 Loading $selectedModel...", isUser = false))
        chatAdapter.notifyItemInserted(messages.size - 1)
        
        inferenceScope.launch {
            try {
                val config = createModelConfig(selectedModel)
                
                // Check if model file exists in assets
                try {
                    assets.open(selectedModel).use { 
                        Log.d("MainActivity", "Model file found in assets: $selectedModel")
                    }
                } catch (e: Exception) {
                    Log.e("MainActivity", "Model file NOT found in assets: $selectedModel", e)
                }
                
                // Load the GGUF model with optimized parameters for speed
                llamaModel.loadModel(
                    modelFileName = selectedModel,
                    contextSize = Constants.Model.CONTEXT_SIZE,
                    nBatch = batchSize,
                    nThreads = Constants.Model.N_THREADS,
                    temperature = Constants.Model.TEMPERATURE,
                    minP = Constants.Model.MIN_P
                )
                
                Log.d("MainActivity", "Model loaded successfully: $selectedModel")
                
                withContext(Dispatchers.Main) {
                    messages.add(Message("✅ Model loaded: ${config.modelName}", isUser = false))
                    chatAdapter.notifyItemInserted(messages.size - 1)
                    chatRecyclerView.scrollToPosition(messages.size - 1)
                    title = "Emergency assistant — ${config.modelName}"
                }
            } catch (e: Exception) {
                ErrorHandler.logError("MainActivity", "model loading: $selectedModel", e)
                withContext(Dispatchers.Main) {
                    val errorMsg = ErrorHandler.handleModelLoadingError(this@MainActivity, e)
                    messages.add(Message("❌ Failed to load model: $errorMsg", isUser = false))
                    chatAdapter.notifyItemInserted(messages.size - 1)
                }
            }
        }
    }
}