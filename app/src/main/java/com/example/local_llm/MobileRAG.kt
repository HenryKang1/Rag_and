package com.example.local_llm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Mobile RAG (Retrieval-Augmented Generation) implementation for Android
 * Provides document indexing and semantic search capabilities
 */
class MobileRAG(private val context: Context) {
    
    companion object {
        const val TAG = "MobileRAG"
    }
    
    enum class ChunkingMode {
        FIXED_SIZE,    // Original fixed-size chunking
        SEMANTIC       // Topic-based semantic chunking
    }
    
    // Current chunking mode (can be changed for testing)
    var chunkingMode = ChunkingMode.FIXED_SIZE
    
    private var embeddings: Array<FloatArray>? = null
    private var chunks: List<String> = emptyList()
    private var embeddingModel: EmbeddingModel? = null
    private var isInitialized = false
    
    data class SearchResult(
        val chunk: String,
        val score: Float,
        val chunkId: Int
    )
    
    /**
     * Initialize the RAG system with embedding model
     */
    suspend fun initialize() = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Initializing RAG system...")
            embeddingModel = EmbeddingModel(context)
            embeddingModel?.initialize()
            
            // Load existing index if available
            loadIndex()
            
            isInitialized = true
            Log.d(TAG, "RAG system initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize RAG: ${e.message}")
            throw e
        }
    }
    
    /**
     * Index a document by creating embeddings for chunks
     * @param useSemanticChunking - if true, uses semantic chunking; otherwise uses fixed-size chunking
     */
    suspend fun indexDocument(text: String, docName: String = "document", useSemanticChunking: Boolean = false) = withContext(Dispatchers.IO) {
        if (!isInitialized) {
            throw IllegalStateException("RAG system not initialized")
        }
        
        LoggingUtils.logSectionStart(TAG, "INDEXING")
        LoggingUtils.logRAGDebug(TAG, "Indexing document: $docName")
        LoggingUtils.logRAGDebug(TAG, "Document length: ${text.length} characters")
        
        // Choose chunking method
        chunkingMode = if (useSemanticChunking) ChunkingMode.SEMANTIC else ChunkingMode.FIXED_SIZE
        LoggingUtils.logRAGDebug(TAG, "Using ${chunkingMode} chunking mode")
        
        // Create chunks with overlap
        val newChunks = when (chunkingMode) {
            ChunkingMode.SEMANTIC -> createSemanticChunks(text)
            ChunkingMode.FIXED_SIZE -> createChunks(text)
        }
        LoggingUtils.logRAGDebug(TAG, "Created ${newChunks.size} chunks using ${chunkingMode} mode")
        
        // Debug: Find which chunks contain important keywords
        val keyTerms = listOf("cholera", "avpu", "shock", "dehydration")
        newChunks.forEachIndexed { idx, chunk ->
            for (term in keyTerms) {
                if (chunk.contains(term, ignoreCase = true)) {
                    LoggingUtils.logRAGDebug(TAG, "⭐ Chunk $idx contains '$term'")
                    LoggingUtils.logRAGDebug(TAG, "Chunk $idx content: ${chunk.take(300)}...")
                }
            }
        }
        
        newChunks.take(3).forEachIndexed { idx, chunk ->
            LoggingUtils.logRAGDebug(TAG, "Chunk ${idx+1} preview: ${chunk.take(100).replace("\n", " ")}...")
        }
        
        // Generate embeddings for chunks
        val newEmbeddings = mutableListOf<FloatArray>()
        for (chunk in newChunks) {
            val embedding = embeddingModel?.encode(chunk) 
                ?: throw IllegalStateException("Embedding model not initialized")
            newEmbeddings.add(embedding)
        }
        
        // Append to existing index or create new
        if (embeddings != null) {
            val combined = embeddings!!.toMutableList()
            combined.addAll(newEmbeddings)
            embeddings = combined.toTypedArray()
            chunks = chunks + newChunks
        } else {
            embeddings = newEmbeddings.toTypedArray()
            chunks = newChunks
        }
        
        // Save index to storage
        saveIndex()
        
        LoggingUtils.logRAGDebug(TAG, "Document indexed successfully. Total chunks: ${chunks.size}")
        LoggingUtils.logSectionEnd(TAG, "INDEXING")
    }
    
    /**
     * Search for relevant chunks given a query
     */
    suspend fun search(query: String, k: Int = Constants.RAG.TOP_K): List<SearchResult> = withContext(Dispatchers.IO) {
        if (!isInitialized || embeddings == null || chunks.isEmpty()) {
            Log.w(TAG, "[RAG DEBUG] No documents indexed yet")
            Log.w(TAG, "[RAG DEBUG] isInitialized: $isInitialized, embeddings: ${embeddings?.size ?: 0}, chunks: ${chunks.size}")
            return@withContext emptyList()
        }
        
        val searchStartTime = System.currentTimeMillis()
        LoggingUtils.logSectionStart(TAG, "SEARCH")
        LoggingUtils.logRAGDebug(TAG, "Query: '$query'")
        LoggingUtils.logRAGDebug(TAG, "Total chunks available: ${chunks.size}")
        
        // === 1. 키워드 스캔 단계 ===
        val keywordScanStartTime = System.currentTimeMillis()
        LoggingUtils.logRAGDebug(TAG, "=== KEYWORD SCAN PHASE ===")
        LoggingUtils.logRAGDebug(TAG, "Performing keyword search...")
        val keywordResults = mutableListOf<Pair<Int, Float>>()
        val queryLower = query.lowercase()
        val importantKeywords = extractImportantKeywords(queryLower)
        LoggingUtils.logRAGDebug(TAG, "Important keywords: $importantKeywords")
        
        // If no important keywords found, skip keyword search
        if (importantKeywords.isEmpty()) {
            LoggingUtils.logRAGDebug(TAG, "No specific keywords found, using embedding search only")
        } else {
            // 모든 청크를 소문자로 변환한 뒤 키워드와 contains 비교
            chunks.forEachIndexed { index, chunk ->
                val chunkLower = chunk.lowercase()
                
                // Count how many keywords match
                val matchingKeywords = importantKeywords.filter { keyword ->
                    chunkLower.contains(keyword)
                }
                
                // Only add if at least one keyword matches
                if (matchingKeywords.isNotEmpty()) {
                    // Score based on number of matching keywords
                    val keywordScore = matchingKeywords.size.toFloat() / importantKeywords.size
                    
                    // Special boost for exact matches of primary terms
                    val primaryBoost = when {
                        queryLower.contains("avpu") && chunkLower.contains("avpu scale") -> 2.0f
                        queryLower.contains("cholera") && chunkLower.contains("cholera") -> 2.0f
                        queryLower.contains("gcs") && chunkLower.contains("glasgow coma scale") -> 2.0f
                        else -> 1.0f
                    }
                    
                    val finalScore = keywordScore * primaryBoost
                    
                    // Only log high-scoring matches
                    if (finalScore > 0.5f && keywordResults.size < 20) {
                        LoggingUtils.logRAGDebug(TAG, "Chunk $index: keywords=${matchingKeywords}, score=$finalScore")
                        if (keywordResults.size < 5) {
                            LoggingUtils.logRAGDebug(TAG, "Preview: ${chunk.take(150)}...")
                        }
                    }
                    
                    keywordResults.add(index to finalScore)
                }
            }
            
            // Sort by score and keep top matches
            keywordResults.sortByDescending { it.second }
            val topKeywordResults = keywordResults.take(minOf(50, keywordResults.size))  // Limit to top 50
            keywordResults.clear()
            keywordResults.addAll(topKeywordResults)
        }
        
        val keywordScanEndTime = System.currentTimeMillis()
        val keywordScanTime = keywordScanEndTime - keywordScanStartTime
        LoggingUtils.logRAGDebug(TAG, "=== KEYWORD SCAN COMPLETED ===")
        LoggingUtils.logRAGDebug(TAG, "- Total chunks scanned: ${chunks.size}")
        LoggingUtils.logRAGDebug(TAG, "- Chunks containing keywords: ${keywordResults.size}")
        LoggingUtils.logRAGDebug(TAG, "- Keyword scan time: ${keywordScanTime}ms")
        LoggingUtils.logRAGDebug(TAG, "- Scan speed: ${chunks.size * 1000.0 / keywordScanTime} chunks/sec")
        
        // PerformanceLogger를 사용한 상세 분석
        PerformanceLogger.logSearchPerformance(
            keywordScanTime = keywordScanTime,
            embeddingTime = 0L, // 아직 측정되지 않음
            sortingTime = 0L,  // 아직 측정되지 않음
            totalChunks = chunks.size,
            keywordMatches = keywordResults.size,
            embeddingCount = 0
        )
        
        // If we found direct keyword matches, prioritize them
        if (keywordResults.isNotEmpty()) {
            LoggingUtils.logRAGDebug(TAG, "Found ${keywordResults.size} keyword matches, using them as priority")
        }
        
        // === 2. 임베딩 유사도 계산 단계 ===
        val embeddingStartTime = System.currentTimeMillis()
        LoggingUtils.logRAGDebug(TAG, "=== EMBEDDING SIMILARITY PHASE ===")
        LoggingUtils.logRAGDebug(TAG, "Encoding query for semantic search...")
        val queryEmbedding = embeddingModel?.encode(query)
            ?: throw IllegalStateException("Embedding model not initialized")
        LoggingUtils.logRAGDebug(TAG, "Query embedding generated, dimension: ${queryEmbedding.size}")
        
        // Calculate cosine similarities
        LoggingUtils.logRAGDebug(TAG, "Calculating similarities with ${embeddings?.size ?: 0} chunks...")
        val scores = mutableListOf<Pair<Int, Float>>()
        embeddings?.forEachIndexed { index, chunkEmbedding ->
            val similarity = cosineSimilarity(queryEmbedding, chunkEmbedding)
            scores.add(index to similarity)
            if (index < 5) {  // Log first 5 for debugging
                LoggingUtils.logRAGDebug(TAG, "Chunk $index similarity: $similarity")
            }
        }
        
        val embeddingEndTime = System.currentTimeMillis()
        val embeddingTime = embeddingEndTime - embeddingStartTime
        val embeddingCount = embeddings?.size ?: 0
        val operationsPerSecond = embeddingCount * 384 * 1000.0 / embeddingTime  // 384차원 벡터 연산
        LoggingUtils.logRAGDebug(TAG, "=== EMBEDDING SIMILARITY COMPLETED ===")
        LoggingUtils.logRAGDebug(TAG, "- Embeddings processed: $embeddingCount")
        LoggingUtils.logRAGDebug(TAG, "- Embedding similarity time: ${embeddingTime}ms")
        LoggingUtils.logRAGDebug(TAG, "- Operations per second: ${String.format("%.0f", operationsPerSecond)} ops/sec")
        LoggingUtils.logRAGDebug(TAG, "- Total operations: ${embeddingCount * 384} (${embeddingCount} × 384 dimensions)")
        
        // === 3. 정렬과 재랭킹 단계 ===
        val sortingStartTime = System.currentTimeMillis()
        LoggingUtils.logRAGDebug(TAG, "=== SORTING AND RERANKING PHASE ===")
        
        // Sort and get top-k (remove duplicates)
        scores.sortByDescending { it.second }
        
        // Remove duplicate chunks with same content
        val uniqueResults = mutableListOf<Pair<Int, Float>>()
        val seenChunks = mutableSetOf<String>()
        
        for ((index, score) in scores) {
            val chunkContent = chunks[index].take(100) // Use first 100 chars as signature
            if (!seenChunks.contains(chunkContent)) {
                uniqueResults.add(index to score)
                seenChunks.add(chunkContent)
                if (uniqueResults.size >= k) break
            }
        }
        
        val topResults = uniqueResults
        LoggingUtils.logRAGDebug(TAG, "Top $k results (before reranking):")
        topResults.forEachIndexed { idx, (chunkIdx, score) ->
            val fullChunk = chunks[chunkIdx].replace("\n", " ")
            LoggingUtils.logRAGDebug(TAG, "  ${idx+1}. Chunk #$chunkIdx (score: $score):")
            LoggingUtils.logLongText(TAG, "[RAG DEBUG]", fullChunk, Constants.Logging.CONTEXT_CHUNK_SIZE)
        }
        
        // Combine keyword results with embedding results
        val combinedResults = if (keywordResults.isNotEmpty()) {
            LoggingUtils.logRAGDebug(TAG, "Prioritizing ${keywordResults.size} keyword matches")
            
            // Create a map to track best scores
            val scoreMap = mutableMapOf<Int, Float>()
            
            // Add keyword matches with high boost
            keywordResults.forEach { (idx, _) ->
                // Get the embedding score for this chunk if available
                val embScore = topResults.find { it.first == idx }?.second ?: 0.5f
                // Combine: 80% keyword match + 20% embedding score
                scoreMap[idx] = 0.8f + (embScore * 0.2f)
            }
            
            // Add top embedding results with lower scores
            topResults.forEach { (idx, score) ->
                if (idx !in scoreMap) {
                    scoreMap[idx] = score * 0.5f  // Reduce score for non-keyword matches
                }
            }
            
            // Sort and take top k
            val results = scoreMap.entries
                .sortedByDescending { it.value }
                .take(k)
                .map { it.key to it.value }
            
            LoggingUtils.logRAGDebug(TAG, "Combined results: ${results.size} chunks")
            results
        } else {
            LoggingUtils.logRAGDebug(TAG, "No keyword matches found, using embedding similarity only")
            // Apply normal reranking with keyword matching
            rerankWithKeywords(query, topResults)
        }
        
        val rerankedResults = combinedResults
        val sortingEndTime = System.currentTimeMillis()
        val sortingTime = sortingEndTime - sortingStartTime
        
        LoggingUtils.logRAGDebug(TAG, "=== SORTING AND RERANKING COMPLETED ===")
        LoggingUtils.logRAGDebug(TAG, "- Sorting and reranking time: ${sortingTime}ms")
        LoggingUtils.logRAGDebug(TAG, "- Results processed: ${rerankedResults.size}")
        
        LoggingUtils.logRAGDebug(TAG, "After reranking:")
        rerankedResults.take(3).forEachIndexed { idx, (chunkIdx, score) ->
            val preview = chunks[chunkIdx].take(100).replace("\n", " ")
            LoggingUtils.logRAGDebug(TAG, "  ${idx+1}. Chunk #$chunkIdx (final score: $score): $preview...")
        }
        LoggingUtils.logRAGDebug(TAG, "Found ${rerankedResults.size} relevant chunks")
        
        // === 전체 검색 성능 요약 ===
        val searchEndTime = System.currentTimeMillis()
        val totalSearchTime = searchEndTime - searchStartTime
        LoggingUtils.logRAGDebug(TAG, "=== SEARCH PERFORMANCE SUMMARY ===")
        LoggingUtils.logRAGDebug(TAG, "- Total search time: ${totalSearchTime}ms")
        LoggingUtils.logRAGDebug(TAG, "- Keyword scan: ${keywordScanTime}ms (${String.format("%.1f", keywordScanTime * 100.0 / totalSearchTime)}%)")
        LoggingUtils.logRAGDebug(TAG, "- Embedding similarity: ${embeddingTime}ms (${String.format("%.1f", embeddingTime * 100.0 / totalSearchTime)}%)")
        LoggingUtils.logRAGDebug(TAG, "- Sorting and reranking: ${sortingTime}ms (${String.format("%.1f", sortingTime * 100.0 / totalSearchTime)}%)")
        LoggingUtils.logRAGDebug(TAG, "- Total chunks processed: ${chunks.size}")
        LoggingUtils.logRAGDebug(TAG, "- Search throughput: ${chunks.size * 1000.0 / totalSearchTime} chunks/sec")
        
        // PerformanceLogger를 사용한 최종 검색 성능 분석
        PerformanceLogger.logSearchPerformance(
            keywordScanTime = keywordScanTime,
            embeddingTime = embeddingTime,
            sortingTime = sortingTime,
            totalChunks = chunks.size,
            keywordMatches = keywordResults.size,
            embeddingCount = embeddings?.size ?: 0
        )
        
        LoggingUtils.logSectionEnd(TAG, "SEARCH")
        
        return@withContext rerankedResults.map { (index, score) ->
            SearchResult(
                chunk = chunks[index],
                score = score,
                chunkId = index
            )
        }
    }
    
    /**
     * Get augmented context for generation with token-based limiting
     * @param query User query
     * @param k Number of top results to retrieve
     * @param tokenCounter Optional token counter function (text -> token count)
     * @param maxTokens Maximum tokens allowed (0 = use character limit only)
     */
    suspend fun getContext(
        query: String, 
        k: Int = Constants.RAG.TOP_K,
        tokenCounter: ((String) -> Int)? = null,
        maxTokens: Int = Constants.RAG.MAX_CONTEXT_TOKENS
    ): String = withContext(Dispatchers.IO) {
        LoggingUtils.logSectionStart(TAG, "GET CONTEXT")
        LoggingUtils.logRAGDebug(TAG, "Getting context for: '$query'")
        LoggingUtils.logRAGDebug(TAG, "Max tokens limit: $maxTokens (${if (tokenCounter != null) "enabled" else "disabled"})")
        
        val results = search(query, k)
        if (results.isEmpty()) {
            Log.w(TAG, "[RAG DEBUG] No results found, returning empty context")
            return@withContext ""
        }
        
        LoggingUtils.logRAGDebug(TAG, "Building context from ${results.size} chunks")
        
        // Build context with size limit (characters and optionally tokens)
        val contextBuilder = StringBuilder()
        var currentTokenCount = 0
        val useTokenLimit = tokenCounter != null && maxTokens > 0
        
        for ((index, result) in results.withIndex()) {
            val chunk = result.chunk.trim()
            val separator = if (contextBuilder.isNotEmpty()) "\n\n" else ""
            val potentialAddition = separator + chunk
            
            // Check character limit
            if (contextBuilder.length + potentialAddition.length > Constants.RAG.MAX_CONTEXT_CHARS) {
                val remainingSpace = Constants.RAG.MAX_CONTEXT_CHARS - contextBuilder.length - separator.length
                if (remainingSpace > 100) {
                    contextBuilder.append(separator)
                    val truncated = chunk.take(remainingSpace - 3)
                    contextBuilder.append(truncated)
                    contextBuilder.append("...")
                    LoggingUtils.logRAGDebug(TAG, "Context truncated at chunk ${index + 1} (char limit)")
                }
                break
            }
            
            // Check token limit if token counter is available
            if (useTokenLimit) {
                val chunkTokens = tokenCounter!!(potentialAddition)
                if (currentTokenCount + chunkTokens > maxTokens) {
                    // Try to add partial chunk if there's room
                    val remainingTokens = maxTokens - currentTokenCount
                    if (remainingTokens > 10) { // Only add if meaningful tokens left (>10)
                        // Estimate characters per token (rough: 4 chars/token)
                        val estimatedChars = remainingTokens * 4
                        val truncated = chunk.take(min(estimatedChars.toInt(), chunk.length))
                        if (truncated.isNotEmpty() && truncated.length > 50) {
                            contextBuilder.append(separator)
                            contextBuilder.append(truncated)
                            contextBuilder.append("...")
                            currentTokenCount += tokenCounter(separator + truncated + "...")
                            LoggingUtils.logRAGDebug(TAG, "Context truncated at chunk ${index + 1} (token limit: $currentTokenCount/$maxTokens)")
                        }
                    }
                    break
                }
                currentTokenCount += chunkTokens
            }
            
            contextBuilder.append(separator)
            contextBuilder.append(chunk)
        }
        
        val context = contextBuilder.toString()
        
        LoggingUtils.logRAGDebug(TAG, "Final context length: ${context.length} characters")
        if (useTokenLimit) {
            val finalTokens = tokenCounter!!(context)
            LoggingUtils.logRAGDebug(TAG, "Final context tokens: $finalTokens (limit: $maxTokens)")
        }
        LoggingUtils.logRAGDebug(TAG, "Full context being returned:")
        LoggingUtils.logLongText(TAG, "[RAG DEBUG] Context", context, Constants.Logging.CONTEXT_CHUNK_SIZE)
        LoggingUtils.logSectionEnd(TAG, "GET CONTEXT")
        
        // Return pure context without "Context:" prefix (buildRAGPrompt will add it)
        return@withContext context
    }
    
    /**
     * Create chunks from text with overlap
     */
    private fun createChunks(text: String): List<String> {
        // Clean text first to fix PDF extraction issues
        val cleanedText = text
            .replace(Regex("\\s+"), " ")  // Replace multiple spaces with single space
            .replace(Regex("([a-z])([A-Z])"), "$1 $2")  // Add space between camelCase
            .trim()
        
        // Split into sentences more carefully
        val sentencePattern = Regex("(?<=[.!?•])\\s+(?=[A-Z])|\\n\\n|\\n(?=[A-Z•])")
        val sentences = cleanedText.split(sentencePattern).filter { it.isNotBlank() }
        
        val chunks = mutableListOf<String>()
        val currentChunk = mutableListOf<String>()
        var currentWordCount = 0
        
        for (sentence in sentences) {
            val trimmedSentence = sentence.trim()
            if (trimmedSentence.isEmpty()) continue
            
            val words = trimmedSentence.split(Regex("\\s+"))
            
            // If adding this sentence would exceed chunk size
            if (currentWordCount + words.size > Constants.RAG.CHUNK_SIZE && currentChunk.isNotEmpty()) {
                // Save current chunk
                val chunkText = currentChunk.joinToString(" ").trim()
                if (chunkText.isNotEmpty()) {
                    chunks.add(chunkText)
                }
                
                // Create overlap from last sentences
                val overlapSentences = currentChunk.takeLast(2)  // Take last 2 sentences for overlap
                currentChunk.clear()
                currentChunk.addAll(overlapSentences)
                currentWordCount = overlapSentences.sumOf { it.split(Regex("\\s+")).size }
            }
            
            // Add sentence to current chunk
            currentChunk.add(trimmedSentence)
            currentWordCount += words.size
        }
        
        // Add remaining chunk
        if (currentChunk.isNotEmpty()) {
            val chunkText = currentChunk.joinToString(" ").trim()
            if (chunkText.isNotEmpty()) {
                chunks.add(chunkText)
            }
        }
        
        LoggingUtils.logRAGDebug(TAG, "Created ${chunks.size} chunks from ${sentences.size} sentences")
        
        return chunks
    }
    
    /**
     * Create semantic chunks based on topic boundaries
     */
    private fun createSemanticChunks(text: String): List<String> {
        LoggingUtils.logRAGDebug(TAG, "Starting semantic chunking...")
        
        // Clean text first
        val cleanedText = text
            .replace(Regex("\\s+"), " ")
            .replace(Regex("([a-z])([A-Z])"), "$1 $2")
            .trim()
        
        val chunks = mutableListOf<String>()
        val sections = detectTopicBoundaries(cleanedText)
        
        LoggingUtils.logRAGDebug(TAG, "Detected ${sections.size} topic sections")
        
        for ((idx, section) in sections.withIndex()) {
            val sectionChunks = splitSectionIntoChunks(section)
            chunks.addAll(sectionChunks)
            if (idx < 3) {  // Log first 3 sections for debugging
                LoggingUtils.logRAGDebug(TAG, "Section $idx created ${sectionChunks.size} chunks")
            }
        }
        
        LoggingUtils.logRAGDebug(TAG, "Semantic chunking complete: ${chunks.size} total chunks")
        return chunks
    }
    
    /**
     * Detect topic boundaries in text (medical document optimized)
     */
    private fun detectTopicBoundaries(text: String): List<String> {
        val sections = mutableListOf<String>()
        
        // Split by common medical document headers
        val majorSplitPattern = Regex(
            "(?i)(?=\\n\\s*(?:" +
            "MODULE \\d+|" +
            "CHAPTER \\d+|" +
            "SECTION \\d+|" +
            "\\d+\\.\\s+[A-Z][A-Za-z]+|" +  // Numbered sections
            "(?:SIGNS AND SYMPTOMS)|" +
            "(?:ASSESSMENT)|" +
            "(?:MANAGEMENT)|" +
            "(?:TREATMENT)|" +
            "(?:DIAGNOSIS)|" +
            "(?:KEY TERMS)|" +
            "(?:CONDITION)|" +
            "(?:Common causes)|" +
            "(?:Definition:)|" +
            "(?:ABCDE)|" +
            "(?:SAMPLE HISTORY)" +
            "))"
        )
        
        val splits = text.split(majorSplitPattern).filter { it.trim().length > 50 }
        
        for (split in splits) {
            // Further split by bullet points if section is too large
            if (split.length > 3000) {
                val bulletSplits = split.split(Regex("\\n\\s*[••●]")).filter { it.trim().length > 50 }
                sections.addAll(bulletSplits)
            } else {
                sections.add(split)
            }
        }
        
        // If no sections found, fall back to paragraph splitting
        if (sections.isEmpty()) {
            LoggingUtils.logRAGDebug(TAG, "No topic boundaries found, using paragraph splitting")
            sections.addAll(text.split(Regex("\\n\\n+")).filter { it.trim().length > 50 })
        }
        
        return sections
    }
    
    /**
     * Split a section into appropriately sized chunks
     */
    private fun splitSectionIntoChunks(section: String): List<String> {
        val chunks = mutableListOf<String>()
        val words = section.split(Regex("\\s+"))
        
        // If section is within size limits, return as single chunk
        if (words.size <= Constants.RAG.MAX_CHUNK_SIZE && words.size >= Constants.RAG.MIN_CHUNK_SIZE) {
            chunks.add(section)
            return chunks
        }
        
        // If too small, skip (will be merged with adjacent in real implementation)
        if (words.size < Constants.RAG.MIN_CHUNK_SIZE) {
            if (section.trim().isNotEmpty()) {
                chunks.add(section)  // Still add small chunks to not lose information
            }
            return chunks
        }
        
        // Split large sections by sentences
        val sentences = section.split(Regex("(?<=[.!?])\\s+")).filter { it.isNotBlank() }
        val currentChunk = mutableListOf<String>()
        var currentWordCount = 0
        
        for (sentence in sentences) {
            val sentenceWords = sentence.split(Regex("\\s+")).size
            
            if (currentWordCount + sentenceWords > Constants.RAG.MAX_CHUNK_SIZE && currentChunk.isNotEmpty()) {
                chunks.add(currentChunk.joinToString(" "))
                
                // Overlap: keep last sentence
                val overlap = if (currentChunk.size > 1) currentChunk.takeLast(1) else emptyList()
                currentChunk.clear()
                currentChunk.addAll(overlap)
                currentWordCount = overlap.sumOf { it.split(Regex("\\s+")).size }
            }
            
            currentChunk.add(sentence)
            currentWordCount += sentenceWords
        }
        
        // Add remaining chunk
        if (currentChunk.isNotEmpty() && currentWordCount >= Constants.RAG.MIN_CHUNK_SIZE / 2) {
            chunks.add(currentChunk.joinToString(" "))
        }
        
        return chunks
    }
    
    /**
     * Calculate cosine similarity between two vectors
     */
    private fun cosineSimilarity(vec1: FloatArray, vec2: FloatArray): Float {
        var dotProduct = 0f
        var norm1 = 0f
        var norm2 = 0f
        
        for (i in vec1.indices) {
            dotProduct += vec1[i] * vec2[i]
            norm1 += vec1[i] * vec1[i]
            norm2 += vec2[i] * vec2[i]
        }
        
        return if (norm1 > 0 && norm2 > 0) {
            dotProduct / (sqrt(norm1) * sqrt(norm2))
        } else {
            0f
        }
    }
    
    /**
     * Extract important keywords from query
     */
    private fun extractImportantKeywords(query: String): List<String> {
        val keywords = mutableListOf<String>()
        val queryLower = query.lowercase()
        
        LoggingUtils.logRAGDebug(TAG, "Extracting keywords from: '$queryLower'")
        
        // Key medical terms with their importance
        val medicalTerms = listOf(
            "avpu", "gcs", "glasgow", "cholera", "shock", "dehydration",
            "trauma", "breathing", "airway", "circulation", "disability", 
            "hemorrhage", "haemorrhage", "pneumothorax", "anaphylaxis",
            "mental", "status", "emergency", "infant", "child", "children",
            "vomiting", "diarrhoea", "fever", "infection"
        )
        
        // Add any medical terms found in the query
        for (term in medicalTerms) {
            if (queryLower.contains(term)) {
                keywords.add(term)
                LoggingUtils.logRAGDebug(TAG, "Found medical term: '$term'")
            }
        }
        
        // Look for uppercase acronyms (like AVPU, GCS)
        val acronymPattern = Regex("\\b[A-Z]{2,5}\\b")
        acronymPattern.findAll(query).forEach { match ->
            val acronym = match.value.lowercase()
            keywords.add(acronym)
            LoggingUtils.logRAGDebug(TAG, "Found acronym: '$acronym'")
        }
        
        // Extract key phrases
        when {
            queryLower.contains("definition of") -> {
                val afterDefinition = queryLower.substringAfter("definition of").trim()
                val targetWord = afterDefinition.split(" ").firstOrNull()?.replace("?", "")
                targetWord?.let {
                    keywords.add(it)
                    LoggingUtils.logRAGDebug(TAG, "Definition target: '$it'")
                }
            }
            queryLower.contains("what is") -> {
                val afterWhatIs = queryLower.substringAfter("what is").trim()
                    .replace("the", "").trim()
                val words = afterWhatIs.split(" ").filter { it.length > 2 }
                words.forEach { word ->
                    val cleanWord = word.replace("?", "")
                    if (cleanWord.isNotEmpty()) {
                        keywords.add(cleanWord)
                        LoggingUtils.logRAGDebug(TAG, "What is target: '$cleanWord'")
                    }
                }
            }
            queryLower.contains("causes of") -> {
                val afterCauses = queryLower.substringAfter("causes of").trim()
                val words = afterCauses.split(" ").filter { it.length > 2 }
                words.forEach { word ->
                    val cleanWord = word.replace("?", "")
                    if (cleanWord.isNotEmpty()) {
                        keywords.add(cleanWord)
                        LoggingUtils.logRAGDebug(TAG, "Causes target: '$cleanWord'")
                    }
                }
            }
        }
        
        LoggingUtils.logRAGDebug(TAG, "Final keywords extracted: ${keywords.distinct()}")
        return keywords.distinct()
    }
    
    /**
     * Rerank results using keyword matching
     */
    private fun rerankWithKeywords(query: String, results: List<Pair<Int, Float>>): List<Pair<Int, Float>> {
        val queryWords = query.lowercase().split(Regex("\\s+")).toSet()
        
        return results.map { (index, embScore) ->
            val chunkWords = chunks[index].lowercase().split(Regex("\\s+"))
            val keywordScore = queryWords.count { word -> 
                chunkWords.any { it.contains(word) }
            }.toFloat() / queryWords.size
            
            // Combine embedding score (70%) and keyword score (30%)
            val combinedScore = embScore * 0.7f + keywordScore * 0.3f
            index to combinedScore
        }.sortedByDescending { it.second }
    }
    
    /**
     * Save index to internal storage
     */
    private fun saveIndex() {
        try {
            // Save embeddings
            val embFile = File(context.filesDir, Constants.File.INDEX_FILE)
            ObjectOutputStream(embFile.outputStream()).use { out ->
                out.writeObject(embeddings)
            }
            
            // Save chunks
            val chunkFile = File(context.filesDir, Constants.File.CHUNKS_FILE)
            ObjectOutputStream(chunkFile.outputStream()).use { out ->
                out.writeObject(chunks)
            }
            
            Log.d(TAG, "Index saved successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save index: ${e.message}")
        }
    }
    
    /**
     * Load index from internal storage
     */
    private fun loadIndex() {
        try {
            val embFile = File(context.filesDir, Constants.File.INDEX_FILE)
            val chunkFile = File(context.filesDir, Constants.File.CHUNKS_FILE)
            
            if (embFile.exists() && chunkFile.exists()) {
                ObjectInputStream(embFile.inputStream()).use { input ->
                    embeddings = input.readObject() as Array<FloatArray>
                }
                
                ObjectInputStream(chunkFile.inputStream()).use { input ->
                    @Suppress("UNCHECKED_CAST")
                    chunks = input.readObject() as List<String>
                }
                
                Log.d(TAG, "Index loaded: ${chunks.size} chunks")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load index: ${e.message}")
        }
    }
    
    /**
     * Clear the index
     */
    fun clearIndex() {
        embeddings = null
        chunks = emptyList()
        
        File(context.filesDir, Constants.File.INDEX_FILE).delete()
        File(context.filesDir, Constants.File.CHUNKS_FILE).delete()
        
        Log.d(TAG, "Index cleared")
    }
    
    /**
     * Get index statistics
     */
    fun getIndexStats(): String {
        return "Indexed chunks: ${chunks.size}\n" +
               "Embedding dimensions: ${Constants.RAG.EMBEDDING_DIM}\n" +
               "Chunk size: ${Constants.RAG.CHUNK_SIZE} words\n" +
               "Overlap: ${Constants.RAG.CHUNK_OVERLAP} words"
    }
}