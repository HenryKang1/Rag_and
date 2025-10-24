package com.example.local_llm

/**
 * 애플리케이션 전반에서 사용되는 상수들을 정의
 */
object Constants {
    
    // Model Configuration - Optimized for speed
    object Model {
        const val DEFAULT_MODEL = "Qwen3-0.6B-Q8_0.gguf"
        const val CONTEXT_SIZE = 2048  // Reduced for faster inference
        const val N_THREADS = 6        // Increased for better parallelization
        const val TEMPERATURE = 0.5f   // Lower for more focused responses
        const val MIN_P = 0.1f        // Slightly higher for better quality
    }
    
    // RAG Configuration - Optimized for speed
    object RAG {
        const val CHUNK_SIZE = 128
        const val CHUNK_OVERLAP = 16
        const val TOP_K = 5
        const val EMBEDDING_DIM = 384
        const val MAX_CONTEXT_CHARS = 3000
        const val MIN_CHUNK_SIZE = 50
        const val MAX_CHUNK_SIZE = 300
        const val SEMANTIC_OVERLAP = 30
    }
    
    // File Configuration
    object File {
        const val INDEX_FILE = "rag_index.bin"
        const val CHUNKS_FILE = "rag_chunks.bin"
        const val MAX_FILE_SIZE_MB = 10
    }
    
    // UI Configuration
    object UI {
        const val SCROLL_DELAY_MS = 300
        const val MEMORY_CHECK_DELAY_MS = 500
        const val CLEANUP_DELAY_MS = 100
    }
    
    // Token Configuration
    object Token {
        val END_TOKEN_IDS = setOf(151643, 151645)
        const val MAX_TOKENS_AUTO = 0
        const val MAX_TOKENS_128 = 128
        const val MAX_TOKENS_256 = 256
        const val MAX_TOKENS_384 = 384
        const val MAX_TOKENS_512 = 512
        const val MAX_TOKENS_768 = 768
    }
    
    // Memory Configuration
    object Memory {
        const val LOW_MEMORY_THRESHOLD = 50
        const val MODERATE_MEMORY_THRESHOLD = 70
        const val HIGH_MEMORY_THRESHOLD = 80
        const val MEMORY_BOOST_FACTOR = 1.5f
    }
    
    // Logging Configuration
    object Logging {
        const val CHUNK_SIZE = 1000
        const val CONTEXT_CHUNK_SIZE = 800
        const val PROMPT_CHUNK_SIZE = 1000
        const val RAG_CHUNK_SIZE = 1000
    }
}

