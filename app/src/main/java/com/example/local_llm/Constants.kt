package com.example.local_llm

/**
 * 애플리케이션 전반에서 사용되는 상수들을 정의
 */
object Constants {
    
    // Model Configuration - Optimized for speed
    object Model {
        const val DEFAULT_MODEL = "Qwen3-0.6B-Q8_0.gguf"
        const val CONTEXT_SIZE = 2048
    const val BATCH_SIZE_DEFAULT = 1024
    const val BATCH_SIZE_OFF = 1
        const val N_THREADS = 4      // Reduced to avoid thermal throttling
        const val TEMPERATURE = 0.4f   // Balanced for speed and quality
        const val MIN_P = 0.05f       // Balanced for speed and quality
    }
    
    // RAG Configuration - Optimized for speed
    object RAG {
        const val CHUNK_SIZE = 128
        const val CHUNK_OVERLAP = 16
        const val TOP_K = 4  // Balanced for quality and speed
        const val EMBEDDING_DIM = 384
        const val MAX_CONTEXT_CHARS = 3000
        const val MAX_CONTEXT_TOKENS = 300  // 토큰 단위 제한 (프리필 속도 개선)
        const val MIN_CHUNK_SIZE = 50
        const val MAX_CHUNK_SIZE = 250  // Balanced chunk size
        const val SEMANTIC_OVERLAP = 30
        const val OPTIMAL_MOBILE_CONTEXT = 1800  // Sweet spot for mobile (good speed)
    }
    
    // File Configuration
    object File {
        const val INDEX_FILE = "rag_index.bin"
        const val CHUNKS_FILE = "rag_chunks.bin"
        const val MAX_FILE_SIZE_MB = 10
    }
    
    // UI Configuration
    object UI {
        const val SCROLL_DELAY_MS = 200
        const val MEMORY_CHECK_DELAY_MS = 300
        const val CLEANUP_DELAY_MS = 50
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
        const val MAX_TOKENS_1024 = 1024
        const val MAX_TOKENS_1536 = 1536
    }
    
    // Memory Configuration
    object Memory {
        const val LOW_MEMORY_THRESHOLD = 70
        const val MODERATE_MEMORY_THRESHOLD = 80
        const val HIGH_MEMORY_THRESHOLD = 90
        const val MEMORY_BOOST_FACTOR = 2.5f
    }
    
    // Logging Configuration
    object Logging {
        const val CHUNK_SIZE = 1000
        const val CONTEXT_CHUNK_SIZE = 800
        const val PROMPT_CHUNK_SIZE = 1000
        const val RAG_CHUNK_SIZE = 1000
    }
}

