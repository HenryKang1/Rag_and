package com.example.local_llm

import android.util.Log
import kotlin.math.min

/**
 * 언어 블로킹을 위한 클래스
 * Qwen 모델에서 중국어, 일본어, 한국어 등 외국어 토큰을 차단
 */
class LanguageBlocker {
    
    companion object {
        private const val TAG = "LanguageBlocker"
        
        // 중국어 범위
        private val CHINESE_RANGES = listOf(
            0x4E00 to 0x9FFF,   // CJK Unified Ideographs
            0x3400 to 0x4DBF,   // CJK Unified Ideographs Extension A
            0x20000 to 0x2A6DF, // CJK Unified Ideographs Extension B
            0xF900 to 0xFAFF    // CJK Compatibility Ideographs
        )
        
        // 일본어 범위
        private val JAPANESE_RANGES = listOf(
            0x3040 to 0x309F,  // Hiragana
            0x30A0 to 0x30FF,  // Katakana
            0x31F0 to 0x31FF   // Katakana Phonetic Extensions
        )
        
        // 러시아어 범위
        private val RUSSIAN_RANGES = listOf(
            0x0400 to 0x04FF,  // Cyrillic
            0x0500 to 0x052F   // Cyrillic Supplement
        )
        
        // 남아시아 언어 범위
        private val SOUTH_ASIAN_RANGES = listOf(
            0x0900 to 0x097F,  // Devanagari
            0x0980 to 0x09FF,  // Bengali
            0x0A00 to 0x0A7F,  // Gurmukhi
            0x0A80 to 0x0AFF,  // Gujarati
            0x0B00 to 0x0B7F,  // Oriya
            0x0B80 to 0x0BFF,  // Tamil
            0x0C00 to 0x0C7F,  // Telugu
            0x0C80 to 0x0CFF,  // Kannada
            0x0D00 to 0x0D7F,  // Malayalam
            0x0D80 to 0x0DFF   // Sinhala
        )
        
        // 아랍어 범위
        private val ARABIC_RANGES = listOf(
            0x0600 to 0x06FF,  // Arabic
            0x0750 to 0x077F,  // Arabic Supplement
            0x08A0 to 0x08FF   // Arabic Extended-A
        )
        
        // 한국어 범위
        private val KOREAN_RANGES = listOf(
            0xAC00 to 0xD7AF,  // Hangul Syllables
            0x1100 to 0x11FF,  // Hangul Jamo
            0x3130 to 0x318F   // Hangul Compatibility Jamo
        )
        
        // 모든 외국어 범위 통합
        private val ALL_FOREIGN_RANGES = CHINESE_RANGES + JAPANESE_RANGES + 
                                       RUSSIAN_RANGES + SOUTH_ASIAN_RANGES + 
                                       ARABIC_RANGES + KOREAN_RANGES
    }
    
    private var foreignLangMask: BooleanArray? = null
    private var tokenizer: BpeTokenizer? = null
    
    /**
     * 토크나이저를 설정하고 외국어 마스크를 초기화
     */
    fun initialize(tokenizer: BpeTokenizer) {
        this.tokenizer = tokenizer
        initializeForeignLangMask()
        LoggingUtils.logRAGDebug(TAG, "Language blocker initialized with ${foreignLangMask?.size ?: 0} tokens")
    }
    
    /**
     * 외국어 토큰 마스크를 초기화
     */
    private fun initializeForeignLangMask() {
        tokenizer?.let { tok ->
            val vocabSize = tok.vocabSize
            foreignLangMask = BooleanArray(vocabSize) { false }
            
            LoggingUtils.logRAGDebug(TAG, "Initializing foreign language mask for vocab size: $vocabSize")
            
            // 각 토큰에 대해 외국어 문자 포함 여부 확인
            for (tokenId in 0 until vocabSize) {
                try {
                    val token = tok.decode(intArrayOf(tokenId))
                    if (containsForeignLanguage(token)) {
                        foreignLangMask!![tokenId] = true
                    }
                } catch (e: Exception) {
                    // 토큰 디코딩 실패 시 무시
                    LoggingUtils.logRAGDebug(TAG, "Failed to decode token $tokenId: ${e.message}")
                }
            }
            
            val blockedCount = foreignLangMask!!.count { it }
            LoggingUtils.logRAGDebug(TAG, "Blocked $blockedCount foreign language tokens out of $vocabSize")
        }
    }
    
    /**
     * 텍스트에 외국어 문자가 포함되어 있는지 확인
     */
    private fun containsForeignLanguage(text: String): Boolean {
        return text.any { char ->
            val codePoint = char.code
            ALL_FOREIGN_RANGES.any { (start, end) ->
                codePoint in start..end
            }
        }
    }
    
    /**
     * 로짓 점수에 외국어 토큰 마스킹 적용
     * @param logits 로짓 점수 배열
     * @return 마스킹이 적용된 로짓 점수 배열
     */
    fun applyLanguageBlocking(logits: FloatArray): FloatArray {
        val mask = foreignLangMask ?: return logits
        val vocabSize = mask.size
        val actualSize = logits.size
        
        LoggingUtils.logRAGDebug(TAG, "Applying language blocking: vocab=$vocabSize, actual=$actualSize")
        
        // 마스크 크기와 로짓 크기가 다른 경우 처리
        val effectiveSize = min(vocabSize, actualSize)
        
        for (i in 0 until effectiveSize) {
            if (mask[i]) {
                logits[i] = Float.NEGATIVE_INFINITY
            }
        }
        
        return logits
    }
    
    /**
     * 토큰 ID 배열에 언어 블로킹 적용
     * @param tokenIds 토큰 ID 배열
     * @return 필터링된 토큰 ID 배열
     */
    fun filterForeignTokens(tokenIds: IntArray): IntArray {
        val mask = foreignLangMask ?: return tokenIds
        val vocabSize = mask.size
        
        return tokenIds.filter { tokenId ->
            if (tokenId >= vocabSize) {
                true // 범위를 벗어난 토큰은 허용
            } else {
                !mask[tokenId] // 외국어 토큰이 아닌 경우만 허용
            }
        }.toIntArray()
    }
    
    /**
     * 생성된 텍스트에서 외국어 부분을 제거
     * @param text 원본 텍스트
     * @return 외국어가 제거된 텍스트
     */
    fun removeForeignLanguage(text: String): String {
        return text.filter { char ->
            val codePoint = char.code
            !ALL_FOREIGN_RANGES.any { (start, end) ->
                codePoint in start..end
            }
        }
    }
    
    /**
     * 언어 블로킹 통계 정보 반환
     */
    fun getBlockingStats(): String {
        val mask = foreignLangMask ?: return "Language blocker not initialized"
        val totalTokens = mask.size
        val blockedTokens = mask.count { it }
        val blockedPercentage = if (totalTokens > 0) (blockedTokens * 100.0 / totalTokens) else 0.0
        
        return "Language Blocker Stats:\n" +
               "- Total tokens: $totalTokens\n" +
               "- Blocked tokens: $blockedTokens\n" +
               "- Blocked percentage: ${String.format("%.2f", blockedPercentage)}%"
    }
}
