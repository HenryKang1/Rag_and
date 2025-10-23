package com.example.local_llm

import android.util.Log

/**
 * 로깅 관련 유틸리티 함수들
 */
object LoggingUtils {
    
    /**
     * 긴 텍스트를 청크 단위로 나누어 로깅
     */
    fun logLongText(tag: String, prefix: String, text: String, chunkSize: Int = Constants.Logging.CHUNK_SIZE) {
        if (text.length <= chunkSize) {
            Log.d(tag, "$prefix: $text")
            return
        }
        
        var offset = 0
        var chunkNum = 1
        while (offset < text.length) {
            val end = minOf(offset + chunkSize, text.length)
            Log.d(tag, "$prefix Part $chunkNum: '${text.substring(offset, end).replace("\n", "\\n")}'")
            offset = end
            chunkNum++
        }
    }
    
    /**
     * RAG 디버그 로깅
     */
    fun logRAGDebug(tag: String, message: String) {
        Log.d(tag, "[RAG DEBUG] $message")
    }
    
    /**
     * 메모리 디버그 로깅
     */
    fun logMemoryDebug(tag: String, message: String) {
        Log.d(tag, "[MEMORY DEBUG] $message")
    }
    
    /**
     * 토큰 디버그 로깅
     */
    fun logTokenDebug(tag: String, message: String) {
        Log.d(tag, "[TOKEN DEBUG] $message")
    }
    
    /**
     * 컨텍스트 디버그 로깅
     */
    fun logContextDebug(tag: String, message: String) {
        Log.d(tag, "[CONTEXT DEBUG] $message")
    }
    
    /**
     * 프롬프트 디버그 로깅
     */
    fun logPromptDebug(tag: String, message: String) {
        Log.d(tag, "[PROMPT DEBUG] $message")
    }
    
    /**
     * 생성 디버그 로깅
     */
    fun logGenerationDebug(tag: String, message: String) {
        Log.d(tag, "[GENERATION DEBUG] $message")
    }
    
    /**
     * 모드 디버그 로깅
     */
    fun logModeDebug(tag: String, message: String) {
        Log.d(tag, "[MODE DEBUG] $message")
    }
    
    /**
     * 섹션 구분자 로깅
     */
    fun logSectionStart(tag: String, sectionName: String) {
        Log.d(tag, "========== $sectionName START ==========")
    }
    
    fun logSectionEnd(tag: String, sectionName: String) {
        Log.d(tag, "========== $sectionName END ==========")
    }
}
