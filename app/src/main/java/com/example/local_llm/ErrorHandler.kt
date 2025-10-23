package com.example.local_llm

import android.content.Context
import android.util.Log
import android.widget.Toast

/**
 * 애플리케이션 전반의 예외 처리를 담당하는 유틸리티 클래스
 */
object ErrorHandler {
    
    private const val TAG = "ErrorHandler"
    
    /**
     * 모델 로딩 관련 에러 처리
     */
    fun handleModelLoadingError(context: Context, error: Exception): String {
        return when {
            error.message?.contains("not found", ignoreCase = true) == true -> 
                "Model file not found in assets"
            error.message?.contains("memory", ignoreCase = true) == true -> 
                "Not enough memory to load model"
            else -> error.message ?: "Unknown error"
        }
    }
    
    /**
     * RAG 관련 에러 처리
     */
    fun handleRAGError(context: Context, error: Exception): String {
        return when {
            error.message?.contains("memory", ignoreCase = true) == true -> 
                "❌ Memory error. Try shorter prompts or disable RAG."
            error.message?.contains("not found", ignoreCase = true) == true -> 
                "❌ RAG index not found. Please upload documents first."
            else -> "❌ RAG Error: ${error.message ?: "Unknown error."}"
        }
    }
    
    /**
     * 문서 처리 관련 에러 처리
     */
    fun handleDocumentProcessingError(context: Context, error: Exception): String {
        return when {
            error.message?.contains("file too large", ignoreCase = true) == true -> 
                "❌ File too large (max ${Constants.File.MAX_FILE_SIZE_MB}MB)"
            error.message?.contains("unsupported format", ignoreCase = true) == true -> 
                "❌ Unsupported file format"
            else -> "❌ Failed to process document: ${error.message}"
        }
    }
    
    /**
     * 메모리 관련 에러 처리
     */
    fun handleMemoryError(context: Context, error: OutOfMemoryError): String {
        Log.e(TAG, "OutOfMemoryError: ${error.message}")
        return "❌ Out of memory! Try shorter prompts or disable RAG."
    }
    
    /**
     * 일반적인 에러 처리
     */
    fun handleGeneralError(context: Context, error: Exception, operation: String): String {
        Log.e(TAG, "Error in $operation: ${error.message}", error)
        return "❌ Error in $operation: ${error.message ?: "Unknown error."}"
    }
    
    /**
     * 에러 메시지를 토스트로 표시
     */
    fun showErrorToast(context: Context, message: String) {
        Toast.makeText(context, message, Toast.LENGTH_LONG).show()
    }
    
    /**
     * 에러 메시지를 로그에 기록
     */
    fun logError(tag: String, operation: String, error: Exception) {
        Log.e(tag, "Error in $operation: ${error.message}", error)
    }
}

