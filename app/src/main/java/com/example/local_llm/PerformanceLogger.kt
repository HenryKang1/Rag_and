package com.example.local_llm

import android.util.Log
import java.text.SimpleDateFormat
import java.util.*

/**
 * 모바일 RAG + LLM 성능 측정 및 로깅 유틸리티
 */
object PerformanceLogger {
    
    private const val TAG = "PerformanceLogger"
    private val dateFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault())
    
    data class PerformanceMetrics(
        val timestamp: Long,
        val phase: String,
        val duration: Long,
        val details: Map<String, Any> = emptyMap()
    )
    
    private val metrics = mutableListOf<PerformanceMetrics>()
    
    /**
     * 성능 측정 시작
     */
    fun startMeasurement(phase: String): Long {
        val startTime = System.currentTimeMillis()
        Log.d(TAG, "🚀 [$phase] Started at ${dateFormat.format(Date(startTime))}")
        return startTime
    }
    
    /**
     * 성능 측정 종료 및 로깅
     */
    fun endMeasurement(phase: String, startTime: Long, details: Map<String, Any> = emptyMap()) {
        val endTime = System.currentTimeMillis()
        val duration = endTime - startTime
        
        val metric = PerformanceMetrics(
            timestamp = startTime,
            phase = phase,
            duration = duration,
            details = details
        )
        metrics.add(metric)
        
        Log.d(TAG, "🏁 [$phase] Completed in ${duration}ms")
        details.forEach { (key, value) ->
            Log.d(TAG, "   - $key: $value")
        }
    }
    
    /**
     * 검색 성능 분석
     */
    fun logSearchPerformance(
        keywordScanTime: Long,
        embeddingTime: Long,
        sortingTime: Long,
        totalChunks: Int,
        keywordMatches: Int,
        embeddingCount: Int
    ) {
        val totalSearchTime = keywordScanTime + embeddingTime + sortingTime
        
        Log.d(TAG, "=== SEARCH PERFORMANCE ANALYSIS ===")
        Log.d(TAG, "📊 Search Metrics:")
        Log.d(TAG, "   - Total time: ${totalSearchTime}ms")
        Log.d(TAG, "   - Keyword scan: ${keywordScanTime}ms (${String.format("%.1f", keywordScanTime * 100.0 / totalSearchTime)}%)")
        Log.d(TAG, "   - Embedding similarity: ${embeddingTime}ms (${String.format("%.1f", embeddingTime * 100.0 / totalSearchTime)}%)")
        Log.d(TAG, "   - Sorting/reranking: ${sortingTime}ms (${String.format("%.1f", sortingTime * 100.0 / totalSearchTime)}%)")
        Log.d(TAG, "📈 Throughput:")
        Log.d(TAG, "   - Chunks scanned: $totalChunks")
        Log.d(TAG, "   - Keyword matches: $keywordMatches")
        Log.d(TAG, "   - Embeddings processed: $embeddingCount")
        Log.d(TAG, "   - Scan speed: ${totalChunks * 1000.0 / totalSearchTime} chunks/sec")
        Log.d(TAG, "   - Embedding ops: ${embeddingCount * 384 * 1000.0 / embeddingTime} ops/sec")
    }
    
    /**
     * 인퍼런스 성능 분석
     */
    fun logInferencePerformance(
        totalTime: Long,
        tokenCount: Int,
        modelName: String,
        thinkingMode: Boolean,
        contextLength: Int,
        memoryUsageMB: Long
    ) {
        val tokensPerSecond = if (totalTime > 0) tokenCount * 1000.0 / totalTime else 0.0
        val avgTimePerToken = if (tokenCount > 0) totalTime.toDouble() / tokenCount else 0.0
        
        Log.d(TAG, "=== INFERENCE PERFORMANCE ANALYSIS ===")
        Log.d(TAG, "🤖 Inference Metrics:")
        Log.d(TAG, "   - Total time: ${totalTime}ms")
        Log.d(TAG, "   - Tokens generated: $tokenCount")
        Log.d(TAG, "   - Speed: ${String.format("%.2f", tokensPerSecond)} tokens/sec")
        Log.d(TAG, "   - Avg time per token: ${String.format("%.2f", avgTimePerToken)}ms")
        Log.d(TAG, "📱 Mobile Performance:")
        Log.d(TAG, "   - Model: $modelName")
        Log.d(TAG, "   - Thinking mode: $thinkingMode")
        Log.d(TAG, "   - Context length: $contextLength chars")
        Log.d(TAG, "   - Memory usage: ${memoryUsageMB}MB")
        
        // 성능 등급 평가
        val performanceGrade = when {
            tokensPerSecond >= 10 -> "Excellent (≥10 tok/s)"
            tokensPerSecond >= 5 -> "Good (5-10 tok/s)"
            tokensPerSecond >= 2 -> "Fair (2-5 tok/s)"
            else -> "Slow (<2 tok/s)"
        }
        Log.d(TAG, "   - Performance grade: $performanceGrade")
        
        // 메모리 효율성 평가
        val memoryEfficiency = when {
            memoryUsageMB < 100 -> "Excellent (<100MB)"
            memoryUsageMB < 200 -> "Good (100-200MB)"
            memoryUsageMB < 400 -> "Fair (200-400MB)"
            else -> "High usage (>400MB)"
        }
        Log.d(TAG, "   - Memory efficiency: $memoryEfficiency")
    }
    
    /**
     * 전체 RAG + 인퍼런스 성능 요약
     */
    fun logTotalPerformance(
        totalTime: Long,
        ragTime: Long,
        inferenceTime: Long,
        tokenCount: Int,
        tokensPerSecond: Double,
        memoryUsageMB: Long,
        modelName: String,
        ragEnabled: Boolean
    ) {
        Log.d(TAG, "=== TOTAL MOBILE PERFORMANCE SUMMARY ===")
        Log.d(TAG, "🏆 Complete RAG + Inference Analysis:")
        Log.d(TAG, "   - Total time: ${totalTime}ms")
        Log.d(TAG, "   - RAG search: ${ragTime}ms (${String.format("%.1f", ragTime * 100.0 / totalTime)}%)")
        Log.d(TAG, "   - Inference: ${inferenceTime}ms (${String.format("%.1f", inferenceTime * 100.0 / totalTime)}%)")
        Log.d(TAG, "   - Total tokens: $tokenCount")
        Log.d(TAG, "   - Overall speed: ${String.format("%.2f", tokensPerSecond)} tokens/sec")
        Log.d(TAG, "   - Memory usage: ${memoryUsageMB}MB")
        Log.d(TAG, "   - Model: $modelName")
        Log.d(TAG, "   - RAG enabled: $ragEnabled")
        
        // 전체 성능 등급
        val overallGrade = when {
            tokensPerSecond >= 8 -> "Excellent"
            tokensPerSecond >= 4 -> "Good"
            tokensPerSecond >= 2 -> "Fair"
            else -> "Needs Optimization"
        }
        Log.d(TAG, "   - Overall grade: $overallGrade")
    }
    
    /**
     * 성능 메트릭 내보내기
     */
    fun exportMetrics(): String {
        val summary = StringBuilder()
        summary.appendLine("=== PERFORMANCE METRICS EXPORT ===")
        summary.appendLine("Timestamp: ${dateFormat.format(Date())}")
        summary.appendLine()
        
        metrics.forEach { metric ->
            summary.appendLine("[$metric.phase] ${metric.duration}ms")
            metric.details.forEach { (key, value) ->
                summary.appendLine("  - $key: $value")
            }
            summary.appendLine()
        }
        
        return summary.toString()
    }
    
    /**
     * 메트릭 초기화
     */
    fun clearMetrics() {
        metrics.clear()
        Log.d(TAG, "Performance metrics cleared")
    }
}
