package com.example.local_llm

import android.app.Activity
import android.util.Log
import kotlinx.coroutines.delay

/**
 * 메모리 관리 관련 유틸리티 클래스
 */
object MemoryManager {
    
    private const val TAG = "MemoryManager"
    
    /**
     * 메모리 압박 상태 확인
     */
    fun checkMemoryPressure(activity: Activity): Boolean {
        val memInfo = DebugHelper.getMemoryInfo(activity)
        val heapUsagePercent = (memInfo.usedHeapMB.toFloat() / memInfo.maxHeapMB.toFloat()) * 100
        
        LoggingUtils.logMemoryDebug(TAG, "Heap usage: ${heapUsagePercent.toInt()}% (${memInfo.usedHeapMB}MB/${memInfo.maxHeapMB}MB)")
        LoggingUtils.logMemoryDebug(TAG, "Available system memory: ${memInfo.availableMemoryMB}MB")
        
        return heapUsagePercent > Constants.Memory.HIGH_MEMORY_THRESHOLD
    }
    
    /**
     * 메모리 정리 수행
     */
    suspend fun performMemoryCleanup() {
        LoggingUtils.logMemoryDebug(TAG, "Performing memory cleanup")
        
        // Force garbage collection
        System.gc()
        System.runFinalization()
        
        // Small delay to let memory settle
        delay(Constants.UI.CLEANUP_DELAY_MS.toLong())
        
        LoggingUtils.logMemoryDebug(TAG, "Memory cleanup complete")
    }
    
    /**
     * 메모리 상태에 따른 최대 토큰 수 계산
     */
    fun calculateMaxTokens(activity: Activity, isThinkingMode: Boolean, forceMaxTokens: Int): Int {
        if (forceMaxTokens > 0) {
            LoggingUtils.logTokenDebug(TAG, "Using forced max tokens: $forceMaxTokens")
            return forceMaxTokens
        }
        
        val memInfo = DebugHelper.getMemoryInfo(activity)
        val heapUsagePercent = (memInfo.usedHeapMB.toFloat() / memInfo.maxHeapMB.toFloat()) * 100
        
        val baseTokens = when {
            heapUsagePercent < Constants.Memory.LOW_MEMORY_THRESHOLD -> 1024
            heapUsagePercent < Constants.Memory.MODERATE_MEMORY_THRESHOLD -> 768
            heapUsagePercent < Constants.Memory.HIGH_MEMORY_THRESHOLD -> 512
            else -> 256
        }
        
        val maxTokens = if (!isThinkingMode) {
            LoggingUtils.logModeDebug(TAG, "Non-thinking mode: boosting tokens by ${Constants.Memory.MEMORY_BOOST_FACTOR}x")
            (baseTokens * Constants.Memory.MEMORY_BOOST_FACTOR).toInt().coerceAtMost(Constants.Token.MAX_TOKENS_768)
        } else {
            LoggingUtils.logModeDebug(TAG, "Thinking mode: using standard tokens")
            baseTokens
        }
        
        LoggingUtils.logMemoryDebug(TAG, "Adjusted max tokens to: $maxTokens")
        LoggingUtils.logModeDebug(TAG, "Thinking mode: ${if (isThinkingMode) "ON" else "OFF (Non-thinking)"}")
        
        return maxTokens
    }
    
    /**
     * 메모리 상태 로깅
     */
    fun logMemoryStatus(activity: Activity, context: String) {
        DebugHelper.logMemoryStatus(activity, context)
    }
}
