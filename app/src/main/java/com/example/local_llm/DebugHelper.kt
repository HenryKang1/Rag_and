package com.example.local_llm

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.util.Log

object DebugHelper {
    private const val TAG = "DebugHelper"
    
    fun getMemoryInfo(context: Context): MemoryInfo {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        
        val runtime = Runtime.getRuntime()
        val usedMemory = runtime.totalMemory() - runtime.freeMemory()
        val maxHeap = runtime.maxMemory()
        
        return MemoryInfo(
            availableMemoryMB = memInfo.availMem / 1048576L,
            totalMemoryMB = memInfo.totalMem / 1048576L,
            usedHeapMB = usedMemory / 1048576L,
            maxHeapMB = maxHeap / 1048576L,
            lowMemory = memInfo.lowMemory
        )
    }
    
    fun logMemoryStatus(context: Context, prefix: String = "") {
        val info = getMemoryInfo(context)
        Log.d(TAG, "$prefix Memory Status:")
        Log.d(TAG, "  System Available: ${info.availableMemoryMB}MB / ${info.totalMemoryMB}MB")
        Log.d(TAG, "  App Heap Used: ${info.usedHeapMB}MB / ${info.maxHeapMB}MB")
        Log.d(TAG, "  Low Memory Warning: ${info.lowMemory}")
    }
    
    fun checkMemoryPressure(context: Context): Boolean {
        val info = getMemoryInfo(context)
        val heapUsagePercent = (info.usedHeapMB.toFloat() / info.maxHeapMB.toFloat()) * 100
        
        if (heapUsagePercent > 80) {
            Log.w(TAG, "High memory pressure! Heap usage: ${heapUsagePercent.toInt()}%")
            return true
        }
        return false
    }
    
    data class MemoryInfo(
        val availableMemoryMB: Long,
        val totalMemoryMB: Long,
        val usedHeapMB: Long,
        val maxHeapMB: Long,
        val lowMemory: Boolean
    )
}