package com.example.local_llm

import android.content.Context
import android.net.Uri
import android.util.Log
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Document processor for handling various file formats
 * Supports PDF, TXT, and other text-based documents
 */
class DocumentProcessor(private val context: Context) {
    
    companion object {
        const val TAG = "DocumentProcessor"
        const val MAX_FILE_SIZE = 10 * 1024 * 1024  // 10MB limit
        
        @Volatile
        private var isPDFBoxInitialized = false
        
        fun initPDFBox(context: Context) {
            if (!isPDFBoxInitialized) {
                // Initialize PDFBox (required for Android)
                PDFBoxResourceLoader.init(context.applicationContext)
                isPDFBoxInitialized = true
            }
        }
    }
    
    init {
        // Initialize PDFBox when DocumentProcessor is created
        initPDFBox(context)
    }
    
    /**
     * Process a document from URI
     */
    suspend fun processDocument(uri: Uri): String = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "[DOC DEBUG] ========== DOCUMENT PROCESSING START ==========")
            Log.d(TAG, "[DOC DEBUG] Processing document URI: $uri")
            
            // Get MIME type first
            val mimeType = context.contentResolver.getType(uri)
            Log.d(TAG, "[DOC DEBUG] MIME type: $mimeType")
            
            val fileName = getFileName(uri)
            var fileExtension = fileName.substringAfterLast('.', "").lowercase()
            
            // If extension not found from filename, determine from MIME type
            if (fileExtension.isEmpty() || fileExtension == fileName.lowercase()) {
                fileExtension = when (mimeType) {
                    "application/pdf" -> "pdf"
                    "text/plain" -> "txt"
                    "text/csv" -> "csv"
                    "text/markdown" -> "md"
                    else -> {
                        // Try to extract from MIME type (e.g., "application/pdf" -> "pdf")
                        mimeType?.substringAfterLast("/")?.lowercase() ?: ""
                    }
                }
                Log.d(TAG, "[DOC DEBUG] Extension determined from MIME type: $fileExtension")
            }
            
            Log.d(TAG, "[DOC DEBUG] File name: $fileName, Extension: $fileExtension, MIME: $mimeType")
            
            val text = when {
                fileExtension == "pdf" || mimeType == "application/pdf" -> processPdf(uri)
                fileExtension in listOf("txt", "text") || mimeType == "text/plain" -> processText(uri)
                fileExtension in listOf("md", "markdown") || mimeType == "text/markdown" -> processText(uri)
                fileExtension == "csv" || mimeType == "text/csv" -> processCsv(uri)
                else -> {
                    Log.w(TAG, "[DOC DEBUG] Unsupported file type: extension=$fileExtension, mime=$mimeType")
                    throw UnsupportedOperationException("File type not supported (extension: .$fileExtension, MIME: $mimeType)")
                }
            }
            
            Log.d(TAG, "Document processed successfully. Length: ${text.length} characters")
            return@withContext text
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process document: ${e.message}")
            throw e
        }
    }
    
    /**
     * Process PDF document
     */
    private fun processPdf(uri: Uri): String {
        val inputStream = context.contentResolver.openInputStream(uri)
            ?: throw IllegalArgumentException("Cannot open file")
        
        return inputStream.use { stream ->
            val document = PDDocument.load(stream)
            val stripper = PDFTextStripper()
            
            // Configure text extraction
            stripper.sortByPosition = true
            stripper.startPage = 1
            stripper.endPage = document.numberOfPages
            
            val text = stripper.getText(document)
            document.close()
            
            Log.d(TAG, "[DOC DEBUG] PDF text extracted: ${text.length} characters")
            
            // Debug: Check if specific keywords are in the extracted text
            val searchTerms = listOf("cholera", "AVPU", "shock", "dehydration")
            for (term in searchTerms) {
                if (text.contains(term, ignoreCase = true)) {
                    Log.d(TAG, "[DOC DEBUG] ✅ Found '$term' in PDF text")
                    val termIndex = text.indexOf(term, ignoreCase = true)
                    val contextStart = maxOf(0, termIndex - 100)
                    val contextEnd = minOf(text.length, termIndex + 100)
                    Log.d(TAG, "[DOC DEBUG] Context around '$term': ...${text.substring(contextStart, contextEnd)}...")
                } else {
                    Log.d(TAG, "[DOC DEBUG] ❌ '$term' NOT found in PDF text")
                }
            }
            
            // Clean up text
            cleanText(text)
        }
    }
    
    /**
     * Process plain text file
     */
    private fun processText(uri: Uri): String {
        val inputStream = context.contentResolver.openInputStream(uri)
            ?: throw IllegalArgumentException("Cannot open file")
        
        return inputStream.use { stream ->
            val reader = BufferedReader(InputStreamReader(stream))
            val text = reader.readText()
            cleanText(text)
        }
    }
    
    /**
     * Process CSV file
     */
    private fun processCsv(uri: Uri): String {
        val inputStream = context.contentResolver.openInputStream(uri)
            ?: throw IllegalArgumentException("Cannot open file")
        
        return inputStream.use { stream ->
            val reader = BufferedReader(InputStreamReader(stream))
            val lines = reader.readLines()
            
            // Convert CSV to readable text
            val textBuilder = StringBuilder()
            for (line in lines) {
                val values = line.split(",").map { it.trim() }
                textBuilder.append(values.joinToString(" "))
                textBuilder.append("\n")
            }
            
            cleanText(textBuilder.toString())
        }
    }
    
    /**
     * Clean and normalize text
     */
    private fun cleanText(text: String): String {
        return text
            .replace(Regex("\\s+"), " ")  // Normalize whitespace
            .replace(Regex("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F]"), "")  // Remove control characters
            .trim()
    }
    
    /**
     * Get file name from URI
     */
    private fun getFileName(uri: Uri): String {
        var fileName = "unknown"
        
        // Try to get display name from content resolver
        try {
            val projection = arrayOf(android.provider.OpenableColumns.DISPLAY_NAME)
            context.contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    if (nameIndex >= 0) {
                        fileName = cursor.getString(nameIndex) ?: "unknown"
                        Log.d(TAG, "[DOC DEBUG] Got display name from cursor: $fileName")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "[DOC DEBUG] Error querying display name: ${e.message}")
        }
        
        // If still unknown, try other methods
        if (fileName == "unknown" || !fileName.contains(".")) {
            // Try to get from URI path
            uri.path?.let { path ->
                val segments = path.split("/")
                segments.lastOrNull { it.contains(".") }?.let {
                    fileName = it
                    Log.d(TAG, "[DOC DEBUG] Got filename from path: $fileName")
                }
            }
        }
        
        // Last fallback
        if (fileName == "unknown" || !fileName.contains(".")) {
            fileName = uri.lastPathSegment ?: "unknown"
            Log.d(TAG, "[DOC DEBUG] Using lastPathSegment: $fileName")
        }
        
        Log.d(TAG, "[DOC DEBUG] Final file name: $fileName from URI: $uri")
        return fileName
    }
    
    /**
     * Check if file size is within limits
     */
    suspend fun checkFileSize(uri: Uri): Boolean = withContext(Dispatchers.IO) {
        try {
            val inputStream = context.contentResolver.openInputStream(uri)
            val size = inputStream?.available() ?: 0
            inputStream?.close()
            
            if (size > MAX_FILE_SIZE) {
                Log.w(TAG, "File too large: $size bytes (max: $MAX_FILE_SIZE)")
                return@withContext false
            }
            
            return@withContext true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to check file size: ${e.message}")
            return@withContext false
        }
    }
    
    /**
     * Extract metadata from document
     */
    fun extractMetadata(uri: Uri): Map<String, String> {
        val metadata = mutableMapOf<String, String>()
        
        try {
            metadata["file_name"] = getFileName(uri)
            metadata["uri"] = uri.toString()
            
            val inputStream = context.contentResolver.openInputStream(uri)
            metadata["size"] = "${(inputStream?.available() ?: 0) / 1024} KB"
            inputStream?.close()
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to extract metadata: ${e.message}")
        }
        
        return metadata
    }
}