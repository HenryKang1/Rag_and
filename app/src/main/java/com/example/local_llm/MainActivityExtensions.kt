package com.example.local_llm

import android.content.Context
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.*

/**
 * MainActivity의 확장 함수들 - 긴 메서드들을 작은 단위로 분리
 */
class MainActivityExtensions(private val activity: MainActivity) {
    
    /**
     * UI 초기화 - 모든 UI 요소를 한 번에 초기화
     */
    fun initializeUI() {
        initializeUIElements()
        setupUIEventListeners()
    }
    
    /**
     * UI 요소들을 findViewById로 초기화
     */
    private fun initializeUIElements() {
        // Initialize UI elements
        activity.thinkingToggle = activity.findViewById(R.id.thinkingToggle)
        activity.ragToggle = activity.findViewById(R.id.ragToggle)
        activity.uploadButton = activity.findViewById(R.id.uploadButton)
        activity.clearIndexButton = activity.findViewById(R.id.clearIndexButton)
        activity.inputEditText = activity.findViewById(R.id.userInput)
        activity.sendButton = activity.findViewById(R.id.sendButton)
        activity.stopButton = activity.findViewById(R.id.stopButton)
        activity.chatRecyclerView = activity.findViewById(R.id.chatRecyclerView)
    }
    
    /**
     * UI 이벤트 리스너들을 설정
     */
    private fun setupUIEventListeners() {
        setupToggles(activity.thinkingToggle, activity.ragToggle)
        setupButtons(activity.uploadButton, activity.clearIndexButton, activity.sendButton, activity.stopButton)
        setupInputField(activity.inputEditText)
    }
    
    private fun setupToggles(thinkingToggle: CheckBox, ragToggle: CheckBox) {
        // Set default to non-thinking mode for better performance
        thinkingToggle.isChecked = false
        
        // RAG controls setup
        ragToggle.setOnCheckedChangeListener { _, isChecked ->
            activity.isRagEnabled = isChecked
            val message = if (isChecked) activity.getString(R.string.rag_enabled) else activity.getString(R.string.rag_disabled)
            Toast.makeText(activity, message, Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun setupButtons(uploadButton: Button, clearIndexButton: Button, sendButton: Button, stopButton: Button) {
        uploadButton.setOnClickListener {
            Log.d("MainActivity", "[MAIN DEBUG] Upload button clicked")
            activity.openFilePicker()
        }
        
        // Long press to toggle semantic chunking mode
        uploadButton.setOnLongClickListener {
            activity.useSemanticChunking = !activity.useSemanticChunking
            val mode = if (activity.useSemanticChunking) 
                activity.getString(R.string.chunking_mode_semantic) 
            else 
                activity.getString(R.string.chunking_mode_fixed)
            Toast.makeText(activity, mode, Toast.LENGTH_LONG).show()
            Log.d("MainActivity", "[CHUNKING MODE] Switched to ${if (activity.useSemanticChunking) "SEMANTIC" else "FIXED-SIZE"} chunking")
            true
        }
        
        clearIndexButton.setOnClickListener {
            activity.clearRAGIndex()
        }
        
        sendButton.setBackgroundResource(R.drawable.btn_send)
        sendButton.backgroundTintList = null
        
        // Long press on send button to adjust max tokens
        sendButton.setOnLongClickListener {
            showTokenSelectionDialog()
            true
        }
        
        stopButton.setBackgroundResource(R.drawable.btn_stop)
        stopButton.backgroundTintList = null
    }
    
    private fun setupInputField(inputEditText: EditText) {
        inputEditText.setOnFocusChangeListener { _, hasFocus ->
            if (hasFocus) {
                activity.chatRecyclerView.postDelayed({
                    activity.chatRecyclerView.scrollToPosition(activity.chatAdapter.itemCount - 1)
                }, Constants.UI.SCROLL_DELAY_MS.toLong())
            }
        }
    }
    
    private fun showTokenSelectionDialog() {
        val options = arrayOf("Auto", "128 tokens", "256 tokens", "384 tokens", "512 tokens", "768 tokens")
        val values = intArrayOf(
            Constants.Token.MAX_TOKENS_AUTO,
            Constants.Token.MAX_TOKENS_128,
            Constants.Token.MAX_TOKENS_256,
            Constants.Token.MAX_TOKENS_384,
            Constants.Token.MAX_TOKENS_512,
            Constants.Token.MAX_TOKENS_768
        )
        
        AlertDialog.Builder(activity)
            .setTitle("Max Output Tokens")
            .setItems(options) { _, which ->
                activity.forceMaxTokens = values[which]
                val message = if (activity.forceMaxTokens == 0) "Auto token adjustment" else "Max tokens: ${activity.forceMaxTokens}"
                Toast.makeText(activity, message, Toast.LENGTH_LONG).show()
                LoggingUtils.logTokenDebug("MainActivity", "Set max tokens to: ${if (activity.forceMaxTokens == 0) "Auto" else activity.forceMaxTokens.toString()}")
            }
            .show()
    }
    
    /**
     * 모델 설정 초기화
     */
    fun initializeModelConfiguration() {
        val config = activity.createModelConfig(activity.selectedModel)
        
        if (config.IsThinkingModeAvailable) {
            activity.thinkingToggle.visibility = View.VISIBLE
            activity.thinkingToggle.isChecked = false
            
            activity.thinkingToggle.setOnCheckedChangeListener { _, isChecked ->
                val mode = if (isChecked) {
                    "🤔 Thinking Mode ON\n(Slower but more thoughtful responses)"
                } else {
                    "⚡ Fast Mode ON\n(Quicker responses, less reasoning)"
                }
                Toast.makeText(activity, mode, Toast.LENGTH_LONG).show()
                LoggingUtils.logModeDebug("MainActivity", "Thinking mode: $isChecked")
            }
        } else {
            activity.thinkingToggle.visibility = View.GONE
        }
        
        activity.title = "Emergency assistant — ${config.modelName}"
    }
    
    /**
     * 채팅 UI 설정
     */
    fun setupChatUI() {
        activity.chatAdapter = ChatAdapter(activity.messages)
        activity.chatRecyclerView.layoutManager = androidx.recyclerview.widget.LinearLayoutManager(activity)
        activity.chatRecyclerView.adapter = activity.chatAdapter
        
        // Hide keyboard when user scrolls the chat
        activity.chatRecyclerView.setOnTouchListener { _, _ ->
            val imm = activity.getSystemService(Context.INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager
            imm.hideSoftInputFromWindow(activity.inputEditText.windowToken, 0)
            false
        }
        
        activity.chatRecyclerView.addOnLayoutChangeListener { _, _, _, _, bottom, _, _, _, oldBottom ->
            if (bottom < oldBottom) {
                activity.chatRecyclerView.post {
                    activity.chatRecyclerView.scrollToPosition(activity.chatAdapter.itemCount - 1)
                }
            }
        }
    }
}
