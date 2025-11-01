*** Begin Patch
*** Update File: LLMInference.cpp
@@
-// existing includes...
+// existing includes...
+#include <chrono>
+
+using clock_us = std::chrono::time_point<std::chrono::steady_clock>;
@@
-// class members (somewhere in the class definition)
-// llama_context* _ctx;
-// std::vector<llama_token> _promptTokens;
-// llama_token _currToken;
+// class members (somewhere in the class definition)
+// llama_context* _ctx;
+// std::vector<llama_token> _promptTokens;
+// llama_token _currToken;
+// llama_pos _currPos {0};                  // ✅ explicit position for decode phase
+// size_t _prefillKernelUs {0};             // ✅ accumulates llama_decode kernel time (prefill)
+// clock_us _prefillStartWall;              // ✅ wall clock start
@@
-bool LLMInference::loadModel(const std::string& modelPath,
-                             int nThreads,
-                             int contextSize) {
+bool LLMInference::loadModel(const std::string& modelPath,
+                             int nThreads,
+                             int contextSize) {
     // ... open model etc.
     llama_context_params ctx_params = llama_context_default_params();
     ctx_params.n_ctx     = contextSize;
     ctx_params.n_threads = nThreads;
+    ctx_params.n_batch   = std::min<int>(1024, contextSize); // ✅ important for fast prefill
+    ctx_params.no_perf   = true;                              // reduce perf overhead in hot path
 
     _ctx = llama_init_from_model(_model, ctx_params);
     return _ctx != nullptr;
 }
@@
-void LLMInference::startCompletion(const std::string& prompt) {
+void LLMInference::startCompletion(const std::string& prompt) {
     // --- tokenize prompt ---
     _promptTokens = common_tokenize(llama_model_get_vocab(_model), prompt, /*add_bos*/true, /*special*/true);
 
-    // clear kv if needed (previous behavior may have done this unconditionally)
-    llama_kv_cache_clear(_ctx);
+    // ✅ Do NOT clear KV here by default. Let explicit reset() decide lifecycle.
+    // llama_kv_cache_clear(_ctx); // (optional) only if the app requested a hard reset
 
-    // prefill: naive (previous) single batch without pos/seq config
-    // ...
+    // ===== PREFILL (chunked by n_batch, with explicit pos/seq and logits on last token) =====
+    const int n_prompt = static_cast<int>(_promptTokens.size());
+    const int n_batch  = llama_n_batch(_ctx);
+
+    _prefillKernelUs = 0;
+    _prefillStartWall = std::chrono::steady_clock::now();
+
+    llama_batch batch = llama_batch_init(n_batch, /*embd*/0, /*n_seq_max*/1);
+    int consumed = 0;
+    while (consumed < n_prompt) {
+        llama_batch_clear(&batch);
+
+        const int take = std::min(n_batch, n_prompt - consumed);
+        for (int i = 0; i < take; ++i) {
+            const int idx = consumed + i;
+            const llama_token tok = _promptTokens[idx];
+            const bool need_logits = (idx == n_prompt - 1); // only last prefill token
+            llama_batch_add(&batch, tok, /*pos*/ idx, /*seq_id*/{0}, /*logits*/ need_logits);
+        }
+
+        const auto t0 = std::chrono::steady_clock::now();
+        const int rc = llama_decode(_ctx, batch);
+        const auto t1 = std::chrono::steady_clock::now();
+        _prefillKernelUs += (size_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
+        if (rc < 0) {
+            llama_batch_free(batch);
+            throw std::runtime_error("llama_decode prefill failed");
+        }
+        consumed += take;
+    }
+
+    const auto prefillWallUs = (size_t) std::chrono::duration_cast<std::chrono::microseconds>(
+        std::chrono::steady_clock::now() - _prefillStartWall).count();
+
+    LOGi("[PERF] PREFILL done: %d tokens | wall %.2f ms | kernel %.2f ms | %.2f tok/s",
+         n_prompt,
+         prefillWallUs / 1000.0,
+         _prefillKernelUs / 1000.0,
+         n_prompt > 0 ? (1e6 * (double)n_prompt / (double)prefillWallUs) : 0.0);
+
+    // prepare decode state
+    _currPos   = (llama_pos)n_prompt;   // ✅ decode starts after prompt
+    _currToken = llama_token_eos(_ctx); // placeholder; will be replaced by sampler output
+
+    llama_batch_free(batch); // free prefill batch (we'll create lightweight one-token batches below)
 }
@@
-bool LLMInference::stepDecodeOnce() {
-    // previously: llama_batch_get_one(&_currToken, 1) with implicit pos
-    llama_batch b = llama_batch_get_one(&_currToken, 1);
-    int rc = llama_decode(_ctx, b);
-    // sample next ...
-    return rc >= 0;
+bool LLMInference::stepDecodeOnce() {
+    // ===== 1-token decode with explicit position =====
+    llama_batch dec = llama_batch_init(/*cap*/1, /*embd*/0, /*n_seq_max*/1);
+    llama_batch_add(&dec, _currToken, /*pos*/ _currPos++, /*seq*/{0}, /*logits*/ true);
+
+    const int rc = llama_decode(_ctx, dec);
+    llama_batch_free(dec);
+    if (rc < 0) return false;
+
+    // ===== sample next token (existing sampler chain) =====
+    // ... existing top_k/top_p/temperature/mirostat, etc. keep as-is ...
+    // result stored to _currToken (and pushed to output buffer by caller)
+    return true;
 }
@@
-void LLMInference::stopCompletion(bool hardReset) {
-    if (hardReset) {
-        llama_kv_cache_clear(_ctx); // mixed usage before
-    } else {
-        // partial clear previously
-    }
+void LLMInference::stopCompletion(bool hardReset) {
+    // ✅ Unify KV policy: hardReset=true clears; false leaves cache for potential reuse.
+    if (hardReset) {
+        llama_kv_cache_clear(_ctx);
+    }
 }
*** End Patch
