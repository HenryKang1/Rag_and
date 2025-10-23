package com.example.local_llm

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.InputStream
import java.text.Normalizer

class BpeTokenizer(context: Context) {

    private val vocab: Map<String, Int>
    private val idToToken: Map<Int, String>
    private val merges: List<Pair<String, String>>
    private val bpeRanks: Map<Pair<String, String>, Int>
    private val specialTokens: Map<String, Int>
    private val nfcNormalize: Boolean
    
    // 언어 블로킹을 위한 속성
    private var languageBlocker: LanguageBlocker? = null
    private var isLanguageBlockingEnabled = false

    init {
        val tokenizerJson = loadTokenizerJson(context)

        // Read vocab
        vocab = tokenizerJson.getJSONObject("model")
            .getJSONObject("vocab").toIntMap()
        idToToken = vocab.entries.associate { (k, v) -> v to k }

        // Read merges
        val mergeList = tokenizerJson.getJSONObject("model").getJSONArray("merges")
        merges = (0 until mergeList.length()).map { i ->
            when (val entry = mergeList.get(i)) {
                is String -> {
                    val parts = entry.split(" ")
                    require(parts.size == 2) { "Invalid merge string: $entry" }
                    parts[0] to parts[1]
                }
                is org.json.JSONArray -> {
                    require(entry.length() == 2) { "Invalid merge array: $entry" }
                    entry.getString(0) to entry.getString(1)
                }
                else -> throw IllegalArgumentException("Unsupported merge entry type: ${entry::class.java}")
            }
        }
        bpeRanks = merges.withIndex().associate { it.value to it.index }

        // Special tokens
        val addedTokens = tokenizerJson.optJSONArray("added_tokens")
        specialTokens = if (addedTokens != null) {
            (0 until addedTokens.length()).associate {
                val obj = addedTokens.getJSONObject(it)
                obj.getString("content") to obj.getInt("id")
            }
        } else emptyMap()

        // Normalize?
        nfcNormalize = tokenizerJson.optJSONObject("normalizer")
            ?.optString("type") == "NFC"
    }
    
    /**
     * 어휘 크기 반환
     */
    val vocabSize: Int
        get() = vocab.size + specialTokens.size

    fun tokenize(text: String, addSpecialTokens: Boolean = false): IntArray {
        val tokens = mutableListOf<Int>()

        if (addSpecialTokens) {
            specialTokens["<|im_start|>"]?.let { tokens.add(it) }
        }

        val processed = if (nfcNormalize) Normalizer.normalize(text, Normalizer.Form.NFC) else text

        val bpeTokens = bpe(processed)
        bpeTokens.forEach { bpeToken ->
            vocab[bpeToken]?.let { tokens.add(it) }
        }
        if (addSpecialTokens) {
            specialTokens["<|im_end|>"]?.let { tokens.add(it) }
        }
        Log.d("tokenized", "ID=${tokens}")

        return tokens.toIntArray()
    }


    fun decode(tokenIds: IntArray): String {
        val builder = StringBuilder()
        for (id in tokenIds) {
            val token = idToToken[id]
            if (token != null) {
                builder.append(token)
            } else {
                val special = specialTokens.entries.find { it.value == id }?.key
                if (special != null) {
                    builder.append(special)
                } else {
                    builder.append("<unk>")
                }
            }
        }
        val raw = builder.toString()
        val cleaned = raw
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("▁", " ")
        return if (nfcNormalize) Normalizer.normalize(cleaned, Normalizer.Form.NFC) else cleaned
    }

    private val decodedTokenCache: Map<Int, String> = buildMap {
        idToToken.forEach { (id, token) ->
            val decoded = when {
                token == "Ċ" -> "\n"
                token == "▁" -> " "
                token.startsWith("Ġ") -> " " + token.removePrefix("Ġ")
                token.contains("Ċ") -> token.replace("Ċ", "\n")
                token.contains("▁") -> token.replace("▁", " ")
                else -> token
            }
            put(id, decoded)
        }

        specialTokens.forEach { (key, id) ->
            putIfAbsent(id, key)
        }
    }

    fun decodeSingleToken(tokenId: Int): String {
        return decodedTokenCache[tokenId] ?: "<unk>"
    }

    fun getTokenId(token: String): Int {
        return specialTokens[token]
            ?: vocab[token]
            ?: throw IllegalArgumentException("Token '$token' not found in vocab or special tokens.")
    }


    /* Regex as per tokenizer
    private fun preTokenize(text: String): List<String> {
        val regex = Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
        return regex.findAll(text).map { it.value }.toList()
    }*/

    private fun preTokenize(text: String): List<String> {
        return Regex("""\S+|\s+""").findAll(text).map { it.value }.toList()
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        return (0 until word.size - 1).map { Pair(word[it], word[it + 1]) }.toSet()
    }

    private fun bpe(token: String): List<String> {
        var word = token.toCharArray().map { it.toString() }.toMutableList()
        var pairs = getPairs(word)
        while (true) {
            val best = pairs.minByOrNull { bpeRanks[it] ?: Int.MAX_VALUE } ?: break
            if (!bpeRanks.containsKey(best)) break

            val (first, second) = best
            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }
            word = newWord
            pairs = getPairs(word)
        }
        return word
    }

    private fun loadTokenizerJson(context: Context): JSONObject {
        val inputStream: InputStream = context.assets.open("tokenizer.json")
        val jsonStr = inputStream.bufferedReader().use { it.readText() }
        return JSONObject(jsonStr)
    }

    private fun JSONObject.toIntMap(): Map<String, Int> {
        return keys().asSequence().associateWith { this.getInt(it) }
    }
    
    /**
     * 언어 블로킹 활성화/비활성화
     */
    fun setLanguageBlocking(enabled: Boolean) {
        isLanguageBlockingEnabled = enabled
        if (enabled && languageBlocker == null) {
            languageBlocker = LanguageBlocker()
            languageBlocker?.initialize(this)
        }
        Log.d("BpeTokenizer", "Language blocking ${if (enabled) "enabled" else "disabled"}")
    }
    
    /**
     * 언어 블로킹이 활성화되어 있는지 확인
     */
    fun isLanguageBlockingEnabled(): Boolean = isLanguageBlockingEnabled
    
    /**
     * 토큰화 시 언어 블로킹 적용
     */
    fun tokenizeWithLanguageBlocking(text: String, addSpecialTokens: Boolean = false): IntArray {
        val tokens = tokenize(text, addSpecialTokens)
        
        if (isLanguageBlockingEnabled && languageBlocker != null) {
            return languageBlocker!!.filterForeignTokens(tokens)
        }
        
        return tokens
    }
    
    /**
     * 디코딩 시 언어 블로킹 적용
     */
    fun decodeWithLanguageBlocking(tokenIds: IntArray): String {
        val decoded = decode(tokenIds)
        
        if (isLanguageBlockingEnabled && languageBlocker != null) {
            return languageBlocker!!.removeForeignLanguage(decoded)
        }
        
        return decoded
    }
    
    /**
     * 언어 블로킹 통계 정보 반환
     */
    fun getLanguageBlockingStats(): String {
        return languageBlocker?.getBlockingStats() ?: "Language blocking not initialized"
    }
}
