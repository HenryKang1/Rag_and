# Qwen QA Style App with RAG Support

An Android application that runs Qwen models locally with Retrieval-Augmented Generation (RAG) capabilities using llama.cpp.

## Features

- **Local LLM Inference**: Run Qwen3 models directly on Android devices
- **RAG Support**: Upload documents (PDF, TXT) and query them with context-aware responses
- **Thinking Mode**: Toggle between fast mode and thinking mode for different response qualities
- **Memory Management**: Dynamic token adjustment based on available memory
- **Semantic Chunking**: Optional semantic-based document chunking for better context retrieval

## Prerequisites

- Android Studio Arctic Fox or later
- NDK 21.0 or higher
- CMake 3.22.1 or higher
- Minimum Android SDK 24 (Android 7.0)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/HenryKang1/Rag_and.git
cd Rag_and
```

llama.cpp is vendored in `third_party/llama.cpp/`, so no submodule setup is needed.

### 2. Place the model file

Download a GGUF model file and place it in `app/src/main/assets/`:
- Recommended: `qwen3-0.6b-q4_k_m.gguf`
- The model must be under 2GB due to Android asset limitations

### 3. Build the app

```bash
./gradlew assembleDebug
```

Or open the project in Android Studio and build from there.

## Architecture

### Native Integration
- Uses llama.cpp for GGUF model inference
- JNI bindings in `app/src/main/cpp/qwenllm.cpp`
- C++ wrapper in `LLMInference.cpp/h`

### RAG Implementation
- Document processing with PDFBox for Android
- FAISS-like vector search using MiniLM embeddings
- Hybrid search combining keyword and semantic similarity

### UI Features
- **Thinking Mode Toggle**: Switch between fast and thoughtful responses
- **RAG Toggle**: Enable/disable document context
- **Document Upload**: Support for PDF and text files
- **Memory Status**: Real-time memory monitoring

## Usage

1. **Upload Documents**: Tap "Upload Doc" to add PDF or text files to the knowledge base
2. **Toggle RAG**: Enable RAG to use uploaded documents as context
3. **Toggle Thinking Mode**: 
   - OFF (default): Fast responses without internal reasoning
   - ON: Slower but more thoughtful responses with reasoning process
4. **Ask Questions**: Type your question and tap Send

## Long Press Features
- **Upload Button**: Toggle between FIXED-SIZE and SEMANTIC chunking modes
- **Send Button**: Adjust maximum output tokens (Auto, 128-768)

## Project Structure

```
Rag_and/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/example/local_llm/
│   │   │   │   ├── MainActivity.kt          # Main app logic
│   │   │   │   ├── LlamaCppModel.kt        # JNI wrapper
│   │   │   │   ├── MobileRAG.kt            # RAG implementation
│   │   │   │   └── DocumentProcessor.kt     # Document handling
│   │   │   ├── cpp/
│   │   │   │   ├── CMakeLists.txt          # Build config
│   │   │   │   ├── qwenllm.cpp             # JNI bindings
│   │   │   │   └── LLMInference.cpp/h      # C++ wrapper
│   │   │   └── assets/
│   │   │       └── [place GGUF models here]
│   └── build.gradle.kts
├── third_party/
│   └── llama.cpp/                          # Vendored llama.cpp
└── README.md
```

## Dependencies

- llama.cpp (vendored in `third_party/`)
- ONNX Runtime Android (for embeddings)
- PDFBox Android (for PDF processing)
- Kotlin Coroutines
- AndroidX libraries

## Troubleshooting

### Build Issues
- Ensure `third_party/llama.cpp/` is present
- Check NDK installation in Android Studio
- Verify CMake version is 3.22.1+

### Runtime Issues
- OutOfMemoryError: Reduce max tokens or disable RAG
- Model not loading: Check model file exists in assets/
- Slow inference: Toggle off Thinking Mode for faster responses

## License

MIT. See [LICENSE](LICENSE). llama.cpp in `third_party/llama.cpp/` keeps its own MIT license.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF inference
- Qwen team for the models
- PDFBox for document processing