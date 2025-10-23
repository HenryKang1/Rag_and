plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.local_llm"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.local_llm"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        
        // NDK configuration for llama.cpp
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
        
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                // Increase stack size for native threads (8MB)
                arguments += listOf("-DANDROID_STL=c++_shared")
            }
        }
    }
    
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        debug {
            isDebuggable = true
            isJniDebuggable = true
            isMinifyEnabled = false
            
            // Enable native debugging (without ASan)
            externalNativeBuild {
                cmake {
                    arguments += listOf("-DCMAKE_BUILD_TYPE=Debug")
                    cppFlags += listOf("-g", "-O0", "-DDEBUG", "-fno-omit-frame-pointer")
                    // Note: Address Sanitizer is not included as it requires additional runtime libraries
                }
            }
            
            // Keep symbols for better crash reports
            packaging {
                jniLibs {
                    keepDebugSymbols += listOf("**/*.so")
                }
            }
        }
        
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            
            externalNativeBuild {
                cmake {
                    arguments += listOf("-DCMAKE_BUILD_TYPE=Release")
                }
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
        viewBinding = true
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.appcompat)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    implementation (libs.json.json)
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation("io.noties.markwon:core:4.6.2")
    implementation("androidx.recyclerview:recyclerview:1.4.0")
    implementation ("com.google.android.material:material:1.12.0")
    implementation ("com.google.code.gson:gson:2.10.1")
    
    // PDF Processing
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")
    
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // Math operations for vector similarity
    implementation("org.apache.commons:commons-math3:3.6.1")
    
    // ONNX Runtime for embedding model (RAG)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.3")
}