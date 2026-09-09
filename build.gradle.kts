plugins {
    id("com.android.library") version "8.7.3"
    kotlin("android") version "2.0.21"
}

android {
    namespace = "com.piu.ocr"
    compileSdk = 35
    // opencv-mobile linkea -static-openmp y el libomp del NDK 27 no trae
    // __kmpc_dispatch_deinit. El 29 sí.
    ndkVersion = "29.0.14206865"
    defaultConfig {
        minSdk = 24
        // Solo arm64: agregar armeabi-v7a duplica el .so por un parque de
        // dispositivos que ya no importa.
        ndk { abiFilters += "arm64-v8a" }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
    buildTypes {
        release {
            // sin esto el .so viaja con símbolos de debug: 12.8 MB contra 2.5
            ndk { debugSymbolLevel = "none" }
        }
    }
    sourceSets["main"].kotlin.srcDir("src/main/kotlin")
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}
