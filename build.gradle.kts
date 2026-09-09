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
        // Reglas que viajan al consumidor: R8 no debe renombrar PiuOcr ni sus
        // métodos native, el .so los busca por nombre (Java_com_piu_ocr_PiuOcr_*).
        consumerProguardFiles("consumer-rules.pro")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
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
}

dependencies {
    // Test de paridad (src/test): corre interpret() + SongMatcher en la JVM
    // sobre salidas grabadas del CLI de host. org.json real porque el de
    // android.jar es un stub.
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    // Test en device (src/androidTest): tools/parity/device.sh
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

// kotlinOptions está deprecado en Kotlin 2.x.
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
