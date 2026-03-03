plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "com.yourname.deepfakedetector"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.yourname.deepfakedetector"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += setOf("arm64-v8a", "x86_64")
        }
    }

    // Chaquopy config — phải nằm NGOÀI defaultConfig trong Kotlin DSL
    chaquopy {
        defaultConfig {
            version = "3.11"
            pip {
                install("numpy")
                install("scikit-learn")
                install("lightgbm")
                install("opencv-python-headless")
                install("yt-dlp")
                install("pillow")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
}