plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "com.application.AIChecker"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.application.AIChecker"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += setOf("arm64-v8a", "x86_64")
        }
    }

    // ✅ BẮT BUỘC: bật ViewBinding
    buildFeatures {
        viewBinding = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    chaquopy {
        defaultConfig {
            version = "3.8"

            // ✅ Trỏ đúng Python 3.12 trên máy bạn
            buildPython("C:/Users/ASUS/AppData/Local/Programs/Python/Python312/python.exe")

            pip {
                install("numpy")
                install("scikit-learn")
                // ✅ lightgbm không có prebuilt wheel cho Android
                // Dùng phiên bản cũ hơn có sẵn trên Chaquopy mirror
                install("lightgbm==3.2.1")
                install("opencv-python-headless")
                install("yt-dlp")
                install("pillow")
            }
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.coordinatorlayout:coordinatorlayout:1.2.0")
    implementation("androidx.cardview:cardview:1.0.0")
}