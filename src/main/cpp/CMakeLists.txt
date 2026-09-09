cmake_minimum_required(VERSION 3.22)
project(piuocr CXX)
set(CMAKE_CXX_STANDARD 17)

set(TP ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party)

# opencv-mobile: solo core+imgproc (~2 MB). El SDK completo son 60 MB y este
# módulo usa 10 funciones — ver MODULO_ANDROID.md §2.
set(OpenCV_DIR ${TP}/opencv-mobile-4.13.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# ncnn SIN Vulkan: el detector corre con use_vulkan_compute=false, así que
# glslang+SPIRV+MachineIndependent (5.3 MB de .a) eran peso muerto.
set(ncnn_DIR ${TP}/ncnn-20240820-android/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(piuocr SHARED
    jni_bridge.cpp detector.cpp segment.cpp text.cpp badge.cpp recognize.cpp)
target_link_libraries(piuocr ${OpenCV_LIBS} ncnn jnigraphics log)
target_compile_options(piuocr PRIVATE -O3 -ffast-math)
