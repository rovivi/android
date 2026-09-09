# El .so resuelve los métodos por nombre JNI: ni la clase ni los native se renombran.
-keepclasseswithmembernames class com.piu.ocr.PiuOcr { native <methods>; }
-keep class com.piu.ocr.PiuOcr { *; }
