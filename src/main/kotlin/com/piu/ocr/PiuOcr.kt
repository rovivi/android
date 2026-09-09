package com.piu.ocr

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import org.json.JSONObject
import java.io.File
import kotlin.math.sqrt

/**
 * Lectura de una pantalla de resultado de PIU, sin LLM.
 *
 * `reason != null` significa "no confío en esto, escalalo". **Un campo con
 * reason NO se debe escribir en la base.** La política es un nombre si el gate
 * pasa, si no se escala — nunca dos candidatos, porque está medido que debajo
 * del gate el segundo candidato casi nunca acierta.
 */
data class Field<T>(val value: T?, val confidence: Float, val reason: String? = null)

data class Reading(
    val song: Field<String>,
    val level: Field<Int>,
    val chartType: Field<String>,
    val rawTitle: String,
) {
    val needsLlm: List<String> get() = buildList {
        if (song.reason != null || song.value == null) add("song")
        if (level.reason != null || level.value == null) add("level")
        if (chartType.reason != null || chartType.value == null) add("chart_type")
    }
    /** Una llamada al VLM devuelve todos los campos: el costo es por pantalla,
     *  no por campo. Alcanza con que uno no pase para pagarla entera. */
    val needsVlmCall: Boolean get() = needsLlm.isNotEmpty()
}

class PiuOcr private constructor(
    handle: Long,
    private val matcher: SongMatcher,
) : AutoCloseable {

    @Volatile private var handle = handle

    fun read(bitmap: Bitmap): Reading {
        check(handle != 0L) { "PiuOcr ya está cerrado" }
        // El .so solo lee ARGB_8888 con píxeles bloqueables. Un HARDWARE bitmap
        // (lo que devuelve ImageDecoder por defecto en API 28+) o un RGB_565
        // fallaban en lockPixels y volvían vacíos en silencio.
        val bmp = if (bitmap.config == Bitmap.Config.ARGB_8888 && !isHardware(bitmap)) bitmap
                  else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        val j = JSONObject(nativeRead(handle, bmp))

        // chart_type primero: acota qué canciones son posibles.
        val ctRaw = j.optString("chart_type").ifEmpty { null }
        val ctConf = j.optDouble("chart_conf", 0.0).toFloat()
        val chart = Field(
            if (ctConf >= MIN_BADGE_CONF) ctRaw else null, ctConf,
            if (ctRaw != null && ctConf >= MIN_BADGE_CONF) null else "baja_confianza")

        // Una pantalla 2P imprime el título dos veces y los dos recortes fallan
        // distinto (reflejo, recorte), así que se juntan los puntajes: una
        // lectura limpia le gana a una colapsada. Vale +7 pts, medido.
        val titles = j.optJSONArray("titles") ?: org.json.JSONArray()
        val pooled = HashMap<String, Double>()
        val raws = ArrayList<String>()
        for (i in 0 until minOf(titles.length(), MAX_SONG_BOXES)) {
            val t = titles.getJSONObject(i)
            val raw = t.getString("raw")
            if (raw.isEmpty()) continue
            raws += raw
            // Ponderar por la confianza de la caja: una caja de 0.005 puede ser
            // la respuesta cuando es la única, sin ganarle a una de 0.5.
            val w = sqrt(maxOf(t.optDouble("conf", 1.0), 0.02))
            for (c in matcher.match(raw, chart.value, topK = 8)) {
                val e = pooled[c.name] ?: 0.0
                pooled[c.name] = maxOf(e, c.score * w) + 0.15 * minOf(e, c.score * w)
            }
        }
        val ranked = pooled.entries.sortedByDescending { it.value }
        val margin = if (ranked.size > 1) ranked[0].value - ranked[1].value else 1.0
        val songOk = ranked.isNotEmpty() && margin >= MIN_SONG_MARGIN
        val song = Field(if (songOk) ranked[0].key else null, margin.toFloat(),
                         if (songOk) null else "margen_bajo")

        // Con la canción resuelta el catálogo dice qué niveles son legales, y
        // eso reordena los dígitos leídos en vez de solo aceptar el argmax.
        val level = readLevel(j, song.value, chart.value)
        return Reading(song, level, chart, raws.joinToString(" | "))
    }

    private fun readLevel(j: JSONObject, song: String?, chartType: String?): Field<Int> {
        val sc = j.optJSONArray("level_scores") ?: return Field(null, 0f, "sin_glifos")
        if (sc.length() == 0) return Field(null, 0f, "sin_glifos")
        val legal = song?.let { matcher.levelsFor(it, chartType) }.orEmpty()
            .filter { it.toString().length == sc.length() }
        if (legal.isEmpty()) {
            val d = j.optJSONArray("level_digits") ?: return Field(null, 0f, "sin_glifos")
            val v = (0 until d.length()).joinToString("") { d.getInt(it).toString() }
                .toIntOrNull() ?: return Field(null, 0f, "no_numerico")
            return Field(v, 0f, if (v in 1..28) null else "fuera_de_rango")
        }
        val scored = legal.map { cand ->
            cand to cand.toString().withIndex().sumOf { (i, ch) ->
                sc.getJSONArray(i).optDouble(ch - '0', -1.0)
            }
        }.sortedByDescending { it.second }
        val m = if (scored.size > 1)
            (scored[0].second - scored[1].second) / sc.length() else 1.0
        return Field(scored[0].first, m.toFloat(), null)
    }

    /** Idempotente: un segundo close() era un double free en el .so. */
    @Synchronized
    override fun close() {
        if (handle != 0L) { nativeDestroy(handle); handle = 0L }
    }

    private fun isHardware(b: Bitmap) =
        Build.VERSION.SDK_INT >= 26 && b.config == Bitmap.Config.HARDWARE

    companion object {
        private var loaded = false

        /**
         * Carga la librería nativa.
         *
         * Si este módulo va como **dynamic feature de Play**, un `.so` adentro
         * NO se carga con `System.loadLibrary`: hay que pasar por
         * `SplitCompat.install()` + `SplitInstallHelper.loadLibrary()`. Y falla
         * SOLO en builds firmados de Play — nunca en debug ni en un APK local —
         * así que se descubre en producción si no se contempla.
         *
         * Se resuelve por reflexión para no obligar a la dependencia de Play
         * Core cuando el módulo va embebido normal.
         */
        @Synchronized
        private fun ensureLoaded(context: Context) {
            if (loaded) return
            try {
                val helper = Class.forName(
                    "com.google.android.play.core.splitcompat.SplitInstallHelper")
                helper.getMethod("loadLibrary", Context::class.java, String::class.java)
                    .invoke(null, context, "piuocr")
            } catch (_: Throwable) {
                System.loadLibrary("piuocr")     // módulo embebido normal
            }
            loaded = true
        }

        // Gates medidos end-to-end con LOSO por foto. Ver MODULO_ANDROID.md §3:
        // cambiarlos degrada el sistema EN SILENCIO.
        //   canción  0.010 -> cob 0.889 / prec 0.900
        //            0.015 -> cob 0.800 / prec 0.972   <- este
        //            0.030 -> cob 0.756 / prec 0.971
        const val MIN_SONG_MARGIN = 0.015
        // 0.35 no compraba precisión, solo la tiraba: sobre 55 bolitas 0.15 da
        // 0.873 y 0.35 da 0.655, porque convierte lecturas buenas en null.
        const val MIN_BADGE_CONF = 0.15f
        const val MAX_SONG_BOXES = 3

        private val ASSETS = listOf("chars.bin", "level.bin", "catalog.json",
                                    "piu_yolo.param", "piu_yolo.bin")

        /**
         * Copia los assets a filesDir y crea el lector. Falla con
         * [IllegalStateException] si el .so no pudo cargar los modelos: antes
         * devolvía un handle a medias que leía vacío para siempre.
         *
         * La copia se hace una vez POR VERSIÓN de la app: si solo se mira
         * `exists()`, una actualización del APK con modelos nuevos sigue usando
         * los viejos. Y cada archivo se escribe a `.tmp` y se renombra, para que
         * un crash a mitad de copia no deje un `.bin` truncado que "existe".
         */
        @JvmStatic
        fun create(context: Context): PiuOcr {
            val dir = File(context.filesDir, "piu_ocr").apply { mkdirs() }
            val stamp = File(dir, ".version")
            val want = appVersion(context)
            if (stamp.takeIf { it.exists() }?.readText() != want) {
                for (n in ASSETS) {
                    val tmp = File(dir, "$n.tmp")
                    context.assets.open("piu_ocr/$n").use { i ->
                        tmp.outputStream().use { o -> i.copyTo(o) }
                    }
                    val dst = File(dir, n)
                    if (!tmp.renameTo(dst)) { dst.delete(); check(tmp.renameTo(dst)) }
                }
                stamp.writeText(want)
            }
            ensureLoaded(context)
            val h = nativeCreate(dir.absolutePath)
            check(h != 0L) { "piuocr: no se pudieron cargar los modelos de ${dir.absolutePath}" }
            return PiuOcr(h, SongMatcher(File(dir, "catalog.json").readText()))
        }

        private fun appVersion(context: Context): String {
            val pi = context.packageManager.getPackageInfo(context.packageName, 0)
            val code = if (Build.VERSION.SDK_INT >= 28) pi.longVersionCode
                       else @Suppress("DEPRECATION") pi.versionCode.toLong()
            return "$code:${pi.lastUpdateTime}"
        }

        @JvmStatic private external fun nativeCreate(assetDir: String): Long
        @JvmStatic private external fun nativeDestroy(handle: Long)
        @JvmStatic private external fun nativeRead(handle: Long, bitmap: Bitmap): String
    }
}
