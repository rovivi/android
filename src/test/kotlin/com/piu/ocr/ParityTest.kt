package com.piu.ocr

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Test de paridad de la mitad Kotlin del módulo.
 *
 * `tools/parity/parity.py --fixture` corre el C++ del .so (CLI de host) sobre
 * las fotos de dataset_v2 y graba, por foto, el JSON crudo que devolvería
 * `nativeRead` más lo que una réplica Python de [PiuOcr.interpret] saca de él.
 * Acá se corre el `interpret` REAL sobre ese mismo JSON y se exige igualdad:
 * si alguien toca un gate en Kotlin y no en la réplica (o al revés), esto
 * falla. Además mide acierto contra el ground truth y lo compara con el
 * baseline grabado, para que una "mejora" que empeora no pase en silencio.
 *
 *     gradle testReleaseUnitTest
 */
class ParityTest {

    private val fixture: JSONObject by lazy {
        val res = javaClass.getResourceAsStream("/parity_fixture.json")
            ?: error("falta src/test/resources/parity_fixture.json: correr tools/parity/parity.py --fixture")
        JSONObject(res.reader().readText())
    }

    private val matcher: SongMatcher by lazy {
        SongMatcher(File("src/main/assets/piu_ocr/catalog.json").readText())
    }

    @Test
    fun interpretMatchesPythonReplica() {
        val rows = fixture.getJSONArray("rows")
        assertTrue("fixture vacío", rows.length() > 0)
        val bad = ArrayList<String>()
        for (i in 0 until rows.length()) {
            val r = rows.getJSONObject(i)
            val exp = r.getJSONObject("expected")
            val got = PiuOcr.interpret(r.getJSONObject("native").toString(), matcher)
            val diffs = ArrayList<String>()
            val expSong = if (exp.isNull("song")) null else exp.getString("song")
            if (got.song.value != expSong) diffs += "song kt=${got.song.value} py=$expSong"
            val expLevel = if (exp.isNull("level")) null else exp.getInt("level")
            if (got.level.value != expLevel) diffs += "level kt=${got.level.value} py=$expLevel"
            val expChart = if (exp.isNull("chart_type")) null else exp.getString("chart_type")
            if (got.chartType.value != expChart) diffs += "chart kt=${got.chartType.value} py=$expChart"
            val expScore = if (exp.isNull("score")) null else exp.getInt("score")
            if (got.score.value != expScore) diffs += "score kt=${got.score.value} py=$expScore"
            if (got.rawTitle != exp.getString("raw")) diffs += "raw kt=${got.rawTitle} py=${exp.getString("raw")}"
            if (diffs.isNotEmpty()) bad += "${r.getString("key")}: ${diffs.joinToString("; ")}"
        }
        assertEquals("interpret() difiere de la réplica Python en:\n" + bad.joinToString("\n"),
                     0, bad.size)
    }

    /** El score sale del C++ ya validado (0..1000000); acá solo el gate. */
    @Test
    fun scoreAccuracyAgainstGroundTruth() {
        val rows = fixture.getJSONArray("rows")
        var n = 0; var cov = 0; var ok = 0
        for (i in 0 until rows.length()) {
            val r = rows.getJSONObject(i)
            val gt = r.getJSONObject("gt")
            if (gt.isNull("score")) continue
            n++
            val v = PiuOcr.interpret(r.getJSONObject("native").toString(), matcher).score.value
                ?: continue
            cov++
            val g = gt.get("score")
            val hit = if (g is org.json.JSONArray)
                (0 until g.length()).any { g.getInt(it) == v } else g == v
            if (hit) ok++
        }
        val precision = if (cov > 0) ok.toDouble() / cov else 0.0
        println("score: n=$n cobertura=${cov.toDouble() / n} precision=$precision")
        assertTrue("precisión de score $precision < 0.90", precision >= 0.90)
    }

    @Test
    fun accuracyAgainstGroundTruthDoesNotRegress() {
        val rows = fixture.getJSONArray("rows")
        var n = 0; var ok = 0; var cov = 0
        for (i in 0 until rows.length()) {
            val r = rows.getJSONObject(i)
            val gt = r.getJSONObject("gt")
            if (gt.isNull("song")) continue
            n++
            val got = PiuOcr.interpret(r.getJSONObject("native").toString(), matcher)
            val v = got.song.value ?: continue
            cov++
            if (SongMatcher.normalize(v) == SongMatcher.normalize(gt.getString("song"))) ok++
        }
        val precision = if (cov > 0) ok.toDouble() / cov else 0.0
        val accuracy = if (n > 0) ok.toDouble() / n else 0.0
        println("song: n=$n cobertura=${cov.toDouble() / n} precision=$precision acierto=$accuracy")
        // Piso: por debajo de esto el gate MIN_SONG_MARGIN dejó de proteger
        // (medido 0.972 a 0.015; a 0.010 cae a 0.90).
        assertTrue("precisión de canción $precision < 0.85", precision >= 0.85)
        assertTrue("acierto de canción $accuracy < 0.60", accuracy >= 0.60)
    }
}
