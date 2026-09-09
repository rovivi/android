package com.piu.ocr

import org.json.JSONObject
import kotlin.math.max
import kotlin.math.min

/**
 * Matching difuso del título contra el catálogo cerrado de 675 canciones.
 *
 * Vive en Kotlin y no en C++ a propósito: es manipulación de strings, no es el
 * cuello de latencia, y evita reimplementar SequenceMatcher en C++.
 *
 * El OCR crudo es feísimo — `waodingGrashorg`, `DDATTUGnux` — y aun así el
 * catálogo cerrado lo rescata. No hace falta leer bien, hace falta leer parecido.
 */
class SongMatcher(catalogJson: String) {

    data class Song(val name: String, val charts: List<Pair<String, Int>>) {
        val norm = normalize(name)
        val trigrams = trigramsOf(norm)
        val compact = norm.replace(" ", "")
    }

    data class Candidate(val name: String, val score: Double, val levels: List<Int>)

    private val songs: List<Song> = JSONObject(catalogJson).let { root ->
        root.keys().asSequence().filter { !it.startsWith("_") }.map { k ->
            val o = root.getJSONObject(k)
            val ch = o.optJSONArray("c")
            val charts = buildList {
                for (i in 0 until (ch?.length() ?: 0)) {
                    val c = ch!!.getJSONArray(i)
                    if (!c.isNull(1)) add(c.getString(0) to c.getInt(1))
                }
            }
            Song(o.getString("n"), charts)
        }.toList()
    }

    /**
     * @param chartType single/double/halfdouble/coop, o null.
     *   El NIVEL a propósito no es parámetro: `match` lo usaría como filtro duro
     *   y el nivel que tenemos en inferencia es LEÍDO, no sabido. Un nivel mal
     *   leído borra la canción correcta. Medido: el cruce con el nivel vale
     *   +3 pts cuando el nivel es correcto y es pérdida neta cuando no.
     */
    fun match(text: String, chartType: String? = null, topK: Int = 2): List<Candidate> {
        val q = normalize(text)
        if (q.isEmpty()) return emptyList()
        val qt = trigramsOf(q)
        val qc = q.replace(" ", "")

        val want = when (chartType) {
            "single" -> "s"; "double" -> "d"; "halfdouble" -> "hd"; "coop" -> "c"
            else -> null
        }
        var pool = songs.filter { s ->
            want == null || s.charts.isEmpty() ||
                s.charts.any { it.first == want || it.first.isEmpty() }
        }
        if (pool.isEmpty()) pool = songs

        // difflib es el 90 % del costo del matching (dos pasadas por canción,
        // 675 canciones). El Jaccard de trigramas es aritmética de conjuntos y
        // ordena la misma región del catálogo, así que se puntúa todo con eso y
        // solo se paga la parte cara sobre la cabeza. Medido, 60/120/240 y sin
        // prefiltro dan el mismo top-1.
        if (pool.size > PREFILTER) {
            pool = pool.sortedByDescending { jaccard(qt, it.trigrams) }.take(PREFILTER)
        }

        return pool.map { s ->
            val tri = jaccard(qt, s.trigrams)
            val ratio = similarity(qc, s.compact)
            val partial = longestCommon(qc, s.compact).toDouble() /
                max(min(qc.length, s.compact.length), 1)
            Candidate(s.name, 0.45 * tri + 0.35 * ratio + 0.20 * partial,
                      s.charts.map { it.second }.distinct().sorted())
        }.sortedByDescending { it.score }.take(topK)
    }

    /** Niveles que el catálogo permite para esa canción y ese step type. */
    fun levelsFor(name: String, chartType: String?): List<Int> {
        val s = songs.firstOrNull { it.name == name || it.norm == normalize(name) }
            ?: return emptyList()
        val want = when (chartType) {
            "single" -> "s"; "double" -> "d"; "halfdouble" -> "hd"; "coop" -> "c"
            else -> null
        }
        val lv = s.charts.filter { want == null || it.first == want || it.first.isEmpty() }
            .map { it.second }.distinct().sorted()
        return lv.ifEmpty { s.charts.map { it.second }.distinct().sorted() }
    }

    companion object {
        /** Holgado respecto de los 60 que ya bastaban: el margen del gate
         *  depende del segundo candidato, no solo del primero. */
        const val PREFILTER = 120

        fun normalize(s: String): String = s.lowercase()
            .replace(Regex("[^a-z0-9\\s]"), " ")
            .replace(Regex("\\s+"), " ").trim()

        fun trigramsOf(n: String): Set<String> {
            val s = "  ${n.replace(" ", "")}  "
            return if (s.length >= 3) (0..s.length - 3).map { s.substring(it, it + 3) }.toSet()
                   else setOf(s)
        }

        private fun jaccard(a: Set<String>, b: Set<String>): Double {
            val inter = a.count { it in b }
            val union = a.size + b.size - inter
            return if (union > 0) inter.toDouble() / union else 0.0
        }

        /** Equivalente a SequenceMatcher.ratio() de difflib. */
        private fun similarity(a: String, b: String): Double {
            if (a.isEmpty() && b.isEmpty()) return 1.0
            val m = lcs(a, b)
            return 2.0 * m / (a.length + b.length)
        }

        private fun lcs(a: String, b: String): Int {
            val dp = IntArray(b.length + 1)
            for (i in a.indices) {
                var prev = 0
                for (j in b.indices) {
                    val tmp = dp[j + 1]
                    dp[j + 1] = if (a[i] == b[j]) prev + 1 else max(dp[j + 1], dp[j])
                    prev = tmp
                }
            }
            return dp[b.length]
        }

        /** Corrida común más larga: un recorte de título viene cortado por el
         *  borde del cuadro, así que un prefijo que calza no debe penalizarse. */
        private fun longestCommon(a: String, b: String): Int {
            if (a.isEmpty() || b.isEmpty()) return 0
            var best = 0
            val dp = IntArray(b.length + 1)
            for (i in a.indices) {
                var prev = 0
                for (j in b.indices) {
                    val tmp = dp[j + 1]
                    dp[j + 1] = if (a[i] == b[j]) prev + 1 else 0
                    best = max(best, dp[j + 1])
                    prev = tmp
                }
            }
            return best
        }
    }
}
