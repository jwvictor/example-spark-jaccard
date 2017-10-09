package org.jwvictor.example

/**
  * Copyright 2017 Jason Victor
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */

import org.apache.spark.SparkContext

/**
  * Jaccard distance example
  *
  * This is an example of a clever way to calculate Jaccard distances between words in a corpus in a
  * highly scalable manner -- implemented here in Spark. The crux of the algorithm is the use of the set-
  * theoretic identity
  *   |X \/ Y| = |X| + |Y| - |X /\ Y|
  * which computes the union of the sets using only the co-occurrence counts and single word counts.
  *
  * Note: the Jaccard distance for two words w1 and w2 is defined as |X /\ Y| / |X \/ Y|, where X and Y
  * are the number of documents in the corpus containing w1 and w2, respectively.
  *
  */
object JaccardDistancesJob {

  /**
    * Tokenize an input document into "words" (tokens)
    *
    * @param document document as string
    * @return array of tokens
    */
  private def tokenizeInputString(document: String): Array[String] = {
    document.split("\\W+").map(_.toLowerCase)
  }

  /**
    * Create a key from input tokens that is unique regardless of order of arguments
    *
    * @param s1 token 1
    * @param s2 token 2
    * @return key
    */
  private def getPairKeyFrom(s1: String, s2: String): (String, String) = {
    if (s1 > s2) (s2, s1) else (s1, s2)
  }

  /**
    * Is a valid pair key predicate. Used to filter the cartesian product
    * of word count RDDs.
    *
    * @param k a possible co-occurrence key
    * @return validity
    */
  private def isValidPairKey(k: (String, String)): Boolean = {
    k._1 <= k._2
  }

  /**
    * Is a key non-empty (i.e. worth considering)?
    *
    * @param k the tuple
    * @return is valid key
    */
  private def nonEmptyKey(k: Tuple2[String, String]): Boolean = {
    !(k._1.isEmpty || k._2.isEmpty)
  }

  /**
    * Gets co-occurrences of words in an input document
    *
    * @param inputString input document
    * @return array of pairs and a 1 suitable for Spark grouping
    */
  private def getWordCoOccurrences(inputString: String): Array[((String, String), Int)] = {
    val words = tokenizeInputString(inputString)
    words.
      flatMap(w => words.map(x => {
        // Impose ordering
        val (a, b) = getPairKeyFrom(x, w)
        // Increment count
        ((a, b), 1)
      })).
      distinct // Only count each pair once per input
  }

  /**
    * Main function
    *
    * @param args
    */
  def main(args: Array[String]): Unit = {

    // Session setup
    val sc = new SparkContext
    val inputFileName = "example.txt"
    val outFileNameSuffix = "output.dat" // Suffix of where to write to
    val COALESCE_TO = 8 // This should be selected in some intelligent way based on the infrastructure setup.

    //
    // CORPUS PREPARATION
    //

    // Treat each line in file as a "document"
    val cachedText = sc.textFile(inputFileName)


    //
    // FINDING CO-OCCURRENCES (I.E. THE INTERSECTION SIZE)
    //

    val coOccurrenceCountsRdd = cachedText.
      flatMap(getWordCoOccurrences).
      filter(t => nonEmptyKey(t._1)). // Some keys may be invalid
      coalesce(COALESCE_TO). // Coalesce after possible uneven filter
      reduceByKey(_ + _) // Reduce to get keyed numbers of documents with co-occurrences


    //
    // FINDING THE WORD COUNTS
    //

    val occurrenceRdd = cachedText.
      flatMap(x => tokenizeInputString(x).map(x => (x, 1)).distinct). // Distinct because we want number of documents
      reduceByKey(_ + _) // Reduce by key to obtain (token, number of documents containing a token) pairs


    //
    // GETTING THE CARTESIAN OF WORD COUNTS
    //

    // Get the cartesian product of the occurrences, which we can join against co-occurrences.
    //
    // NOTE: cartesian products naturally take a lot of memory, so I filter out invalid keys as quickly as
    // possible, to avoid extra strain on the subsequent `map` (but mostly `join`) operations. However,
    // this doesn't filter out _unused_ values, which would be even better, although that seems like it might
    // not be possible without relying on some kind of intermediate persistence.
    val occCartesianRdd = occurrenceRdd.
      cartesian(occurrenceRdd). // Cartesian the RDD against itself
      filter(x => isValidPairKey((x._1._1, x._2._1))). // Filter invalid keys ASAP
      map(x => ((x._1._1, x._2._1), (x._1._2, x._2._2))) // Unroll the tuples logically


    //
    // JOINING THE CO-OCCURRENCE COUNTS AGAINST THE CARTESIAN OF COUNTS
    // TO COMPUTE THE JACCARD DISTANCE
    //

    // Compute Jaccard distances by joining the co-occurrence counts with the cartesian product of word counts.
    // Note: out of concern that the hash join could cause some unevenness, I coalesce again, to be on the safe side.
    val jaccards = coOccurrenceCountsRdd.
      join(occCartesianRdd).
      coalesce(COALESCE_TO).
      map(x => {
        // The key:
        val k = x._1
        // The intersection:
        val aAndb = x._2._1
        // The word counts:
        val ct1 = x._2._2._1
        val ct2 = x._2._2._2
        // The union, given by the set-theoretic identity |X \/ Y| = |X| + |Y| - |X /\ Y|
        val aOrb = ct1 + ct2 - aAndb
        // Definition of Jaccard distance:
        val jacc = aAndb.toDouble / aOrb
        // Make a keyed RDD
        (k, jacc)
      })

    // Persist to disk.
    jaccards.map(_.toString).saveAsTextFile(s"djac.$outFileNameSuffix")

    // Print locally -- for expository purposes.
    jaccards.foreach(c => println(s"[OUTPUT]  Distributed pair ${c._1} has Jaccard index ${c._2}"))


  }
}