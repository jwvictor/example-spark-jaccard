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
  * In this demonstration, I go through two approaches to computing the Jaccard distances of words in a corpus:
  *
  *   1. a variant that broadcasts the source documents, hence assuming that the full corpus can fit on a single node; and
  *   2. a variant that uses only RDD operations until the final persistence call.
  *
  * I am skeptical of whether variant #2 has an advantage over #1 in terms of scalability. Ultimately, I think either
  * will end up trying to pull the corpus into memory on a single node.
  *
  * Notes on liberties I took in this demonstration:
  *
  *  - I only calculate the co-occurrences (intersection) data once, using a fully distributed method. In the variant with
  *    the corpus available as a broadcast variable, one could use various methods to speed up the processing by using
  *    the data in the broadcast variable. However, this is a tricky question, and one I did not address here.
  *
  *  - Long chains of operators are avoided. This is for readability.
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
    val inputFileName = "example.txt" //"README.md"
    val outFileNameSuffix = "output.dat" // Suffix of where to write to
    val COALESCE_TO = 32 // This should be selected in some intelligent way based on the infrastructure setup.

    //
    // CORPUS PREPARATION
    //

    // Treat each line in file as a "document"
    val cachedText = sc.textFile(inputFileName)
    val tokenizedText = cachedText.map(tokenizeInputString).cache

    //
    // FINDING CO-OCCURRENCES (I.E. THE INTERSECTION SIZE)
    //   (common to both approaches)
    //

    // Get co-occurrences -- i.e. X /\ Y -- the scalable way
    // (not using broadcast variable for expository purposes).
    val coOccurrenceCountsRdd = cachedText.
      flatMap(getWordCoOccurrences).
      filter(t => nonEmptyKey(t._1)). // Some keys may be invalid
      coalesce(COALESCE_TO).  // Coalesce after possible uneven filter
      reduceByKey(_ + _).
      cache

    //
    // APPROACH #1: COLLECT AND BROADCAST TOKENIZED TEXT
    // COMPUTE UNIONS USING THE BROADCAST VALUE.
    //

    // Broadcast tokenized text to each node for union computation
    val bcTokenizedText = sc.broadcast(tokenizedText.collect)

    // Get the union values -- i.e. X \/ Y -- the less scalable way
    val distinctPairs = coOccurrenceCountsRdd.map(_._1).cache
    val keyedOrCounts = distinctPairs.
      map(x => {
        val w1 = x._1
        val w2 = x._2
        val z = bcTokenizedText.value.count(doc => doc.contains(w1) || doc.contains(w2))
        ((w1, w2), z)
      })

    // And get the indices
    val jaccardDistances = coOccurrenceCountsRdd.
      join(keyedOrCounts).
      map(x => (x._1, x._2._1.toDouble / x._2._2)).
      sortByKey(true)

    //
    // APPROACH #2: MAKE THE WHOLE SPACE AS AN RDD AND USE KEYS TO GROUP THE OR OPERATIONS.
    //

    // Cartesian the pairs against the documents, group by pairs, and compute the unions over
    // an iterable of documents.
    val distribOrs = distinctPairs.
      cartesian(cachedText).
      groupByKey.
      map(pairDocs => {
        val pair = pairDocs._1
        val docs = pairDocs._2
        val unionSize = docs.count(d => d.contains(pair._1) || d.contains(pair._2))
        (pair, unionSize)
      })

    val distribJaccard = coOccurrenceCountsRdd.
      join(distribOrs).
      map(x => (x._1, x._2._1.toDouble / x._2._2)).
      sortByKey(true)

    // Persist to disk -- for expository purposes
    jaccardDistances.map(_.toString).saveAsTextFile(s"jac.$outFileNameSuffix")
    distribJaccard.map(_.toString).saveAsTextFile(s"djac.$outFileNameSuffix")
    coOccurrenceCountsRdd.map(_.toString).saveAsTextFile(s"cooc.$outFileNameSuffix")
    // Print locally -- for expository purposes
    jaccardDistances.foreach(c => println(s"[OUTPUT]  Pair ${c._1} has Jaccard index ${c._2}"))
    distribJaccard.foreach(c => println(s"[OUTPUT]  Distributed pair ${c._1} has Jaccard index ${c._2}"))
    keyedOrCounts.foreach(c => println(s"[OUTPUT]  OR ${c._1} has value ${c._2}"))
    coOccurrenceCountsRdd.foreach(c => println(s"[OUTPUT]  AND ${c._1} has value ${c._2}"))

    //occurrenceRdd.foreach(x => println(s"Occurrence: $x"))


  }
}