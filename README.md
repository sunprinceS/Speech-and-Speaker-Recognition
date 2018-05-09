DT2119: Speech and Speaker Recognition
===

Lecturer: Prof. [Giampiero Salvi](http://www.speech.kth.se/~giampi/)

This is the [course](https://www.kth.se/student/kurser/kurs/DT2119?l=en) I took during my exchange in KTH in Sweden.

The course note is [here](https://sunprinces.github.io/learning/topics/speech-processing/) (it's in Chinese because I think there is sufficient resource in English on the internet 😃)

## Content

* Assignment1: Feature extraction and comparison
  * Step-by-step MFCC and survey in detail through experiment
  * Dynamic Time Wrapping algorithm to compare utterances
* Assignment2: Hidden Markov Model
  * Forward/Backward algorithm to evaluate the likelihood of state sequence
  * Viterbi algorithm to find the optimal state sequence
  * Baum-Welch Algorithm to train the phoneme model
* Assignment3: Phoneme Recognition with DNN
  * Use Assignment1 to extract audio feature
  * Use Assignment2 to force-align phoneme as label
  * Train phoneme recognticizer implemented in Deep Neural Network
