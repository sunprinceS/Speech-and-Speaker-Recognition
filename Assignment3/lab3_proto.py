import numpy as np
import copy
from sklearn.mixture import log_multivariate_normal_density
from lab3_tools import *
from proto import *
from proto2 import *
from states import *

def words2phones(wordList, pronDict, addSilence=True, addShortPause=False):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phs = copy.copy(pronDict[wordList[0]])
    for word in wordList[1:]:
        if addShortPause:
            phs += ['sp']
        phs += pronDict[word]

    if addSilence:
        phs = ['sil'] + phs + ['sil']
    return phs

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans, filename):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """
    utteranceHMM = concatHMMs(phoneHMMs,phoneTrans,digit=filename[:-1])
    stateTrans = [phone +  '_' + str(stateid) for phone in phoneTrans for stateid in range(nstates[phone])]

    from sklearn.mixture import log_multivariate_normal_density
    import warnings; warnings.simplefilter('ignore')
    obsloglik = log_multivariate_normal_density(lmfcc,utteranceHMM['means'],utteranceHMM['covars'],'diag')
    _,viterbiStateIdTrans = viterbi(obsloglik,np.log(utteranceHMM['startprob']),np.log(utteranceHMM['transmat'][:-1,:-1]))
    return [stateTrans[idx] for idx in viterbiStateIdTrans]


def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """
