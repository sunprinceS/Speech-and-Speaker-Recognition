import numpy as np

from tools2 import *

def concatHMMs(hmmmodels, namelist, digit):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of states in each HMM model (could be different for each)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    dim_feat = 13
    num_state = len(namelist) * 3 # including the last sink state
    # for ph in namelist:
        # num_state += hmmmodels[ph]['means'].shape[0]

    ret = {}
    ret['digit'] = digit
    ret['startprob'] = np.zeros(num_state+1)
    ret['startprob'][0] = 1.0

    ret['means'] = np.zeros((num_state,dim_feat))
    ret['covars'] = np.zeros((num_state,dim_feat))
    ret['transmat'] = np.zeros((num_state+1,num_state+1))

    for idx,ph in enumerate(namelist):
        ret['means'][3*idx:3*(idx+1),:] = hmmmodels[ph]['means']
        ret['covars'][3*idx:3*(idx+1),:] = hmmmodels[ph]['covars']
        ret['transmat'][3*idx:3*idx+4,3*idx:3*idx+4] = hmmmodels[ph]['transmat']

    ret['transmat'][-1,-1] = 1.0 # bug in sample???

    return ret

# def concatAnyHMMs(hmmmodels,namelist):
    # ret = {}
    # ret['startprob'] = np
    # ret['means'] = np
    # ret['covars'] = np
    # ret['transmat'] = np
    # return ret

def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    num_state = log_startprob.shape[0] - 1
    num_frame = log_emlik.shape[0]
    
    logalpha = np.zeros((num_frame,num_state))

    for j in range(num_state):
        logalpha[0][j] = log_startprob[j] + log_emlik[0][j]

    for i in range(1,num_frame):
        for j in range(num_state):
            logalpha[i][j] = logsumexp(logalpha[i-1,:]+log_transmat[:,j]) +log_emlik[i][j]

    return logalpha

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    num_state = log_startprob.shape[0] - 1
    num_frame = log_emlik.shape[0]

    logbeta = np.zeros((num_frame,num_state))

    for j in range(num_state-1,-1,-1):
        logbeta[num_frame-1][j] = 0

    for i in range(num_frame-2,-1,-1):
        for j in range(num_state):
            logbeta[i][j] = logsumexp(logbeta[i+1,:]+log_emlik[i+1,:]+log_transmat[j,:])

    return logbeta

def viterbi(log_emlik, log_startprob, log_transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    num_state = log_startprob.shape[0] - 1
    num_frame = log_emlik.shape[0]
    logdelta = np.zeros((num_frame,num_state))
    ptr = np.zeros((num_frame,num_state),dtype=np.int8) # the first row is dummy
    best_path = np.zeros(num_frame,dtype=np.int8)

    for j in range(num_state):
        logdelta[0][j] = log_startprob[j] + log_emlik[0][j]

    for i in range(1,num_frame):
        for j in range(num_state):
            tmp = logdelta[i-1,:] + log_transmat[:,j]
            logdelta[i][j] = np.max(tmp) + log_emlik[i][j]
            ptr[i][j] = np.argmax(tmp)
 
    best_path[-1] = np.argmax(logdelta[-1,:])
    for i in range(2,num_frame+1):
        best_path[-i] = ptr[-i+1][best_path[-i+1]]

    return np.max(logdelta[-1,:]), best_path

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    loggamma = np.zeros(log_alpha.shape)
    num_frame,num_state = loggamma.shape

    for i in range(num_frame):
        loggamma[i] = log_alpha[i,:] + log_beta[i,:] - logsumexp(log_alpha[-1])
    return loggamma

def updateMeanAndVar(X, log_gamma, prev_means, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """

    num_state = log_gamma.shape[1]
    num_feat = X.shape[1]

    means = np.zeros((num_state,num_feat))
    covars = np.zeros((num_state,num_feat))

    normFactor = np.exp(logsumexp(log_gamma))

    means = np.dot(np.exp(log_gamma).T,X)
    for i in range(num_state):
        covars[i] = np.dot(np.exp(log_gamma[:,i]),(X-prev_means[i])**2)
    for i in range(num_state):
        means[i] /= normFactor[i]
        covars[i] /= normFactor[i]
    np.clip(covars,5.0,None,covars)
    return means,covars
