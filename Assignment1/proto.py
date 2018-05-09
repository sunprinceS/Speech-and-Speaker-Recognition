# DT2119, Lab 1 Feature Extraction

import numpy as np
from scipy import signal
from scipy import fftpack
from tools import trfbank, lifter
# Function given by the exercise ----------------------------------

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    return lifter(ceps, liftercoeff)

def melspec(samples,winlen = 400,winshift = 200,preempcoeff=0.97,nfft=512,samplingrate=20000):
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)
# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    frames = np.array(samples[0:winlen])
    for i in range(winshift,len(samples)-winlen,winshift):
        frames = np.vstack([frames,samples[i:i+winlen]])
    return frames

    # frames = np.array(np.take(samples,range(0,winlen),mode='wrap'))
    # for i in range(winshift,len(samples)-winlen,winshift):
        # frames = np.vstack([frames,np.take(samples,range(i,i+winlen),mode='wrap')])
    # return frames
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    return signal.lfilter([1,-p],[1],input,axis=1) #default is the last axis (so the same!)

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    ham_window = signal.hamming(input.shape[1],sym=False)
    return (ham_window * input)

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    return np.abs(fftpack.fft(input,nfft))**2


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    mel_filter_bank = trfbank(samplingrate,input.shape[1])

    return np.log(np.dot(input,mel_filter_bank.T))

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    # mfcc = fftpack.realtransforms.dct(input,norm='ortho')[:,:nceps]
    # lmfcc = lifter(mfcc[:,:nceps])
    return fftpack.realtransforms.dct(input,norm='ortho')[:,:nceps]

def localDist(utt1,utt2,dist = None):
    if dist is None:
        dist = lambda utt1,utt2: np.linalg.norm(utt1-utt2)

    N, M = utt1.shape[0], utt2.shape[0]

    ret = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            ret[i][j] = dist(utt1[i],utt2[j])
    return ret

def dtw(x, y, normalize = True,dist = None):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    if dist is None:
        dist = lambda utt1,utt2: np.linalg.norm(utt1-utt2)
    N,M = x.shape[0], y.shape[0]

    LD = localDist(x,y,dist)
    AD = np.zeros((N,M))

    AD[0][0] = LD[0][0]
    for i in range(1,N):
        AD[i][0] = AD[i-1][0] + LD[i][0]
    for j in range(1,M):
        AD[0][j] = AD[0][j-1] + LD[0][j]

    for i in range(1,N):
        for j in range(1,M):
            AD[i][j] = min(AD[i-1][j],AD[i-1][j-1],AD[i][j-1]) + LD[i][j]
    # d = AD[N-1][M-1] * (N+M) / np.linalg.norm(AD[N-1][M-1])

    # path = []
    # i, j = N-1, M-1
    # while (i,j) != (0,0):
        # path.append((i,j))
        # i,j = min((i-1,j),(i-1,j-1),(i,j-1),key = lambda e:AD[e[0]][e[1]])
    # path.append((0,0))
    if normalize:
        return AD[N-1][M-1]/(N+M), LD, AD
    else:
        return AD[N-1][M-1], LD, AD



if __name__ == "__main__":

    ## Check DTW correctness
    a = np.array([[1],[3],[5],[7]])
    b = np.array([[1],[5],[7],[0]])
    d,LD,AD,path = dtw(a,b,lambda x,y: sum(abs(x-y)))
    print(d)
    print(LD)
    print(AD)
    print(path)
