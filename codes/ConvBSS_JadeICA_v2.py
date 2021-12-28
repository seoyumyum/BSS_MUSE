#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
from numpy.linalg import*
import matplotlib.pyplot as plt
from numpy import r_, exp, cos, sin, pi, zeros, ones, hanning, sqrt, log, floor, reshape, mean
from scipy import signal
from numpy.fft import fft
import math
import time
import scipy.optimize as opt

import scipy.io as sio
import scipy.io.wavfile
import sounddevice as sd
from IPython.display import Audio

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import copy
get_ipython().run_line_magic('matplotlib', 'inline')
import librosa
import librosa.display

import importlib
import jade as jica
importlib.reload(jica)

import Library_HS_210609 as lib_HS 
importlib.reload(lib_HS)

import itertools 

def ConvBSS_JadeICA(X, fs_true, N_fft=1024, N_hop=0.25*1024, M=4, K=None, f_LPF=None):
    
    ### STFT block ###
    f, t, _ = signal.stft(X[0,:], fs_true, nperseg=N_fft, noverlap = N_fft-N_hop)
    X_ft = np.zeros((M,len(f),len(t)), dtype='complex')
    print('(M,f,t): ', X_ft.shape)
    for i in range(M):
        _, _, X_ft[i,:,:] = signal.stft(X[i,:], fs_true, nperseg=N_fft, noverlap = N_fft-N_hop)   
    # hopsize H = nperseg - noverlap


    #magnitudeZxx = np.abs(X_ft[0,:,:])
    #log_spectrogramZxx = librosa.amplitude_to_db(magnitudeZxx)#-60

    #plt.figure(figsize=(10,4))
    #librosa.display.specshow(log_spectrogramZxx, sr=fs_true, x_axis='time', y_axis='linear', hop_length=N_hop) #,cmap=plt.cm.gist_heat)
    #plt.xlabel("Time")
    #plt.ylabel("Frequency")
    #plt.colorbar(format='%+2.0f dB')
    #plt.title("Spectrogram (dB)")


    ### Separation Step Using JadeICA ###

    A_f = np.zeros((len(f),M,K), dtype='complex')
    Y_ft = np.zeros((len(f), K, len(t)), dtype='complex')
    W_f = np.zeros((len(f), K, K), dtype='complex')
    V_f = np.zeros((len(f), K, M), dtype='complex')

    # at each f
    for i in range(len(f)):
        A_tmp,Y_tmp,V_tmp,W_tmp = jica.jade(X_ft[:,i,:],m=None,max_iter=3000)
        A_f[i,:,:] = A_tmp    # freq x M x K   (estimated mixing matrix)
        Y_ft[i,:,:] = Y_tmp    # freq x K x T   (estimated source signals)
        W_f[i,:,:] = W_tmp    # freq x K x K   (unmixing matrix)
        V_f[i,:,:] = V_tmp    # freq x K x M   (Sphering/Whitening matrix)


    ### Permutation Resolve method: PowerRatio-correlation approach ###

    # Calculate PowerRatio
    PowR = np.zeros((len(f), K, len(t)))

    for freq in range(len(f)):
        for tau in range(len(t)):
            PowSum=0
            for i in range(K):    
                PowR[freq,i,tau] = np.linalg.norm(A_f[freq,:,i]*Y_ft[freq,i,tau],ord=2)**2    
                PowSum += PowR[freq,i,tau]
            PowR[freq,:,tau] /= PowSum     

            
    ##### Correlation Algorithm - centroid 

    cent_k = np.sum(PowR,axis=0)/len(f)    ## initialization: K x T

    #for freq in f:
    Sigma_f = np.arange(K)
    Y_ft_PowR = copy.deepcopy(Y_ft)
    A_f_PowR = copy.deepcopy(A_f)

    index_argmax = np.zeros((len(f),K),dtype='int')

    flag = 0
    N_trial = 100
    for n in range(N_trial):
        #print('(---n =',n,'---)')    
        #index_argmax_tmp = copy.deepcopy(index_argmax)
        cent_k_temp = copy.deepcopy(cent_k)
        PowR_sum = np.zeros((K,len(t)))
        for freq in range(len(f)):
            #print('---f =',freq,'---')
            index_argmax[freq,:]=Sigma_f

            Sum_max = -1
            for i in np.array(list(itertools.permutations(Sigma_f))):

                Sum_l = np.zeros(K)
                for l in range(K):
                    Sum_l[l] = np.corrcoef(PowR[freq,i[l],:],cent_k[Sigma_f[l],:])[1,0]
                Sum = np.sum(Sum_l)

                if Sum > Sum_max:
                    Sum_max = Sum
                    index_argmax[freq,:] = i

            PowR_sum += PowR[freq,index_argmax[freq,:],:]

        cent_k = PowR_sum/len(f)

        #print(cent_k[:,:3])
        ##print('(iteration = ',n,')','loss = ',np.sum(np.abs(cent_k_temp-cent_k)))
        if np.sum(np.abs(cent_k_temp-cent_k))<1e-5:
            flag += 1
        #print('flagCount = ', flag)
        if flag == 5:
            print('flagCount = ', flag)
            break

    for freq in range(len(f)):        
        Y_ft_PowR[freq,:,:] = Y_ft[freq,index_argmax[freq,:] ,:]
        A_f_PowR[freq,:,:] = A_f[freq,:,index_argmax[freq,:]].T
        
    ### Scaling after Clustering
    Y_ft_scale = copy.deepcopy(Y_ft_PowR)
    for freq in range(len(f)):
        for i in range(K):
            Y_ft_scale[freq,i,:] = A_f_PowR[freq,0,i]*Y_ft_PowR[freq,i,:]

    ## istft check
    y_t = np.zeros((K,len(X[0,:])))
    for i in range(K):
        _, y = signal.istft(Y_ft_scale[:,i,:], fs_true, nperseg=N_fft, noverlap = N_fft-N_hop)
        y_t[i,:]=y[:len(X[0,:])]
        if f_LPF != None:
            y_t[i,:] = lib_HS.butter_lowpass_filter(y_t[i,:], f_LPF, fs_true, 10)   ## LPF 
       
    return y_t,Y_ft_scale, A_f_PowR 

