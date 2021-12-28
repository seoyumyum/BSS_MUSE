#!/usr/bin/env python
# coding: utf-8

# In[7]:


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

import Library_HS_210609 as lib_HS 
importlib.reload(lib_HS)

import itertools 


def AoA_MVDR(Rxx_f, Rxx_f_inv, freq_s, Ns, vp, d):

    # Raw spectrum of AoA
    dtheta = 1          #Angle resolution
    theta_start = -90
    theta_end = 90
    N_Angle = 1+int((theta_end - theta_start)/dtheta)
    Angle_sweep = np.linspace(theta_start,theta_end,N_Angle)  
    AoA_sweep = np.zeros((N_Angle,Ns))       # These are amplitudes in real number (not complex value)

    for j in range(Ns):
        AoA_sweep[:,j] = lib_HS.AoA_MVDR_WB(Rxx_f[:,:,j], Rxx_f_inv[:,:,j], freq_s[j], Angle_sweep, vp, d)
    AoA_sweep_norm = AoA_sweep/np.max(AoA_sweep)        
    ##############
    fig1 = plt.figure(figsize = (8.6,4))
    CS = plt.contourf(Angle_sweep, freq_s[:len(AoA_sweep_norm[0,:])]*1e-3, 10*np.log10(AoA_sweep_norm).T, levels= np.linspace(-50,0,21),cmap='Reds')
    plt.colorbar(CS,orientation='vertical')
    plt.xlabel('AoA [deg]')
    plt.ylabel('Freq[kHz]')
    plt.tight_layout()
    plt.grid(True,axis='x')
    plt.xticks(np.arange(-90,100,30))
    plt.xlim(-90,90)
    plt.ylim(0,5)
    plt.show()    
    #############################    
        
        
       
     

    AoA_1D = np.linalg.norm(AoA_sweep_norm,ord=2, axis=1)**2
    
    fig = plt.figure(figsize = (8,2))
    plt.xlim(-90, 90)
    #plt.ylim(-80, 20)
    plt.xticks(np.arange(-90,100,10))
    plt.grid(True)
    plt.plot(Angle_sweep,10*np.log10(AoA_1D))
    plt.xlabel('Incident angle [deg]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Angle of Arrival Estimation')

    peaks_AoA, _ = signal.find_peaks(AoA_1D, prominence=np.max(AoA_1D)*0.1)
    print("peak_ratio:",Angle_sweep[peaks_AoA])
    return Angle_sweep[peaks_AoA]

def ConvBSS_MVDR_AoA(X, fs_true, d, N_snap, N_samp, M, K, vp=340, f_LPF=None, f_ADC=None):
  
    I = N_snap
    Ns = N_samp 

    ############## Lowpass filter #############
    X_LP = np.zeros((M,len(X[0,:])))
    if f_LPF == None:
        X_LP = X
    else:
        for i in range(M):
            X_LP[i,:] =  lib_HS.butter_lowpass_filter(X[i,:], f_LPF, fs_true, 10)

    ############## Sampling #############
    if f_ADC == None:
        f_ADC = fs_true   # 8820 * 2 Hz
    factor_ADC = int(fs_true/f_ADC)
    X_ADC = X_LP[:,::factor_ADC]        # in time domain

    for i in range(len(X_ADC[:,0])):
        X_ADC[i,:] = X_ADC[i,:]-np.average(X_ADC[i,:])

    t_ADC = np.arange(Ns*I)/f_ADC
 
    ############ Covariance Matrix generation    ###############
    Rxx_f, Rxx_f_inv, Xs_f_tensor = lib_HS.Rxx_Gen_WB_t2f(X_ADC,I,Ns)  # input: time series, output: covariance matrix function of freq.
    
    
    ##### MVDR block for look direction ########

    
    if Ns%2==0:
        freq_s = np.linspace(-f_ADC/2, f_ADC/2-f_ADC/Ns, Ns)
    else:
        freq_s = np.linspace(-f_ADC/2 + 0.5*f_ADC/Ns, f_ADC/2-0.5*f_ADC/Ns, Ns)
    
    
    
    Angle_Tar= AoA_MVDR(Rxx_f, Rxx_f_inv, freq_s, Ns, vp, d)
    print("Target Angles [deg]:",Angle_Tar)
    SigVec_Tar = np.real(lib_HS.BF_MVDR_WB(Xs_f_tensor, Rxx_f,Rxx_f_inv, Angle_Tar, freq_s,d))          # MVDR weight and read
    
    if f_LPF != None:
        SigVec_Tar[i,:] = lib_HS.butter_lowpass_filter(SigVec_Tar[i,:], f_LPF, fs_true, 10)   ## LPF 

    return SigVec_Tar
    
    
    
def ConvBSS_MVDR(X, fs_true, d, Angle, N_snap, N_samp, M, K, vp=340, f_LPF=None, f_ADC=None):
  
    I = N_snap
    Ns = N_samp 

    ############## Lowpass filter #############
    X_LP = np.zeros((M,len(X[0,:])))
    if f_LPF == None:
        X_LP = X
    else:
        for i in range(M):
            X_LP[i,:] =  lib_HS.butter_lowpass_filter(X[i,:], f_LPF, fs_true, 10)

    ############## Sampling #############
    if f_ADC == None:
        f_ADC = fs_true   # 8820 * 2 Hz
    factor_ADC = int(fs_true/f_ADC)
    X_ADC = X_LP[:,::factor_ADC]        # in time domain

    for i in range(len(X_ADC[:,0])):
        X_ADC[i,:] = X_ADC[i,:]-np.average(X_ADC[i,:])

    t_ADC = np.arange(Ns*I)/f_ADC
    print(X_ADC.shape)
    ############ Covariance Matrix generation    ###############
    Rxx_f, Rxx_f_inv, Xs_f_tensor = lib_HS.Rxx_Gen_WB_t2f(X_ADC,I,Ns)          # input: time series, output: covariance matrix function of freq.
    
    
    ##### MVDR block for look direction ########

    Angle_Tar= Angle
    if Ns%2==0:
        freq_s = np.linspace(-f_ADC/2, f_ADC/2-f_ADC/Ns, Ns)
    else:
        freq_s = np.linspace(-f_ADC/2 + 0.5*f_ADC/Ns, f_ADC/2-0.5*f_ADC/Ns, Ns)

    print("Target Angles [deg]:",Angle_Tar)
    SigVec_Tar = np.real(lib_HS.BF_MVDR_WB(Xs_f_tensor, Rxx_f,Rxx_f_inv, Angle_Tar, freq_s,d))          # MVDR weight and read
    
    if f_LPF != None:
        SigVec_Tar[i,:] = lib_HS.butter_lowpass_filter(SigVec_Tar[i,:], f_LPF, fs_true, 10)   ## LPF 

    return SigVec_Tar






