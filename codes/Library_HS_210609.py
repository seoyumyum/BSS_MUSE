#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

import itertools 

def Tshift(arr,tau,fs):
    num = (tau*fs).astype(int)
    arr=np.roll(arr,num)
    if num<0:
         np.put(arr,range(len(arr)+num,len(arr)),0)
    elif num > 0:
         np.put(arr,range(num),0)
    return arr

def Tshift_Ups(arr,tau,fs,factor):
    arr_ups = signal.resample(arr, len(arr)*factor)
    fs_ups = fs*factor
    
    num = (tau*fs_ups).astype(int)
    
    arr_ups=np.roll(arr_ups,num)

    if num<0:
         np.put(arr_ups,range(len(arr_ups)+num,len(arr_ups)),0)
    elif num > 0:
         np.put(arr_ups,range(num),0)
         
    arr_ups_LPF = butter_lowpass_filter(arr_ups, fs, fs_ups, 10)        
    return arr_ups_LPF[::factor]

def ArrayVec_deg_WB(M,Angle_deg, freq, vp, d):
    phi = 2*np.pi*freq*d*np.sin(Angle_deg*np.pi/180)/vp
    a=[]
    for i in range(M):
        a.append(np.exp(i*-1j*phi))
    return np.asarray(a)

def ArrayVec_deg_WB_CentRef(M,Angle_deg, freq, vp,d):
    phi = 2*np.pi*freq*d*np.sin(Angle_deg*np.pi/180)/vp
    a=[]
    for i in range(M):
        a.append(math.exp(((-1)**i)*-1j*phi/2))
    return np.asarray(a)

def ArrayVec_deg_WB_COS(M,Angle_deg, freq, vp,d):
    phi = 2*np.pi*freq*d*np.cos(Angle_deg*np.pi/180)/vp
    a=[]
    for i in range(M):
        a.append(math.e**((i*1j*phi)))
    return np.asarray(a)



def deg2ind(deg_list):              #Convert angle in degree to index of Angle_Sweep   Updated on 15th Nov. 2020
    if type(deg_list) != list:
        deg_list = np.array([deg_list])
    index_list = np.array([])
    for deg in deg_list:
        index_list = np.append(index_list, (N_Angle-1)*(deg-theta_start)/(theta_end - theta_start))
    index_list = index_list.astype(int)
    return index_list

def MixGen(SigVec, Angle, M, fs_true, SNR, factor_ups, vp, d):
    X = np.zeros((M,len(SigVec[0,:])))
    K = len(SigVec[:,0])
    ## Convolutive Mixing using upsampling 
    for i in range(M):
        for j in range(K):
            X[i,:] += Tshift_Ups(SigVec[j,:], -i*d*np.cos(Angle[j]*np.pi/180)/vp, fs_true, factor_ups)
            #X[i,:] += Tshift(SigVec[j,:], -i*d*np.cos(Angle[j]*np.pi/180)/vp, fs_true)

    Disp = 500

    # Noise vector
    #np.random.seed(0)
    SigMaxVar = np.max(np.var(SigVec,axis=1))/2
    noiseVec =  np.random.normal(0,np.sqrt(SigMaxVar/(10**(SNR/10))),size=(M,len(SigVec[0,:])))    #Noise in time domain is real

    X = X + noiseVec
    return X


def Rxx_Gen_WB_t2f(Xs,I,Ns):
    mu = 1e-7
    N = len(Xs[:,0])
    Xs_f = np.zeros((N,1,I,Ns), dtype='complex')     # MIC number x 1 x snapshot x basis number

    for i in range(I):
        Xs_f[:,0,i,:] = np.fft.fft(Xs[:,Ns*i:Ns*i+Ns],axis=1)
    Xs_f = np.fft.fftshift(Xs_f,axes=3)                     # DC is centered 

    # Covariance Matrix
    Rxx_f = np.zeros((N,N,Ns),dtype=complex)
    Rxx_f_inv = np.zeros((N,N,Ns),dtype=complex)
    Rxx_f_raw = np.zeros((N,N,I,Ns),dtype=complex)
    for j in range(len(Xs_f[0,0,0,:])):
        for i in range(I):
            Rxx_f_raw[:,:,i,j] = Xs_f[:,:,i,j] @ np.conjugate(Xs_f[:,:,i,j].T) 
        Rxx_f = np.average(Rxx_f_raw, axis = 2 )
        Rxx_f[:,:,j] = Rxx_f[:,:,j] + mu * np.eye(N)
        Rxx_f_inv[:,:,j] = np.linalg.inv(Rxx_f[:,:,j])
    return Rxx_f, Rxx_f_inv, Xs_f              # All in frequency domain


def BF_MVDR_WB_f(X_f, Rxx, Rxx_inv, Angle_I, freq, vp=340):

    BF = np.zeros((len(Angle_I),1),dtype='complex')      
    for i in range(len(Angle_I)):
        a_theta = ArrayVec_deg_WB(N,Angle_I[i], freq, vp)
        W = (Rxx_inv@a_theta)/(np.conjugate(a_theta.T)@Rxx_inv@a_theta)
        BF[i]=np.conjugate(W.T)@X_f
    return BF

def BF_MVDR_WB_Weights(Rxx, Rxx_inv, Angle_I, freq, d, vp=340):

    M = len(Rxx[:,0])
    W = np.zeros((M,len(Angle_I)), dtype='complex')      
    for k in range(len(Angle_I)):
        a_theta = ArrayVec_deg_WB(M,Angle_I[k], freq, vp, d)
        W[:,k] = (Rxx_inv@a_theta)/(np.conjugate(a_theta.T)@Rxx_inv@a_theta)
    return W


def BF_MVDR_WB(Xs_f_tensor, Rxx_f,Rxx_f_inv, Angle_I, freq, d):
    Ns = len(Xs_f_tensor[0,0,0,:])
    I = len(Xs_f_tensor[0,0,:,0])
    BF_f = np.zeros((len(Angle_I),Ns*I), dtype='complex')
    BF = np.zeros((len(Angle_I),Ns*I), dtype='complex')
    
    for i in range(I):
        Xs_f = Xs_f_tensor[:,0,i,:]
        #print(Xs_f.shape)

        for j in range(Ns):
  
            BF_f[:,Ns*i+j] = np.conjugate(BF_MVDR_WB_Weights(Rxx_f[:,:,j], Rxx_f_inv[:,:,j], Angle_I, freq[j],d).T)@Xs_f[:,j]
            
        BF[:,Ns*i:Ns*i+Ns] = np.fft.ifft(np.fft.fftshift(BF_f[:,Ns*i:Ns*i+Ns],axes=1),axis=1)
    return BF

def AoA_MVDR_WB(Rxx, Rxx_inv, freq, Angle_sweep, vp, d):
    M = len(Rxx[:,0])
    N_Angle = len(Angle_sweep)
    AoA_sweep = np.zeros((N_Angle))      
    for i in range(N_Angle):
        a_theta = ArrayVec_deg_WB(M,Angle_sweep[i],freq, vp, d) 
        W = (Rxx_inv@a_theta)/(np.conjugate(a_theta.T)@Rxx_inv@a_theta)
        AoA_sweep[i]=np.abs(np.conjugate(W.T)@Rxx@W)
    return AoA_sweep 

def AoA_MVDR_WB_COS(Rxx, Rxx_inv, freq, Angle_sweep, vp, d,N_Angle):
    M = len(Rxx[:,0])

    AoA_sweep = np.zeros((N_Angle))      
    for i in range(N_Angle):
        a_theta = ArrayVec_deg_WB_COS(M,Angle_sweep[i],freq, vp, d) 
        W = (Rxx_inv@a_theta)/(np.conjugate(a_theta.T)@Rxx_inv@a_theta)
        AoA_sweep[i]=np.abs(np.conjugate(W.T)@Rxx@W)
    return AoA_sweep 

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #y = signal.lfilter(b, a, data)           #With actual delay
    y = signal.filtfilt(b,a, data)            #without delay
    return y
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    #y = signal.lfilter(b, a, data)           #With actual delay
    y = signal.filtfilt(b,a, data)            #without delay
    return y

