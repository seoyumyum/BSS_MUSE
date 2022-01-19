

## Get measured signals at MIC.
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


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import copy
get_ipython().run_line_magic('matplotlib', 'inline')
import librosa
import librosa.display
import itertools 

import rir_generator as rir

def MixGen_RIR(SigVec, Angle, Angle_rot, fs_true, SNR, vp, d, Range, Room_len, z, T_reverb, nsample, dim, order):
    H = []
    X = np.zeros((2,len(SigVec[0,:])))
    for i in range(len(Angle)):
        Angle_i = Angle[i]
        Room_loc = np.array([Room_len, Room_len, 2*z])
        MIC_loc = [[(Room_len-d*np.cos(Angle_rot*np.pi/180))/2, (Room_len-d*np.sin(Angle_rot*np.pi/180))/2, z],         
                   [(Room_len+d*np.cos(Angle_rot*np.pi/180))/2, (Room_len+d*np.sin(Angle_rot*np.pi/180))/2, z]]       
                                                                        # Receiver position(s) [x y z] (m)
        s_loc = Room_loc/2 + [Range*np.cos(Angle[i]*np.pi/180), Range*np.sin(Angle[i]*np.pi/180), 0]

        h = rir.generate(
            c = vp,                  # Sound velocity (m/s)
            fs = fs_true,                  # Sample frequency (samples/s)
            r = MIC_loc ,
            s = s_loc,          # Source position [x y z] (m)
            L = Room_loc,            # Room dimensions [x y z] (m)
            reverberation_time=T_reverb, # Reverberation time (s)
            nsample=nsample,           # Number of output samples
            dim = dim,
            order = order,
        )
        X[0,:] += (signal.convolve(h[:,0],SigVec[i,:]))[:len(SigVec[0,:])]
        X[1,:] += (signal.convolve(h[:,1],SigVec[i,:]))[:len(SigVec[0,:])]
        H.append(h)
    return X, H

def MixGen_RIR_single(SigVec, Angle, Angle_rot, fs_true, SNR, vp, d, Range, Room_len, z, T_reverb, nsample, dim, order):
    H = []
    X = np.zeros((2,len(SigVec)))
    

    Room_loc = np.array([Room_len, Room_len, 2*z])
    MIC_loc = [[(Room_len-d*np.cos(Angle_rot*np.pi/180))/2, (Room_len-d*np.sin(Angle_rot*np.pi/180))/2, z],         
               [(Room_len+d*np.cos(Angle_rot*np.pi/180))/2, (Room_len+d*np.sin(Angle_rot*np.pi/180))/2, z]]       
                                                                    # Receiver position(s) [x y z] (m)
    s_loc = Room_loc/2 + [Range*np.cos(Angle*np.pi/180), Range*np.sin(Angle*np.pi/180), 0]

    h = rir.generate(
        c = vp,                  # Sound velocity (m/s)
        fs = fs_true,                  # Sample frequency (samples/s)
        r = MIC_loc ,
        s = s_loc,          # Source position [x y z] (m)
        L = Room_loc,            # Room dimensions [x y z] (m)
        reverberation_time=T_reverb, # Reverberation time (s)
        nsample=nsample,           # Number of output samples
        dim = dim,
        order = order,
    )
    X[0,:] += (signal.convolve(h[:,0],SigVec))[:len(SigVec)]
    X[1,:] += (signal.convolve(h[:,1],SigVec))[:len(SigVec)]
    H.append(h)
    return X, H

############# need to define func to get MIC data

 #def getMIC():
        