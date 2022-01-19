#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
from scipy.io.wavfile import read,write
from scipy import signal,stats 
import math
import numpy.ma as ma
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
import math
import importlib
import tfanalysis as tfanalysis
import tfsynthesis as tfsynthesis
import twoDsmooth as twoDsmooth
importlib.reload(tfanalysis)
importlib.reload(tfsynthesis)
importlib.reload(twoDsmooth)
from tfanalysis import tfanalysis
from tfsynthesis import tfsynthesis
from twoDsmooth import twoDsmooth


################################################
#     setp 1,2,3
################################################ 
# 1. analyze the signals - STFT
# 1) Create the spectrogram of the Left and Right channels.

#constants used

### Mode =1 : prominence will be used for peak detection
### Mode =2 : height will be used.

def DUET(x, fs, N_fft, timestep,p,q,prom_const, max_const,peak_mode,graph_mode):
    eps=2.2204e-16
    numfreq = N_fft
    wlen = numfreq
    awin=np.hamming(wlen) #analysis window is a Hamming window Looks like Sine on [0,pi]
    x1=x[0,:]
    x2=x[1,:]
    ### normalizing needed? for now let's not.
    x1=x1/np.max(x1) # Dividing by maximum to normalise
    x2=x2/np.max(x2) # Dividing by maximum to normalise
    tf1=tfanalysis(x1,awin,timestep,numfreq) #time-freq domain
    tf2=tfanalysis(x2,awin,timestep,numfreq) #time-freq domain
    x1=np.asmatrix(x1)
    x2=np.asmatrix(x2)
    tf1=np.asmatrix(tf1)
    tf2=np.asmatrix(tf2)

    #removing DC component
    tf1=tf1[1:,:]
    tf2=tf2[1:,:]
    #eps is the a small constant to avoid dividing by zero frequency in the delay estimation

    #calculate pos/neg frequencies for later use in delay calc ??

    a=np.arange(1,((numfreq/2)+1))
    b=np.arange((-(numfreq/2)+1),0)
    freq=(np.concatenate((a,b)))*((2*np.pi)/numfreq) #freq looks like saw signal

    a=np.ones((tf1.shape[1],freq.shape[0]))
    freq=np.asmatrix(freq)
    a=np.asmatrix(a)
    for i in range(a.shape[0]):
        a[i]=np.multiply(a[i],freq)
    fmat=a.transpose()

    
    ####################################################

    #2.calculate alpha and delta for each t-f point
    #2) For each time/frequency compare the phase and amplitude of the left and
    #   right channels. This gives two new coordinates, instead of time-frequency 
    #   it is phase-amplitude differences.

    R21 = (tf2+eps)/(tf1+eps)
    #2.1HERE WE ESTIMATE THE RELATIVE ATTENUATION (alpha)
    a=np.absolute(R21) #relative attenuation between the two mixtures
    #alpha = a
    alpha=a-1./a #'alpha' (symmetric attenuation)
    #2.2HERE WE ESTIMATE THE RELATIVE DELAY (delta)
    delta = -(np.imag((np.log(R21)/fmat)))                  ### delta = delay * fs
    
    # imaginary part, 'delta' relative delay
    ####################################################

    # 3.calculate weighted histogram
    # 3) Build a 2-d histogram (one dimension is phase, one is amplitude) where 
    #    the height at any phase/amplitude is the count of time-frequency bins that
    #    have approximately that phase/amplitude.

    #p=1; q=0;
    #p=2; q=2;
    h1=np.power(np.multiply(np.absolute(tf1),np.absolute(tf2)),p) #refer to run_duet.m line 45 for this. 
                                                                  #It's just the python translation of matlab 
    h2=np.power(np.absolute(fmat),q)

    tfweight=np.multiply(h1,h2) #weights vector 
    maxa=0.7;
    maxd=2*3.6;#histogram boundaries for alpha, delta

    abins=35;
    dbins=2*50;#number of hist bins for alpha, delta


    # only consider time-freq points yielding estimates in bounds
    amask=(abs(alpha)<maxa)&(abs(delta)<maxd)
    amask=np.logical_not(amask)
    alphavec = np.asarray(ma.masked_array(alpha, mask=(amask)).transpose().compressed())[0]
    deltavec = np.asarray(ma.masked_array(delta, mask=(amask)).transpose().compressed())[0]
    tfweight = np.asarray(ma.masked_array(tfweight, mask=(amask)).transpose().compressed())[0]
    # to do masking the same way it is done in Matlab/Octave, after applying a mask we must take transpose and compress

    #determine histogram indices (sampled indices?)

    alphaind=np.around((abins-1)*(alphavec+maxa)/(2*maxa))
    deltaind=np.around((dbins-1)*(deltavec+maxd)/(2*maxd))

    #FULL-SPARSE TRICK TO CREATE 2D WEIGHTED HISTOGRAM
    #A(alphaind(k),deltaind(k)) = tfweight(k), S is abins-by-dbins
    A=sp.sparse.csr_matrix((tfweight, (alphaind, deltaind)),shape=(abins,dbins)).todense()
    #smooththehistogram-localaverage3-by-3neighboringbins

    A=twoDsmooth(A,3)
    X=np.linspace(-maxd,maxd,dbins)
    Y=np.linspace(-maxa,maxa,abins)

    
    if peak_mode==0:
        col_peak,_=signal.find_peaks(np.max(A,axis=0),prominence=np.max(A)*prom_const)
    else:
        col_peak,_=signal.find_peaks(np.max(A,axis=0),height=np.max(A)*max_const)
 
    row_peak = []
    for idx in col_peak:
        row_peak.append(np.argmax(A[:,idx]))

    if graph_mode == 1:
        fig = plt.figure(0)
        ax = fig.add_subplot(111, projection='3d')
        X_grid, Y_grid = np.meshgrid(X, Y)
        ax.plot_wireframe(X_grid,Y_grid,A)
        plt.xlabel('delta')
        plt.ylabel('alpha')
        plt.tight_layout()
        plt.show()

        # You can have a look at the histogram to look at the local peaks and what not

        ## Peak location detection
        plt.figure(1)
        #plt.imshow(A,origin='upper')
        #plt.colorbar()
        #plt.figure(1)
        plt.plot(X,np.max(A,axis=0))
        plt.xlabel('delta')
        plt.tight_layout()
        plt.show()        
        
        
        
        
        
    ######################################    step 4,5,6,7
    ######################################
    #4.peak centers (determined from histogram) THIS IS DONE BY HUMAN.
    #4) Determine how many peaks there are in the histogram.
    #5) Find the location of each peak. 

    numsources=len(col_peak);

    peakdelta=X[col_peak]
 
    return peakdelta

