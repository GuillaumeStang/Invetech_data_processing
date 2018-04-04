# -*- coding: utf-8 -*-
"""
Created on Sat Apr 08 16:56:20 2017

@author: guill
"""


#%% import libraries
#This step is paramount and specific to your application. You import only the libraries you are using or need
import numpy as np # you will always have to import this library, it brings the matrice structure you need to interact effcetively with your data 
import matplotlib.pyplot as plt # this library brings plotting capabilities similar to what matlab does 
import scipy.integrate as integrate # this library is more specific and brings different numerical integration methods
from scipy import interpolate # this library is more specific and brings different numerical interpolation  methods
import scipy.signal as scs # this library is more specific and brings algorithm to filter process signals (noise filtering.
import scipy.fftpack as FFT
 
import pandas as pd # this library is written specifically for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. It also has good capabilities in importing and exporting data. 

#%% quick signal analysis exercice

data_excercice = np.genfromtxt('exercice_data.csv', delimiter=',',skip_header=1)

Ts = data_excercice[1,0]-data_excercice[0,0] # sampling interval
Fs = 1.0/Ts  # sampling rate
t = data_excercice[:,0] # time vector

y = data_excercice[:,1] #A 5 Hz waveform sampled at 500 Hz for 1 second

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)] # one side frequency range

Y = FFT.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

fig, ax = plt.subplots(3, 1)

ax[0].plot(t,y)
ax[0].set_xlabel('Time(s)')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('raw signal')
ax[0].grid()

ax[1].plot(t,y)
ax[1].set_xlabel('Time(s)')
ax[1].set_ylabel('Amplitude')
ax[1].set_xlim([0,0.02])
ax[1].set_title('zoomed raw signal')
ax[1].grid()

ax[2].plot(frq,abs(Y),'r') # plotting the spectrum
ax[2].set_xlabel('Freq (Hz)')
ax[2].set_ylabel('|Y(freq)|')
ax[2].set_xlim([-10,5000])
ax[2].set_title('signal spectrum')
ax[2].grid()
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.suptitle('Fourier transform', fontsize=32)
