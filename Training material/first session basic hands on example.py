# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:21:48 2016

@author: zzgs
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


#%% Array generation

x = np.linspace(0, 10, 21) # ctrl I + click on linspace to access the documentation 
xx = np.arange(0, 10, 0.5) # ctrl I + click on arange to access the documentation 
y1 = x**2 # x^2
y2 = x**3 # x^3
y3 = x**4 # x^4

# see also: empty, empty_like, zeros, zeros_like, ones, ones_like, fill_diagonal


#%%concatenate
Conca = np.concatenate((x, y1, y2, y3), axis=0) # look at variable explorer to see what it does
column_stack = np.column_stack((x, y1, y2, y3)) # look at variable explorer to see what it does

Conca_easy = np.array([x, y1, y2, y3]) # look at variable explorer to see what it does
Conca_easy_t = np.transpose(Conca_easy) # look at variable explorer to see what it does

#%% export to CSV 

df = pd.DataFrame(data = Conca_easy_t, columns=['x', 'x^2', 'x^3', 'x^4']) # this is very specific to the pandas library, this line creates what they call a dataframe which is a more flexible numpy array. data = conca_easy_t populate the data frame. columns=['x', 'x^2', 'x^3', 'x^4'] names the different columns.
df.to_csv('export_file.csv', index=False) # export the dataframe into a csv file

#%reset

#%% import csv
data = np.genfromtxt('export_file.csv', delimiter=',', skip_header=1)

#%%slicing

# array index starts with 0!!!

print data[0,2] # return value stored in row 0, colum 2

print data[0,:] # return row 0, ":" means all elements of that axis 
print data[1,:] # return row 1, ":" means all elements of that axis 

print data[:,0] # return Column 0, ":" means all elements of that axis 
print data[:,1] # return column 1, ":" means all elements of that axis 
print data[:,2] # return column 2, ":" means all elements of that axis 

print data[3:5,2] # return subset of Conca_easy_t, try to figure out which part by looking in the variable explorer. Take a specific attention to 3:5, what does it mean?  
print data[3:,2] # return subset of Conca_easy_t, try to figure out which part by looking in the variable explorer. Take a specific attention to 3:, it means from element 3 to the last.    
print data[3:-2,2] # return subset of Conca_easy_t, try to figure out which part by looking in the variable explorer. Take a specific attention to 3:-2, it means from element 3 to the third las!!

#%% plotting
plt.figure() 
plt.plot(data[:,0], data[:,1], label='x^2')
plt.plot(data[:,0], data[:,2], '.-', label='x^3')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('THIS IS A TITLE')
plt.legend()
plt.grid()

#%% plotting with more control and complexity 
fig, ax = plt.subplots(1, 2)
for power in range(0,3):
    ax[0].plot(data[:,0], data[:,0]**power, label='x^{0}'.format(*[power]))
ax[0].set_xlabel('xlabel')
ax[0].set_ylabel('ylabel')
ax[0].set_title('THIS IS A TITLE')
ax[0].legend()
ax[0].grid()

for power in range(3,6):
    ax[1].plot(data[:,0], data[:,0]**power, label='x^{0}'.format(*[power]))
ax[1].set_xlabel('xlabel')
ax[1].set_ylabel('ylabel')
ax[1].set_title('THIS IS A TITTLE')
ax[1].legend()
ax[1].grid()

plt.suptitle("Main title", fontsize=32)

#%%  derivation and integration

Fs = 500;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector

ff = 5;   # frequency of the signal
y = scs.square(2 * np.pi * ff * t) #A 5 Hz waveform sampled at 500 Hz for 1 second

fig, ax = plt.subplots(1, 2)
ax[0].plot(t, y, label='x^{0}'.format(*[power]))
ax[0].set_xlabel('xlabel')
ax[0].set_ylabel('ylabel')
ax[0].set_title('SQUARE SIGNAL')
ax[0].set_ylim([-1.2,1.2])
ax[0].legend()
ax[0].grid()

dydt = np.diff(y)/np.diff(t)

ax[1].plot(t[:-1], dydt, label='x^{0}'.format(*[power]))
ax[1].set_xlabel('xlabel')
ax[1].set_ylabel('ylabel')
ax[1].set_title('DERIVATIVE')
ax[1].set_ylim([-1200,1200])
ax[1].legend()
ax[1].grid()

print "integration of half of first period: ", integrate.simps(y[0:50],t[0:50]), "Y.s"

#%% Fourier transform (fast fourier transform algorithm) 

# The packing of the result is “standard”: If A = fft(a, n), then A[0] contains the zero-frequency term, A[1:n/2] contains the positive-frequency terms, and A[n/2:] contains the negative-frequency terms, in order of decreasingly negative frequency. So for an 8-point transform, the frequencies of the result are [0, 1, 2, 3, -4, -3, -2, -1]. To rearrange the fft output so that the zero-frequency component is centered, like [-4, -3, -2, -1, 0, 1, 2, 3], use fftshift.

#For n even, A[n/2] contains the sum of the positive and negative-frequency terms. For n even and x real, A[n/2] will always be real.

#This function is most efficient when n is a power of two, and least efficient when n is prime.

#If the data type of x is real, a “real FFT” algorithm is automatically used, which roughly halves the computation time. To increase efficiency a little further, use rfft, which does the same calculation, but only outputs half of the symmetrical spectrum. If the data is both real and symmetrical, the dct can again double the efficiency, by generating half of the spectrum from half of the signal.

# The first bin in the FFT is DC (0 Hz), the second bin is Fs / N, where Fs is the sample rate and N is the size of the FFT. The next bin is 2 * Fs / N. To express this in general terms, the nth bin is n * Fs / N.

Fs = 500;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector

ff = 5;   # frequency of the signal
y = scs.square(2 * np.pi * ff * t) #A 5 Hz waveform sampled at 500 Hz for 1 second

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)] # one side frequency range

Y = FFT.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

fig, ax = plt.subplots(2, 1)

ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[0].set_ylim([-1.2,1.200])
ax[0].grid()

ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')
ax[1].grid()

plt.suptitle('Fourier transform', fontsize=32)
