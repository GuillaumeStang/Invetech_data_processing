# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:27:37 2017

@author: zzgs
"""

#%% import libraries
#This step is paramount and specific to your application. You import only the libraries you are using or need
import numpy as np # you will always have to import this library, it brings the matrice structure you need to interact effcetively with your data 
import matplotlib.pyplot as plt # this library brings plotting capabilities similar to what matlab does 
import matplotlib.animation as animation

import scipy.integrate as integrate # this library is more specific and brings different numerical integration methods
from scipy import interpolate # this library is more specific and brings different numerical interpolation  methods
import scipy.signal as scs # this library is more specific and brings algorithm to filter process signals (noise filtering.
import scipy.fftpack as FFT
 
import pandas as pd # this library is written specifically for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. It also has good capabilities in importing and exporting data. 

#%% import data
file_name = 'OldSpec_WarmUp_USB2U072911_09-50-07-759.txt'
data = np.genfromtxt(file_name, delimiter='\t', skip_header=16)
time = data[1:,1]
time = time - time[0]
wavelength = data[0,2:]
data = data[1:, 2:]

#%% plotting
fig, ax = plt.subplots()

line, = ax.plot(wavelength, data[0,:])
ax.set_ylim([0, 18000])
ax.grid()
ax.set_title("spectrum animation", fontsize=32)
ax.set_xlabel("wavelength(nm)", fontsize=22)
ax.set_ylabel("light intensity", fontsize=22)

def init():    
    line.set_label()

def animate(i):
    line.set_ydata(data[i,:])  # update the data
    return line, 
    
ani = animation.FuncAnimation(fig, animate, range(1,len(time)), 
                              interval=50, blit=False, repeat = False)
#    
  
#%%  links animation
#http://matplotlib.org/examples/animation/basic_example.html
#http://matplotlib.org/examples/animation/simple_anim.html