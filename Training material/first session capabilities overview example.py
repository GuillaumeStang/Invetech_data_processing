# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:28:45 2016

@author: zzgs
"""
#%% import libraries
#This step is paramount and specific to your application. You import only the libraries you are using or need
import numpy as np # you will always have to import this library, it brings the matrice structure you need to interact effcetively with your data 
import matplotlib.pyplot as plt # this library brings plotting capabilities similar to what matlab does 
import scipy.integrate as integrate # this library is more specific and brings different numerical integration methods
from scipy import interpolate # this library is more specific and brings different numerical interpolation  methods
import scipy.signal as scs # this library is more specific and brings algorithm to filter process signals (noise filtering...) 

#%% import of data 

file_name = 'ssm2_bankB__69.dat'
data_raw = np.genfromtxt(file_name, delimiter='\t', skip_header=8) #read txt file and import the data into an numpy array. the sam efunction is used for CSV files. 
#%% plot of the raw data 

plt.plot(data_raw[:,0],data_raw[:,1]) # plot data
plt.xlabel('time') # write the x axis label
plt.ylabel('voltage from the sensor (V)')# write the y axis label
plt.title('raw data') # write the title of the plot 
plt.grid() #add a grid structure to the plot

#%% quick data processing

data = data_raw
data[:, 0] = (data[:, 0] - data[0, 0]) * 1000 #offset time column so that it starts at 0 and change the time unit from s to ms
data[:, 1] = 25 * (data[:, 1] - 4.5) - .065 #converting voltage value into pressure.(FYI, applying a calibration equation)
data[:, 1] = data[:, 1] - 1 * np.mean(data[0:2000, 1]) # calculate the data offset and substract it to the data
print 'offset correction:', np.mean(data[0:2000, 1]) # print the offset in the Ipython console 

#%% plot after quick data processing

plt.figure() # open a new figure window, if you don't add this line, it will plot within the previous figure
plt.plot(data[:,0],data[:,1], label='differential')
plt.xlabel('time (ms)')
plt.ylabel('Pressure(kPa)')
plt.title('plot after offset correction, calibration correction, time offset and unit correction')
plt.grid()

#%% how to extract dip (find start and end of a pressure dip) -> use of differential

differential = np.diff(data[:, 1]) # calculate the differential of the data[:,1] array
plt.figure()
plt.plot(differential)
plt.xlabel('row number')
plt.ylabel('Pressure differential(d(kPa)')
plt.title('differential')
plt.grid()

#%% how to handle noisy signal, sinfnal filtering

filtered_differential = np.diff(scs.medfilt(data[:, 1], 11)) # medfilt Apply a median filter to the input array
plt.plot(filtered_differential, label='filtered differential')
plt.legend(loc='upper right') 

#%% dip extraction 

extraction_para = 0.1 #parameter used to select differential values

down_index = np.where(filtered_differential < -extraction_para) # extract the index of the matrix filtered differential where filtered differential is less than -0.2
up_index = np.where(filtered_differential > extraction_para) # extract the index of the matrix filtered differential where filtered differential is less than -0.2
down_index = down_index[0] #convert tupple of array into one array 
up_index = up_index[0] #convert tupple of array into one array 


down_out = np.array([down_index[0]])
up_out = np.array([])

# loop which extracts the index of the start of the pressure dip 
j = 1
for i in range(0, len(down_index)):

    if down_index[i] - down_index[i - 1] > 80:
        down_out = np.concatenate((down_out, [down_index[i]]))
        j += 1

# loop which extracts the index of the end of the pressure dip 
j = 0
for i in range(0, len(up_index)-1):

    if up_index[i] - up_index[i - 1] > 80:
        up_out = np.concatenate((up_out, [up_index[i-1]]))
        j += 1

up_out = np.concatenate((up_out, [up_index[-1]]))
cycle_out = np.column_stack((down_out, up_out))
extracted_dip = []

# loop which extract every single pressure dip from the index extacted previously and append them in a python list. 
#one list element contains 1 two numpy array with time and pressure. 
for i in range(0, np.size(cycle_out, axis=0)):
    extracted_dip.append(data[cycle_out[i, 0]:cycle_out[i, 1], :])

    # time correction (for every pressure, you make the time start )
for i in range(0, len(extracted_dip)):
    extracted_dip[i][:, 0] = extracted_dip[i][:, 0] - extracted_dip[i][0, 0]
print len(extracted_dip), 'dip(s) extracted'

#%% plotting of a couple of pressure dip 
dip_number = 10
plt.figure()
for i in range(0, dip_number):
    plt.plot(extracted_dip[i][:, 0], extracted_dip[i][:, 1],'.-')
plt.xlabel('time(ms)')
plt.ylabel('Pressure kPa')
plt.title('Extracted pressure dips')
plt.grid()


#%% Pressure dip analysis and statistical plot

Raw_Analysis = np.empty([len(extracted_dip), 5]) #np.empty declare an empty array ready to be populated
Raw_Analysis_header = ['u-scavenge integration', 'peak', 'duration']
analysis_output_header=['SSM', 'micro_scavenge duration', 'Nb 1/2 open', 'nb full open','integration mean', 'integration 3 std', 'dip mean', 'dip 3 std', 'real duration mean', 'real duration 3std ']
analysis_output=np.empty([4, len(analysis_output_header)])

for i in range(0, len(extracted_dip), 1):
    Raw_Analysis[i, 0] = integrate.simps(extracted_dip[i][:, 1], extracted_dip[i][:, 0]) # integrate extracted_dip[i][:, 1] over extracted_dip[i][:, 0] (pressure over time)
    Raw_Analysis[i, 1] = np.min(extracted_dip[i][:, 1]) # calculate the minimum of extracted_dip[i][:, 1] array
    Raw_Analysis[i, 2] = extracted_dip[i][-1, 0]

#%% statistical plot

plt.figure() # create a new figure windows
plt.subplot(1, 3, 1) # create a subplot within this figure
plt.hist(Raw_Analysis[:, 0], label='mean=' +
                                  str(np.average(Raw_Analysis[:, 0]))+'\n3std='+str(3*np.std(Raw_Analysis[:, 0]))) #automatically generate an histogramme from Raw_Analysis[:, 0]
plt.xlabel('Kpa.ms')
plt.ylabel('Count')
plt.title("Pressure integration")
plt.legend(loc='upper right', shadow=False, fontsize='medium')
plt.grid()

plt.subplot(1, 3, 2)# create a second subplot within thefigure
plt.hist(Raw_Analysis[:, 1], label='mean=' +
                                  str(np.average(Raw_Analysis[:, 1]))+'\n3std='+str(3*np.std(Raw_Analysis[:, 1])))
plt.xlabel('Kpa.ms')
plt.xlabel('Pressure (kPa)')
plt.ylabel('Count')
plt.title("Dip")
plt.legend(loc='upper right', shadow=False, fontsize='medium')
plt.grid()

plt.subplot(1, 3, 3)
plt.hist(Raw_Analysis[:, 2], label='mean=' +
                                  str(np.average(Raw_Analysis[:, 2]))+'\n3std='+str(3*np.std(Raw_Analysis[:, 2])))
plt.xlabel('Kpa.ms')
plt.xlabel('duration (ms)')
plt.ylabel('Count')
plt.title("Real duration")
plt.legend(loc='upper right', shadow=False, fontsize='medium')
plt.grid()
plt.suptitle('Statistical', fontsize=18)

