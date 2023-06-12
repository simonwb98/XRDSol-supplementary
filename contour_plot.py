# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:08:44 2022

@author: Simon (Python 3.8)
Given a full 2\theta integration scan from XRDSol, 
plot the contour plot (x, y, z(x,y)) -> (time, q, intensity)
and save individual arrays/matrices as .csv files in a separate folder called
'output'.

Ideas for improvements:
    - label orientations or give separate table with q values and reflexes
    - eliminate user input necessities
    - Make get data more efficient by pre-defining q array with random values
    and then replace consecutively
    - Check if csv file already exists and only convert if it doesn't
    
    
Known bugs:
    - Depending on the host computer settings, XRDSol saves the scan images with 
different time formats, for example 1:23 PM instead of 13:23. Pandas 
interprets the AM/PM part as additional column. To eliminate this bug, append/
delete last column name in the array -names- and placeholder variable in
line.split(',')  (for-loop in get_data).

"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import ntpath
from matplotlib import cm
from matplotlib.ticker import LinearLocator


### USER INPUTS:
    
# The output scan seperators are irregular tab spaces. 
# The following saves the output scan as a csv file:
file_path = r"C:\Users\User\Desktop\LBL\BT_December\ZnO_SAM\GIWAXS\ZnO_SAM.dat" # absolute location of 
                                                    # scan, must be .dat file!
time_per_frame = 1.844   # in sec
num_frames = 675 # input the number of frames in the scan
# plt.style.use('seaborn-white') # uncomment, restart kernel for other style

### OTHER:
  
file_name = ntpath.basename(file_path)
sample_name = file_name.rstrip('.dat') # under which name data is saved later
Path(os.path.join(file_path.rstrip(file_name), 'output_' + sample_name)).mkdir(parents=True, exist_ok=True)
save_path = os.path.join(file_path.rstrip(file_name), 'output_' + sample_name)
# print('Converting to csv...')
file = pd.read_csv(file_path, sep='\s+', header=0, names=np.array(['image_num', 'twotheta', 'twotheta_cuka', 'dspacing', 'qvalue', 'intensity', 'frame_number', 'izero', 'date', 'time', 'am']))
csv_path = os.path.join(save_path, sample_name + '.csv')
file.to_csv(csv_path, index=0)
length = len(file)


### CODE:

# =================Part 1: Loading Data======================

def get_data(csv_path, frames, sample_name, save_path, time_per_frame):
    '''
    Parameters
    ----------
    csv_path : path object, 
        points towards the saved csv file.
    frames : int,
        imports the total number of frames taken for the scan
    sample_name : str,
        name of the sample. Default is the name under which scan is saved.
    save_path : path object
        where the output is saved.

    Returns
    -------
    three arrays, q, frame_num and full_intensity which are one, one and two 
    dimensional, respectively. These are saved as csv files.

    '''
    with open(csv_path, 'r') as data_file:
        counter = 1
        q = np.array([])
        q_size = int(length/frames)
        full_intensity = np.zeros(shape=(frames, q_size))
        frame_intensity = np.array([])
        data_file.readline()
        for line in tqdm(data_file, desc = 'Gathering data'):
            imagenum, twotheta, twotheta_cuka, dspacing, qvalue, intensity, frame_number, izero, date, time, am = line.split(',')
            if int(frame_number) == counter:
                intensity = np.array([float(intensity)])
                q = np.append(q, float(qvalue)) 
                frame_intensity = np.append(frame_intensity, intensity)
            else:
                full_intensity[int(counter) - 1] = frame_intensity
                counter += 1
                # clearing the arrays that save the data of each frame
                q = np.array([]) 
                frame_intensity = np.array([])
                # save data of current line
                q = np.append(q, float(qvalue))
                frame_intensity = np.append(frame_intensity, intensity)
        # save last frame values
        full_intensity[frames - 1] = frame_intensity
    data_file.close()
    frame_num = np.linspace(0, int(frames)-1, int(frames))
        
    # saving the data in separate csv files
    df_q = pd.DataFrame(q)
    df_t = pd.DataFrame(frame_num * time_per_frame + time_per_frame/2)
    df_i = pd.DataFrame(full_intensity.T)
    
    df_q.to_csv(os.path.join(save_path, 'reciprocal_' + sample_name + '.csv'), header = ['Q r$(1/\AA\)$'], index = None)
    df_t.to_csv(os.path.join(save_path, 'time_' + sample_name + '.csv'), header = ['Time (s)'], index = None)
    df_i.to_csv(os.path.join(save_path, 'intensity_' + sample_name + '.csv'), header = frame_num, index = None)
    # Note: intensity matrix saved with frames for columns and qs for rows
    print('\n Saved data in: ' + save_path)
    
    return (q, frame_num, full_intensity.T)

# =================Part 2: Contour plot======================

def plot_contour(csv_path, frames, sample_name, save_path, time_per_frame):
    '''
    Parameters
    ----------
    csv_path : path object, 
        points towards the saved csv file.
    frames : int,
        imports the total number of frames taken for the scan
    sample_name : str,
        name of the sample. Default is the name under which scan is saved.
    save_path : path object
        where the output is saved.
    time_per_frame: float.

    Returns
    -------
    Contour plot

    '''
    # call/define variables for contour plot
    q, frame_num, intensity = get_data(csv_path, frames, sample_name, save_path, time_per_frame)
    
    # create an empty figure with the following dimensions
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    
    # add the contour plot and a colorbar
    cp = plt.contourf(frame_num*time_per_frame, q, intensity, np.linspace(0, np.amax(intensity), 100))
    plt.colorbar(cp, location='left', ticks = np.arange(0, np.amax(intensity), 1000))
    
    # define axis names, ticks, etc.
    q_min, q_max = (q[0], q[-1])
    q_min, q_max = (0.5, 3.5)
    y_ticks = np.linspace(q_min, q_max, 20) # number of tickmarks 
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Q $(\AA^{-1})$')
    ax.set_yticks(y_ticks)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim(q_min, q_max)
    ax.set_title(sample_name)
    plt.savefig(os.path.join(save_path, 'Contour_plot_' + str(sample_name)), dpi = 300, bbox_inches = "tight")
    plt.show()

# =================Part 3: Surface plot (WORK IN PROGRESS)======================

def plot_surface(csv_path, frames, sample_name, save_path, time_per_frame, q_range, frame_range):
    q, frame_num, intensity = get_data(csv_path, frames, sample_name, save_path, time_per_frame)
    
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    
    surf = ax.plot_surface(frame_num*time_per_frame + time_per_frame/2, q, intensity, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)



plot_contour(csv_path, num_frames, sample_name, save_path, time_per_frame)
# plot_surface(csv_path, num_frames, sample_name, save_path, time_per_frame, (0.5, 2.0), (25, 45))
