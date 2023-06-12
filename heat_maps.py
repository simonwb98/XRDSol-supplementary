# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 13:37:11 2022

@author: Simon (Python 3.8)

Setup for first plot (3x1):
    1. PL contour
    2. XRD contour plot
    3. Instrument parameters
All with shared time axis. Acquisition durations different in some cases --> unequal time axis. 

Script works under the assumption that loaded PL and GIWAXS starting times are the same in the
sense that mainPL.py script was used to fix time shift in PL acquisition. The logging file, 
however, records time with a slight offset, which needs to be accounted for.

Setup for second plot (2x2):
    1. left axis: Single Integrated diffraction peak intensity
       right axis: temperature (1x1)
    2. left axis: diffraction peak FWHM 
       right axis: (100) q value (1x2)
    3. left axis: Integrated PL intensity 
       right axis: temperature (2x1)
    4. left axis: PL peak FWHM
       right axis: peak center (2x2)

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
from cycler import cycler
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator

### PLOT SETTINGS:
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['font.size'] = 12

### USER INPUTS:
    
sample_name = "ZnO"  # saving stacked plot under sample_name in save_path
save_path = "C:\XRDSol\data\BT_April_22\stacked_plots"
pl_directory = r"C:\XRDSol\data\BT_April_22\PL\ZnO"
log_directory = r"C:\XRDSol\data\BT_April_22\log_files\output_ZnO"
xrd_directory = r"C:\XRDSol\data\BT_April_22\GIWAXS\Sample_5-ZnO\output_ZnO_3"

from_when = 15.87 # in s, define what time should be set as new zero 
up_to = 70 # in s, define what time should be set as end in old time axis
relative_shift = 0 # in s, time by which logging time lags behind, (>0) or leads (<0) wrt GIWAXS/PL 
as_drop = 48.32

### Getting absolute paths:
    
xrd_fit_path = glob.glob(xrd_directory + '\*' + 'peak_fit_results' + '*.csv')[0]
xrd_contour_path = glob.glob(xrd_directory + '\*' + 'intensity' + '*.csv')[0]
xrd_time_path = glob.glob(xrd_directory + '\*' + 'time' + '*.csv')[0]
xrd_q_path = glob.glob(xrd_directory + '\*' + 'reciprocal' + '*.csv')[0]
xrd_paths = (xrd_contour_path, xrd_time_path, xrd_q_path)
    
pl_contour_path = glob.glob(pl_directory + '\*' + '2D' + '*.csv')[0]
pl_time_path = glob.glob(pl_directory + '\*' + 'time' + '*.csv')[0]
pl_wavelength_path = glob.glob(pl_directory + '\*' + 'wavelength' + '*.csv')[0]
pl_fit_path = glob.glob(pl_directory + '\*' + 'FitResults' + '*.csv')[0]
pl_paths = (pl_contour_path, pl_time_path, pl_wavelength_path, pl_fit_path)

temperature_path = glob.glob(log_directory + '\*' + 'Pyrometer' + '*.csv')[0]
spin_path = glob.glob(log_directory + '\*' + 'Spin_Motor' + '*.csv')[0]
time_path = glob.glob(log_directory + '\*' + 'Time' + '*.csv')[0]
instrument_paths = (temperature_path, spin_path, time_path)
line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))

### CODE:

def stacked_plot1(pl_paths, xrd_paths):
    # define subplots
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, constrained_layout=True,) # gridspec_kw={'height_ratios': [2, 2, 1]}
    
    # PL plot
    intensity = pd.read_csv(pl_paths[0])
    time = np.loadtxt(pl_paths[1], skiprows=3)
    energy = np.loadtxt(pl_paths[2], skiprows=2)
    center = pd.read_csv(pl_paths[3], usecols=[0], skiprows=1).values.tolist()
    # FWHM = pd.read_csv(pl_paths[3], usecols=[2], skiprows=1).values.tolist()
    # time_indices = np.where(time <= from_when)[0].max() # gives the largest time index with value less or equal to from_when
    i_max = intensity.max(numeric_only = True).max()
    cp1 = ax1.contourf(time - from_when, energy, intensity.T/i_max, np.linspace(0, 1, 100), cmap = plt.get_cmap('gist_heat'))
    cax1 = ax1.inset_axes([1.2, 0.2, 0.05, 0.6], transform=ax1.transAxes)
    ax1.axvline(x=as_drop - from_when + relative_shift, linestyle="dashed", label="", color="#1f77b4")
    cb1 = fig.colorbar(cp1, ax = ax1, ticks = np.linspace(0, 1, 2), cax=cax1, shrink=0.6,)
    mask = [i for i, v in enumerate(time) if v > (as_drop+relative_shift)]
    time_of_interest = time[mask[0]:mask[-1]]-from_when
    center_of_interest = np.array(center[mask[0]:mask[-1]]).flatten()
    # FWHM_of_interest = np.array(FWHM[mask[0]:mask[-1]]).flatten()*(5/8)
    ax1.plot(time_of_interest, center_of_interest, label="peak center", color="darkorange")
    # ax1.fill_between(time_of_interest, center_of_interest+FWHM_of_interest, center_of_interest-FWHM_of_interest, color="#ABABAB", alpha=0.15, label=r"$3\sigma$ uncertainty band")
    cb1.set_label('Norm. Intensity', fontsize = 12)
    ax1.set_ylabel('Energy (eV)')
    # ax1.set_xlabel("Time (s)")
    # ax1.set_xlim(20, up_to + from_when)
    ax1.set_ylim(1.52, 1.82)
    
    ax1.set_title(sample_name, fontweight='bold')
    ax1.legend(loc='lower right')
    ax2 = ax1.twinx()
    # spin_speed = np.loadtxt(instrument_paths[1], skiprows=1)
    temperature = np.loadtxt(instrument_paths[0], skiprows=1)
    spin_speed = np.loadtxt(instrument_paths[1], skiprows=1)
    time = np.loadtxt(instrument_paths[2], skiprows=1)
    ax2.plot(time - from_when + relative_shift, temperature, color = 'red', linestyle = "dotted")  #- relative_shift
    ax2.set_ylabel(r'Temperature ($^{\circ}$C)', color='red')
    ax2.spines['right'].set_color("red")
    ax2.tick_params(axis='y', color="red")
    
    # XRD plot
    intensity = pd.read_csv(xrd_paths[0])
    xrd_time = np.loadtxt(xrd_paths[1], skiprows=1)
    q = np.loadtxt(xrd_paths[2], skiprows=1)
    i_max = intensity.max(numeric_only = True).max() 
    i_min = intensity.min(numeric_only = True).min()
    
    # i_min = min(i for i in intensity.to_numpy().flatten() if i > 0)
    # limits for q range (in \AA^{-1})
    q_range = (0.35, 3.65)
    cp3 = ax3.contourf(xrd_time - from_when, q, intensity/i_max, np.linspace(i_min/i_max, 0.7, 100), cmap=plt.get_cmap('YlGnBu'))
    cax3 = ax3.inset_axes([1.2, 0.2, 0.05, 0.6], transform=ax3.transAxes)
    cb3 = fig.colorbar(cp3, ax = ax3, ticks = np.linspace(i_min/i_max, 1, 2), cax=cax3, shrink=0.6)
    cb3.set_label('Norm. Intensity', fontsize = 12)
    ax3.set_ylim(q_range[0], q_range[1])
    ax3.set_ylabel(r'q ($\AA^{-1}$)')
    ax3.set_xlim(20, 400) 
    ax3.axvline(x=as_drop - from_when + relative_shift, linestyle="dashed", label="", color="#1f77b4")
    ax4 = ax3.twinx()
    # ax5 = ax3.twinx()
    ax4.plot(time - from_when + relative_shift, temperature, color = 'red', linestyle = "dotted", alpha = 0.6)  #- relative_shift
    ax4.set_ylabel(r'Temperature ($^{\circ}$C)', color='red')
    ax4.spines['right'].set_color("red")
    ax4.tick_params(axis='y', color="red")
    # ax5.plot(time - from_when + relative_shift, spin_speed, color = 'blue', linestyle = "dotted", alpha = 0.6)  #- relative_shift
    # ax5.set_ylabel(r'Spin Speed ($rpm$)', color='blue')
    # ax5.spines['right'].set_color("blue")
    # ax5.tick_params(axis='y', color="blue")
    # ax3.legend(loc='lower right')
    ax3.set_xlabel('Time (s)')
    # plt.savefig(os.path.join(save_path, sample_name + '_stack_with_spin_speed' '.png'), dpi = 300, bbox_inches = "tight")
    
    plt.show()

def stacked_plot2(xrd_fit_path, pl_fit_path, xrd_time_path, pl_time_path, instrument_paths):
    # define subplots
    fig, axs_a = plt.subplots(2, 2, figsize=(9, 6), sharex=True, constrained_layout=True)
    ((ax1a, ax2a),(ax3a, ax4a)) = axs_a
    # define variables
    xrd_fit_data = pd.read_csv(xrd_fit_path)
    amplitude = xrd_fit_data['amplitude (au)'].values
    center = xrd_fit_data['center ($\AA$)'].values
    sigma = xrd_fit_data['sigma ($\AA$)'].values
    xrd_time = np.loadtxt(xrd_time_path, skiprows=1)
    
    pl_center = pd.read_csv(pl_fit_path, usecols=[0])
    pl_amplitude = pd.read_csv(pl_fit_path, usecols=[1])
    pl_time = pd.read_csv(pl_time_path, skiprows=1)
    pl_fwhm = pd.read_csv(pl_fit_path, usecols=[2]).values.tolist()
    
    temperature = np.loadtxt(instrument_paths[0], skiprows=1)
    instrument_time = np.loadtxt(instrument_paths[2], skiprows=1)
    
    # upper left (1x1) plot
    ax1a.plot(xrd_time, amplitude, color=colors[0])
    ax1b = ax1a.twinx()
    ax1b.plot(instrument_time, temperature, colors[1], linestyle='dotted')
    ax1b.set_yticks([30, 45, 60, 75, 90, 105])
    ax1b.spines['right'].set_color(colors[1])
    ax1b.tick_params(axis='y', color=colors[1])
    ax1a.set_ylabel('Integrated Intensity (au)')
    ax1b.set_ylabel('Temperature ($^{\circ}$C)', color=colors[1])
    mask = np.isnan(amplitude)   
    
    # upper right (1x2) plot
    ax2a.plot(xrd_time, 2*sigma, colors[2])
    ax2a.set_ylabel('FWHM ($\AA$)', color=colors[2])
    ax2a.tick_params(axis='y', color=colors[2])
    ax2b = ax2a.twinx()
    ax2b.plot(xrd_time, center, colors[3])
    ax2b.spines['left'].set_color(colors[2])
    ax2b.spines['right'].set_color(colors[3])
    ax2b.set_ylabel('q ($\AA$)', color=colors[3])
    
    #loawer left (2x1) plot
    ax3a.plot(pl_time, pl_amplitude, colors[0])
    ax3a.set_yscale('log')
    ax3a.set_ylabel('PL Integrated Intensity (au)')
    ax3a.set_xlabel('Time (s)')
    ax3b = ax3a.twinx()
    ax3b.plot(instrument_time, temperature, colors[1], linestyle='dotted')
    ax3b.set_yticks([30, 45, 60, 75, 90, 105])
    ax3b.set_ylabel('Temperature ($^{\circ}$C)', color=colors[1])
    ax3b.spines['right'].set_color(colors[1])
    ax3b.tick_params(axis='y', color=colors[1])
    
    #lower right (2x2) plot
    #apply mask to pl data
    ax4a.plot(pl_time, pl_fwhm, color=colors[2])
    ax4a.set_xlabel('Time (s)')
    
    ax4a.set_ylabel('FWHM (eV)', color=colors[2])
    ax4a.tick_params(axis='y', color=colors[2])
    ax4b = ax4a.twinx()
    ax4b.plot(pl_time, pl_center, colors[3])
    ax4b.set_ylabel('Peak Position (eV)', color=colors[3])
    ax4b.spines['left'].set_color(colors[2])
    ax4b.spines['right'].set_color(colors[3])
    ax4a.set_ylim(0.02, 0.3)
    ax4b.set_ylim(1.58, 1.7)
    
    plt.suptitle(sample_name, fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(save_path, 'stacked_plot_2_' + str(sample_name)), dpi = 300, bbox_inches = "tight")
    
    
stacked_plot1(pl_paths, xrd_paths)
# stacked_plot2(xrd_fit_path, pl_fit_path, xrd_time_path, pl_time_path, instrument_paths)