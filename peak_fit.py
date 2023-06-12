# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:14:09 2022

@author: Simon (Python 3.8)

Peak fitting procedure using python module lmfit:
    
    Implements a peak+background model and fits each frame in the selected q
    range with suitably chosen initial parameters. Fit data is then saved in
    separate folder for plotting, etc...
    
    The model consists of a linear background, with a Pseudo-Voigt peak. 
    Modify the init_params values below to according to the peak to be fitted.
    The fitting algorithm will try to fit the entire q range with the above 
    described model. Choose q_range such that only the desired peak is visible.
    
    The algorithm assumes that contour_plot.py was previously run, i.e. that
    the output scan .dat file from XRDSol has been converted to .csv and that
    the 'output' folder exists in the same directory as the corresponding .dat
    file. The fit params will be saved in the same output folder with specified
    (hkl) value for the specific reflex.

Improvement ideas:
    
    - different method of determining if peak exists then by prominence
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import ntpath

from scipy import signal
from tqdm import tqdm

from lmfit.models import LinearModel, PseudoVoigtModel

### USER INPUTS:

file_path = r"C:\Users\User\Desktop\LBL\GaAs_SAM\GIWAXS\GaAs_SAM.dat"
q_range = (0.8, 0.95)                # define q range to be fitted  
   
init_params = {                     # initial guess parameters
               'amplitude' : 2,     # default: 2
               'center' : 0.9,        # 1 (in angstrom)
               'sigma' : 0.005,      # 0.01
               'fraction' : 0.5,    # 0.5
               'slope' : 0,         # 0
               'intercept' : 1000    # 700
               }
show_every = 100              # int n, shows every n'th frame with fit
show_these = [int(i) for i in range(20, 60, 5)]
hkl = 'pbi2'
### OTHER:
  
file_name = ntpath.basename(file_path)
sample_name = file_name.rstrip('.dat')
import_path = os.path.join(file_path.rstrip(file_name), 'output_' + sample_name)
#Path(os.path.join(file_path.rstrip(file_name), hkl + '_peak_fit_' + sample_name)).mkdir(parents=True, exist_ok=True)
save_path = import_path # os.path.join(file_path.rstrip(file_name), hkl + '_peak_fit_' + sample_name)
csv_path = os.path.join(import_path, sample_name + '.csv')
num_frames = max(pd.read_csv(os.path.join(import_path, 'intensity_' + sample_name + '.csv'), 
                   nrows=1, header=None))


# =================Part 1: Fitting======================

    
def fit_single_frame(csv_path, frame_index, sample_name, import_path, 
                     frames_to_plot, q_range = (0.9, 1.1), init_params=init_params):
    '''
    Parameters
    ----------
    csv_path : path object, 
        points towards the saved csv file.
    frame_index : int, 
        frame to be fit.
    sample_name : str,
        name of the sample. Default is the name under which scan is saved.
    save_path : path object,
        where the output is saved.
    frames_to_plot : list,
        contains frame numbers. If frame_index is included, prints the fit, 
        with initial guess.
    q_range : tuple, optional
        (q_start, q_end) - range over which the fitting algorithm will be 
        implemented. The default is (0.9, 1.1).

    Returns
    -------
    params : list,
        fit parameters for fit.
    red_chi : float,
        reduced chi squared value for the implemented fit.

    '''
    
    df_i = pd.read_csv(os.path.join(import_path, 'intensity_' + sample_name + '.csv'), 
                       usecols = [frame_index - 1])
    df_q = pd.read_csv(os.path.join(import_path, 'reciprocal_' + sample_name + '.csv'))
     
    # define ranges for q and I to be shown
    index_list = df_q.index[(df_q['Q r$(1/\AA\)$'] > q_range[0]) &
                            (df_q['Q r$(1/\AA\)$'] < q_range[1])] # search for index values in df_q
    first_index, last_index = (index_list[0], index_list[-1] + 1)
    df_sliced_q, df_sliced_i = (df_q.iloc[first_index:last_index, :], 
                                df_i.iloc[first_index:last_index, :])
    
    x = df_sliced_q.to_numpy().squeeze()
    y = df_sliced_i.to_numpy().squeeze()
    # define fitting models (so far, one peak and a background function)
    peak = PseudoVoigtModel()
    background = LinearModel()
    mod = peak + background
    # initial values
    pars = mod.make_params(amplitude = init_params['amplitude'], 
                           center = init_params['center'], 
                           sigma = init_params['sigma'], 
                           fraction = init_params['fraction'], 
                           slope = init_params['slope'], 
                           intercept = init_params['intercept'])
    # bounds
    pars.add('amplitude', value=init_params['amplitude'], min=0)
    pars.add('center', value=init_params['center'], min=min(q_range)*1.05, max=max(q_range)*0.95)
    # pars.add('sigma', value=init_params['sigma'], max=0.01)
    
    # # "hints" - don't change
    # mod.set_param_hint('amplitude', min=0)
    # mod.set_param_hint('center', min=0.95, max=1.05)
    # mod.set_param_hint('sigma', max=0.01)
    
    # determine if peak in data, promninence of 190 is chosen by hand, doesn't
    # need to be ideal for every sample
    peak_in_frame = False #initially false
    peaks = signal.find_peaks(y, prominence=60)[0]
    if len(peaks) > 0:
        
        peak_in_frame = True
        
        # fitting call
        result = mod.fit(y, pars, x=x)

        redchi = result.redchi
        dely = result.eval_uncertainty(sigma=3)
        params = []
        std_error = []
        for name, param in result.params.items():
            params.append(param.value)
            std_error.append(param.stderr) 
        if frame_index in frames_to_plot:
            plt.plot(x, y, 'o', label='intensity')
            plt.plot(x[peaks], y[peaks], 'r.', label='found peak')
            plt.plot(x, result.init_fit, '--', label='initial guess')
            plt.plot(x, result.best_fit, '-', label='best fit')
            # plt.fill_between(x, result.best_fit-dely, result.best_fit+dely, 
            #                  color='#ABABAB', label='3$\sigma$ - uncertainty band')
    
            plt.xlabel(r'q $(\AA)$')
            plt.ylabel(r'Intensity (au)')
            # result.plot(data_kws={'markersize': 1})
            plt.legend()
            plt.title('Frame: ' + str(frame_index))
            plt.show()
            

    elif len(peaks) == 0:
        if frame_index in frames_to_plot:
            plt.plot(x, y, 'o', label='intensity')
            plt.xlabel(r'q $(\AA^{-1})$')
            plt.ylabel(r'Intensity (au)')
            # result.plot(data_kws={'markersize': 1})
            plt.legend()
            plt.title('Frame: ' + str(frame_index))
            plt.show()
        params = [None]*6
        std_error = [None]*3
        redchi = [None]
        
    return (params, std_error, redchi, peak_in_frame)        

def fit_several_frames(start, end, show_every = 10, q_range = (0.9, 1.1)):
    '''
    Parameters
    ----------
    start : int, 
        starting frame number.
    end : int, 
        final frame number.
    show_every : int, optional
        show every n'th frame and the corresponding fit. Useful for evaluating
        initial params and success of fit. The default is 10.
    q_range : tuple, optional
        (q_start, q_end) - range over which the fitting algorithm will be 
        implemented. The default is (0.9, 1.1).

    Returns
    -------
    None.

    '''
    amplitude, unc_a = [], []
    center, unc_c = [], []
    sigma, unc_s = [], []
    fraction = []
    slope = []
    
    intercept = []
    red_chi = []
    all_params = [amplitude, center, sigma, fraction, slope, intercept]
    peak_unc = [unc_a, unc_c, unc_s]
    frames = range(start, end + 2)
    frames_to_plot = [i for i in frames if i % show_every == 0]
    frames_to_plot.extend(show_these)
    for frame in tqdm(frames, desc='Fitting frames'):
        params, std_error, redchi, peak_in_frame = fit_single_frame(csv_path, frame, sample_name, 
                                          import_path, frames_to_plot, q_range, init_params)
        red_chi.append(redchi)

        for index, param in enumerate(all_params):
            param.append(params[index])
        for index, unc in enumerate(peak_unc):
            unc.append(std_error[index])
        # if peak_in_frame:
        #     # for higher efficiency, the init_params are now changed to the 
        #     # fit values for next scan. However, if the initial frame is 
        #     # wrongly identified to contain a peak, this might lead to problems
        #     init_params['amplitude'] = params[0]
        #     init_params['center'] = params[1]
        #     init_params['sigma'] = params[2]
        #     init_params['fraction'] = params[3]
        #     init_params['slope'] = params[4]
        #     init_params['intercept'] = params[5]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    plot1, = ax1.plot(frames, center, label='center')
    ax2 = ax1.twinx()
    plot2, = ax2.plot(frames, sigma, 'g', label='$\sigma$')
    ax1.set_xlabel('Frame #')
    ax1.set_ylabel(r'q ($\AA^{-1}$)')
    ax2.set_ylabel(r' $\sigma$ ($\AA^{-1}$)')
    fig.suptitle('Fit Results ' + sample_name, fontsize=14)
    fig.legend()
    
    # saving peak fit params in separate csv files
    params_to_save = {'amplitude (au)' : amplitude, 
                      'center ($\AA$)' : center, 
                      'sigma ($\AA$)' : sigma, 
                      'std error amplitude (au)' : unc_a, 
                      'std error center ($\AA$)' : unc_c, 
                      'std error sigma ($\AA$)' : unc_s}
    df = pd.DataFrame(params_to_save)
    df.to_csv(os.path.join(save_path, hkl + '_peak_fit_results_' + sample_name + '.csv'), index=None)
    
fit_several_frames(1, num_frames, show_every = show_every, q_range=q_range)