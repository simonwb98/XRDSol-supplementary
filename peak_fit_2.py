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
import winsound
import imageio
import time
import numpy as np

from scipy import signal
from tqdm import tqdm

from lmfit.models import LinearModel, PseudoVoigtModel
plt.style.use('default')

### USER INPUTS:

file_path = r"C:\Users\User\Desktop\LBL\BT_December\GaAs_SAM\GIWAXS\GaAs_SAM.dat"
q_range = (0.85, 0.97)                # define q range to be fitted  
double_peak = False
strict_bounds = (True, 0.85, 0.95)
make_gif = True
   
init_params = {                     # initial guess parameters
               'amplitude' : 3,     # default: 2
               'center' : 0.91,        # 1 (in angstrom)
               'sigma' : 0.01,      # 0.01
               'fraction' : 0.5,    # 0.5
               'slope' : 0,         # 0
               'intercept' : 500    # 700
               }
show_every = 20             # int n, shows every n'th frame with fit
show_these = [int(i) for i in range(1, 680, 1)]
hkl = 'PbI'
hkl2 = 'alpha_pvsk_no_avg'
avg_frames = 1 # int, number of frames to average each iteration
### OTHER:
  
file_name = ntpath.basename(file_path)
sample_name = file_name.rstrip('.dat')
import_path = os.path.join(file_path.rstrip(file_name), 'output_' + sample_name)
#Path(os.path.join(file_path.rstrip(file_name), hkl + '_peak_fit_' + sample_name)).mkdir(parents=True, exist_ok=True)
save_path = import_path # os.path.join(file_path.rstrip(file_name), hkl + '_peak_fit_' + sample_name)
csv_path = os.path.join(import_path, sample_name + '.csv')
num_frames = max(pd.read_csv(os.path.join(import_path, 'intensity_' + sample_name + '.csv'), 
                   nrows=1, header=None))

if avg_frames != 1 or avg_frames != None:
    assert(type(avg_frames)==int)
    avg = True

if double_peak: 
    init_params['p2_amplitude'] = 2     # default: 2
    init_params['p2_center'] = 1.0        # 1 (in angstrom)
    init_params['p2_sigma'] = 0.005      # 0.01
    init_params['p2_fraction'] = 0.5    # 0.5
    
if make_gif:
    filenames = []
    
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
    # select data to average over
    avg_cols = list(range(frame_index - avg_frames, frame_index))
    df_i = pd.read_csv(os.path.join(import_path, 'intensity_' + sample_name + '.csv'), 
                           usecols = avg_cols)
    
    # average intensities stored in dataframe
    if avg_frames != 1:
        column_names = df_i.columns.tolist()
        df_i = pd.DataFrame(df_i[column_names].mean(axis=1), columns = ['mean'])
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
    
    if double_peak:
        peak2 = PseudoVoigtModel(prefix='p2_')
        
        mod = peak + peak2 + background
        # initial values
        pars = mod.make_params(amplitude = init_params['amplitude'], 
                               center = init_params['center'], 
                               sigma = init_params['sigma'], 
                               fraction = init_params['fraction'], 
                               
                               p2_amplitude = init_params['p2_amplitude'], 
                               p2_center = init_params['p2_center'], 
                               p2_sigma = init_params['p2_sigma'], 
                               p2_fraction = init_params['p2_fraction'],
                               
                               slope = init_params['slope'], 
                               intercept = init_params['intercept'])
    else:
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
    if strict_bounds[0]:
        pars.add('center', value=init_params['center'], min=strict_bounds[1], max=strict_bounds[2])
        pars.add('intercept', value=init_params['intercept'], min=init_params['intercept']*0.8, max=init_params['intercept']*1.2)
    else:
        pars.add('center', value=init_params['center'], min=min(q_range)*1.05, max=max(q_range)*0.95)
        pars.add('intercept', value=init_params['intercept'], min=init_params['intercept']*0.8, max=init_params['intercept']*1.2)
    pars.add('sigma', value=init_params['sigma'], max=0.05)
    
    if double_peak:
        pars.add('p2_amplitude', value=init_params['p2_amplitude'], min=0)
        pars.add('p2_center', value=init_params['p2_center'], min=0.95, max=1.1)
    
    
    # # "hints" - don't change
    # mod.set_param_hint('amplitude', min=0)
    # mod.set_param_hint('center', min=0.95, max=1.05)
    # mod.set_param_hint('sigma', max=0.01)
    
    # determine if peak in data, promninence of 190 is chosen by hand, doesn't
    # need to be ideal for every sample
    peak_in_frame = False #initially false
    peaks = signal.find_peaks(y, prominence=30)[0]
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
            if make_gif:
                plt.ylim([0, 6000])
                plt.savefig(f'{save_path}\{sample_name}_{frame_index}.jpg', dpi=200)
                filenames.append(f'{save_path}\{sample_name}_{frame_index}.jpg')
            plt.clf()

    elif len(peaks) == 0:
        if frame_index in frames_to_plot:
            plt.plot(x, y, 'o', label='intensity')
            plt.xlabel(r'q $(\AA^{-1})$')
            plt.ylabel(r'Intensity (au)')
            # result.plot(data_kws={'markersize': 1})
            plt.legend()
            plt.title('Frame: ' + str(frame_index))
            plt.clf()
           
        
        if not double_peak:
            params = [None]*6
            std_error = [None]*3
            redchi = [None]
        else:
            params = [None]*10
            std_error = [None]*6
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
    
    
    if double_peak:
        amplitude_2, unc_a_2 = [], []
        center_2, unc_c_2 = [], []
        sigma_2, unc_s_2 = [], []
        fraction_2 = []
        all_params = [amplitude, center, sigma, fraction, amplitude_2, center_2, sigma_2, slope, intercept]
        peak_unc = [unc_a, unc_c, unc_s, unc_a_2, unc_c_2, unc_s_2]
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
    
    if double_peak:
        plot1, = ax1.plot(frames, center_2, 'r', label='center 2nd peak')
        plot2, = ax2.plot(frames, sigma_2, 'y', label='sigma 2nd peak')
        
    ax1.set_xlabel('Frame #')
    ax1.set_ylabel(r'q ($\AA^{-1}$)')
    ax1.set_ylim(q_range)
    ax2.set_ylabel(r' $\sigma$ ($\AA^{-1}$)')
    fig.suptitle('Fit Results ' + sample_name, fontsize=14)
    fig.legend()
    plt.savefig(f'{save_path}\{sample_name}_{hkl}_fit_results.jpg', dpi=100)
    # saving peak fit params in separate csv files
    time = np.linspace(-1.844/2, num_frames*1.844 - 1.844/2, num_frames + 1)
    print(len(time), len(amplitude), len(sigma))
    if double_peak:
        params_to_save = {'time (s)' : time,
                          'amplitude (au)' : amplitude, 
                          'center ($\AA$)' : center, 
                          'sigma ($\AA$)' : sigma, 
                          'std error amplitude (au)' : unc_a, 
                          'std error center ($\AA$)' : unc_c, 
                          'std error sigma ($\AA$)' : unc_s,
                          
                          
                          'amplitude 2 (au)' : amplitude_2, 
                          'center 2 ($\AA$)' : center_2, 
                          'sigma 2 ($\AA$)' : sigma_2, 
                          'std error amplitude 2 (au)' : unc_a_2, 
                          'std error center 2 ($\AA$)' : unc_c_2, 
                          'std error sigma 2 ($\AA$)' : unc_s_2
                          }
    else:
        params_to_save = {'time (s)' : time,
                          'amplitude (au)' : amplitude, 
                          'center ($\AA$)' : center, 
                          'sigma ($\AA$)' : sigma, 
                          'std error amplitude (au)' : unc_a, 
                          'std error center ($\AA$)' : unc_c, 
                          'std error sigma ($\AA$)' : unc_s,
                          }
    df = pd.DataFrame(params_to_save)
    df.to_csv(os.path.join(save_path, hkl + '_peak_fit_results_' + sample_name + '.csv'), index=None)
    
fit_several_frames(avg_frames, num_frames, show_every = show_every, q_range=q_range)

if make_gif:
    with imageio.get_writer(f'{save_path}\{sample_name}_{hkl}.gif', duration=0.3, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
print('Done making gif')
    # os.remove(f'{save_path}\{sample_name}_*.jpg')
# make noise when done (optional)
duration = 200  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
time.sleep(.1)
winsound.Beep(freq, 2*duration)