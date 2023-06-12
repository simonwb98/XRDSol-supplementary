# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:55:34 2022

@author: Simon (Python 3.8)

Exporting logging data
"""

import numpy as np
from pathlib import Path
import ntpath
import os
import pandas as pd

### USER INPUTS:

file_path = r"C:\XRDSol\data\BT_April_22\log_files\TK10_Epi_wAS 001576.txt" # absolute location of scan
skip = 16 # skip + 1 = rows to skip 

### CODE:
  
file_name = ntpath.basename(file_path)
sample_name = file_name.rstrip('.txt') # under which name data is saved later
Path(os.path.join(file_path.rstrip(file_name), 'output_' + sample_name)).mkdir(parents=True, exist_ok=True)
save_path = os.path.join(file_path.rstrip(file_name), 'output_' + sample_name)
names=np.array(['Time of Day', 'Time', 'Pyrometer', 'Dispense X', 'Spin_Motor', 'BK Set Amps', 'BK Set Volts', 'BK Amps', 'BK Volts', 'BK Power', 'Sine'])
file = pd.read_csv(file_path, sep='\t', header = 0, names = names, skiprows = skip)
csv_path = os.path.join(save_path, sample_name + '.csv')
file.to_csv(csv_path, index = 0)
care_abouts = ['Time', 'Pyrometer', 'Spin_Motor']
for name in names:
    if name in care_abouts:    
        file.to_csv(os.path.join(save_path, name + '.csv'), index = 0, columns = np.array([name]))