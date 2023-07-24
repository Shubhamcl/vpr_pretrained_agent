from os.path import join
from glob import glob

import h5py
import cv2
import numpy as np
from alive_progress import alive_bar

# Insert address to resize images
address = ''

all_h5_files = glob(join(address, '*'))

with alive_bar(len(all_h5_files)) as bar:

    for h5_file_address in all_h5_files:

        # Open each file in read-write model:
        with h5py.File(h5_file_address, 'r+') as f:
            for key in f.keys():
                
                step_number = key.split('_')[1] # in str form
                
                # Copy the frame
                f.copy(f'/step_{step_number}/obs/central_rgb/frame', 
                    f'/step_{step_number}/obs/small_central_rgb/frame')
                
                # Getting, resizing and saving image, all on the fly
                f[f'/step_{step_number}/obs/small_central_rgb/data'] = cv2.resize(f[f'/step_{step_number}/obs/central_rgb/data'][:], (224, 224))
        bar()
