import os
import random
import numpy as np
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join


CONTACT_MAPS_BASE_DIR = '../data/example_files/output/'
selected_targets = os.listdir(os.path.join(CONTACT_MAPS_BASE_DIR,'validTrue_maps'))

selected_targets=['5OHQA.npy']
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
method="ccmpred"

ax[0].set_title('ContactGAN Map')

for target in selected_targets:
    predicted_map = np.load(join(CONTACT_MAPS_BASE_DIR, '{}'.format(target)))[0][0]
    im = ax[0].imshow(predicted_map, interpolation='none')

plt.savefig('tmp.jpg')