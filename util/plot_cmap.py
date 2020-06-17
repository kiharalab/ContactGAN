import os
import numpy as np
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", nargs=2, help="Path to contact map")

args = parser.parse_args()
pathG=args.G_path
pathD=args.D_path

selected_targets=['5OHQA.npy']
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
method="ccmpred"

ax[0].set_title('ContactGAN Map')

for target in selected_targets:
    predicted_map = np.load(args.input)[0][0]
    im = ax[0].imshow(predicted_map, interpolation='none')

plt.savefig('predicted_map.jpg')