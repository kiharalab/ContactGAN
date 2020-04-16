import os
import random
import numpy as np
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join


CONTACT_MAPS_BASE_DIR = '/net/kihara/home/smaddhur/tensorFlow/CPGAN/paper/output_npy/validation/'
NB_TARGETS = 3

# Main code



# selected_targets=['1UJ8A', '4WFTC', '2JMSA', '2RNGA']
# selected_targets=['1BHUA', '2RNGA', '5GZTA', '1HP8A', '2CWYA', '2K49A', '2JMSA', '1VQOZ', '2KRXA', '3ID1A', '1UJ8A', '5YWRB', '5APGB', '2I9XA', '3BW6A', '5OHQA', '1IM3P', '3S8SA', '5M0WA', '3PESB', '2KSNA', '4XALA', '6FC0B', '1TM9A', '4WFTC', '6H9HB', '3BAMA', '1X9BA']
# create a figure
selected_targets = os.listdir(os.path.join(CONTACT_MAPS_BASE_DIR,'validTrue_maps'))
random_seed= 42
np.random.seed(random_seed)
np.random.shuffle(selected_targets)
selected_targets = selected_targets[:NB_TARGETS]
# fig, ax = plt.subplots(len(selected_targets), 3, figsize=(12, 12))
# fig, ax = plt.subplots(NB_TARGETS, 3, figsize=(12, 12))

selected_targets=['5OHQA.npy']
fig, ax = plt.subplots(1, 4, figsize=(12, 12))
print(len(selected_targets))
method="ccmpred_deepcontact_deepcov"
# ax[0].set_title('True Map')
# ax[1].set_title(method+' Map')
# ax[2].set_title('PCDGAN Map')

# ax[0].set_title('True Map')
ax[0].set_title('CCMPred Map')
ax[1].set_title('DeepCov Map')
ax[2].set_title('DeepContact Map')
ax[3].set_title('contactGAN Map')

# ax[0,0].set_title('True Map')
# # ax[0,1].set_title(method+' Map')
# ax[0,1].set_title('CCMPRED Map')
# ax[0,2].set_title('PCDGAN Map')

# ax[1,0].set_title('True Map')
# # ax[0,1].set_title(method+' Map')
# ax[1,1].set_title('CCMPRED Map')
# ax[1,2].set_title('PCDGAN Map')

# ax[2,0].set_title('True Map')
# # ax[0,1].set_title(method+' Map')
# ax[2,1].set_title('CCMPRED Map')
# ax[2,2].set_title('PCDGAN Map')
print(selected_targets)
# ax[0, 0].set_title('CCMPRED Map')
# ax[0, 1].set_title('CPGAN Map')

# show in figure 10 random 64x64 slices
i = 0
for target in selected_targets:
    # print(join(CONTACT_MAPS_BASE_DIR, 'validTrue_maps','{}.npy'.format(target)))
    # print(join(CONTACT_MAPS_BASE_DIR, 'validPred_maps','deepcov_in','{}.npy'.format(target)))
    true_map = np.load(join(CONTACT_MAPS_BASE_DIR, 'validTrue_maps','{}'.format(target)))[0][0]
    method_map = np.load(join(CONTACT_MAPS_BASE_DIR, method,'{}'.format(target)))[0][0]
    method_map1 = np.load(join(CONTACT_MAPS_BASE_DIR, method,'{}'.format(target)))[0][1]
    method_map2 = np.load(join(CONTACT_MAPS_BASE_DIR, method,'{}'.format(target)))[0][2]
    predicted_map = np.load(join(CONTACT_MAPS_BASE_DIR, 'validPred_'+method,'{}'.format(target)))[0][0]
    # print(true_map.shape)
    # print(method_map.shape)
    # print(predicted_map.shape)

    # im1 = ax[0].imshow(true_map, interpolation='none')
    # im1 = ax[1].imshow(method_map, interpolation='none')
    # im2 = ax[2].imshow(predicted_map, interpolation='none')

    # im1 = ax[0].imshow(true_map, interpolation='none')
    im1 = ax[0].imshow(method_map, interpolation='none')
    im2 = ax[1].imshow(method_map1, interpolation='none')
    im3 = ax[2].imshow(method_map2, interpolation='none')
    im4 = ax[3].imshow(predicted_map, interpolation='none')

    # im1 = ax[i, 0].imshow(true_map, interpolation='none')
    # im2 = ax[i, 1].imshow(method_map, interpolation='none')
    # im3 = ax[i, 2].imshow(predicted_map, interpolation='none')
    i += 1

plt.savefig('tmp.jpg')