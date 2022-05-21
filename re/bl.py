import numpy as np
import os
import imageio

# convert the initial projections using Beer-lambert law

for idx in range(3):
    idx=idx+19
    angluar_sub_sampling = 1
    data_path='./walnut_%d/noisy/'%idx
    preData_path = './walnut_%d/noisy_proj/'%idx

    projs_idx = range(1,501, angluar_sub_sampling)
    projs_name = 'scan_{:06}.tif'
    dark_name = 'di000000.tif'
    flat_name = ['io000000.tif', 'io000001.tif']
    projs_rows = 768
    projs_cols = 972
    projs = np.zeros((len(projs_idx), projs_rows, projs_cols), dtype=np.float32)
    dark = imageio.imread(os.path.join(data_path, dark_name))
    flat = np.zeros((2, projs_rows, projs_cols), dtype=np.float32)
    for i, fn in enumerate(flat_name):
        flat[i] = imageio.imread(os.path.join(data_path, fn))
    flat =  np.mean(flat,axis=0)

    for i in range(len(projs_idx)):
        projs[i] = imageio.imread(os.path.join(data_path, projs_name.format(projs_idx[i])))

    print('pre-process data', flush=True)
    projs -= dark
    projs /= (flat - dark)
    np.log(projs, out=projs)
    np.negative(projs, out=projs)

    #save data as tif format
    for i in range(len(projs_idx)):
        print(projs_idx[i])
        imageio.imsave(preData_path + '%d' % (projs_idx[i]) + '.tif', projs[i,:,:])

