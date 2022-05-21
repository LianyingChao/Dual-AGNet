import numpy as np
import imageio

# cut the central regions of the projections 

for idx in range(3):
    print(idx)
    idx=idx+19
    angluar_sub_sampling = 1
    data_path='./walnut_%d/noisy_proj/'%idx
    path = './walnut_%d/pre_test/'%idx

    data_path_full = data_path
    projs_idx = range(1,501, angluar_sub_sampling)
    projs_rows = 544
    projs_cols = 576
    projs = np.zeros((projs_rows, projs_cols, len(projs_idx)), dtype=np.float32)

    trafo = lambda image : np.transpose(np.flipud(image))

    index=range(1,501,angluar_sub_sampling)
    # load projection data
    for i in range(len(projs_idx)):
        a=imageio.imread(data_path_full + '%d'%index[i] + '.tif')[384-272:384+272,486-288:486+288]
        projs[:,:,i]=(a-a.min())/(a.max()-a.min())

    # five consecutive projections as input, and the middle projection as output
    # It would be better to input nine or more consecutive projections
    for i in range(500):
        batch = np.zeros((544, 576, 5, 1), dtype=np.float32)
        for j in range(5):
            batch[:,:,j,0] = projs[:,:,(498+j+i)%500]
        np.save(path+'%d'%(i+1)+'.npy',batch)


