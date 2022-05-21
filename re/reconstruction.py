import numpy as np
import astra
import imageio
import scipy.io as io


idx=19
print(idx)
angluar_sub_sampling = 1
voxel_per_mm = 10
data_path_full='./walnut_%d/pre_proj/'%idx
recon_path = './walnut_%d/pre_re/'%idx
vecs=io.loadmat('vectors.mat')['vectors'] 

projs_idx = range(1,501, angluar_sub_sampling)
projs_name = 'scan_{:06}.tif'

projs_rows = 544
projs_cols = 576
projs = np.zeros((len(projs_idx), projs_rows, projs_cols), dtype=np.float32)
trafo = lambda image : np.transpose(np.flipud(image))
index=range(1,501,1)

# load projection data
for i in range(len(projs_idx)):
    projs[i]=imageio.imread(data_path_full + '%d'%index[i] + '.tif')

projs = np.transpose(projs, (1,0,2))
projs = np.ascontiguousarray(projs)

### compute FDK reconstruction ###
vol_sz  = 3*(44 * 10 + 8,)
vox_sz  = 1/voxel_per_mm
vol_rec = np.zeros(vol_sz, dtype=np.float32)
vol_geom = astra.create_vol_geom(vol_sz)
vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * vox_sz
vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * vox_sz
vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * vox_sz
vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * vox_sz
vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * vox_sz
vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * vox_sz
proj_geom = astra.create_proj_geom('cone_vec', projs_rows, projs_cols, vecs)
vol_id  = astra.data3d.link('-vol', vol_geom, vol_rec)
proj_id = astra.data3d.link('-sino', proj_geom, projs)
cfg = astra.astra_dict('FDK_CUDA')
cfg['ProjectionDataId'] = proj_id
cfg['ReconstructionDataId'] = vol_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 1)
astra.algorithm.delete(alg_id)
astra.data3d.delete(proj_id)
astra.data3d.delete(vol_id)
### save reconstruction ###
print('save results', flush=True)
np.transpose(vol_rec,[0,1,2])
for i in range(200):
    a=(vol_rec[:,:,i+150]-vol_rec[:,:,i+150].min())/(vol_rec[:,:,i+150].max()-vol_rec[:,:,i+150].min())
    imageio.imsave(recon_path+'%d'%(i+1)+'.tif', a)



idx=20
print(idx)
angluar_sub_sampling = 1
voxel_per_mm = 10
data_path_full='./walnut_%d/pre_proj/'%idx
recon_path = './walnut_%d/pre_re/'%idx
vecs=io.loadmat('vectors.mat')['vectors'] 

projs_idx = range(1,501, angluar_sub_sampling)
projs_name = 'scan_{:06}.tif'

projs_rows = 544
projs_cols = 576
projs = np.zeros((len(projs_idx), projs_rows, projs_cols), dtype=np.float32)
trafo = lambda image : np.transpose(np.flipud(image))
index=range(1,501,1)

# load projection data
for i in range(len(projs_idx)):
    projs[i]=imageio.imread(data_path_full + '%d'%index[i] + '.tif')

projs = np.transpose(projs, (1,0,2))
projs = np.ascontiguousarray(projs)

### compute FDK reconstruction ###
vol_sz  = 3*(44 * 10 + 8,)
vox_sz  = 1/voxel_per_mm
vol_rec = np.zeros(vol_sz, dtype=np.float32)
vol_geom = astra.create_vol_geom(vol_sz)
vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * vox_sz
vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * vox_sz
vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * vox_sz
vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * vox_sz
vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * vox_sz
vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * vox_sz
proj_geom = astra.create_proj_geom('cone_vec', projs_rows, projs_cols, vecs)
vol_id  = astra.data3d.link('-vol', vol_geom, vol_rec)
proj_id = astra.data3d.link('-sino', proj_geom, projs)
cfg = astra.astra_dict('FDK_CUDA')
cfg['ProjectionDataId'] = proj_id
cfg['ReconstructionDataId'] = vol_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 1)
astra.algorithm.delete(alg_id)
astra.data3d.delete(proj_id)
astra.data3d.delete(vol_id)
### save reconstruction ###
print('save results', flush=True)
np.transpose(vol_rec,[0,1,2])
for i in range(200):
    a=(vol_rec[:,i+150,:]-vol_rec[:,i+150,:].min())/(vol_rec[:,i+150,:].max()-vol_rec[:,i+150,:].min())
    imageio.imsave(recon_path+'%d'%(i+1)+'.tif', a)



idx=21
print(idx)
angluar_sub_sampling = 1
voxel_per_mm = 10
data_path_full='./walnut_%d/pre_proj/'%idx
recon_path = './walnut_%d/pre_re/'%idx
vecs=io.loadmat('vectors.mat')['vectors'] 

projs_idx = range(1,501, angluar_sub_sampling)
projs_name = 'scan_{:06}.tif'

projs_rows = 544
projs_cols = 576
projs = np.zeros((len(projs_idx), projs_rows, projs_cols), dtype=np.float32)
trafo = lambda image : np.transpose(np.flipud(image))
index=range(1,501,1)

# load projection data
for i in range(len(projs_idx)):
    projs[i]=imageio.imread(data_path_full + '%d'%index[i] + '.tif')

projs = np.transpose(projs, (1,0,2))
projs = np.ascontiguousarray(projs)

### compute FDK reconstruction ###
vol_sz  = 3*(44 * 10 + 8,)
vox_sz  = 1/voxel_per_mm
vol_rec = np.zeros(vol_sz, dtype=np.float32)
vol_geom = astra.create_vol_geom(vol_sz)
vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * vox_sz
vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * vox_sz
vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * vox_sz
vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * vox_sz
vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * vox_sz
vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * vox_sz
proj_geom = astra.create_proj_geom('cone_vec', projs_rows, projs_cols, vecs)
vol_id  = astra.data3d.link('-vol', vol_geom, vol_rec)
proj_id = astra.data3d.link('-sino', proj_geom, projs)
cfg = astra.astra_dict('FDK_CUDA')
cfg['ProjectionDataId'] = proj_id
cfg['ReconstructionDataId'] = vol_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 1)
astra.algorithm.delete(alg_id)
astra.data3d.delete(proj_id)
astra.data3d.delete(vol_id)
### save reconstruction ###
print('save results', flush=True)
np.transpose(vol_rec,[0,1,2])
for i in range(200):
    a=(vol_rec[:,:,i+150]-vol_rec[:,:,i+150].min())/(vol_rec[:,:,i+150].max()-vol_rec[:,:,i+150].min())
    imageio.imsave(recon_path+'%d'%(i+1)+'.tif', a)

