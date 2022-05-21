import numpy as np
import imageio
for idx in range(3):
    idx=idx+19
    print(idx)
    data_path='./walnut_%d/pre_re/'%idx
    path = './walnut_%d/post_test/'%idx
    recon = np.zeros((448, 448, 200), dtype=np.float32)
    for i in range(200):
        a=imageio.imread(data_path + '%d.tif'%(i+1))
        recon[:,:,i]=(a-a.min())/(a.max()-a.min())

    batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
    batch[:,:,0,0] = recon[:,:,0]
    batch[:,:,1,0] = recon[:,:,0]
    batch[:,:,2,0] = recon[:,:,0]
    batch[:,:,3,0] = recon[:,:,0]
    batch[:,:,4,0] = recon[:,:,0]
    batch[:,:,5,0] = recon[:,:,1]
    batch[:,:,6,0] = recon[:,:,2]
    batch[:,:,7,0] = recon[:,:,3]
    batch[:,:,8,0] = recon[:,:,4]
    np.save(path+'1.npy',batch)

    batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
    batch[:,:,0,0] = recon[:,:,0]
    batch[:,:,1,0] = recon[:,:,0]
    batch[:,:,2,0] = recon[:,:,0]
    batch[:,:,3,0] = recon[:,:,0]
    batch[:,:,4,0] = recon[:,:,1]
    batch[:,:,5,0] = recon[:,:,2]
    batch[:,:,6,0] = recon[:,:,3]
    batch[:,:,7,0] = recon[:,:,4]
    batch[:,:,8,0] = recon[:,:,5]
    np.save(path+'2.npy',batch)


    batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
    batch[:,:,0,0] = recon[:,:,0]
    batch[:,:,1,0] = recon[:,:,0]
    batch[:,:,2,0] = recon[:,:,0]
    batch[:,:,3,0] = recon[:,:,1]
    batch[:,:,4,0] = recon[:,:,2]
    batch[:,:,5,0] = recon[:,:,3]
    batch[:,:,6,0] = recon[:,:,4]
    batch[:,:,7,0] = recon[:,:,5]
    batch[:,:,8,0] = recon[:,:,6]
    np.save(path+'3.npy',batch)



    batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
    batch[:,:,0,0] = recon[:,:,0]
    batch[:,:,1,0] = recon[:,:,0]
    batch[:,:,2,0] = recon[:,:,1]
    batch[:,:,3,0] = recon[:,:,2]
    batch[:,:,4,0] = recon[:,:,3]
    batch[:,:,5,0] = recon[:,:,4]
    batch[:,:,6,0] = recon[:,:,5]
    batch[:,:,7,0] = recon[:,:,6]
    batch[:,:,8,0] = recon[:,:,7]
    np.save(path+'4.npy',batch)

    batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
    for i in range(192):
        batch[:,:,0,0] = recon[:,:,i]
        batch[:,:,1,0] = recon[:,:,i+1]
        batch[:,:,2,0] = recon[:,:,i+2]
        batch[:,:,3,0] = recon[:,:,i+3]
        batch[:,:,4,0] = recon[:,:,i+4]
        batch[:,:,5,0] = recon[:,:,i+5]
        batch[:,:,6,0] = recon[:,:,i+6]
        batch[:,:,7,0] = recon[:,:,i+7]
        batch[:,:,8,0] = recon[:,:,i+8]
        np.save(path+'%d'%(i+5)+'.npy',batch)


    batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
    batch[:,:,0,0] = recon[:,:,192]
    batch[:,:,1,0] = recon[:,:,193]
    batch[:,:,2,0] = recon[:,:,194]
    batch[:,:,3,0] = recon[:,:,195]
    batch[:,:,4,0] = recon[:,:,196]
    batch[:,:,5,0] = recon[:,:,197]
    batch[:,:,6,0] = recon[:,:,198]
    batch[:,:,7,0] = recon[:,:,199]
    batch[:,:,8,0] = recon[:,:,199]
    np.save(path+'197.npy',batch)


    batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
    batch[:,:,0,0] = recon[:,:,193]
    batch[:,:,1,0] = recon[:,:,194]
    batch[:,:,2,0] = recon[:,:,195]
    batch[:,:,3,0] = recon[:,:,196]
    batch[:,:,4,0] = recon[:,:,197]
    batch[:,:,5,0] = recon[:,:,198]
    batch[:,:,6,0] = recon[:,:,199]
    batch[:,:,7,0] = recon[:,:,199]
    batch[:,:,8,0] = recon[:,:,199]
    np.save(path+'198.npy',batch)



    batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
    batch[:,:,0,0] = recon[:,:,194]
    batch[:,:,1,0] = recon[:,:,195]
    batch[:,:,2,0] = recon[:,:,196]
    batch[:,:,3,0] = recon[:,:,197]
    batch[:,:,4,0] = recon[:,:,198]
    batch[:,:,5,0] = recon[:,:,199]
    batch[:,:,6,0] = recon[:,:,199]
    batch[:,:,7,0] = recon[:,:,199]
    batch[:,:,8,0] = recon[:,:,199]
    np.save(path+'199.npy',batch)



    batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
    batch[:,:,0,0] = recon[:,:,195]
    batch[:,:,1,0] = recon[:,:,196]
    batch[:,:,2,0] = recon[:,:,197]
    batch[:,:,3,0] = recon[:,:,198]
    batch[:,:,4,0] = recon[:,:,199]
    batch[:,:,5,0] = recon[:,:,199]
    batch[:,:,6,0] = recon[:,:,199]
    batch[:,:,7,0] = recon[:,:,199]
    batch[:,:,8,0] = recon[:,:,199]
    np.save(path+'200.npy',batch)


