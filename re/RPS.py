import skimage.measure as measure
import numpy as np
import imageio
import re




s = 0
ssim0=np.zeros(600)
psnr0=np.zeros(600)
rmse0=np.zeros(600)


##########========19=========##########
for i in range(200):
    print(i+1)
    a = imageio.imread('/mnt/data3/Real-low-dose-ct/testend/19/nd/tif/%d.tif'%(i+1)).astype(float)
    a=(a-a.min())/(a.max()-a.min())
    b = imageio.imread('./walnut_19/post_re/%d.tif'%(i+1)).astype(float)
    ssim0[s] = measure.compare_ssim(a, b, data_range=1, multichannel=False)
    psnr0[s] = measure.compare_psnr(a, b, data_range=1)
    print(ssim0[s])
    print(psnr0[s])
    d=(a-b)**2
    d=np.sqrt(d)
    rmse0[s]=d.mean()
    print(rmse0[s])
    s=s+1



##########========20=========##########
for i in range(200):
    print(i+1)
    a = imageio.imread('/mnt/data3/Real-low-dose-ct/testend/20/nd/tif/%d.tif'%(i+1)).astype(float)
    #a=a[65:435,75:505]
    a=(a-a.min())/(a.max()-a.min())
    b = imageio.imread('./walnut_20/post_re/%d.tif'%(i+1)).astype(float)
    #b=b[65:435,75:505]
    #b=(b-b.min())/(b.max()-b.min())
    ssim0[s] = measure.compare_ssim(a, b, data_range=1, multichannel=False)
    psnr0[s] = measure.compare_psnr(a, b, data_range=1)
    print(ssim0[s])
    print(psnr0[s])
    d=(a-b)**2
    d=np.sqrt(d)
    rmse0[s]=d.mean()
    print(rmse0[s])
    s=s+1


##########========21=========##########
for i in range(200):
    print(i+1)
    a = imageio.imread('/mnt/data3/Real-low-dose-ct/testend/21/nd/tif/%d.tif'%(i+1)).astype(float)
    #a=a[65:435,75:505]
    a=(a-a.min())/(a.max()-a.min())
    b = imageio.imread('./walnut_21/post_re/%d.tif'%(i+1)).astype(float)
    #b=b[65:435,75:505]
    #b=(b-b.min())/(b.max()-b.min())
    ssim0[s] = measure.compare_ssim(a, b, data_range=1, multichannel=False)
    psnr0[s] = measure.compare_psnr(a, b, data_range=1)
    print(ssim0[s])
    print(psnr0[s])
    d=(a-b)**2
    d=np.sqrt(d)
    rmse0[s]=d.mean()
    print(rmse0[s])
    s=s+1


print('****************************************************0')
print('#############')
print(rmse0.mean())
print(np.std(rmse0))

print('#############')
print(psnr0.mean())
print(np.std(psnr0))

print('#############')
print(ssim0.mean())
print(np.std(ssim0))



