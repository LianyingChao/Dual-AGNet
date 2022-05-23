# Dual-AGNet for low-dose CBCT reconstruction

Tensorflow = 1.15    python = 3.6

# Load noisy Data and trained models

Walnuts #19-21: https://zenodo.org/record/3763412 (such as walnut #19, to ./Dual-AGNet/re/walnut_19/noisy/)

Pre-model:  https://drive.google.com/file/d/1BQQMyo-YfV5nL3acAZFs34bFqMl4ZIGJ/view?usp=sharing (to ./Dual-AGNet/pre/Checkpoints/)

Post-model: https://drive.google.com/file/d/13TPRt9mopHNIevyjRmW_pRGwGoGsTpBZ/view?usp=sharing (to ./Dual-AGNet/post/Checkpoints/)


# Dual-AGNet based low-dose CBCT reconstruction
Beer-lambert law:  python ./re/python bl.py 

Prepare the input of Pre-AGNet:  python ./re/python cut2pre.py

Denoise the low-dose projections:  python ./pre/test.py

Reconstruct the CBCT images:  python ./re/reconstruction.py

Prepare the input of Post-AGNet:  python ./re/prepost.py

Refine the pre-processed slices:  python ./post/test.py


# Quantitative evaluation
python ./re/RPS.py

# Please contact us
D20208157@hust.edu.cn


