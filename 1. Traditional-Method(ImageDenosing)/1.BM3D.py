import bm3d
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('/content/1.jpeg', 0)
img_float = img_as_float(img)
denoised_image = bm3d.bm3d(img + 0.5, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

psnr2_fast = peak_signal_noise_ratio(img_float, denoised_image)
mse_fast = mean_squared_error(img_float, denoised_image)
print(f"PSNR (fast, using sigma) = {psnr2_fast:0.2f}")
print(f"MSE (fast, using sigma) = {mse_fast:0.5f}")

plt.yticks([])
plt.xticks([])
plt.imshow(denoised_image, 'gray')

plt.show()

