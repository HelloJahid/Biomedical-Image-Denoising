import matplotlib.pyplot as plt

from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
import cv2


img = cv2.imread('/content/3.jpeg')
original = img_as_float(img)
# original = img_as_float(data.chelsea()[100:250, 50:300])

sigma = 0.12
noisy = random_noise(original, var=sigma**2)

# fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5),
#                        sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print(f"Estimated Gaussian noise standard deviation = {sigma_est}")

im_bayes = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)


psnr_bayes = peak_signal_noise_ratio(original, im_bayes)
wavelat_mse = mean_squared_error(original, im_bayes)
print(f"wavelat_mse = {wavelat_mse:0.5f}")
print(f"psnr_bayes = {psnr_bayes:0.5f}")

plt.imshow(im_bayes) 
plt.axis('off')