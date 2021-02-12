import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from skimage.util import random_noise

img = cv2.imread('/content/3.jpeg')
astro = img_as_float(img)
# astro = astro[30:180, 150:300]

sigma = 0.08
noisy = random_noise(astro, var=sigma**2)

# estimate the noise standard deviation from the noisy image
sigma_est = np.mean(estimate_sigma(noisy, multichannel=True))
print(f"estimated noise standard deviation = {sigma_est}")

patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)



# slow algorithm, sigma provided
denoise2 = denoise_nl_means(noisy, h=0.8 * sigma_est, sigma=sigma_est,
                            fast_mode=False, **patch_kw)


fig.tight_layout()

# print PSNR metric for each case

psnr2 = peak_signal_noise_ratio(astro, denoise2)
mse_fast = mean_squared_error(astro, denoise2_fast)

print(f"PSNR (fast, using sigma) = {psnr2:0.2f}")
print(f"MSE (fast, using sigma) = {mse_fast:0.5f}")


plt.imshow(denoise2)
plt.axis("off")
plt.show()