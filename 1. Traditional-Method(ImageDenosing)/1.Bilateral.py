# Read the image. 
main_img = cv2.imread('/content/otsu_mm.png') 
main_img_float = img_as_float(main_img)

# Apply bilateral filter with d = 15,  
# sigmaColor = sigmaSpace = 75. 
bilateral = cv2.bilateralFilter(main_img, 50, 75, 75) 

# img_c = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGBA)

psnr2_fast = peak_signal_noise_ratio(main_img_float, bilateral)
mse_fast = mean_squared_error(main_img_float, bilateral)
print(f"PSNR (fast, using sigma) = {psnr2_fast:0.2f}")
print(f"MSE (fast, using sigma) = {mse_fast:0.5f}")


plt.imshow(img_c)