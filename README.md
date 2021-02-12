____
# `Medical Image Denosing`

This repository contains material about Bimedical Image Denoising from scratch to advanced. It consists of notebooks as well as python file for various deep learning topics. In most cases, the notebooks lead you through implementing models such as,

#### Traditional Method
+ Bilateral
+ BM3D
+ NLM
+ Wavelet
+ and so on

#### Deep Learning Approach
+ AutoEncoder
+ GAN
+ and so on


___
# Bio-Medical Image and their Challenges
When a signal is transmitted over some distance obviously it is corrupted by thenoise.  Noise is random signal.  It is used to destroy most of the part of image in-formation.  Image distortion is most pleasance problems in image processing.  Imagedistorted due to various types of noise such as Gaussian noise, Poisson noise, Specklenoise,  Salt and Pepper noise and many more are fundamental noise types in caseof  digital  images.   There  are  mainly  two  types  of  noises  which  are  introduced  inOCT images during the process of image acquisition, short noise and speckle noise.Short  noise  is  also  described  by  Additive  White  Gaussian  Noise  (AWGN)  whichis additive in nature,  while speckle noise is multiplicative in nature [9].  However,during acquisition process image is corrupted either by thermal energy due to heatproduced by image sensors or due to physics-like photon nature of light [15].  Imageis acquired by measuring intensity of backscattered light from tissue via MichelsonInterferometer using Low Coherence Interferometry.  Additive white Gaussian noise(AWGN) is a basic noise model used in information theory to mimic the effect ofmany random processes that occur in nature.  Additive because it is added to anynoise that might be intrinsic to the information system.  White refers to the ideathat it has uniform power across the frequency band for the information system.  Itis an analogy to the color white which has uniform emissions at all frequencies inthe spectrum.  Gaussian because it has a normal distribution in the time domainwith an average time domain value of zero.  It is also called as electronic noise be-cause it arises in amplifiers or detectors.  Gaussian noise caused by natural sourcessuch as thermal vibration of atoms and discrete nature of radiation of warm objects3

Gaussian noise generally disturbs the gray values in digital images.  Noise isessentially identified by the noise power.  Noise power spectrum is constant in whitenoise.  In white noise, correlation is not possible because of every pixel value are dif-ferent from their neighbors.  That is why autocorrelation is zero.  So that image pixelvalues are normally disturb positively due to white noise.  Colored noise has manynames such as Brownian noise or pink noise or flicker noise or 1/f noise.  Brownianmotion seen due to the random movement of suspended particles in fluid.  Fractalnoise  is  caused  by  natural  process.   Impulse  Valued  Noise  this  is  also  called  datadrop noise because statistically its drop the original data values.  This noise is alsoreferred as salt and pepper noise.  This noise is seen in data transmission.  Imagepixel values are replaced by corrupted pixel values either maximum ‘or’ minimumpixel value i.e.  255 ‘or’ 0 respectively, if number of bits are 8 for transmission.  Saltand Pepper noise generally corrupted the digital image by malfunctioning of pixelelements in camera sensors, faulty memory space in storage, errors in digitizationprocess  and  many  more.   Image  formed  due  to  the  addition  of  various  crust  andtroughs of coherent waves produces a grainy representation known as speckle.  Noiseis pixel in image which shows different values instead of true value of pixel whichcould alter important meaningful feature use to diagnose disease.  Speckle noise isa special type of noise as it carries information about the image, acting as a majordegrading factor of an image.  This noise is multiplicative noise.  Their appearanceis seen in coherent imaging system such as laser, radar and acoustics etc.  Specklenoise can exist similar in an image as Gaussian noise.  Its probability density func-tion  follows  gamma  distribution  [16]  [13]  [6].However,  coherent  imaging  methodsoften suffer from multiplicative noise known as speckle.  Speckle is caused by theconstructive and destructive interference of the coherent returns scattered by smallreflectors within each resolution cell.  Furthermore, due to local processing natureof some of these methods, they often fail to preserve sharp features such as edgesand often contain block artifacts in the denoised image.  Imaging light is multiplyingscattered and coherently superimposed by the intraocular tissue that can generatethe speckle noise, which is a common problem in OCT images.  Removing specklenoise is challenging due to three factors.  First, speckle noise is a multiplicative noiseinstead of the additive white Gaussian noise.  Second, when imaging different tis-sues of the eye, the intensity of speckle noise is also different.  Finally, speckle noisenot only appears in the background but also exists in the retina, thereby severelydegrading the OCT imaging quality for clinical diagnosis and analysis.  OCT usesa low-coherence interferometry technique, and so the quality of OCT images is de-graded by speckle noise which is inherent in coherent imaging. To remove the specklenoise, many methods have been proposed including hardware- and software-basedmethods.  The latter approaches are flexible and easy to implement, and so manymethods such as median filter, Wiener filter, and wavelet filter have been proposed.Of these, use of a wavelet filter is a powerful method to reduce the speckle noise, andthis filter is often used in ultrasound images.  The model of the speckle noise hasa  multiplicative  nature,  and  conventional  filtering  methods  are  somewhat  ineffec-tive against this speckle noise.  OCT images normally suffer from granular patternscalled speckle noise.  Speckle noise is an inherent property of an OCT images whichaffects the visual quality of the images, hence difficult to diagnosis the patients.  So,speckle noise reduction is an essential preprocessing step, whenever OCT imaging isused for medical imaging.  






_
### Reference
(2) (PDF) Deep Learning Based Retinal OCT Image Denoising with Generative Adversarial Network. Available from: https://www.researchgate.net/publication/346955242_Deep_Learning_Based_Retinal_OCT_Image_Denoising_with_Generative_Adversarial_Network [accessed Feb 12 2021].



## Author
+ Name: Jahid Hasan
+ 𝐏𝐡𝐨𝐧𝐞:   (+880) 1772905097 (Whatsapp)
+ 𝘔𝘢𝘪𝘭:     jahidnoyon36@gmail.com
+ LinkedIn: http://linkedin.com/in/hellojahid
