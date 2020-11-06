import pywt
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
# Lena1 resmi okundu ve resize edildi. imshow fonksiyonu ile ekranda gösterildi.
img = cv.imread('lena1.jpg', 0) #read gray image.
img = cv.resize(img, (256,256))
window_name_lena1 = 'Lena1'
cv.imshow(window_name_lena1,img)
cv.waitKey()
# Wavelet transform
coeffs = pywt.swt2(img, 'db2', 1) 
coeffs = coeffs[0] 
LL, (LH, HL, HH) = coeffs

A1L1 = np.float32(LL) 
H1L1 = np.float32(LH) 
V1L1 = np.float32(HL) 
D1L1 = np.float32(HH)
# Lena2 resmi okundu ve resize edildi. imshow fonksiyonu ile ekranda gösterildi.

img2 = cv.imread('lena2.jpg', 0) #read gray image.
img2 = cv.resize(img2, (256,256))
window_name_lena2 = 'Lena2'
cv.imshow(window_name_lena2,img2)
cv.waitKey()
plt.figure(1)
# Wavelet transform
coeffs2 = pywt.swt2(img2, 'db2', 1) 
coeffs2 = coeffs2[0] 
AA, (AB, BA, BB) = coeffs2

A2L1 = np.float32(AA) 
H2L1 = np.float32(AB) 
V2L1 = np.float32(BA) 
D2L1 = np.float32(BB)
# Fusion başladı. 
AfL1 = 0.5*(A1L1 + A2L1)
D = (np.abs(H1L1) - np.abs(H2L1)) >=0
HfL1 = np.multiply(D, H1L1) + (np.logical_not(D))*H2L1
D = (np.abs(V1L1) - np.abs(V2L1)) >=0
VfL1 = np.multiply(D, V1L1) + (np.logical_not(D))*V2L1
D = (np.abs(D1L1) - np.abs(D2L1)) >=0
DfL1 = np.multiply(D, D1L1) + (np.logical_not(D))*D2L1

coeffs3 = AfL1, (HfL1, VfL1, DfL1)
#Fusion işlemi sonucu oluşan resim
imf = np.uint8(pywt.iswt2(coeffs3,'db2'));
window_name_lena2 = 'Lena2'
cv.imshow(window_name_lena2,imf)

cv.waitKey()
plt.figure(1)
plt.subplot(121);
plt.imshow(img)
plt.subplot(122);
plt.imshow(img2)

plt.figure(2)
plt.imshow(imf)




