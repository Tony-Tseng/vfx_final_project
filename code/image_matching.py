import cv2
import matplotlib.pyplot as plt

prime = cv2.imread("image/prime.jpg")
second = cv2.imread("image/second.jpg")
prime = cv2.cvtColor(prime, cv2.COLOR_BGR2RGB)
second = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)

prime_gray = cv2.cvtColor(prime, cv2.COLOR_BGR2GRAY)
second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
prime_kp = sift.detect(prime_gray, None)
pts = cv2.KeyPoint_convert(prime_kp)
prime_pts_row = [pt[0] for pt in pts]
prime_pts_col = [pt[1] for pt in pts]

second_kp = sift.detect(second_gray, None)
pts = cv2.KeyPoint_convert(second_kp)
second_pts_row = [pt[0] for pt in pts]
second_pts_col = [pt[1] for pt in pts]

_, ax = plt.subplots(2, 1, squeeze=False)

ax[0,0].imshow(prime)
ax[0,0].scatter(prime_pts_row, prime_pts_col, s=3)
ax[0,0].axis('off')

ax[1,0].imshow(second)
ax[1,0].scatter(second_pts_row, second_pts_col, s=3)
ax[1,0].axis('off')

plt.show()