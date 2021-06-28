# https://zhuanlan.zhihu.com/p/69752670
import cv2
import numpy as np
import matplotlib.pyplot as plt
from match_util import *
from KLT import *

# prime = cv2.VideoCapture("../video/prime.mp4")
# second = cv2.VideoCapture("../video/second.mp4")
prime = cv2.VideoCapture("../video/iron2.mp4")
second = cv2.VideoCapture("../video/no_iron2.mp4")

n_frame = 100

feature_params = dict(maxCorners=200, qualityLevel=0.05, minDistance=15 , blockSize=5)
# KLT optical flow param
lk_params = dict(winSize=(24, 24), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), minEigThreshold=1e-3)
color = [[255, 255, 0], [51, 153, 255]] # BGR

out = cv2.VideoWriter('../output/output.avi',0,cv2.VideoWriter_fourcc('M','J','P','G'),20.0,( int(second.get(3)), int(second.get(4))))

# light flow tracking
for i in range(n_frame):
    prime_ret, prime_frame = prime.read()  # height, width
    second_ret, second_frame = second.read()
    if prime_ret is False or second_ret is False:
        break

    # get features in prime video
    prime_frame_gray = cv2.cvtColor(prime_frame, cv2.COLOR_BGR2GRAY)
    # p0 = cv2.goodFeaturesToTrack(prime_frame_gray, mask=None, **feature_params)
    pts_prime = getFeatures(prime_frame_gray, use_shi=False)

    second_frame_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow by KLT
    pts_second = estimateAllTranslation(pts_prime, prime_frame, second_frame)
    # p1, st, err = cv2.calcOpticalFlowPyrLK(prime_frame_gray, second_frame_gray, p0, None, **lk_params)
    # pts_second = p1[st == 1]
    # pts_prime = p0[st == 1]

    # Warp second frame to the first frame
    H, mask = cv2.findHomography(pts_second, pts_prime, cv2.RANSAC, 5.0) # src, dst
    size = (second_frame.shape[1], second_frame.shape[0])
    
    second_frame_warp = cv2.warpPerspective(second_frame, H, size)
    # second_frame_warp_gray = cv2.cvtColor(second_frame_warp, cv2.COLOR_BGR2GRAY)

    second_frame = second_frame_warp
    result_frame = second_frame.copy()
    
    # prime_mask = np.zeros_like()

    diff_frame = abs(np.sum(prime_frame.astype(int) - second_frame_warp.astype(int), axis=2))
    diff_frame[diff_frame<100] = 0
    diff_frame[diff_frame>=100] = 1

    diff_frame = np.dstack([diff_frame]*3)
    add_frame = np.uint8(prime_frame * diff_frame)
    add_frame[:,:,0] = np.uint8(add_frame[:,:,0] * 0.8)
    
    result_frame[diff_frame>0] = 0
    result_frame = np.uint8(add_frame + result_frame)

    prime_concate_frame = np.concatenate((prime_frame, second_frame), axis=0)
    result_concate_frame = np.concatenate((result_frame, add_frame), axis=0)

    total_frame = np.concatenate((prime_concate_frame, result_concate_frame), axis=1)

    # cv2.imshow('diff frame', np.uint8(add_frame))
    cv2.imshow('prime frame', np.uint8(total_frame))
    # cv2.imshow('prime & second frame', np.uint8(prime_concate_frame))
    # cv2.imshow('result frame', np.uint8(result_concate_frame))

    k = cv2.waitKey(27) & 0xff
    if k == 27:
        break
    
    out.write(result_frame)

cv2.destroyAllWindows()
prime.release()
second.release()
out.release()