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

feature_params = dict(maxCorners=200, qualityLevel=0.05, minDistance=15 , blockSize=5)

# KLT optical flow param
lk_params = dict(winSize=(24, 24), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), minEigThreshold=1e-3)
color = [[255, 255, 0], [51, 153, 255]] # BGR

count=0
# light flow tracking
while True:
    prime_ret, prime_frame = prime.read()  # height, width
    second_ret, second_frame = second.read()
    if prime_ret is False or second_ret is False:
        break

    # get features in prime video
    prime_frame_gray = cv2.cvtColor(prime_frame, cv2.COLOR_BGR2GRAY)
    # p0 = cv2.goodFeaturesToTrack(prime_frame_gray, mask=None, **feature_params)
    p0 = getFeatures(prime_frame_gray, use_shi=False)

    second_frame_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow by KLT
    p1 = estimateAllTranslation(p0, prime_frame, second_frame)
    # p1, st, err = cv2.calcOpticalFlowPyrLK(prime_frame_gray, second_frame_gray, p0, None, **lk_params)
    # choose the match keypoints
    # pts_second = p1[st == 1] # width, height
    # pts_prime = p0[st == 1]
    pts_second = p1
    pts_prime = p0
    # print(pts_prime.shape)
    # print(pts_second.shape)

    # Warp second frame to the first frame
    H, mask = cv2.findHomography(pts_second, pts_prime, cv2.RANSAC, 5.0) # src, dst
    size = (second_frame.shape[1], second_frame.shape[0])
    
    second_frame_warp = cv2.warpPerspective(second_frame, H, size)
    second_frame_warp_gray = cv2.cvtColor(second_frame_warp, cv2.COLOR_BGR2GRAY)

    # pixel_prob = pixel_consistency(prime_frame, second_frame, pts_prime, pts_second)
    # motion_prob, new_pts_second = motion_consistency(pts_prime, pts_second, H)

    # frame_prob = calculate_whole_prob(prime_frame, pts_prime, pixel_prob, motion_prob)
    # prime_frame_arrow = prime_frame.copy()
    # second_frame_arrow = second_frame.copy()

    # draw the tracking line
    # for i, (pt_prime, pt_second) in enumerate(zip(pts_prime, pts_second)):
    #     a,b = pt_prime.ravel()
    #     c,d = pt_second.ravel()
    #     a, b, c, d = int(a), int(b), int(c), int(d)
        
    #     point_color = color[0]
    #     if(pixel_prob[i] + motion_prob[i] < 5e-3):
    #         point_color = color[1]

    #     prime_frame_arrow = cv2.arrowedLine(prime_frame_arrow, (a,b),(c,d), point_color, 2)
    #     second_frame_arrow = cv2.arrowedLine(second_frame_arrow, (c,d),(a,b), point_color, 2)
        # prime_frame_arrow = cv2.circle(prime_frame,(a,b),5,point_color,-1)
        # second_frame_arrow = cv2.circle(second_frame,(c,d),5,point_color,-1)
    
    # frame = np.concatenate((prime_frame_arrow, second_frame_arrow), axis=0)
    # cv2.imshow('frame', frame)

    # prob_threshold = threshold_yen(frame_prob)
    # rescale_frame_prob = rescale_intensity(frame_prob, (0, prob_threshold), (0, 255))
    # frame_prob[frame_prob<1e-2] = 1
    # frame_prob[frame_prob>=1e-2] = 0

    # normalize_frame = np.uint8((np.max(frame_prob) - frame_prob) / (np.max(frame_prob) - np.min(frame_prob))*255)
    # normalize_frame[normalize_frame>200] = 255
    # normalize_frame[normalize_frame<=200] = 0
    # cv2.imshow('prob frame', normalize_frame)

    # mask_frame = np.dstack([normalize_frame]*1)
    # mask_frame = normalize_frame
    # diff_frame = prime_frame_gray * normalize_frame
    # cv2.imshow('diff frame', np.uint8(diff_frame))

    second_frame = second_frame_warp

    diff_frame = (prime_frame[:,:,0].astype(int) - second_frame[:,:,0].astype(int)) # * mask_frame
    diff_frame[diff_frame<100] = 0

    diff_frame = np.dstack([diff_frame]*3)
    second_frame = diff_frame + second_frame
    cv2.imshow('diff frame', np.uint8(diff_frame))
    cv2.imshow('diff frame', np.uint8(second_frame))

    k = cv2.waitKey(27) & 0xff
    if k == 27:
        break
    count+=1

cv2.destroyAllWindows()
prime.release()
second.release()