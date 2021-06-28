# https://zhuanlan.zhihu.com/p/69752670
import cv2
import numpy as np
import matplotlib.pyplot as plt
from match_util import *
from KLT import *

def merge_frame(num_frame, source_path, background_path):
    feature_params = dict(maxCorners=200, qualityLevel=0.05, minDistance=15 , blockSize=5)
    lk_params = dict(winSize=(24, 24), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), minEigThreshold=1e-3)
    mask_frame = np.empty((num_frame,) ,dtype=np.ndarray)
    first_frame_warp = np.empty((num_frame,) ,dtype=np.ndarray)

    source = cv2.VideoCapture(source_path)
    background = cv2.VideoCapture(background_path)

    # light flow tracking
    for i in range(num_frame):
        prime_ret, prime_frame = source.read()
        second_ret, second_frame = background.read()
        if prime_ret is False or second_ret is False:
            break

        # get features in prime video
        prime_frame_gray = cv2.cvtColor(prime_frame, cv2.COLOR_BGR2GRAY)
        # pts_prime = getFeatures(prime_frame_gray, use_shi=False)

        # calculate optical flow by KLT
        # pts_second = estimateAllTranslation(pts_prime, prime_frame, second_frame)
        p0 = cv2.goodFeaturesToTrack(prime_frame_gray, mask=None, **feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prime_frame_gray, cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)
        # choose the match keypoints
        pts_second = p1[st == 1]
        pts_prime = p0[st == 1]

        # Warp prime frame to the second frame
        H, mask = cv2.findHomography(pts_second, pts_prime, cv2.RANSAC, 5.0) # src, dst
        size = (second_frame.shape[1], second_frame.shape[0])

        first_frame_warp[i] = cv2.warpPerspective(prime_frame, H, size)

        diff_frame = abs(np.sum(first_frame_warp[i].astype(int) - second_frame.astype(int), axis=2))
        diff_frame[diff_frame<100] = 0
        diff_frame[diff_frame>=100] = 1

        mask_frame[i] = np.dstack([diff_frame]*3)

        cv2.imshow('diff_frame', np.uint8(diff_frame*255))
        cv2.imshow('diff_frame 2', np.uint8(first_frame_warp[i] * mask_frame[i]))
        cv2.waitKey(10)

    cv2.destroyAllWindows()
    background.release()
    source.release()
        
    return mask_frame, first_frame_warp
    

if __name__ == '__main__':
    num_frame = 50
    background_path = "../video/room.mp4"
    prime_path = "../video/room_prime.mp4"
    second_path = "../video/room_second.mp4"

    first_mask, first_warp = merge_frame(num_frame, prime_path, background_path)
    print("Finish first clip!")
    # second_mask, second_warp = merge_frame(num_frame, second_path, background_path)
    # print("Finish second clip!")

    # out = cv2.VideoWriter('../output/kit.avi',0,cv2.VideoWriter_fourcc('M','J','P','G'),20.0,( int(background.get(3)), int(background.get(4))))
    background = cv2.VideoCapture(background_path)
    prime = cv2.VideoCapture(prime_path)
    second = cv2.VideoCapture(second_path)

    for i in range(num_frame):
        prime_ret, prime_frame = prime.read()
        second_ret, second_frame = second.read()
        background_ret, background_frame = background.read()

        result_frame = first_warp[i].copy()

        first_mask_frame = np.uint8(background_frame * first_mask[i])
        # second_mask_frame = np.uint8(second_frame * second_mask[i])

        result_frame[first_mask[i]>0] = 0
        result_frame = np.uint8(first_mask_frame + result_frame)

        # first_mask[i] = first_mask[i] * 255

        prime_concate_frame = np.concatenate((prime_frame, background_frame), axis=0)
        result_concate_frame = np.concatenate((result_frame, first_mask[i]*255), axis=0)

        cv2.imshow('prime & second frame', np.uint8(prime_concate_frame))
        cv2.imshow('result frame', np.uint8(result_concate_frame))
        cv2.waitKey(100)

        # out.write(result_frame)

    background.release()
    prime.release()
    second.release()

    # out.release()