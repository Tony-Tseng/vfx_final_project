from skimage.feature import corner_harris, corner_shi_tomasi, peak_local_max
from numpy.linalg import inv
from utils import interp2
from scipy import signal
import numpy as np
import cv2

def getFeatures(img, use_shi=False):
    if use_shi:
        corner_response = corner_shi_tomasi(img)
    else:
        corner_response = corner_harris(img)
    coordinates = peak_local_max(corner_response, min_distance=15, num_peaks=300, exclude_border=20)
    # coordinates = peak_local_max(corner_response, min_distance=10, num_peaks=20, exclude_border=2)
    # return np.float32(coordinates.reshape(-1,1,2))
    return np.float32(coordinates)

def estimateAllTranslation(points, img1, img2):
    I = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    I = cv2.GaussianBlur(I,(5,5),0.2)
    Iy, Ix = np.gradient(I.astype(float))

    startXs_flat = points[:,1].flatten()
    startYs_flat = points[:,0].flatten()
    newXs = np.full(startXs_flat.shape,-1,dtype=float)
    newYs = np.full(startYs_flat.shape,-1,dtype=float)
    for i in range(np.size(points[:,1])):
        if startXs_flat[i] != -1:
            newXs[i], newYs[i] = estimateFeatureTranslation(startXs_flat[i], startYs_flat[i], Ix, Iy, img1, img2)
    newXs = newXs.reshape(-1, 1)
    newYs = newYs.reshape(-1, 1)
    new_points = np.concatenate((newYs, newXs), axis=1)
    return np.float32(new_points)

def estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2, WINDOW_SIZE = 25):
    X=startX
    Y=startY
    mesh_x,mesh_y=np.meshgrid(np.arange(WINDOW_SIZE),np.arange(WINDOW_SIZE))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    mesh_x_flat_fix =mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
    mesh_y_flat_fix =mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
    coor_fix = np.vstack((mesh_x_flat_fix,mesh_y_flat_fix))
    I1_value = interp2(img1_gray, coor_fix[[0],:], coor_fix[[1],:])
    Ix_value = interp2(Ix, coor_fix[[0],:], coor_fix[[1],:])
    Iy_value = interp2(Iy, coor_fix[[0],:], coor_fix[[1],:])
    I=np.vstack((Ix_value,Iy_value))
    A=I.dot(I.T)+np.eye(I.shape[0])*1e-9
   
    for _ in range(15):
        mesh_x_flat=mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
        mesh_y_flat=mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
        coor=np.vstack((mesh_x_flat,mesh_y_flat))
        I2_value = interp2(img2_gray, coor[[0],:], coor[[1],:])
        Ip=(I2_value-I1_value).reshape((-1,1))
        b=-I.dot(Ip)
        solution=inv(A).dot(b)
        if( abs(solution[0,0])>0.2 and abs(solution[1,0])>0.2):
          X += solution[0,0]
          Y += solution[1,0]
    return X, Y

if __name__ == "__main__":
    cap = cv2.VideoCapture("../video/iron2.mp4")
    # cap = cv2.VideoCapture("Easy.mp4")
    ret, frame1 = cap.read()  # get first frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2,cv2.COLOR_RGB2GRAY)

    points = getFeatures(frame1_gray)
    new_points = estimateAllTranslation(points, frame1, frame2)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()
    diff = np.subtract(frame1_gray.astype(int),frame2_gray.astype(int))
    ax.imshow(diff,cmap='gray')
    ax.scatter(new_points[:,1], new_points[:,0],color=(0,1,0))
    ax.scatter(points[:,1], points[:,0],color=(1,0,0))
    plt.show()
