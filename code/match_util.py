import numpy as np
import cv2
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import scipy.stats

def pixel_consistency(prime_frame, second_frame, pts_prime, pts_second, window_size=3):
    # prime_threshold = threshold_yen(prime_frame)
    # second_threshold = threshold_yen(second_frame)
    # I_prime = rescale_intensity(prime_frame, (0, prime_threshold), (0, 255))
    # I_second = rescale_intensity(second_frame, (0, second_threshold), (0, 255))
    height, width, channel = second_frame.shape
    kernel = np.ones((window_size, window_size),np.uint8)
    
    pixel_consistency_loss = np.zeros((pts_prime.shape[0]), dtype=float)
    # distribution  
    rv = scipy.stats.norm(loc=0, scale= 36)
    pixel_consistency_prob = []

    for idx in range(3):
        I_second_max = cv2.dilate(second_frame[:, :, idx], kernel, iterations=1).astype(int)
        I_second_min = cv2.erode(second_frame[:, :, idx], kernel, iterations=1).astype(int)
        I_prime = prime_frame[:, :, idx].astype(int)

        for i, (pt_prime, pt_second) in enumerate(zip(pts_prime, pts_second)):
            b,a = pt_prime.ravel()
            d,c = pt_second.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)

            if(d>=width): d = width-1
            elif(d<0): d = 0
            if(c>=height): c = height-1
            elif(c<0): c = 0

            pixel_loss = max(0, max(I_prime[a, b]-I_second_max[c, d], I_second_min[c, d]-I_prime[a, b]))
            pixel_consistency_loss[i] = pixel_loss
    
    for loss in pixel_consistency_loss:
        prob = rv.pdf(loss)
        pixel_consistency_prob.append(prob)

    return np.array(pixel_consistency_prob)

def motion_consistency(pts_prime, pts_second, H): 
    """ 
        pts_prime: feature points of prime frame
        pts_second: feature points of second frame
        H: homography matrix, calculated from second->prime
    """
    motion_consistency_prob = []
    motion_pts_second = []

    # distribution
    rv = scipy.stats.norm(loc=0, scale=100)
    
    for i, (pt_prime, pt_second) in enumerate(zip(pts_prime, pts_second)):
        b,a = pt_prime.ravel()
        d,c = pt_second.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)

        new_d, new_c, _ =np.dot(H,np.array([d, c, 1]))
        motion_pts_second.append([new_d, new_c])
        motion_loss = ( (new_d-b)**2 + (new_c-a)**2 )**(1/2)
        motion_prob = rv.pdf(motion_loss)
        motion_consistency_prob.append(motion_prob)

    return np.array(motion_consistency_prob), np.array(motion_pts_second)

def calculate_whole_prob(frame, pts_second, pixel_prob, motion_prob):
    height, width, channel = frame.shape
    prob_frame = np.zeros((height, width), dtype=float)

    pixel_motion_prob = (motion_prob+pixel_prob).reshape(-1,1)
    distance_mean_width = nearest_feat_dist(pts_second[:,0])
    distance_mean_height = nearest_feat_dist(pts_second[:,1])

    width_points, width_prob = local_weighted_regression(width, pts_second[:,0], pixel_motion_prob, distance_mean_width)
    height_points, height_prob = local_weighted_regression(height, pts_second[:,1], pixel_motion_prob, distance_mean_height)

    # fig,a =  plt.subplots(2,2)
    # a[0][0].plot(pts_second[:,0], pixel_motion_prob, 'b.')
    # a[0][0].plot(width_points, width_prob, 'r.') # Predictions in red color.
    # a[0][0].set_title('width')

    # a[0][1].plot(pts_second[:,1], pixel_motion_prob, 'b.')
    # a[0][1].plot(height_points, height_prob, 'r.') # Predictions in red color.
    # a[0][1].set_title('height')
    # plt.show()

    for i in range(width):
        for j in range(height):
            prob_frame[j, i] = width_prob[i] + height_prob[j]

    return prob_frame

def local_weighted_regression(size, pts_axis_second, pixel_motion_prob, distance_mean):
    m = pts_axis_second.shape[0]
    pts_axis_second = pts_axis_second.reshape(-1,1)

    sample_points = np.linspace(0, size-1, size)
    sample_pred = []

    for sample in sample_points:
        theta, pred = predict(pts_axis_second, pixel_motion_prob, sample, distance_mean)
        sample_pred.append(pred)

    return sample_points, np.array(sample_pred).reshape(-1,1)

def nearest_feat_dist(pts_second):
    distance_mean = []
    
    for pt in pts_second:
        distance = np.mean(np.sort(pts_second - pt)[1:11])
        distance_mean.append(distance)

    return np.array(distance_mean).reshape(-1,1)

def wm(point, X, distance_mean):
    """
        distance_mean --> bandwidth
        X --> Training data.
        point --> the x where we want to make the prediction.
    """
    m = X.shape[0] 
    # Initialising W as an identity matrix.
    w = np.mat(np.eye(m)) 
    
    # Calculating weights for all training examples [x(i)'s].
    for i in range(m): 
        xi = X[i] 
        d = (-2 * distance_mean[i] * distance_mean[i]) - 1e-5
        w[i, i] = np.exp(np.dot((xi-point), (xi-point).T)/d) 
        
    return w

def predict(X, y, point, tau):
    # m = number of training examples. 
    m = X.shape[0]
    X_ = np.append(X, np.ones(m).reshape(m,1), axis=1) 
    
    # point is the x where we want to make the prediction. 
    point_ = np.array([point, 1])
    # Calculating the weight matrix using the wm function we wrote      #  # earlier. 
    w = wm(point_, X_, tau)
    # Calculating parameter theta using the formula.
    # print(X_.shape)
    # print(w.shape)
    # print(y.shape)

    theta = np.linalg.pinv(X_.T*(w * X_))*(X_.T*(w * y)) 
    # Calculating predictions.  
    pred = np.dot(point_, theta) 
    
    # Returning the theta and predictions 
    return theta, pred

def plot_predictions(X, y, tau, nval):
    """
        X --> Training data. 
        y --> Output sequence.
        nval --> number of values/points for which we are going to
        predict.
        tau --> the bandwidth.
        The values for which we are going to predict.
        X_test includes nval evenly spaced values in the domain of X.
    """
    X_test = np.linspace(-3, 3, nval)
    # Empty list for storing predictions. 
    preds = [] 
    
    # Predicting for all nval values and storing them in preds. 
    for point in X_test:
        theta, pred = predict(X, y, point, tau)
        preds.append(pred)

    # Reshaping X_test and preds
    X_test = np.array(X_test).reshape(nval,1)
    preds = np.array(preds).reshape(nval,1)

    plt.plot(X, y, 'b.')
    plt.plot(X_test, preds, 'r.') # Predictions in red color.

    plt.show()


if __name__ == '__main__':
    np.random.seed(8)
    sample_size = 500
    X = np.random.randn(sample_size,1)
    y = 2*np.sum(X**3, axis=1).reshape(-1,1) + 4.6*np.random.randn(sample_size,1)

    distance_mean = nearest_feat_dist(X)
    plot_predictions(X, y, distance_mean, sample_size)