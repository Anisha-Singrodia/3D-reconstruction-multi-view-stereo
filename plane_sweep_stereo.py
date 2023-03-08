import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane using intrinsics and extrinsics
    
    Hint:
    depth * corners = K @ T @ y, where y is the output world coordinates and T is the 4x4 matrix of Rt (3x4)

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points, here 2x2 correspinds to 4 corners
    """

    points = np.array(
        (
            (0, 0, 1),
            (width, 0, 1),
            (0, height, 1),
            (width, height, 1),
        ),
        dtype=np.float32,
    ).reshape(2, 2, 3)

    """ YOUR CODE HERE
    """
    R = Rt[:, :-1]
    t = Rt[:, -1:]
    for i in range(2):
        for j in range(2):
            calib = (np.linalg.inv(K)@(depth*points[i][j])).reshape((3,1))
            a = R.T@(calib - t)
            points[i][j][0] = a[0]
            points[i][j][1] = a[1]
            points[i][j][2] = a[2]

    """ END YOUR CODE
    """
    return points


def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    
    Hint:
    Z * projections = K @ T @ p, where p is the input points and projections is the output, T is the 4x4 matrix of Rt (3x4)
    
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    R = Rt[:, :-1]
    T = Rt[:, -1:]
    points_2d = np.array(
        (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ),
        dtype=np.float32,
    ).reshape(2, 2, 2)
    for i in range(2):
        for j in range(2):
            point = points[i][j]
            new_point = K@((R@point).reshape((3,1)) + T)
            points_2d[i][j][0] = new_point[0]/new_point[2]
            points_2d[i][j][1] = new_point[1]/new_point[2]
    """ END YOUR CODE
    """
    return points_2d


def warp_neighbor_to_ref(
    backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor
):
    """
    Warp the neighbor view into the reference view
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective
    
    ! Note, when you use cv2.warpPerspective, you should use the shape (width, height), NOT (height, width)
    
    Hint: you should do the follows:
    1.) apply backproject_corners on ref view to get the virtual 3D corner points in the virtual plane
    2.) apply project_fn to project these virtual 3D corner points back to ref and neighbor views
    3.) use findHomography to get teh H between neighbor and ref
    4.) warp the neighbor view into the reference view

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]
    

    """ YOUR CODE HERE
    """
    backproject = backproject_fn(K_ref, width, height, depth, Rt_ref)
    proj_ref = project_fn(K_ref, Rt_ref, backproject).reshape((-1,2))
    proj_neighbor = project_fn(K_neighbor, Rt_neighbor, backproject).reshape((-1,2))
    H, mask = cv2.findHomography(proj_neighbor, proj_ref, cv2.RANSAC, 5.0)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, H, (width, height))





    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """
    Compute the cost map between src and dst patchified images via the ZNCC metric

    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value,
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """
    height = src.shape[0]
    width = src.shape[1]
    zncc = np.zeros((height,width))
    for h in range(height):
        for w in range(width):
                    err = 0
                    for i in range(3):
                        w1_ = (np.sum(src[h,w,:,i]))/(src.shape[2])
                        w2_ = (np.sum(dst[h,w,:,i]))/(src.shape[2])

                        sigma_w1 = np.sqrt((np.linalg.norm(src[h,w, :,i]-w1_)**2)/src.shape[2])
                        sigma_w2 = np.sqrt((np.linalg.norm(dst[h,w, :,i]-w2_)**2)/src.shape[2])
                        err += (np.sum((src[h,w, :,i]-w1_)@((dst[h,w, :,i]-w2_).T)))/(sigma_w1*sigma_w2 + EPS)
                    zncc[h][w] = err
    # print(zncc)
    return zncc        

    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """
    height = dep_map.shape[0]
    width = dep_map.shape[1]
    xyz_cam = np.zeros((height, width, 3))
    for h in range(height):
        for w in range(width):
            pixel = np.array([w,h,1]).reshape((3,1))
            point = dep_map[h][w]*(np.linalg.inv(K)@pixel)
            xyz_cam[h][w][0] = point[0]
            xyz_cam[h][w][1] = point[1]
            xyz_cam[h][w][2] = point[2]
    """ END YOUR CODE
    """
    return xyz_cam
