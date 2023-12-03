import cv2
import numpy as np

points = []

def click_event(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x,y])

def point_reader(img):
    global points
    points = []
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.array(points)

def show(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_split(video_path, frames_per_second):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps > frames_per_second:
        mod = int(video_fps/frames_per_second)
    else:
        mod = 1
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            if frame_count%mod == 0:
                frames.append(frame)
            frame_count+=1
        else:
            break
    return frames

def find_homography(to_transform, transformed):
    A = np.array([
        [to_transform[0][0], to_transform[1][0], to_transform[2][0]],
        [to_transform[0][1], to_transform[1][1], to_transform[2][1]],
        [1, 1, 1]
    ])
    op = np.array([[to_transform[3][0]], [to_transform[3][1]], [1]])
    scaleA = np.matmul(np.linalg.inv(A), op)

    B = np.array([
        [transformed[0][0], transformed[1][0], transformed[2][0]],
        [transformed[0][1], transformed[1][1], transformed[2][1]],
        [1, 1, 1]
    ])
    op = np.array([[transformed[3][0]], [transformed[3][1]], [1]])
    scaleB = np.matmul(np.linalg.inv(B), op)

    for i in range(3):
        for j in range(3):
            A[i][j] = A[i][j]*scaleA[j]
            B[i][j] = B[i][j]*scaleB[j]
    H = np.matmul(B, np.linalg.inv(A))
    return H

def find_affine(to_transform, transformed):
    A = np.zeros((6,6))
    B = np.zeros((6,1))
    for i in range(3):
        A[i*2] = np.array([to_transform[i][0], to_transform[i][1], 1, 0, 0, 0])
        A[i*2+1] = np.array([0, 0, 0, to_transform[i][0], to_transform[i][1], 1])

        B[i*2] = transformed[i][0]
        B[i*2+1] = transformed[i][1]
    
    x = np.matmul(np.linalg.inv(A), B)
    M = np.zeros((2,3))
    for i in range(2):
        for j in range(3):
            M[i][j] = x[i*3+j]
    return M

def warp_perspective(img, M, out_shape):
    ret = np.zeros((out_shape[1], out_shape[0], 3))
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            new_x = round((M[0][0]*i + M[0][1]*j + M[0][2])/(M[2][0]*i + M[2][1]*j + M[2][2]))
            new_y = round((M[1][0]*i + M[1][1]*j + M[1][2])/(M[2][0]*i + M[2][1]*j + M[2][2]))
            ret[new_y][new_x] = img[j][i]
    return ret

def warp_affine(img, M, out_shape):
    ret = np.zeros((out_shape[1], out_shape[0], 3))
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            new_x = round(M[0][0]*i + M[0][1]*j + M[0][2])
            new_y = round(M[1][0]*i + M[1][1]*j + M[1][2])
            ret[new_y][new_x] = img[j][i]
    return ret

def self_warping(pts1, pts2, img, out_shape, dof=8):
    to_transform = np.float32(pts1)
    transformed = np.float32(pts2)
    if dof == 8:
        M = find_homography(to_transform, transformed)
        out = warp_perspective(img, M, out_shape)
    elif dof == 6:
        M = find_affine(to_transform[:3], transformed[:3])
        out = warp_affine(img, M, out_shape)
    return out

def video_projector(projected_video1, projected_video2, projection_img, alpha):

    frames1 = video_split(projected_video1, 10)
    frames2 = video_split(projected_video2, 10)
    img = cv2.resize(cv2.imread(projection_img), (800, 600))
    h,w,c = img.shape

    print("Mark 4 points on the image shown")
    pts_frame1 = point_reader(frames1[0])
    print("Monitor video points:\n", pts_frame1)

    print("Mark 4 corresponding points on the monitor screen")
    pts_monitor = point_reader(img)
    print("Monitor screen points:\n", pts_monitor)

    print("Mark 4 points on the image shown")
    pts_frame2 = point_reader(frames2[0])
    print("TV video points:\n", pts_frame2)

    print("Mark 4 corresponding points on the TV screen")
    pts_tv = point_reader(img)
    print("TV screen points:\n", pts_tv)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 10, (int(w),int(h)))
    for frame_id in range(len(frames1)):
        print(frame_id)
        wf1 = self_warping(pts_frame1, pts_monitor, frames1[frame_id], (w,h))
        wf2 = self_warping(pts_frame2, pts_tv, frames2[frame_id], (w,h))
        img_mask1 = wf1*img
        img_mask2 = wf2*img
        final_frame = (img_mask1*img_mask2==0)*img + np.uint8((wf1*alpha)) + np.uint8((img_mask1!=0)*img*(1-alpha)) + np.uint8((wf2*alpha)) + np.uint8((img_mask2!=0)*img*(1-alpha))
        out.write(final_frame)
    out.release()



video_projector('Monitor_clip.mp4', "TV_clip.mp4.mp4", 'Figure1.jpg', 0.9)
