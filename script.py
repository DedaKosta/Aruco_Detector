import numpy as np
import cv2 as cv
import cv2.aruco as aruco
import pathlib
import glob
import os
import sys
sys.path.append('/usr/local/python/3.5')


def calibrate_aruco(marker_length, marker_separation):

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    arucoParams = aruco.DetectorParameters_create()
    board = aruco.GridBoard_create(5, 7, marker_length, marker_separation, aruco_dict)

    counter, corners_list, id_list = [], [], []
    first = 0

    images = glob.glob('calib_image_*.jpg')

    for fname in images:
        image = cv.imread(fname)
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            img_gray,
            aruco_dict,
            parameters=arucoParams
        )

        if first == 0:
            corners_list = corners
            id_list = ids
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list, ids))
        first = first + 1
        counter.append(len(ids))

    counter = np.array(counter)

    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
        corners_list,
        id_list,
        counter,
        board,
        img_gray.shape,
        None,
        None
    )
    return [ret, mtx, dist, rvecs, tvecs]


MARKER_LENGTH = 3
MARKER_SEPARATION = 0.25

ret, cameraMatrix, distCoeffs, rvecs, tvecs = calibrate_aruco(
    MARKER_LENGTH,
    MARKER_SEPARATION
)

images = glob.glob('calib_image_*.jpg')

for fname in images:
    img = cv.imread(fname)
    h, w = img.shape[:2]
    newCameraMatrix, validPixROI = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))

    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newCameraMatrix, (w, h), cv.CV_32FC1)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    x, y, w, h = validPixROI
    dst = dst[y:y+h, x:x+w]

    cv.imwrite("Calibration/" + fname, dst)
    cv.waitKey(1)

image_size = (1920, 1080)
map1, map2 = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, cameraMatrix, image_size, cv.CV_16SC2)

aruco_dict = aruco.Dictionary_get( aruco.DICT_6X6_1000 )
markerLength = 40
markerSeparation = 8
board = aruco.GridBoard_create(5, 7, markerLength, markerSeparation, aruco_dict)

arucoParams = aruco.DetectorParameters_create()

videoFile = "Aruco_board.mp4"
cap = cv.VideoCapture(videoFile)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('redrawn_video.mp4', fourcc, 30.0, (1920, 1080))

while(True):
    ret, frame = cap.read()
    if ret == True:

        frame_remapped = cv.remap(frame, map1, map2, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
        frame_remapped_gray = cv.cvtColor(frame_remapped, cv.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_remapped_gray, aruco_dict, parameters=arucoParams)
        aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejectedImgPoints)

        if ids is not None:
            im_with_aruco_board = aruco.drawDetectedMarkers(frame_remapped, corners, ids, (0,255,0))
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, None, None)
            if retval != 0:
                im_with_aruco_board = aruco.drawAxis(im_with_aruco_board, cameraMatrix, distCoeffs, rvec, tvec, 100)
        else:
            im_with_aruco_board = frame_remapped

        cv.imshow("arucoboard", im_with_aruco_board)
        cv.imwrite('output.jpg', im_with_aruco_board)
        out.write(im_with_aruco_board)

        if cv.waitKey(2) & 0xFF == ord('q'):
            break
    else:
        break

out.release()
cv.waitKey(0)
cv.destroyAllWindows()