import cv2
import numpy as np


if __name__ == '__main__':
    img = cv2.imread('image.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp_img, des_img = orb.detectAndCompute(gray_img, None)

    vid = cv2.VideoCapture('vid.mp4')
    countFrame = 0
    # frameTime set to 100 as it's then close to real time video
    frameTime = 100
    vid.set(cv2.CAP_PROP_POS_MSEC, countFrame * frameTime)
    success, img_vid = vid.read()
    while success:
        gray_vid = cv2.cvtColor(img_vid, cv2.COLOR_BGR2GRAY)
        kp_vid, des_vid = orb.detectAndCompute(gray_vid, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_img, des_vid)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:20]

        src_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_vid[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = gray_img.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, m)

        result_img = cv2.polylines(img_vid, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        result_img = cv2.resize(result_img, (1500, 800))
        cv2.imshow('', result_img)
        cv2.waitKey(1)
        countFrame += 1
        vid.set(cv2.CAP_PROP_POS_MSEC, countFrame * frameTime)
        success, img_vid = vid.read()
