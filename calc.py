import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def calculateMatchedCoordinate(img_org_mono,img_tgt_mono):
    w, h = img_org_mono.shape[::-1]

    # Generate A-KAZE Detector
    akaze = cv2.AKAZE_create()

    # 特徴量の検出と特徴量ベクトルの計算
    kp1, des1 = akaze.detectAndCompute(img_tgt_mono, None)
    kp2, des2 = akaze.detectAndCompute(img_org_mono, None)

    # Brute-Force Matcher生成
    bf = cv2.BFMatcher()

    # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
    matches = bf.knnMatch(des1, des2, k=2)

    # データを間引きする
    ratio = 0.5
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    n = len(good)
    if n >= 3:
        kp_query = []
        kp_train = []
        for i in range(n):
            gq = good[i][0].queryIdx
            gt = good[i][0].trainIdx
            kp_query.append((kp1[gq].pt[0],kp1[gq].pt[1]))
            kp_train.append((kp2[gt].pt[0],kp2[gt].pt[1]))
            # print(kp_query[i],'=',kp_train[i]) #対応点群の表示
    else:
        raise RuntimeError('特徴点が少ないよ')
    return kp_query,kp_train,n

def convertByAffin(kp_query,kp_train,kp_num):
    left = np.zeros((2*kp_num,3),dtype=np.float)
    right = np.zeros((2*kp_num,1),dtype=np.float)
    for i in range(kp_num):
        left[2*i][0] = kp_train[i][0]
        left[2*i][1] = 1
        left[2*i+1][0] = kp_train[i][1]
        left[2*i+1][2] = 1
        right[2*i][0] = kp_query[i][0]
        right[2*i+1][0] = kp_query[i][1]
    # print(left)
    # print(right)
    w_t,s_t,t_t = np.dot(np.linalg.pinv(left),right)
    mag = w_t[0]
    s = s_t[0]
    t = t_t[0]
    # print(mag,s,t) #拡大率、平行移動の表示
    return mag,s,t

def convertImageSize(img_org,mag_rate):
    imged = Image.fromarray(img_org)
    width,height  = img_org.shape
    zoomed = np.asarray(imged.resize((int(height*mag_rate),int(width*mag_rate))))
    return zoomed

def searchMatchingArea(img_tgt_mono,img_org_zoomed,methods):
    w, h = img_org_zoomed.shape[::-1]
    method = eval(methods)

    # Apply template Matching
    res = cv2.matchTemplate(img_tgt_mono,img_org_zoomed,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return res,top_left,bottom_right