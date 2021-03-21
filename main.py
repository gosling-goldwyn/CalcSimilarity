import cv2
import numpy as np
from matplotlib import pyplot as plt
from calc import calculateMatchedCoordinate, convertByAffin,convertImageSize, searchMatchingArea

def main(fp1,fp2,num):

    img_tgt = cv2.imread(fp1, cv2.COLOR_BGR2RGB)
    img_tgt_mono = cv2.imread(fp1, 0)
    img_org = cv2.imread(fp2, cv2.COLOR_BGR2RGB)
    img_org_mono = cv2.imread(fp2, 0)

    kp_query,kp_train,kp_num = calculateMatchedCoordinate(img_org_mono,img_tgt_mono)
    mag,_,_ = convertByAffin(kp_query,kp_train,kp_num)
    img_org_zoomed = convertImageSize(img_org_mono,mag)

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    res,top_left,bottom_right = searchMatchingArea(img_tgt_mono,img_org_zoomed,methods[num])
    cv2.rectangle(img_tgt, top_left, bottom_right,(255,255,255),20)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(cv2.cvtColor(img_tgt, cv2.COLOR_BGR2RGB))
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(methods[num])
    plt.show()

if __name__ == "__main__":
    fp1 = 'filepath'
    fp2 = 'filepath'
    num = 0
    main(fp1,fp2,num)