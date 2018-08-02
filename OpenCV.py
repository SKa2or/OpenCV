import cv2
import numpy as np
import math


cap = cv2.VideoCapture(0)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel=np.ones((3,3),np.uint8)
fgbg=None
captured = False
i=0
skin_ycrcb_mint = np.array([0,133,77],np.uint8)
skin_ycrcb_maxt = np.array([255,173,127],np.uint8)

rect=(250,150,300,450)
# x1=320
# x2=520
# y1=100
# y2=340
x1=190
x2=490
y1=180
y2=480
previous_defects = 0
previous_defects_counter = 0
defects_valid = False
defectsList=[]
averageDefects = 0
mode = 0
startCount = False
while(cap.isOpened()):
    # read image

    ret, img=cap.read()
    
    #bgdmodel=np.zeros((1,65),np.float64)
    #fgdmodel=np.zeros((1,65),np.float64)
    # maskfg=np.zeros(img.shape[:2],np.uint8)
    # cv2.grabCut(img,maskfg,rect,bgdmodel,fgdmodel,5,cv2.GC_INIT_WITH_RECT)
    # maskfg2=np.where((maskfg==2)|(maskfg==0),0,1).astype('uint8')
    # img2=img*maskfg2[:,:,np.newaxis]
    # cv2.imshow('fg',img2)


    img=cv2.bilateralFilter(img,5,50,100)
    #img = cv2.flip(img,0)
    #img=cv2.medianBlur(img,5)
    # get hand data from the rectangle sub window on the screen
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0),0)
    crop_img = img[y1:y2, x1:x2]

    #blurred2=cv2.GaussianBlur(img,(35,35),0)
    #cv2.imshow('1st blur',blurred2)
    if captured==True:
        print ("start2")
        
        fgmask=fgbg.apply(crop_img,learningRate=0)
        
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res=cv2.bitwise_and(crop_img, crop_img,mask=fgmask)
        cv2.imshow('res',res)
        cv2.imshow('fgmask', fgmask)
        # convert to grayscale
        grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        #grey=cv2.cvtColor(res, cv2.COLOR_BGR2YCR_CB)
        #grey = cv2.inRange(grey, skin_ycrcb_mint, skin_ycrcb_maxt)
        #grey=cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        
        # applying gaussian blur
        value = (5, 5)
        blurred = cv2.GaussianBlur(grey, value, 0)

        # thresholdin: Otsu's Binarization method
        _, thresh1 = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                                   #cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow('blur', blurred)
        # show thresholded image
        cv2.imshow('Thresholded', thresh1)

        # check OpenCV version to avoid unpacking error
        (version, _, _) = cv2.__version__.split('.')

        if version == '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        elif version == '2':
            contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
                   cv2.CHAIN_APPROX_NONE)

        # find contour with max area
        try:
            cnt = max(contours, key = lambda x: cv2.contourArea(x))
            hull=cv2.convexHull(cnt,returnPoints=False)
            defects=cv2.convexityDefects(cnt, hull)
            count_defects = 0
            defects_direction = 0
            yDirection = False
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]

                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                # apply cosine rule here
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c))*57 

                # ignore angles > 90 and highlight rest with red dots
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(crop_img, far, 5, [255,0,0], -1)
                    midpoint=(start[1]+end[1])/2
                    dirFinger=far[1]-midpoint
                    if dirFinger>0 :
                        defects_direction += 1
                #dist = cv2.pointPolygonTest(cnt,far,True)

                # draw a line from start to end i.e. the convex points (finger tips)
                # (can skip this part)
                cv2.line(crop_img,start, end, [0,255,0], 2)
                cv2.circle(crop_img, start, 5, [255,255,0], -1)
                cv2.circle(crop_img, end, 5, [0,255,255], -1) #yellow
                #cv2.circle(crop_img,far,5,[0,0,255],-1)
        except AttributeError:
            print("lol")
        except ValueError:
            print("aha")

        if count_defects-defects_direction == 0:
            yDirection=True
        else:
            yDirection=False
        # define actions required
        print(count_defects)
        print("defect and i")
        print(i)
        
        # if j == 19:
        #     startCount = True
        
        defectsList.append(count_defects)
        if len(defectsList)==20:
            averageDefects=sum(defectsList)/len(defectsList)
            defectsList=defectsList[1: ]
            mode = max(set(defectsList), key=defectsList.count)

        # if previous_defects == count_defects:
        #     previous_defects_counter += 1
        # else:
        #     previous_defects_counter = 0

        # if previous_defects_counter>=5:
        #     defects_valid = True
        # else:
        #     defects_valid = False

        previous_defects = count_defects
        print(yDirection)
        print(averageDefects)
        if (averageDefects <= 2)&(averageDefects>=1)&yDirection&(mode <= 2)&(mode>=1):
            cv2.putText(img,"Turning"+str(averageDefects), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0,0,255),thickness=5)
        elif (averageDefects >= 3.8)&(averageDefects<=5)&yDirection&(mode <= 5)&(mode>=4):
            cv2.putText(img,"Landing"+str(averageDefects), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,  color=(0,0,255),thickness=5)
    ##        cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    ##    elif count_defects == 3:
    ##        cv2.putText(img,"This is 4 :P", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    ##    elif count_defects == 4:
    ##        cv2.putText(img,"Hi!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img,"Nope", (50, 50),\
                        cv2.FONT_HERSHEY_SIMPLEX, 2,color=(0,0,255),thickness=5)

    cv2.imshow('Final', img)
    
    
    # if i==10:# and captured==False: 
    #     fgbg=cv2.createBackgroundSubtractorMOG2(varThreshold=60,detectShadows=False)
    #     #fgmask=fgbg.apply(img,learningRate=0)
    #     captured = True
    #     i=0
    #     print ("start")
    k = cv2.waitKey(50)
    if k == 27:
        break
    elif k == ord('b'):
    	fgbg=cv2.createBackgroundSubtractorMOG2(varThreshold=60,detectShadows=False)
    	captured=True
    	print("captured")
