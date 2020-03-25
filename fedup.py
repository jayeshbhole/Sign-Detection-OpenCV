# import the necessary packages
import cv2
import numpy as np

#Difference Variable
minDiff = 12000
minSquareArea = 5000
match = -1

#Frame width & Height
w=640
h=480

#Font Type
font = cv2.FONT_HERSHEY_SIMPLEX

#Reference Images Display name & Original Name
ReferenceImages = ["_turnAround.png", "_park.png", "_left.png", "_right.png", "_spinAround.png", "_charge.png"]
ReferenceTitles = ["_turnAround.png", "_park.png", "_left.png", "_right.png", "_spinAround.png", "_charge.png"]

def dist_dir(pts):
    rect = order_points(pts)
    markerSize = rect[1][0] - rect[0][0]
     
    if markerSize == 0:
        distance = 0
    else:
        distance = int( (10 * 1200 )/ markerSize)
    
    direction = (640) - ((rect[0][0] + rect[1][0] + rect[2][0] + rect[3][0]) // 4)
    return distance, direction

#define class for References Images
class Symbol:
    def __init__(self):
        self.img = 0
        self.name = 0

#define class instances (6 objects for 6 different images)
symbol= [Symbol() for i in range(6)]



def readRefImages():
    for count in range(6):
        image = cv2.imread(ReferenceImages[count], cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        symbol[count].img = cv2.resize(image,(w//2,h//2),interpolation = cv2.INTER_AREA)
        symbol[count].name = ReferenceTitles[count]
        #cv2.imshow(symbol[count].name,symbol[count].img);


def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = order_points(pts)
        (tl, tr, br, bl) = rect                 #top-left top-right bottom-right bottom-left

        maxWidth = w//2
        maxHeight = h//2

        dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype = "float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped


def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged


def resize_and_threshold_warped(image):
        #Resize the corrected image to proper size & convert it to grayscale
        #warped_new =  cv2.resize(image,(w/2, h/2))
        warped_new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Smoothing Out Image
        blur = cv2.GaussianBlur(warped_new_gray,(5,5),0)

        #Calculate the maximum pixel and minimum pixel value & compute threshold
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
        threshold = (min_val + max_val)/2

        #Threshold the image
        ret, warped_processed = cv2.threshold(warped_new_gray, threshold, 255, cv2.THRESH_BINARY)
        warped_processed = cv2.resize(warped_processed,(320, 240))
        
        #return the thresholded image
        return warped_processed




def main():

    # initialize the camera and grab a reference to the raw camera capture
    video = cv2.VideoCapture(0)         # 0 for the inbuilt webcam 1 for the first USB cam and so on....

    #Windows to display frames
    cv2.namedWindow("Main Frame", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Matching Operation", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Corrected Perspective", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Contours", cv2.WINDOW_AUTOSIZE)

    #Read all the reference images
    readRefImages()

    # capture frames from the camera
    while True:
           
            ret, frameread = video.read()
            OriginalFrame = frameread

            
            gray = cv2.cvtColor(OriginalFrame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray,(3,3),0)
            
            #Detecting Edges
            edges = auto_canny(gray)

            #Contour Detection & checking for squares based on the square area
            contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

                    if len(approx)==4:
                            area = cv2.contourArea(approx)

                            if area > minSquareArea:
                                    cv2.drawContours(OriginalFrame,[approx],0,(0,0,255),2)
                                    warped = four_point_transform(OriginalFrame, approx.reshape(4, 2))
                                    warped_eq = resize_and_threshold_warped(warped)
                                    
                                    pts = approx.reshape(4,2)
                                    #list = dist_dir(pts)
                                    di_st ,di_rn = dist_dir(pts)
                                    
                                    
                                    for i in range(6):
#                                       print(warped_eq.shape)
#                                       print(symbol[i].img.shape)
                                        diffImg = cv2.bitwise_xor(warped_eq, symbol[i].img)
                                        diff = cv2.countNonZero(diffImg);

                                        if diff < minDiff:
                                            match = i
                                            
#                                            print symbol[i].name, diff
#                                            print approx.reshape(4,2)[0]

                                            cv2.putText(OriginalFrame,symbol[i].name, tuple(approx.reshape(4,2)[0]), font, 1, (200,0,255), 2, cv2.LINE_AA)
                                            #print(di_rn,"==dirn :::: dist==",di_st)
                                            
                                            diff = minDiff
                                            
                                            break;
#                                    if di_st > 15:
#                                            cv2.putText(OriginalFrame,str(di_st),tuple(approx.reshape(4,2)[2]), font, 2,(255,255,0),2,cv2.LINE_AA)
                                    
                                    cv2.putText(OriginalFrame,str(di_rn),(100,100), font, 2,(0,0,255),2,cv2.LINE_AA)
                                            
                                    cv2.imshow("Corrected Perspective", warped_eq)
                                    cv2.imshow("Matching Operation", diffImg)
            cv2.imshow("Contours", edges)
                                    

            #Display Main Frame
            cv2.imshow("Main Frame", OriginalFrame)
           
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                    break

    video.release()
    cv2.destroyAllWindows()


#Run Main
if __name__ == "__main__" :
    main()
