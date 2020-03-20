import cv2
import numpy as np
import time
import timeit
from timeit import default_timer as timer

def order_points(pts):
   # initialize a list of coordinates that will be ordered
   # such that the first entry in the list is the top-left,
   # the second entry is the top-right, the third is the
   # bottom-right, and the fourth is the bottom-left
   rect = np.zeros((4, 2), dtype="float32")

   # the top-left point will have the smallest sum, whereas
   # the bottom-right point will have the largest sum
   s = pts.sum(axis=1)
   rect[0] = pts[np.argmin(s)]
   rect[2] = pts[np.argmax(s)]

   # now, compute the difference between the points, the
   # top-right point will have the smallest difference,
   # whereas the bottom-left will have the largest difference
   diff = np.diff(pts, axis=1)
   rect[1] = pts[np.argmin(diff)]
   rect[3] = pts[np.argmax(diff)]

   # return the ordered coordinates
   return rect
   

def nothing(x):
  # any operation
    pass


fileNames = ["_turnAround.png", "_park.png", "_left.png", "_right.png", "_spinAround.png", "_charge.png"]
symbolImages = []



cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-E", "Trackbars", 0, 250, nothing)
cv2.createTrackbar("H-E", "Trackbars", 0, 250, nothing)


for i in range(len(fileNames)):
	temp = cv2.imread(fileNames[i])
	temp2 = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
	symbolImages.append(temp2)
print ("Number of images loaded:")
print (len(symbolImages))


while True:
#    print("++++ default stop ++++")
#    start = timer()
#    start = timeit.timeit()
    sign = 0
    le = cv2.getTrackbarPos("L-E", "Trackbars")
    he = cv2.getTrackbarPos("H-E", "Trackbars")
#    l_e = (2*le)+1
#    h_e = (2*he)+1

    _, frame = cap.read()

    imageSizeWidth = 640
    imageSizeHeight = 360

    image = frame
    blur = cv2.blur(image,(7,7))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    blur2= cv2.GaussianBlur(gray, (11,11), cv2.BORDER_DEFAULT)
    #edges = cv2.Canny(blur,10,80)
    edges2 = cv2.Canny(blur2,le,he)

    #cv2.imshow('blur2',blur2)
    cv2.imshow('edges2',edges2)
    
    contours, _ = cv2.findContours(edges2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    foundMatch = False

    for contour in contours:

       #cv2.drawContours(frame, contour, -1, (255, 255, 200), 3)
       approx = cv2.approxPolyDP(contour,0.05*cv2.arcLength(contour,True),True)
       if len(approx) == 4:
           area = cv2.contourArea(approx)
           if area > 4000:
               print("Found something, processing...")

               cv2.drawContours(frame, contour, -1, (0,255,255), 3)		# color is GBR standard, not RGB

               #cv2.imwrite('approx.jpg',image)


               #time.sleep(0.1)

               points = np.array([[approx[0][0][0], approx[0][0][1]], [approx[1][0][0], approx[1][0][1]], [approx[2][0][0], approx[2][0][1]], [approx[3][0][0], approx[3][0][1]]], dtype = "float32")


               #print ("Corner points in the image (after sorting):")
               points = order_points(points)

               #print(points)

               points2 = np.array([[0,0],[255,0],[255,255], [0,255]], dtype = "float32")

               direction = (imageSizeWidth / 2) - ((points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4)
               print("direction =====",direction)



               markerSize = points[1][0] - points[0][0]

               if markerSize == 0:
                   distance = 0
               else:
                   distance = (7 * 550 )/ markerSize

               M = cv2.getPerspectiveTransform(points,points2)

               warped = cv2.warpPerspective(image,M,(256,256))

               if (len(warped.shape) == 2):
                   break

               warpedGrayscale = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

               (thresh, warpedBinImage) = cv2.threshold(warpedGrayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

               sign = -1
               cv2.imshow('warpedBinImage',warpedBinImage)
               for idx, image in enumerate(symbolImages):
                   #tempp= cv2.rotate(warpedBinImage, cv2.ROTATE_90_CLOCKWISE)
                   #WBI= cv2.flip(tempp, 1)
                   #diffImg = image - WBI
                   diffImg = image - warpedBinImage
                   #exor = cv2.bitwise_xor(WBI, image)
                   difference = cv2.countNonZero(diffImg)
                   #xor = cv2.countNonZero(exor)
                   #cv2.imshow('diff',diffImg)
                   #print(difference)
                   #time.sleep(1)

                   if (difference < 14000):
                       sign = idx + 1
                       foundMatch = True
                       print("_______________found match__________________")
                       print("===============",sign)
                      # time.sleep(0.1)
               x = approx.ravel()[0]
               y = approx.ravel()[1]
               font = cv2.FONT_HERSHEY_SIMPLEX

               if (foundMatch):
                   cv2.drawContours(frame, [approx], 0, (0,110,200),5)
                   cv2.putText(frame, "ROI", (x,y), font, 1,(0,110,200))
                   if(direction > 40):
                        print("rotate left")
                   elif(direction < -40 ):
                        print("rotate right")
                   print("**********",distance,"**********")

                   if(distance<20):
                        print("stop")

                   else:
                        print("forward")


                   #cv2.imshow('final',approx)
                   break
                   

    #end = timeit.timeit()
    #end = timer()
    #timetaken = end - start
#    print("time ==  ",timetaken)
#    if(timetaken>0.065):
#
#        time.sleep(0.01)

    cv2.imshow('frame',frame)
    cv2.imshow('blur',blur)
    cv2.imshow('blur2',blur2)
   # print("$$$$ default rotate $$$$")
    time.sleep(0.2)
    if cv2.waitKey(20) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
