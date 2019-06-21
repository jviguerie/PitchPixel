# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import datetime
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=False,
	help="video_in filename")
ap.add_argument("-o", "--output", required=True,
	help="video_out filename")
ap.add_argument("-f", "--fps", type=int, default=20,
	help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="codec of output video")
args = vars(ap.parse_args())

def _bounding_box_of(contour):
    rotbox = cv2.minAreaRect(contour)
    coords = cv2.boxPoints(rotbox)

    xrank = np.argsort(coords[:, 0])

    left = coords[xrank[:2], :]
    yrank = np.argsort(left[:, 1])
    left = left[yrank, :]

    right = coords[xrank[2:], :]
    yrank = np.argsort(right[:, 1])
    right = right[yrank, :]

    #            top-left,       top-right,       bottom-right,    bottom-left
    box_coords = tuple(left[0]), tuple(right[0]), tuple(right[1]), tuple(left[1])
    box_dims = rotbox[1]
    box_centroid = int((left[0][0] + right[1][0]) / 2.0), int((left[0][1] + right[1][1]) / 2.0)

    return box_coords, box_dims, box_centroid                                

# define the lower and upper boundaries of the "cyan"
# target center in the HSV color space
cyanLower = (80, 80, 80)
cyanUpper = (120, 255, 255)

# define the lower and upper boundaries of the "dayglo"
# target center in the HSV color space
daygloLower = (0, 80, 100)
daygloUpper = (80, 200, 255)

# define the lower and upper boundaries of the "blue"
# target center in the HSV color space
blueLower = (110, 120, 120)
blueUpper = (140, 255, 255)

# define the lower and upper boundaries of the "magenta"
# bordered target in the HSV color space
magLower = (80, 0, 0)
magUpper = (180, 180, 80)

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
 
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# initialize the FourCC, video writer, dimensions of the frame
fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None

FirstStrike = True

# keep looping
while True:
    # grab the current frame
    frame = vs.read()
 
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break
 
    ## report dimensions
    fh, fw = frame.shape[:2]
    #print("[INFO] frame1 wlen", fw )
    #print("[INFO] frame1 hlen", fh )

    # init writer
    if writer is None:
        writer = cv2.VideoWriter(args["output"], fourcc, args["fps"],(fw, fh), True)

    # convert it to the HSV color space and Grayscale
    blurred = cv2.medianBlur(frame, 5)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color CYAN, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    cyan_mask = cv2.inRange(hsv, cyanLower, cyanUpper)
    cyan_mask = cv2.erode(cyan_mask, None, iterations=3)
    cyan_mask = cv2.dilate(cyan_mask, None, iterations=2)

    # find contours in the cyan_mask and initialize the current
    # (x, y) center of the ball
    cyan_cnts = cv2.findContours(cyan_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cyan_cnts = imutils.grab_contours(cyan_cnts)
    cyan_center = None

    # only proceed if at least one contour was found
    if len(cyan_cnts) > 0:
        # loop through contours in the mask, then use
        # it to compute the minimum enclosing circle and centroid
        for cyan_c in cyan_cnts:
                cyan_rect = cv2.minAreaRect(cyan_c)
                if cyan_rect[2] < -10 and cyan_rect[2] > -80:
                    FirstStrike = True
#                    print("[REJ] cyan tilt CW - cyan_cArea", cyan_cArea
                    continue
                if cyan_rect[1][0] < 10 or cyan_rect[1][1] < 10:
                    FirstStrike = True
#                    print("[REJ] cyan tilt CW - cyan_cArea", cyan_cArea )
                    continue                
                ((cy_x, cy_y), cyan_radius) = cv2.minEnclosingCircle(cyan_c)
                cyan_M = cv2.moments(cyan_c)
                cyan_center = (int(cyan_M["m10"] / cyan_M["m00"]), int(cyan_M["m01"] / cyan_M["m00"]))

                # only proceed if the radius meets a minimum size, and center is in target ROI
                if (cyan_radius > fh/40) and ((fh / cyan_radius ) > 1.75):

                    # construct a mask for the color DAYGLO, then perform
                    # a series of dilations and erosions to remove any small
                    # blobs left in the mask
                    dayglo_mask = cv2.inRange(hsv, daygloLower, daygloUpper)
                    dayglo_mask = cv2.erode(dayglo_mask, None, iterations=3)
                    dayglo_mask = cv2.dilate(dayglo_mask, None, iterations=2)

                    # find contours in the dayglo_mask and initialize the current
                    # (x, y) center of the ball
                    dayglo_cnts = cv2.findContours(dayglo_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    dayglo_cnts = imutils.grab_contours(dayglo_cnts)
                    dayglo_center = None
                                     
                    # only proceed if at least one contour was found
                    if len(dayglo_cnts) > 0:
                        # loop through contours in the mask, then use
                        # it to compute the minimum enclosing circle and centroid
                        cyan_cArea = cv2.contourArea(cyan_c)
                        for dayglo_c in dayglo_cnts:
                            dayglo_cArea = cv2.contourArea(dayglo_c)
                            if dayglo_cArea < (cyan_cArea / 15):
                                FirstStrike = True
##                                print("[REJ] dayglo too small - cyan_cArea", cyan_cArea )
##                                print("[REJ] dayglo too small - dayglo_cArea", dayglo_cArea )
                                continue
#                            if cyan_cArea < (dayglo_cArea * 6):
##                                print("[REJ] cyan too small - cyan_cArea", cyan_cArea )
##                                print("[REJ] cyan too small - dayglo_cArea", dayglo_cArea )
#                                continue

                            dayglo_rect = cv2.minAreaRect(dayglo_c)
                                
                            if dayglo_rect[2] < -10 and dayglo_rect[2] > -80:
                                FirstStrike = True
##                                print("[REJ] dayglo tilt CW - cyan_cArea", cyan_cArea )
##                                print("[REJ] dayglo tilt CW - dayglo_cArea", dayglo_cArea )
                                continue
                            if dayglo_rect[1][0] < 10 or dayglo_rect[1][1] < 10:
                                FirstStrike = True
##                                print("[REJ] dayglo tilt CCW - cyan_cArea", cyan_cArea )
##                                print("[REJ] dayglo tilt CCW - dayglo_cArea", dayglo_cArea )
                                continue                
                            
                            ((dg_x, dg_y), dayglo_radius) = cv2.minEnclosingCircle(dayglo_c)
                            dayglo_M = cv2.moments(dayglo_c)
                            dayglo_center = (int(dayglo_M["m10"] / dayglo_M["m00"]), int(dayglo_M["m01"] / dayglo_M["m00"]))

                            # only proceed if the radius meets a minimum size, and center is in target ROI
                            if (dayglo_radius > (fh/160)) and (abs(dg_x - cy_x) < fh/40) and (abs(dg_y - cy_y) < fh/40):
##                                print("[INFO] cyan_cArea", cyan_cArea )
##                                print("[INFO] dayglo_cArea", dayglo_cArea )

                                #scale line width
                                line_width = int(36 * (cyan_radius/fh))
                                                                                              
                                # white BB center infill (-1)
##                                    bb_center = (int(dayglo_M["m10"] /dayglo_M["m00"])+36, int(dayglo_M["m01"]-1 / dayglo_M["m00"]))
##                                    cv2.circle(frame, bb_center, int(dayglo_radius/6), (255, 255, 255), -1, lineType=cv2.LINE_AA)
                                cv2.circle(frame, (int(dg_x), int(dg_y) + int(1.125 * dayglo_radius)), int(dayglo_radius/6), (255, 255, 255), -1, lineType=cv2.LINE_AA)

                                # red-infill for magenta TATER SPOT (-1)
                                cv2.circle(frame, (int(dg_x), int(dg_y)), int(dayglo_radius/3.6), (0, 0, 255), -1, lineType=cv2.LINE_AA)

                                # CYAN draw the circle and centroid on the frame
                                cy_box = cv2.boxPoints(cyan_rect)
                                cy_box = np.int0(cy_box)
                                xrank = np.argsort(cy_box[:, 0])

                                left = cy_box[xrank[:2], :]
                                yrank = np.argsort(left[:, 1])
                                left = left[yrank, :]

                                right = cy_box[xrank[2:], :]
                                yrank = np.argsort(right[:, 1])
                                right = right[yrank, :]

                                #            top-left,       top-right,       bottom-right,    bottom-left
                                cy_box_coords = tuple(left[0]), tuple(right[0]), tuple(right[1]), tuple(left[1])
                                cy_box_dims = cyan_rect[1]
                                cy_box_centroid = int((left[0][0] + right[1][0]) / 2.0), int((left[0][1] + right[1][1]) / 2.0)

#                                cv2.drawContours(frame,[cy_box],0,(255, 255, 0), line_width)
                                cy_pts = np.array(cy_box, np.int32)
                                cy_pts = cy_pts.reshape((-1,1,2))
                                cv2.polylines(frame,[cy_pts],True,(255,255,0), line_width)

                                # draw the dayglo bounding box on the frame
                                dayglo_rect = cv2.minAreaRect(dayglo_c)
                                dg_box = cv2.boxPoints(dayglo_rect)
                                dg_box = np.int0(dg_box)
                                xrank = np.argsort(dg_box[:, 0])

                                left = dg_box[xrank[:2], :]
                                yrank = np.argsort(left[:, 1])
                                left = left[yrank, :]

                                right = dg_box[xrank[2:], :]
                                yrank = np.argsort(right[:, 1])
                                right = right[yrank, :]

                                #            top-left,       top-right,       bottom-right,    bottom-left
                                dg_box_coords = tuple(left[0]), tuple(right[0]), tuple(right[1]), tuple(left[1])
                                dg_box_dims = dayglo_rect[1]
                                dg_box_centroid = int((left[0][0] + right[1][0]) / 2.0), int((left[0][1] + right[1][1]) / 2.0)

#                                cv2.drawContours(frame, [dg_box],0,(0, 255, 0), line_width)
                                dg_pts = np.array(dg_box, np.int32)
                                dg_pts = dg_pts.reshape((-1,1,2))
                                cv2.polylines(frame,[dg_pts],True,(0,255,0), line_width)

                                bl_box = cv2.boxPoints(cyan_rect)
                                bl_box = np.int0(bl_box)
                                
##                                pers_bl_box = np.zeroslike(bl_box)
##                                pers_bl_box = np.int0(pers_bl_box)

                                bl_box[0,0] = cy_box[0,0] + ((dg_box[0,0] - cy_box[0,0])/2.0)
                                bl_box[0,1] = cy_box[0,1] + (2.0*(dg_box[0,1] - cy_box[0,1])/3.0)

                                bl_box[1,0] = dg_box[1,0] + ((cy_box[1,0] - dg_box[1,0])/2.0)
                                bl_box[1,1] = cy_box[1,1] + (2.0*(dg_box[1,1] - cy_box[1,1])/3.0)

                                bl_box[2,0] = dg_box[2,0] + ((cy_box[2,0] - dg_box[2,0])/2.0)
                                bl_box[2,1] = dg_box[2,1] + (1.0*(cy_box[2,1] - dg_box[2,1])/3.0)

                                bl_box[3,0] = cy_box[3,0] + ((dg_box[3,0] - cy_box[3,0])/2.0)
                                bl_box[3,1] = dg_box[3,1] + (1.0*(cy_box[3,1] - dg_box[3,1])/3.0)

                                if (bl_box[0,0] - bl_box[1,0]) < (fh/50.0) and (bl_box[1,1] - bl_box[2,1]) < (fh/50.0): 
                                    #cv2.drawContours(frame,[bl_box],0,(255, 0, 0), line_width)
                                    pers_bl_box = bl_box
                                    bl_pts = np.array(bl_box, np.int32)
                                    bl_pts = bl_pts.reshape((-1,1,2))
                                    FirstStrike = False
                                    cv2.polylines(frame,[bl_pts],True,(255,0,0), line_width)
                                else:
                                    if not(FirstStrike):
                                        pers_bl_pts = np.array(pers_bl_box, np.int32)
                                        pers_bl_pts = pers_bl_pts.reshape((-1,1,2))
                                        cv2.polylines(frame,[pers_bl_pts],True,(255,0,0), line_width)
                                        #  and (cy_box[0,1] < pers_bl_box[0,1] < dg_box[0,1]) and (cy_box[0,0] < pers_bl_box[0,0] < dg_box[0,0])


    # show the frame to our screen
    writer.write(frame)
    #cv2.imshow("gry", gry)
    #cv2.imshow("dayglo_mask", dayglo_mask)
    #cv2.imshow("hsv", hsv)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
 
# otherwise, release the camera
else:
	vs.release()
 
# close all windows
cv2.destroyAllWindows()

