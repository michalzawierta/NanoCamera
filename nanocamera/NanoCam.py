# Import the needed libraries
import time
from threading import Thread
import os
import subprocess

import cv2

# (MZ) added functions: HW sensor definition, crop (on full size image), and rescale to desired resolution

class Camera:
    def __init__(self, camera_type=0, device_id=0, source="localhost:8080", flip=0, width=640, height=480, fps=30,
                 enforce_fps=False, debug=False, s_width=0, s_height=0, crop=1, shift_x=-1, shift_y=-1, wbmode=1,
                 exp_manual=False, exp_time=0, exp_gain=0, exp_digitalgain=0, c_left=0, c_right=0, c_top=0, c_bottom=0,
                 rec=0, s_fpsa=1, s_fpsb=1, s_path="/mnt/ramdisk/image%%07d.jpg"):
                 #rec frame rate, rec location
        # initialize all variables
        self.fps = fps
        self.camera_type = camera_type
        self.camera_id = device_id
        # for streaming camera only
        self.camera_location = source
        self.flip_method = flip
        self.width = width
        self.height = height
        self.enforce_fps = enforce_fps
        # (MZ) new settings 
        self.s_width = width if s_width == 0 else s_width #sensor width
        self.s_height = height if s_height == 0 else s_height #sensor height
        # (MZ) shift define how image is shifted during cropping, -1 means it start from point zero
        self.shift_x = shift_x
        self.shift_y = shift_y
        # (MZ) alternatively use cropping coordinates
        # (MZ) based on the final resolution
        self.c_left = c_left
        self.c_right = c_right
        self.c_top = c_top
        self.c_bottom = c_bottom
        # (MZ) white balance: (0): off, (1): auto, (2): incandescent, (3): fluorescent, (4): warm-fluorescent, 
        # (MZ) (5): daylight, (6): cloudy-daylight, (7): twilight, (8): shade, (9): manual
        self.wbmode = wbmode
        # (MZ) cropping image
        self.crop = crop 
        # (MZ) if crop is defined, use crop parameters
        if self.crop<1 and self.crop>0:
            self.c_width = self.s_width * crop
            self.c_height = self.s_height * crop
            if shift_x == -1:
                self.c_left = self.c_width * 1/2
                self.c_right = self.c_width * 3/2
            else:
                self.c_left = self.shift_x
                self.c_right = self.c_width + self.shift_x
            if shift_y == -1:
                self.c_top = self.c_height * 1/2
                self.c_bottom = self.c_height * 3/2
            else:
                self.c_top = self.shift_y
                self.c_bottom = self.c_height + self.shift_y
            self.c_string = 'left=%d right=%d top=%d bottom=%d' % (self.c_left, self.c_right, self.c_top, self.c_bottom)
        # (MZ) if crop is not defined, try to use cropping coordinates
        # (MZ) cropping is done on the raw image before scaling
        elif (self.c_left+self.c_right+self.c_top+self.c_bottom)>0:
            self.c_string = 'left=%d right=%d top=%d bottom=%d' % (self.c_left, self.c_right, self.c_top, self.c_bottom)
        else:
            self.c_string = ''
        # (MZ) manual exposure setting and string generation
        self.exp_manual = exp_manual
        self.exp_time = exp_time
        self.exp_gain = exp_gain
        self.exp_digitalgain = exp_digitalgain
        if self.exp_manual == True:
            self.exp_string = 'aelock=true exposuretimerange="%d %d" gainrange="%d %d" ispdigitalgainrange="%d %d"' % (self.exp_time, 
                    self.exp_time, self.exp_gain, self.exp_gain, self.exp_digitalgain, self.exp_digitalgain)
        else:
            self.exp_string = ''

        # (MZ) rec = 1 to enable recording images, s_fpsa/s_fpsb define how often images are saved and s_path define where to save images
        # for example s_path = "/tmp/photos/name%%05d", default path is not defined at the moment
        self.rec = rec
        self.s_fpsa = s_fpsa
        self.s_fpsb = s_fpsb
        self.s_path = s_path

        self.debug_mode = debug
        # track error value
        '''
        -1 = Unknown error
        0 = No error
        1 = Error: Could not initialize camera.
        2 = Thread Error: Could not read image from camera
        3 = Error: Could not read image from camera
        4 = Error: Could not release camera
        '''
        # Need to keep an history of the error values
        self.__error_value = [0]

        # created a thread for enforcing FPS camera read and write
        self.cam_thread = None
        # holds the frame data
        self.frame = None

        # tracks if a CAM opened was succesful or not
        self.__cam_opened = False

        # create the OpenCV camera inteface
        self.cap = None

        # open the camera interface
        self.open()
        # enable a threaded read if enforce_fps is active
        if self.enforce_fps:
            self.start()

    def __csi_pipeline(self, sensor_id=0):
        return ('nvarguscamerasrc sensor-id=%d wbmode=%d %s ! '
                'video/x-raw(memory:NVMM), '
                'width=(int)%d, height=(int)%d, '
                'format=(string)NV12, framerate=(fraction)%d/1 ! '
                'nvvidconv flip-method=%d %s ! '
                'video/x-raw, width=(int)%d, height=(int)%d, pixel-aspect-ratio=1/1, format=(string)BGRx ! '
                'videoconvert ! '
                'video/x-raw, format=(string)BGR ! appsink' % (sensor_id, self.wbmode, self.exp_string,
                                                               self.s_width, self.s_height, self.fps, self.flip_method,
                                                               self.c_string, self.width, self.height))

    def __csi_pipeline_rec(self, sensor_id=0):
        return ('nvarguscamerasrc sensor-id=%d wbmode=%d %s ! '
                'video/x-raw(memory:NVMM), '
                'width=(int)%d, height=(int)%d, '
                'format=(string)NV12, framerate=(fraction)%d/1 ! '
                'nvvidconv flip-method=%d %s ! video/x-raw(memory:NVMM) ! tee name=t ! queue ! nvvidconv !'
                'video/x-raw ! '
                'videorate drop-only=true ! video/x-raw,framerate=%d/%d ! nvjpegenc ! identity drop-allocation=true ! multifilesink location=%s t. ! '
                'queue ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! '
                'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGR ! identity drop-allocation=true ! appsink' % (sensor_id, self.wbmode, self.exp_string,
                                                               self.s_width, self.s_height, self.fps, self.flip_method, self.c_string,
                                                               self.s_fpsa, self.s_fpsb, self.s_path, self.width, self.height))

    def __usb_pipeline(self, device_name="/dev/video1"):
        return ('v4l2src device=%s ! '
                'video/x-raw, '
                'width=(int)%d, height=(int)%d, '
                'format=(string)YUY2, framerate=(fraction)%d/1 ! '
                'videoconvert ! '
                'video/x-raw, format=BGR ! '
                'appsink' % (device_name, self.width, self.height, self.fps))

    def __rtsp_pipeline_bak(self, location="localhost:8080"):
        return ('rtspsrc location=%s latency=0 ! '
                'rtph264depay ! h264parse ! omxh264dec ! '
                'videorate ! videoscale ! '
                'video/x-raw, '
                'width=(int)%d, height=(int)%d, '
                'format=(string)YUY2, framerate=(fraction)%d/1 ! '
                'videoconvert ! '
                'video/x-raw, format=BGR ! '
                'appsink' % ("rtsp://" + location, self.width, self.height, self.fps))

    def __rtsp_pipeline(self, location="localhost:8080"):
        return ('rtspsrc location=%s ! '
                'rtph264depay ! h264parse ! omxh264dec ! '
                'videorate ! videoscale ! '
                'video/x-raw, '
                'width=(int)%d, height=(int)%d, '
                'framerate=(fraction)%d/1 ! '
                'videoconvert ! '
                'video/x-raw, format=BGR ! '
                'appsink' % ("rtsp://" + location, self.width, self.height, self.fps))

    def __mjpeg_pipeline(self, location="localhost:8080"):
        return ('souphttpsrc location=%s do-timestamp=true is_live=true ! '
                'multipartdemux ! jpegdec ! '
                'videorate ! videoscale ! '
                'video/x-raw, '
                'width=(int)%d, height=(int)%d, '
                'framerate=(fraction)%d/1 ! '
                'videoconvert ! '
                'video/x-raw, format=BGR ! '
                'appsink' % ("http://" + location, self.width, self.height, self.fps))

    def __usb_pipeline_enforce_fps(self, device_name="/dev/video1"):
        return ('v4l2src device=%s ! '
                'video/x-raw, '
                'width=(int)%d, height=(int)%d, '
                'format=(string)YUY2, framerate=(fraction)%d/1 ! '
                'videorate ! '
                'video/x-raw, framerate=(fraction)%d/1 ! '
                'videoconvert ! '
                'video/x-raw, format=BGR ! '
                'appsink' % (device_name, self.width, self.height, self.fps, self.fps))

    def open(self):
        # open the camera inteface
        # determine what type of camera to open
        if self.camera_type == 0:
            # then CSI camera 
            self.__open_csi()
        elif self.camera_type == 2:
            # rtsp camera
            self.__open_rtsp()
        elif self.camera_type == 3:
            # http camera
            self.__open_mjpeg()
        else:
            # it is USB camera
            self.__open_usb()
        return self

    def start(self):
        self.cam_thread = Thread(target=self.__thread_read)
        self.cam_thread.daemon = True
        self.cam_thread.start()
        return self

    # Tracks if camera is ready or not(maybe something went wrong)
    def isReady(self):
        return self.__cam_opened

    # Tracks the camera error state.
    def hasError(self):
        # check the current state of the error history
        latest_error = self.__error_value[-1]
        if latest_error == 0:
            # means no error has occured yet.
            return self.__error_value, False
        else:
            return self.__error_value, True

    # (MZ) added option rec, which will select 
    def __open_csi(self):
        # opens an inteface to the CSI camera
        try:
            # initialize the first CSI camera
            if self.rec == 1:
                self.cap = cv2.VideoCapture(self.__csi_pipeline_rec(self.camera_id), cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(self.__csi_pipeline(self.camera_id), cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                # raise an error here
                # update the error value parameter
                self.__error_value.append(1)
                raise RuntimeError()
            self.__cam_opened = True
        except RuntimeError:
            self.__cam_opened = False
            if self.debug_mode:
                raise RuntimeError('Error: Could not initialize CSI camera.')
        except Exception:
            # some unknown error occurred
            self.__error_value.append(-1)
            self.__cam_opened = False
            if self.debug_mode:
                raise RuntimeError("Unknown Error has occurred")

    def __open_usb(self):
        # opens an interface to the USB camera
        try:
            # initialize the USB camera
            self.camera_name = "/dev/video" + str(self.camera_id)
            # check if enforcement is enabled
            if self.enforce_fps:
                self.cap = cv2.VideoCapture(self.__usb_pipeline_enforce_fps(self.camera_name), cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(self.__usb_pipeline(self.camera_name), cv2.CAP_GSTREAMER)
                if not self.cap.isOpened():
                    # raise an error here
                    # update the error value parameter
                    self.__error_value.append(1)
                    raise RuntimeError()
            self.__cam_opened = True
        except RuntimeError:
            self.__cam_opened = False
            if self.debug_mode:
                raise RuntimeError('Error: Could not initialize USB camera.')
        except Exception:
            # some unknown error occurred
            self.__error_value.append(-1)
            self.__cam_opened = False
            if self.debug_mode:
                raise RuntimeError("Unknown Error has occurred")

    def __open_rtsp(self):
        # opens an interface to the RTSP location
        try:
            # starts the rtsp client
            self.cap = cv2.VideoCapture(self.__rtsp_pipeline(self.camera_location), cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                # raise an error here
                # update the error value parameter
                self.__error_value.append(1)
                raise RuntimeError()
            self.__cam_opened = True
        except RuntimeError:
            self.__cam_opened = False
            if self.debug_mode:
                raise RuntimeError('Error: Could not initialize RTSP camera.')
        except Exception:
            # some unknown error occurred
            self.__error_value.append(-1)
            self.__cam_opened = False
            if self.debug_mode:
                raise RuntimeError("Unknown Error has occurred")

    def __open_mjpeg(self):
        # opens an interface to the MJPEG location
        try:
            # starts the MJEP client
            self.cap = cv2.VideoCapture(self.__mjpeg_pipeline(self.camera_location), cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                # raise an error here
                # update the error value parameter
                self.__error_value.append(1)
                raise RuntimeError()
            self.__cam_opened = True
        except RuntimeError:
            self.__cam_opened = False
            if self.debug_mode:
                raise RuntimeError('Error: Could not initialize MJPEG camera.')
        except Exception:
            # some unknown error occurred
            self.__error_value.append(-1)
            self.__cam_opened = False
            if self.debug_mode:
                raise RuntimeError("Unknown Error has occurred")

    def __thread_read(self):
        # uses thread to read
        time.sleep(1.5)
        while self.__cam_opened:
            try:
                self.frame = self.__read()

            except Exception:
                # update the error value parameter
                self.__error_value.append(2)
                self.__cam_opened = False
                if self.debug_mode:
                    raise RuntimeError('Thread Error: Could not read image from camera')
                break
        # reset the thread object:
        self.cam_thread = None

    def __read(self):
        # reading images
        ret, image = self.cap.read()
        if ret:
            return image
        else:
            # update the error value parameter
            self.__error_value.append(3)

    def read(self):
        # read the camera stream
        try:
            # check if debugging is activated
            if self.debug_mode:
                # check the error value
                if self.__error_value[-1] != 0:
                    raise RuntimeError("An error as occurred. Error Value:", self.__error_value)
            if self.enforce_fps:
                # if threaded read is enabled, it is possible the thread hasn't run yet
                if self.frame is not None:
                    return self.frame
                else:
                    # we need to wait for the thread to be ready.
                    return self.__read()
            else:
                return self.__read()
        except Exception as ee:
            if self.debug_mode:
                raise RuntimeError(ee.args)

    def release(self):
        # destroy the opencv camera object
        try:
            # update the cam opened variable
            self.__cam_opened = False
            # ensure the camera thread stops running
            if self.enforce_fps:
                if self.cam_thread is not None:
                    self.cam_thread.join()
            if self.cap is not None:
                self.cap.release()
            # update the cam opened variable
            self.__cam_opened = False
        except RuntimeError:
            # update the error value parameter
            self.__error_value.append(4)
            if self.debug_mode:
                raise RuntimeError('Error: Could not release camera')

class Filemover:
    def __init__(self,source="/mnt/ramdisk",destination="."):
        self.source = source
        self.destination = destination
        self.command = 'find %s -maxdepth 1 -mmin +0.1 -type f -name "*.jpg" -exec mv --backup=numbered "{}" %s \\;' % (self.source, self.destination)
        self.proc = subprocess.Popen(['watch', self.command], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def __del__(self):
        self.proc.terminate()
