import numpy as np
import io
import struct
import serial
from PIL import Image as PILImage
import cv2
import os
import glob

class Camera:

    def __init__(self, device='/dev/ttyACM0'):
        """Reads images from OpenMV Cam
        Args:
            device (str): Serial device
        Raises:
            serial.SerialException
        """
        self.port = serial.Serial(device, baudrate=115200,
                                  bytesize=serial.EIGHTBITS,
                                  parity=serial.PARITY_NONE,
                                  xonxoff=False, rtscts=False,
                                  stopbits=serial.STOPBITS_ONE,
                                  timeout=None, dsrdtr=True)

        # Important: reset buffers for reliabile restarts of OpenMV Cam
        self.port.reset_input_buffer()
        self.port.reset_output_buffer()

    def read_image(self):
        """Read image from OpenMV Cam
        Returns:
            image (ndarray): Image
        Raises:
            serial.SerialException
        """

        # Sending 'snap' command causes camera to take snapshot
        self.port.write('snap'.encode())
        self.port.flush()

        # Read 'size' bytes from serial port
        size = struct.unpack('<L', self.port.read(4))[0]
        image_data = self.port.read(size)
        
        image = np.array(PILImage.open(io.BytesIO(image_data)))

        return image

# Define the dimensions of checkerboard
CHECKERBOARD = (6, 9)
# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS + 
            cv2.TERM_CRITERIA_MAX_ITER, 22.33, 0.001)
# Vector for 3D points
threedpoints = []
# Vector for 2D points
twodpoints = []
#  3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0] 
                      * CHECKERBOARD[1], 
                      3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                               0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
  
# Extracting path of individual image stored
# in a given directory. Since no path is
# specified, it will take current directory
# jpg files alone
images = glob.glob('*.jpg')

currentFrame = 0
save_count = 0
while(True):
    # Create a camera by just giving the ttyACM depending on your connection value
    # Change the following line depending on your connection
    cap = Camera(device='/dev/ttyACM0')
    # Capture frame-by-frame
    im1 = cap.read_image()

    
    # Our operations on the frame come here
    # gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # Saves image of the current frame in jpg file
    # name = 'frame' + str(currentFrame) + '.jpg'
    # cv2.imwrite(name, frame)

    # Display the resulting frame
    cv2.imshow('im1',im1)
    # cv2.imshow('im1',gray)
    
    if currentFrame%100 == 0:
        cv2.imwrite('%s.jpg'%(save_count), im1)
        save_count += 1
        print(save_count)
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

