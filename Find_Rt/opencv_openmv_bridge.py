import numpy as np
import io
import struct
import serial
from PIL import Image as PILImage
import cv2
import arsenal
from make_Template import make_Ideal_RF, make_Ideal_TF
from timeit import default_timer as timer
from scipy.special import softmax
# Camera object to create the snaps/frames/images that
#  will be deserialized later in the opencv code

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

wid = 128
hei = 128
walldistance = 5
FL = 107.57
hg = 16//2
RI = np.eye(3, 3)
K_03 = np.array([[FL, 0, wid//2 ],
			    [0, FL, hei//2 ],
			    [0, 0, 1, ]])  
# K_03 = np.array([[619.13152082,   0,          61.6084372],
# 			     [  0,         715.72291161,  69.46021523],
# 			     [0, 0, 1, ]])  
print(K_03)
dist_coeffs = np.array([[-1.23001715e+01, -1.67838676e+03, -7.94323900e-02,  3.76330876e-02,  8.30444398e+04]])
wall = arsenal.makewall(walldistance, wid, hei, hg, K_03)

hsv = np.zeros((hei, wid, 3)).astype(np.float32)
hsv[...,1] = 255

hsv0 = np.zeros((hei, wid, 3)).astype(np.float32)
hsv0[...,1] = 255

translation_O = np.array([0, 0, 0]).reshape(3, 1)
AV = 0.25 #perframe
T = 2.3

IdealFlowList = []
DIdealFlowList = []
Ppitch = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, AV, 0, 0)
Pyaw = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, AV, 0)
Proll = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, 0, AV)

# Npitch = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, -AV, 0, 0)
# Nyaw = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, -AV, 0)
# Nroll = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, 0, -AV)
Npitch = -Ppitch
Nyaw = -Pyaw
Nroll = -Proll

IdealFlowList.append(Proll)
IdealFlowList.append(Nroll)
IdealFlowList.append(Ppitch)
IdealFlowList.append(Npitch)
IdealFlowList.append(Pyaw)
IdealFlowList.append(Nyaw)
for i in IdealFlowList:
    new = arsenal.meanOpticalFlow(i)
    DIdealFlowList.append(new.flatten())

IdealTFlowList = []
DIdealTFlowList = []
IdealTFlowListname = []
Px = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, T, 0, 0)
Py = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, T, 0)
Pz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, 0, -T)
# PzT = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, 0, -T)/1.6
# Pxy = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, T/1.6, T/1.6, 0)
# PxNy = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, T/1.6, -T/1.6, 0)
# Pxz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, T/1.6, 0, -T/1.6)
# PxNz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, T/1.6, 0, 0) - PzT
# Pyz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, T/1.6, -T/1.6)
# PyNz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, T/1.6, 0) - PzT

# Nx = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, -T, 0, 0)
# Ny = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, -T, 0)
# Nz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, 0, T)
Nx = -Px
Ny = -Py
Nz = -Pz
# Nxy = -Pxy
# Nxz = -Pxz
# Nyz = -Pyz
# NxPy = -PxNy
# NxPz = -PxNz
# NyPz = -PyNz

IdealTFlowList.append(Px)
IdealTFlowList.append(Nx)
IdealTFlowList.append(Py)
IdealTFlowList.append(Ny)
IdealTFlowList.append(Pz)
IdealTFlowList.append(Nz)

# IdealTFlowList.append(Pxy)
# IdealTFlowList.append(Nxy)
# IdealTFlowList.append(PxNy)
# IdealTFlowList.append(NxPy)
# IdealTFlowList.append(Pxz)
# IdealTFlowList.append(Nxz)
# IdealTFlowList.append(PxNz)
# IdealTFlowList.append(NxPz)
# IdealTFlowList.append(Pyz)
# IdealTFlowList.append(Nyz)
# IdealTFlowList.append(PyNz)
# IdealTFlowList.append(NyPz)

IdealTFlowListname.append('Px')
IdealTFlowListname.append('Nx')
IdealTFlowListname.append('Py')
IdealTFlowListname.append('Ny')
IdealTFlowListname.append('Pz')
IdealTFlowListname.append('Nz')

# IdealTFlowListname.append('Pxy')
# IdealTFlowListname.append('Nxy')
# IdealTFlowListname.append('PxNy')
# IdealTFlowListname.append('NxPy')
# IdealTFlowListname.append('Pxz')
# IdealTFlowListname.append('Nxz')
# IdealTFlowListname.append('PxNz')
# IdealTFlowListname.append('NxPz')
# IdealTFlowListname.append('Pyz')
# IdealTFlowListname.append('Nyz')
# IdealTFlowListname.append('PyNz')
# IdealTFlowListname.append('NyPz')

for i in IdealTFlowList:
    new = arsenal.meanOpticalFlow(i)
    DIdealTFlowList.append(new.flatten())
    # print(len(DIdealFlowList))
    # print(new.shape)

currentFrame = 0
# dis = cv2.DISOpticalFlow_create(0)
# dis.setFinestScale(1)
start_point = np.array([0, 0, 0])

while(True):
    Vectorimage = np.zeros((320, 320, 3), np.uint8)
    # Create a camera by just giving the ttyACM depending on your connection value
    # Change the following line depending on your connection
    strat = timer()
    cap = Camera(device='/dev/ttyACM0')
    # Capture frame-by-frame
    im1 = cap.read_image()
    # im1 = cv2.cvtColor(np.float32(im1), cv2.COLOR_GRAY2BGR)
    # im1 = cv2.undistort(im1, K_03, dist_coeffs, None, K_03)
    # im1 = cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if currentFrame == 0:
        im0 = im1
    # flow = dis.calc(im0, im1, None, )
    flow = cv2.calcOpticalFlowFarneback(im0, im1, None, 0.5, 8, 15, 3, 5, 1.2, 0)
    Dflow = arsenal.meanOpticalFlow(flow)
    DotResult = arsenal.dotWithTemplatesOpt(Dflow.flatten(), DIdealTFlowList)
    
    DotResult[DotResult<25] = 0
    GraphArray = np.round(DotResult/np.sum(DotResult), 3)
    # print(DotResult.index(max(DotResult)))
    if np.mean(DotResult) > 5:
        # DotResult = softmax(DotResult).tolist()
        # print(np.round(DotResult/np.sum(DotResult), 1))
        end_point = np.array([GraphArray[1] - GraphArray[0], GraphArray[3] - GraphArray[2], GraphArray[5] - GraphArray[4]])
        print(np.round(end_point, 3))
    
        
        dist_coef = np.zeros((4, 1))
        KK = np.array([[500, 0, 160], [0, 500, 160], [0, 0, 1]]).astype(np.float32)
        points, _ = cv2.projectPoints(np.array([end_point]), (0, 0, 0), (0, 0, 0), KK, dist_coef)
        start_point_2d = tuple(map(int, points[0][0]))
        end_point_2d = (160, 160)
        cv2.line(Vectorimage, start_point_2d, end_point_2d, (255, 255, 255), 2)
        
        #print(np.round(DotResult, 3))
        # print(IdealTFlowListname[DotResult.index(np.max(DotResult))])

    # DotResult[4] *= 3
    # DotResult[5] *= 3
    # print('LEN', len(DotResult))
    # if currentFrame%5 == 0:
    
    # print('%s \n'%(DotResult))

    # print(Dflow.shape)
    # RollP = np.inner(flow, Proll)
    # RollN = np.inner(flow, Nroll)
    # PitchP = np.inner(flow, Ppitch)
    # PitchN = np.inner(flow, Npitch)
    # YawP = np.inner(flow, Pyaw)
    # YawN = np.inner(flow, Nyaw)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # Saves image of the current frame in jpg file
    # name = 'frame' + str(currentFrame) + '.jpg'
    # cv2.imwrite(name, frame)

    # Display the resulting frame
    # cv2.imshow('im1',im1)
    # cv2.imshow('im1',gray)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv[...,2] = mag*3
    hsv = hsv.astype(np.uint8)
    # print(hsv.shape)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    bgr = cv2.resize(bgr, (320, 320))

    mag0, ang0 = cv2.cartToPolar(Pz[...,0], Pz[...,1])
    hsv0[...,0] = ang0*180/np.pi/2
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv0[...,2] = mag0*3
    hsv0 = hsv0.astype(np.uint8)
    # print(hsv.shape)
    bgr0 = cv2.cvtColor(hsv0,cv2.COLOR_HSV2BGR)

    bgr0 = cv2.resize(bgr0, (320, 320))

    IMG = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    IMG = cv2.resize(IMG, (320, 320))
    merge0 = np.concatenate((IMG, bgr), axis=1)
    merge1 = np.concatenate((Vectorimage, bgr0), axis=1)
    merge = np.concatenate((merge0, merge1), axis=0)
    cv2.imshow('merge', merge)
    # cv2.imshow('sda', Vectorimage)

    end = timer()
    # print('FPS', 1/(end-strat))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1
    im0 = im1