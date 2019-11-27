# Import required modules
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from PIL import Image
from skimage.color import rgb2gray
import os

# Frame grabber
def frame_grabber(file):
    # video_file = file + '.avi'
    frames = []
    # Opens the Video file
    cap = cv2.VideoCapture(file)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        # print(type(frame))
        if ret == False:
            break
        # if (i < 10):
        #     cv2.imwrite(file + '-00' + str(i) + '.png', frame)
        # elif (10 <= i < 100):
        #     cv2.imwrite(file + '-0' + str(i) + '.png', frame)
        # else:
        #     cv2.imwrite(file + '-' + str(i) + '.png', frame)
        # i += 1
        frames.append(frame)
    cap.release()
    cv2.destroyAllWindows()
    frames = np.asarray(frames)

    return frames
# Read File
def read_files(file):
    # Reading the images from the directory
    filename = glob.glob(file + '*png')
    file_name = {}
    for i in range(len(filename)):
        file_name[i] = np.double(Image.open(str(filename[i])).convert('L'))
    return (file_name)

# Background Substraction I
def background_subtraction(Im, background, threshold):
    bgs = {}
    plt.show()
    for i in range(len(Im)):
        bgs[i] = (np.abs(Im[i] - background) > threshold).astype(int)
    plt.show()
    return (bgs)

# Calculate MHI
def MHI(image, delta):
    mhi = np.zeros((np.shape(image[0])[0], np.shape(image[0])[1]), np.uint8)

    row, column = np.shape(image[0])

    for timestamp in range(0, len(image)):
        frame = image[timestamp]

        for y in range(row):
            for x in range(column):
                if (frame[y, x] == 1):
                    mhi[y, x] = timestamp
                else:
                    if (mhi[y, x] < timestamp - delta):
                        mhi[y, x] = 0

    # fig = plt.figure(figsize=(5, 5))
    # fig.suptitle('The final MHI is', fontsize=20)
    # plt.imshow(mhi)
    # plt.gray()
    # plt.axis('off')
    # plt.show()

    return mhi

# Calculate MEI
def MEI(Im):
    mei = np.zeros((np.shape(Im[0])[0], np.shape(Im[0])[1]), np.uint8)

    # The MEI/MHI duration should include all image diff results in the sequence into the final template.
    # So, frames to be considered i.e., delta = 22

    for i in range(len(Im)):
        mei = mei + Im[i]

        mei = mei > 0

    fig = plt.figure(figsize=(5, 5))
    fig.suptitle('The final MEI is', fontsize=20)
    plt.imshow(mei)
    plt.axis('off')
    # plt.show()

    return (np.asarray(mei))

# Calculate MEI as threshold of MHI
def MEI_Thresh(mhi):
    mei = mhi > 0
    return mei

# Normalize MHI and MEI
def normalize(mhi):
    # Normalize MHI and MEI,

    mhi_n = np.maximum(0, np.divide((mhi - (np.min(mhi[np.nonzero(mhi)]) - 1.0)),
                                    (np.max(mhi[np.nonzero(mhi)]) - (np.min(mhi[np.nonzero(mhi)]) - 1.0))))

    print('Maximum value in MHI: ', np.max(mhi_n))
    print('Minimum value in MHI: ', np.min(mhi_n))

    return (mhi_n)

# Calculate similitude moments
def similitude_moments(Im):
    y, x = np.mgrid[range(Im.shape[0]), range(Im.shape[1])]

    similitude_moments = []

    x_bar = np.sum(x * Im) / np.sum(Im)
    y_bar = np.sum(y * Im) / np.sum(Im)

    # Since 2 <= (i+j) <=3, the similitude moments
    for i in range(4):
        for j in range(4):
            if (2 <= (i + j) <= 3):
                s = np.sum(((x - x_bar) ** i) * ((y - y_bar) ** j) * Im) / (np.sum(Im)) ** (((i + j) / 2) + 1)
                similitude_moments.append(s)

    return (similitude_moments)

def get_temporal_template(file_name , bg_file_name):
    # Threshold for Background Substraction
    thresh = 40
    # Delta value for the number of frames to keep
    delta = 20
    # Grab frames from video of action
    imgFrameData = frame_grabber(file_name)
    imgGrayscaleFrameData = {}
    for i,image in enumerate(imgFrameData):
        grayscale = rgb2gray(image)
        # Convert color image to grayscale image
        imgGrayscaleFrameData[i] = grayscale * 255
    # Get image of background
    bgFrameData = frame_grabber(bg_file_name)
    # Convert background image from color to grayscale
    bgImage = rgb2gray(bgFrameData[0]) * 255
    # Perform background substraction on the images captured from the videos
    bgSubsImages = background_subtraction(imgGrayscaleFrameData , bgImage, threshold=thresh)
    # Get the MHI from the background subtracted images
    mhiImg = MHI(bgSubsImages , delta)
    # Normal MHI Image
    normMhiImg = normalize(mhiImg)
    # plt.imshow(normMhiImg , cmap='gray')
    # Get MEI
    meiImg = MEI_Thresh(normMhiImg)
    # plt.imshow(meiImg , cmap='gray')
    # plt.show()
    # Get Similitude moments of MHI and MEI
    mhiSimilitude = similitude_moments(normMhiImg)
    meiSimilitude = similitude_moments(meiImg)
    result = []
    result.extend(mhiSimilitude)
    result.extend(meiSimilitude)
    return result

def create_temporal_template_dict():
    tt_dict = {}
    rootdir = 'Train Dataset'
    background = 'Backgrounds/bg_007.avi'
    for subdir, dir, files in os.walk(rootdir):
        for d in dir:
            dirPath = rootdir + '/' + d
            for file in os.listdir(rootdir + '/' + d):
                filePath = dirPath + '/' + file
                temporal_template = get_temporal_template(filePath, background)
                tt_dict[file] = temporal_template
                print(file + " processed.")
                print(temporal_template)
    print(tt_dict)
    with open('temporal_templates.txt', 'w') as f:
        f.write(json.dumps(tt_dict))

def get_mahalanobis_dist(test_template , train_template):
    test_template = np.asarray(test_template)
    train_template = np.asarray(train_template)
    t = np.vstack((test_template, train_template))
    print(type(t))
    k = np.cov(t.T)
    print(k.shape)
    md = np.matmul(np.matmul((test_template - train_template).T, (np.linalg.inv(k))), (test_template - train_template))
    return (md)

def predict_activity(file , background , temporal_template_dict):
    pred_temporal_template = get_temporal_template(file, background)
    min_dist = float("inf")
    activity = ""
    for k , v in temporal_template_dict.items():
        dist = get_mahalanobis_dist(pred_temporal_template , v)
        print(k)
        print(dist)
        if(dist < min_dist):
            min_dist = dist
            activity = k
    return activity

if __name__ == '__main__':
    pred_file = 'lyova_run.avi'
    background = 'Backgrounds/bg_007.avi'
    with open('temporal_templates.txt' , 'r') as f:
        temporal_template_dict = json.loads(f.read())
    print(temporal_template_dict)
    result_activity = predict_activity(pred_file , background , temporal_template_dict)
    print(result_activity)