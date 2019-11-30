# Import required modules
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from PIL import Image
from skimage.color import rgb2gray
import os
from scipy import spatial

bg_mapping = {'daria' : 'bg_015.avi' , 'denis' : 'bg_026.avi' , 'eli' : 'bg_062.avi' , 'ido' : 'bg_062.avi' , 'ira' : 'bg_007.avi' , 'lena' : 'bg_026.avi' , 'lyova' : 'bg_046.avi' , 'moshe' : 'bg_070.avi' , 'shahar' : 'bg_079.avi'}
# Frame grabber
def frame_grabber(file):
    # video_file = file + '.avi'
    frames = []
    # Opens the Video file
    cap = cv2.VideoCapture(file)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
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
                    mhi[y, x] = timestamp + 1
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

# Normalize MHI
def normalize(mhi):
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
    delta = 30
    # Grab frames from video of action
    imgFrameData = frame_grabber(file_name)
    imgGrayscaleFrameData = {}
    for i,image in enumerate(imgFrameData):
        grayscale = rgb2gray(image)
        # Convert color image to grayscale image
        imgGrayscaleFrameData[i] = (grayscale * 255).astype(int)
    # Get image of background
    bgFrameData = frame_grabber(bg_file_name)
    # Convert background image from color to grayscale
    bgImage = (rgb2gray(bgFrameData[0]) * 255).astype(int)
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
    rootdir = 'Static Train'
    # rootdir = 'Train Dataset'
    background = 'Backgrounds'
    for subdir, dir, files in os.walk(rootdir):
        for file in files:
            act_name = file.split('.')[0].split('_')[0]
            print(act_name)
            bg_path = background + '/' + bg_mapping[act_name]
            filePath = rootdir + '/' + file
            temporal_template = get_temporal_template(filePath, bg_path)
            tt_dict[file] = temporal_template
            print(file + " processed.")
            print(temporal_template)
    print(tt_dict)
    with open('temporal_templates_static_3.txt', 'w') as f:
        f.write(json.dumps(tt_dict))

def get_euclidean_dist(test_template , train_template):
    return spatial.distance.euclidean(test_template , train_template)

def get_mahalanobis_dist(test_template , train_template):
    # Method - 1 (Using mahalanobis library)
    # a = np.asarray(test_template)
    # b = np.asarray(train_template)
    # K = np.cov((a , b) , rowvar=False)
    # k_inv = np.linalg.inv(K)
    # return spatial.distance.mahalanobis(test_template , train_template , k_inv)
    # Method - 2 (Using formula in slides)
    test_template = np.asarray(test_template)
    train_template = np.asarray(train_template)
    t = np.vstack((test_template, train_template))
    # print(type(t))
    k = np.cov(t.T)
    # print(k.shape)
    mean_temp = np.mean(t , axis=0)
    # print(mean.shape)
    md = np.matmul(np.matmul((test_template - mean_temp).T, (np.linalg.pinv(k))), (test_template - mean_temp))
    return (md)
    # return md

def get_mahalanobis_dist_pinv(test_template , train_template):
    t = np.array([train_template, test_template]).T
    k = np.cov(t)
    md = spatial.distance.mahalanobis(train_template, test_template, np.linalg.pinv(k))
    return (md)

def get_cosine_dist(test_template , train_template):
    return spatial.distance.cosine(test_template , train_template)

def predict_activity(file , background , temporal_template_dict):
    pred_temporal_template = get_temporal_template(file, background)
    min_dist = float("inf")
    activity = ""
    for k , v in temporal_template_dict.items():
        # Using mahalanobis distance
        dist = get_mahalanobis_dist(pred_temporal_template , v)
        # Using euclidean distance
        # dist = get_euclidean_dist(pred_temporal_template , v)
        # Using cosine distance
        # dist = get_cosine_dist(pred_temporal_template , v)
        if(dist < min_dist):
            min_dist = dist
            activity = k
    print(min_dist)
    return activity

if __name__ == '__main__':
    # Run below command to train dataset
    # create_temporal_template_dict()
    # Prediction Calculation
    pred_dir = 'Static Test'
    # pred_dir = 'Test Dataset'
    background = 'Backgrounds'
    with open('temporal_templates_static_2.txt' , 'r') as f:
        temporal_template_dict = json.loads(f.read())
    print(temporal_template_dict)
    acc_count = 0
    total = 0
    for subdir, dir, files in os.walk(pred_dir):
        for file in files:
            pred_file_path = pred_dir + '/' + file
            print(file)
            act_activity = file.split('.')[0].split('_')[1]
            act_name = file.split('.')[0].split('_')[0]
            print(act_activity)
            bg_path = background + '/' + bg_mapping[act_name]
            print(bg_path)
            result_activity = predict_activity(pred_file_path , bg_path , temporal_template_dict)
            print(result_activity)
            pred_activity = result_activity.split('.')[0].split('_')[1]
            print(pred_activity)
            print("------------")
            if(act_activity == pred_activity):
                acc_count += 1
            total += 1
    print("Accuracy : ",acc_count / total)