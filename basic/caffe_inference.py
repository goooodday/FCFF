import numpy as np
from numpy import zeros, newaxis
import matplotlib.pyplot as plt
import caffe
import cv2
%matplotlib inline

import os

# set Currnet Directory
os.chdir('/data1/Notebooks/1CAL_Test')
print("caffe-%s, cv2-%s, %s" % (caffe.__version__, cv2.__version__, os.getcwd()))

#--------------------------------------------------------------------------------
MEAN_FILE = 'Train_Datasets/mean.binaryproto'
DEP_NETWORK = "Train_Datasets/deploy.prototxt"
MODEL_WEIGHTS = "Train_Datasets/Snapshots/1stTrain_iter_5000.caffemodel"
TEST_DATA_PATH = "Image_Sets/val_data/"

# set device
caffe.set_device(2)
caffe.set_mode_gpu()

#Read model architecture and trained model's weights
net = caffe.Net(DEP_NETWORK,      # defines the structure of the model
                MODEL_WEIGHTS,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

#Read mean image
mean_blob = caffe.proto.caffe_pb2.BlobProto()

with open(MEAN_FILE, 'rb') as f:
    mean_blob.ParseFromString(f.read())
    
mean_arr = np.asarray(mean_blob.data, dtype=np.float32).reshape((mean_blob.channels, mean_blob.height, mean_blob.width))

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_arr.mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))

print("Set model & network...")

#---------------------------------------------------------------------------
image_path = 'house'
image_file = '1234343_3434.jpg'
#---------------------------------------------------------------------------

# Load image
img = cv2.imread(os.path.join(TEST_DATA_PATH, image_path, image_file), 0)
img_size = (net.blobs['data'].data.shape[2], net.blobs['data'].data.shape[3])

re_img = cv2.resize(img, img_size)
re_img = re_img[:,:, newaxis]

net.blobs['data'].data[...] = transformer.preprocess('data', re_img)
#print(net.blobs['data'].data.shape)

#compute
out = net.forward()
result_label = out['prob'].argmax()
#print(result_label)

# get data image
blob_image = np.squeeze(net.blobs['data'].data)

plt.figure()
plt.title('Label : %d' %(result_label))
plt.axis('off')
plt.imshow(blob_image, cmap='gray')
