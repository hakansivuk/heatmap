from keras import backend as K
K.set_image_data_format('channels_last')
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
import cv2
from keras.models import load_model
import argparse
import os
from PIL import Image as pimg
import glob
import imageio


def findImageFiles(path):
    all_paths = os.listdir(path)
    image_paths = []
    for i in range(len(all_paths)):
        if(all_paths[i].lower().endswith(('.png'))):
            image_paths.append(all_paths[i])
    return image_paths

def loadAndProcessImage(image_path):
    img = image.load_img(image_path, color_mode='grayscale')
    x = image.img_to_array(img)
    x = x.astype('float32') / 255
    x = x.reshape((1, x.shape[0], x.shape[1], 1))
    return x

def createHeatmap(image_path, layer_number, model):
    # This is the entry in the prediction vector we want to examine
    pred_vector_output = model.layers[len(model.layers) - 2].output[:,0]

    # Loaded and processed image
    x = loadAndProcessImage(image_path)

    # It is the output feature map of one of the conv layers we want to visualize
    conv_layer = model.layers[layer_number].output

    # This is the gradient of the predicted vector output w.r.t. the output feature map of the selected conv layer
    grads = K.gradients(pred_vector_output, conv_layer)[0]

    # This i vector of shape (# of channels, ), where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, conv_layer[0]])

    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(model.layers[layer_number].output_shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    return heatmap

def saveHeatmapImage(image_path, heatmap, saveas):
    # Loading the image
    img = cv2.imread(image_path)

    img_heatmap = np.maximum(heatmap, 0)
    img_heatmap /= np.max(img_heatmap)
        
    # We resize the heatmap to have the same size as the original image
    img_hm = cv2.resize(img_heatmap, (img.shape[1], img.shape[0]))

    # We convert the heatmap to RGB
    img_hm = np.uint8(255 * img_hm)

    # We apply the heatmap to the original image
    img_hm = cv2.applyColorMap(img_hm, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = img_hm * intensity_factor + img

    # Save the image to disk
    cv2.imwrite(saveas, superimposed_img)


# Constructing the argument parser
ap = argparse.ArgumentParser()

# Adding an argument for the file which we load the model
ap.add_argument("-m", "--model", required=True, help="name of the hdf5 file")
# Adding an argument for the path to dataset
ap.add_argument("-d", "--dataset", required=True, help="path to dataset we use")

args = vars(ap.parse_args())

# Loading the model
model = load_model(args["model"])
model.summary()

# Setting the intensity factor
intensity_factor = 0.4

output_path = "heatmaps_" + args["dataset"]
if not os.path.exists(output_path):
    os.mkdir(output_path)

# Finding image paths from the dataset file
dataset_file_path = os.getcwd() + '/' + args["dataset"] + '/'
image_names = findImageFiles(dataset_file_path)

for i in range(len(model.layers)):
    if(model.layers[i].__class__.__name__ == 'Conv2D'): # For all conv layers
        # Creating a folder for that conv layer
        if not os.path.exists(output_path + "/conv2d_layer" + str(i)):
            os.mkdir(output_path + "/conv2d_layer" + str(i))

        for j in range(len(image_names)): # For all images
            image_path = dataset_file_path + image_names[j] 
            heatmap = createHeatmap(image_path, i, model) # heatmap of the image
            saveHeatmapImage(image_path, heatmap, output_path + '/conv2d_layer' + str(i) + '/' + image_names[j] + '.jpg')     