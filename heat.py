from keras import backend as K
K.set_image_data_format('channels_last')
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
import cv2
from keras.models import load_model

# Note that we are including the densely-connected classifier on top;
# all previous times, we were discarding it.
model = load_model('weights-from_scratch-47-1.00.hdf5')
model.summary()
# The local path to our target image
img_path = '_ (12).png'

# `img` is a PIL image
img = image.load_img(img_path, color_mode='grayscale')

# `x` is a float32 Numpy array of shape (160, 256, 1)
x = image.img_to_array(img)
x = x.astype('float32') / 255
# We add a dimension to transform our array into a "batch"
# of size (1, 160, 256, 1)
x = x.reshape((1,160,256,1))
# This is the entry in the prediction vector
pred_vector_output = model.layers[18].output[:,0]
int_layer = 10
# The is the output feature map of the given layer
some_conv_layer = model.layers[int_layer].output

# This is the gradient of the predicted class with regard to
# the output feature map of selected block
grads = K.gradients(pred_vector_output, some_conv_layer)[0]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, some_conv_layer[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(model.layers[int_layer].output_shape[-1]):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)

print(heatmap)
# We use cv2 to load the original image
img = cv2.imread(img_path)

img_heatmap = np.maximum(heatmap, 0)
img_heatmap /= np.max(img_heatmap)
    
# We resize the heatmap to have the same size as the original image
img_hm = cv2.resize(img_heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
img_hm = np.uint8(255 * img_hm)

# We apply the heatmap to the original image
img_hm = cv2.applyColorMap(img_hm, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = img_hm * 0.4 + img

# Save the image to disk
cv2.imwrite('./12_{}.jpg'.format('heatmap'), superimposed_img)