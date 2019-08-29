# heatmap

By using weights of a trained network and photoset, 
this code can create heatmap images for each image and each conv layer

You can execute the code by writing 
python heat.py --dataset name_of_the_photo_folder -- model name_of_the_saved_model_file
example: python heat.py --dataset photos --model weights.hdf5

Note that photo folder and hdf5 file should be in the same folder with the code

It creates heatmaps folder and conv2d folders in that heatmaps folder
In each folder you can find heatmap versions of each image