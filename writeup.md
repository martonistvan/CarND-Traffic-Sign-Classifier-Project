#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
You're reading it! and here is a link to my project on github (https://github.com/martonistvan/CarND-Traffic-Sign-Classifier-Project.git)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python and numpy library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

####2. Include an exploratory visualization of the dataset.

for visualization purposes I did 2 things:
1, for each traffic sign I am showing an example from the training dataset

2, I am applying a barchart to show the dispersion of each traffic sign (label) in the training dataset.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As I am using the color images for training purposes I did not convert the pictures!
I only suffled training data set and set parameters like EPOCHS and BATCH_SIZE.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Input: Input = 32x32x3

1, Layer 1:

Convolutional:
    - Output: 28x28x6
    - Padding: VALID
    - stride: 1x1
Activation:
    - RELU
Max Pooling:
    - Output = 14x14x6
    - stride: 2x2
    
2, Layer 2:

Convolutional:
    - Output: 10x10x16
    - Padding: VALID
    - stride: 1x1
Activation:
    - RELU
Max Pooling:
    - Output = 5x5x16
    - stride: 2x2
    
3, Flatten the output of the second convolutional layer:
    - Input: 5x5x16
    - output: 400 (5*5*16)

4, Layer 3:

Fully connected layer:
    - Input: 400
    - Output: 120
Activation:
    - RELU
    
5, Layer 4:

Fully connected layer:
    - Input: 120
    - Output: 84
Activation:
    - RELU

6, Layer 5:

Fully connected layer:
    - Input: 120
    - Output: 43 (number of labels in the data set)

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I:

1, defined tensor placeholders (x, y) which holds the image and label information during the training. In addition to that I define one_hot_y for cross entropy calculation.

2, set rate parameter

3, training pipeline looks like:
    - passing input data to logits function to get the logits
    - using softmax and cross entropy to compare logits to one_hot labels
    - applied reduce_mean to average cross entropy from all the training images
    - Adam optimizer with rate "0.002" to minimize the loss function
    - run the minimize function on the optimizer

4,after setting up the training pipeline it is important to evaluate how good the model is ("evaluate" function).
    - measuring if prediction is correct
    - caclulate the model overall accuracy on  the batches
    
5, to train the model I:
    - create tensorflow session
    - initialize variables
    - Training is carried out "EPOCHS" time on the data set
    - at the beginning of each "EPOCH" I shuffle training data set to assure that the training isn't biased by the order of the images
    - divide the training data into batches and I train the model on each batch
    - at the end of each epoch I evaluate the model on the validation data
    
6, I save the model to use later.
   
   
7, last step is to evaluate the model on the test dataset.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I tried several scenarios to train the model and get at least 0.93 accuracy on the validation set.
I applied the following parameters:
    - EPOCHS = 60
    - BATCH_SIZE = 128
    - RATE = 0.002
    
 with these parameters I could reach 0.931 accuracy in epoch 55 (see attached HTML).
 
Test set accuracy: 0.902

I used LeNet architecture to train the model.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I found the following traffic signs on the web (images are also uploaded to github, in folder "test_images"):
    - 1: speed limit 30 km/h
    - 3: speed limit 60 km/h
    - 11: Right-of-way at the next intersection
    - 12: Priority road
    - 13: Yield
    - 14: Stop
    - 18: General Caution
    - 28: Children crossing

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Image                                           Prediction
---------------------------------------------------------------------------------------
speed limit 30 km/h                             speed limit 30 km/h
speed limit 60 km/h                             Speed limit (80km/h)
Right-of-way at the next intersection           Right-of-way at the next intersection
Priority road                                   Priority road
Yield                                           No passing for vehicles over 3.5 metric tons
Stop                                            Stop
General Caution                                 General Caution
Children crossing                               Road work


The model was able to correctly guess 5 of the 8 traffic signs, which gives an accuracy of 62.5%. This compares favorably to the accuracy on the test set of 0.902

Based on my tests with different parameter values the prediction on my test images worked better when training accuracy was below 0.900 (between 0.850-0.900).

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 22nd cell of the Ipython notebook.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00					| Right-of-way at the next intersection											|
| 1.00         			| speed limit 30 km/h   									| 
| 3.33905126e-11 (3rd)	| speed limit 60 km/h 										|
| 1.00				    | Stop      							|
| 3.89943889e-05 (2nd)  | Children crossing      							|
| 1.00				    | General Caution      							|
| 1.00	      			| Priority road					 				|
| 4.72048082e-37 (3rd)  | Yield      							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Unfortunately I did not have sufficient time for that. I would definitely like to do that to see what characteristics of an image the network finds interesting.

