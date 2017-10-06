# **Traffic Sign Recognition**

---
## Writeup

In this writeup, a ConvNet is trained to recognize traffic signs in the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).  The project is organized into 3 steps:

1. Dataset Summary & Exploration
2. Design and Test a Model Architecture
3. Test the Model on New Images

The code for this project is available in both an [iPython Notebook](https://github.com/patrick-klein/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and [HTML document](https://github.com/patrick-klein/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html) on my [GitHub repo](https://github.com/patrick-klein/CarND-Traffic-Sign-Classifier-Project), per project guidelines.  This writeup will explicitly address the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/481/view).

Code has been tested on my personal computer running macOS 10.13 and an AWS instance, both within the *carnd-term1* environment.

[//]: # (Image References)

[image1]: ./examples/hist1 "Histogram 1"
[image2]: ./examples/hist2 "Histogram 2"
[image3]: ./examples/hist3 "Histogram 3"
[image4]: ./examples/aug_ref "Reference Image"
[image5]: ./examples/aug_noise "Noisy Image"
[image6]: ./examples/aug_rot "Rotated Image"
[image7]: ./examples/aug_blur "Blurred Image"
[image8]: ./test/test1.jpg "Traffic Sign 1"
[image9]: ./test/test2.jpg "Traffic Sign 2"
[image10]: ./test/test3.jpg "Traffic Sign 3"
[image11]: ./test/test4.jpg "Traffic Sign 4"
[image12]: ./test/test5.jpg "Traffic Sign 5"
[image13]: ./examples/test1_topk.png "Traffic Sign 1 Top 5"
[image14]: ./examples/test2_topk.png "Traffic Sign 2 Top 5"
[image15]: ./examples/test3_topk.png "Traffic Sign 3 Top 5"
[image16]: ./examples/test4_topk.png "Traffic Sign 4 Top 5"
[image17]: ./examples/test5_topk.png "Traffic Sign 5 Top 5"

---

### Dataset Exploration

#### Basic Summary of the Dataset

The following statistics about the dataset were calculated using basic Python functions, such as *len*, *shape*, and *set*.

```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

Note that these statistics are valid only for the default dataset, before the dataset has been augmented and pre-processed.

#### Exploratory Visualization of the Dataset

As part of the dataset visualization, histograms were generated to demonstrate the relative distributions in the training, testing, and validation datasets.

![alt text][image1] ![alt text][image2] ![alt text][image3]

These plots show that each of the datasets include a similar distribution of traffic signs relative to each other.  However there is significant variation within each, varying between 250 and 2000 occurrences in the training data.

It would be possible to alter these distributions by selectively augmenting the data, or by using more sophisticated tools in scikit-learn to generate new training/testing/validation sets.  This might increase testing accuracy, but these techniques were not pursued in this project.

#### Augment the Dataset

The training data is augmented in order to increase robustness against minor variances.  This includes duplicating the dataset with small variations in noise, rotation, and blur.  These effects are applied independently to one another (i.e., the effects are not combined).  As a result, the training set is 4x larger after including this additional data, and has the same distributions of signs.

Below is a reference image, with an example of each of these alterations.  

![alt text][image4]

![alt text][image5] ![alt text][image6] ![alt text][image7]

Augmentation techniques inspired by [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

---

### Design and Test a Model Architecture

#### Preprocessing

The pre-processing consisted of only two steps, converting to grayscale and normalization.

Both of these functions were designed in TensorFlow so that it could easily be incorporated into the training and testing pipelines.  This ensures that every image/batch gets pre-processing applied before being run through the model.


#### Model Architecture

The final ConvNet had the following architecture:

|| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
|| Input         		                  | 32x32x1 grayscale image   							        |
|L1<sub>a</sub>| Convolution 3x3   	  | 6 5x5 kernels, 1x1 stride, outputs 28x28x6 	    |
|L1<sub>b</sub>| RELU					        |												                          |
|L1<sub>c</sub>| Max pooling	      	| 2x2 kernel, 2x2 stride,  outputs 14x14x6        |
|L2<sub>a</sub>| Convolution 3x3     	| 16 5x5 kernels, 1x1 stride, outputs 10x10x16 	  |
|L2<sub>b</sub>| RELU					        |	                                                |
|L2<sub>c</sub>| Max pooling	      	| 2x2 kernel, 2x2 stride,  outputs 5x5x16 	      |
|L3<sub>a</sub>| Flatten & Concatenate| flatten & concatenate L1<sub>c</sub> and L2<sub>c</sub>, outputs 1576  |
|L3<sub>b</sub>| Fully Connected      | outputs 120  |
|L3<sub>c</sub>| RELU                 |   |
|L3<sub>e</sub>| Dropout              |   |
|L4<sub>a</sub>| Fully Connected      | outputs 84  |
|L4<sub>b</sub>| RELU                 |   |
|L4<sub>c</sub>| Dropout              |   |
|L5            | Fully Connected      | outputs 43  |

Convolution and pooling layers use 'VALID' padding.

#### Model Training

The Adam optimizer was used during training, using cross entropy on the softmax of the logits as the loss function.  The training pipeline ran for 10 epochs and used mini-batches of 128.  The initial learning rate for the Adam optimizer was set to 0.001.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

#### Solution Approach

The initial architecture that was tested was based on the LeNet-5 implementation shown during the Udacity lessons.  This model achieved an accuracy of approximately 0.88.

The first improvement on this model was the inclusion of pre-processing.  Normalization helps training by centering the data around 0 and scaling the features, so I implemented a basic form of normalization for the images.  Then, based on the suggestions from [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), I also converted all images going into the model to grayscale.  This seems to have helped the training of the first convolution layer by limiting the feature space and reducing variations between images.

Following the steps of Sermanet's & LeCun's paper again, I decided to augment the data to provide robustness against minor variations.  I created copies of the data that included noise, rotations, and blurring, and added them to the training set.

I attempted to add batch normalization to the model, but this drastically *lowered* the accuracy to approximately 0.3.  Considering how this this technique usually increased the quality of the model, I assume I did not implement it correctly.  A commented-out version of my attempt remains in the code.

Another addition to the model that significantly improved the accuracy was connecting the max pooling output of the first convolution layer to the first fully-connected layer.  This increased the accuracy because (a) the two convolution layers contain different types of features, and these were now both included in the fully-connected layers, and (b) the number of trainable parameters increased.

The most important addition to the model was the inclusion of regularization via dropout layers.  These layers were useful in the model because they reduce dependence on individual neurons "remembering" features and help combat overfitting.

The final results of the model were:

```
Training Accuracy = 0.983
Validation Accuracy = 0.953
Test Accuracy = 0.935
```

The model achieved the desired results of 0.93, but there is still some overfitting occurring.  The correct implementation of batch normalization would help in this regard.


### Test a Model on New Images

#### Acquiring New Images

Here are 5 stock images I found online of German traffic signs (after being cropped and resized):

![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12]

Each of these images contain a watermark, although at this size they don't appear distracting.  However, I cropped these images by hand so they may have a different bias in scale and centering than the original dataset.

#### Performance on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 70 km/h      		   | 30 km/h   							|
| Wildlife Crossing  | Wildlife crossing			|
| 50 km/h					   | 60 km/h								|
| Yield	      	     | Yield					 				|
| 30 km/h			       | 30 km/h      					|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. These results are lower than the test set accuracy of 0.935.

#### Model Certainty - Softmax Probabilities

To find the confidence in these results, the softmax of the logits were calculated, and the top 5 probabilities were found, as seen in the images below (with the correct label highlighted):

![alt text][image13] ![alt text][image14] ![alt text][image15]
![alt text][image16] ![alt text][image17]

For each of the images that were correctly predicted, the confidence was >90%.  However, it is important to note that for the other images, it still predicted similar signs with the same shape; it was the digits that were not correctly recognized.  The high confidence in the incorrect prediction of a 30 km/h sign in the first image may also indicate a low precision for signs that had a higher distribution in the training and testing data.
