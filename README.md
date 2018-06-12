# Deep Learning Project Starter Code

The objective of this project is to build a segmentation network, it will be trained, validated and implemented in the project called Follow Me, so the drone will follow a single hero target using the network pipeline.

---
<!--more-->

[//]: # (Image References)

[image0]: ./misc_images/semantic_segmentation.jpg "Semantic Segmentation"


#### How to run the program with your own code

For the execution of your own code, inside the folder /code

```bash
1.  jupyter notebook model_training.ipynb
```

After opening the notebook, at the top, select the tab called *Cell* and select the option *Run All*.

---

The summary of the files and folders int repo is provided in the table below:

| File/Folder                     | Definition                                                                                            |
| :------------------------------ | :---------------------------------------------------------------------------------------------------- |
| code/*                          | Folder that contains contains all the coding files used in deep learning practice.                    |
| data/*                          | Folder that contains the dataset to be used by the segmentation network.                              |
| docs/*                          | Folder that contains documentation about socketio to facilitate communication between QuadSim and the |
|                                 | Python code.                                                                                          |
| misc_images/*                   | Folder containing the images of the project.                                                          |
|                                 |                                                                                                       |
| README.md                       | Contains the project documentation.                                                                   |
| README_udacity.md               | Is the udacity documentation that contains how to configure and install the environment.              |

---

### README_udacity.md

In the following link is the [udacity readme](https://github.com/Abhaycl/RoboND-DeepLearning-1P4/blob/master/README_udacity.md), for this practice provides instructions on how to install and configure the environment.

---


**Steps to complete the project:**  

1. Clone the project repo here
2. Fill out the TODO's in the project code as mentioned here 
3. Optimize your network and hyper-parameters.
4. Train your network and achieve an accuracy of 40% (0.40) using the Intersection over Union IoU metric which is final_grade_score at the bottom of your notebook.
5. Make a brief writeup report summarizing why you made the choices you did in building the network.


## [Rubric Points](https://review.udacity.com/#!/rubrics/1155/view)
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

---

### Network Architecture

If we want to do a *semantic segmentation*, we are interested not only in classifying whether a target person is present or not in the input image, but also in where that person is so that the drone controller can take the necessary actions such as: zoom in if the target is far away or move away from the centre of the image.

![alt text][image0]
###### Trained FCN performing semantic segmentation. Input image (left), ground truth (center) and FCN output (left).


A classical convolution network classifies the probability of a determined class is present in the image, unlike a full convolution network (FCN), which preserve the spatial information throughout the entire network outputting by generating a map probabilities corresponding to each pixel of the input image.

I will create a full convolution network following the suggestions of the classes and this notebook, this FCN will consist of three parts:

1. An encoder network that transforms an image input into feature maps.
2. Followed by 1x1 convolution that combines the feature maps (similar to a fully connected layer).
3. Finally a decoder network that upsample the result from the previous layer back to the same dimensions as the input image.

```python

```

### Data Recording

I didn't record any data from simulator, I was able to do all required steps using the provided Training, Validation, and Sample Evaluation Data.

<table><tbody>
    <tr><th align="center" colspan="3"> Data Set 1</td></tr>
    <tr><th align="center">Folder</th><th align="center">Content</th></tr>
    <tr><td align="left">/data/train</td><td align="left">4,131 images + 4,131 masks</td></tr>
    <tr><td align="left">/data/validation</td><td align="left">1,184 images + 1,184 masks</td></tr>    
    <tr><td align="left">/data/sample_evalution_data/following_images</td>
       <td align="left">542 images + 542 masks</td></tr><tr>
    <td align="left">/data/sample_evalution_data/patrol_non_targ</td>
       <td align="left"> 270 images + 270 masks</td></tr><tr>
    <td align="left">/data/sample_evalution_data/patrol_with_targ</td>
       <td align="left"> 322 images + 322 masks</td></tr>
</tbody></table>

### Hyperparameters

The Hyperparameters use in this project is:

```python
learning_rate = 0.001
batch_size = 16
num_epochs = 50
steps_per_epoch = 4131 / batch_size
validation_steps = 100
workers = 2
```

#### Define and tune hyperparameters

* **learning_rate:** Started at 0.001 and the network had no problem with that value.
* **batch_size:** Is the number of training samples/images that get propagated through the network in a single pass. It is set to 16.
* **num_epochs:** Is the number of times the entire training dataset gets propagated through the network. This value is set to 50.
* **steps_per_epoch:** Is the number of batches of training images that go through the network in 1 epoch. We have provided you with a default value. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size. This value is set to 4131/16 = 258
* **validation_steps:** Is the number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. We have provided you with a default value for this as well. This value is set to 100.
* **workers:** Maximum number of processes to spin up. This can affect your training speed and is dependent on your hardware. We have provided a recommended value to work with. This value is set to 2.


### Observations, possible improvements, things used

