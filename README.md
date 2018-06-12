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

After opening the notebook, at the top, select the tab called "Cell" and select the option "Run All".

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

If we want to do a " semantic segmentation ", we are interested not only in classifying whether a target person is present or not in the input image, but also in where that person is so that the drone controller can take the necessary actions such as: zoom in if the target is far away or move away from the centre of the image.

![alt text][image0]
###### Trained FCN performing semantic segmentation. Input image (left), ground truth (center) and FCN output (left).

A classical convolution network classifies the probability of a determined class is present in the image, unlike a full convolution network (FCN), which preserve the spatial information throughout the entire network outputting by generating a map probabilities corresponding to each pixel of the input image.

I will create a full convolution network following the suggestions of the classes and this notebook, this FCN will consist of three parts:

1) An encoder network that transforms an image input into feature maps.
2) Followed by 1x1 convolution that combines the feature maps (similar to a fully connected layer).
3) Finally a decoder network that upsample the result from the previous layer back to the same dimensions as the input image.

```python

```


### Observations, possible improvements, things used

