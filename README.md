# Deep Learning Project Starter Code

The objective of this project is to build a segmentation network, it will be trained, validated and implemented in the project called *Follow Me*, so the drone will follow a single hero target using the network pipeline.

---
<!--more-->

[//]: # (Image References)

[image0]: ./misc_images/semantic_segmentation.jpg "Semantic Segmentation"
[image1]: ./misc_images/Fully_Convolutional_Network.jpg "Fully Convolutional Network"
[image2]: ./misc_images/convolutional.jpg "1x1 Convolution"
[image3]: ./misc_images/onelayerFCNmodel.jpg "One Layer FCN Model"
[image4]: ./misc_images/twolayerFCNmodel.jpg "Two Layer FCN Model"
[image5]: ./misc_images/threelayerFCNmodel.jpg "Three Layer FCN Model"
[image6]: ./misc_images/fourlayerFCNmodel.jpg "Four Layer FCN Model"

#### How to run the program with your own code

For the execution of your own code, inside the folder ***/code***

```bash
1.  jupyter notebook model_training.ipynb
```

After opening the notebook, at the top, select the tab called ***Cell*** and select the option ***Run All***.

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

### Semantic Segmentation

Semantic Segmentation is the task of assigning meaning to a part of an object. This can be done at the pixel level where we assign each pixel to a target class such as road, car, pedestrian, sign, or any number of other classes. Semantic segmentation helps us derive valuable information about every pixel in the image rather than just slicing sections into bounding boxes. It's mainly relevant to full scene understanding help with perception of the objects.

![alt text][image0]
###### Trained FCN performing semantic segmentation. Input image (left), ground truth (center) and FCN output (left).


### Network Architecture

The goal of the encoder is to extract features from the image. It does that via a several layers that find simple patterns in the first layer and then gradually learn to understand more and more about complex structures and shapes in the deeper layers. Next, the 1x1 convolution layer implements the same function as a fully connected layer but with the advantage of saving spatial information. The 1x1 convolution layer connects to the decoder, in which the goal of it is to up-scale the output of the encoder such that it's the same size as the original image. In addition, there are skip connections which are connected non-adjacent layers together. The use of skip connections here, for example, the output from the first encoder is connected to the input of the final decoder. The reason is to save information that might be lost during the encoding process, as a result, the network is able to make more precise segmentation decisions. At last, the last decoder stage contains a convolution output layer with softmax activation which makes the final pixel-wise segmentation between the three classes..


A classical convolution network classifies the probability of a determined class is present in the image, unlike a full convolution network (FCN), which preserve the spatial information throughout the entire network outputting by generating a map probabilities corresponding to each pixel of the input image.

I will create a full convolution network following the suggestions of the classes and this notebook, this FCN will consist of three parts:

1. An encoder network that transforms an image input into feature maps.
2. Followed by 1x1 convolution that combines the feature maps (similar to a fully connected layer).
3. Finally a decoder network that upsample the result from the previous layer back to the same dimensions as the input image.

![alt text][image1]
###### Example FCN comprised of Encoder block (left) followed by 1x1 Convolution (center) and Decoder block (right)


```python
def fcn_model(inputs, num_classes):
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_block_1 = encoder_block(inputs, filters = 32, strides = 2)
    encoder_block_2 = encoder_block(encoder_block_1, filters = 64, strides = 2)
    encoder_block_3 = encoder_block(encoder_block_2, filters = 128, strides = 2)
    encoder_block_4 = encoder_block(encoder_block_3, filters = 256, strides = 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv1 = conv2d_batchnorm(encoder_block_4, filters = 256, kernel_size = 1, strides = 1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_block_1 = decoder_block(small_ip_layer = conv1, large_ip_layer = encoder_block_3, filters = 128)
    decoder_block_2 = decoder_block(small_ip_layer = decoder_block_1, large_ip_layer = encoder_block_2, filters = 64)
    decoder_block_3 = decoder_block(small_ip_layer = decoder_block_2, large_ip_layer = encoder_block_1, filters = 32)
    decoder_block_4 = decoder_block(small_ip_layer = decoder_block_3, large_ip_layer = inputs, filters = num_classes)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation = 'softmax', padding = 'same')(decoder_block_4)
```


#### Encoder Block

The fist step in building our network is to add feature detectors capable of transforming the input image into semantic representation. It squeezes the spacial dimensions at the same time that it increases the depth (or number of filters maps), by using a series of convolution layers, forcing the network to find generic representations of the data.


```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```
```python
def encoder_block(input_layer, filters, strides):
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```


#### 1x1 Convolution

In between the encoder block and the decoder block is a 1x1 convolution layer that computes a semantic representation by combining the feature maps from the encoder. It that acts like a fully connected layer where the number of kernels is equivalent to the number of outputs of a fully connected layer.


![alt text][image2]
###### 1x1 Convolution combines feature maps (depth) preserving spacial information


The reason for using a 1x1 convolution instead of a fully connected layer is that it preserves spacial information. Fully connected layers, on the other hand, all its dimensions are flattened into a single vector losing the original spacial structure of the input. One last reason for using convolution: it works for different input sizes while fully connected layers are constrained with a fixed output size.

This is what the code for the 1x1 convolution layer looks like:

```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```


#### Decoder Block

Finally the decoder block upsamples the output from the 1x1 convolution back to the original input format, through the use of a series of transpose convolution layers.

This is what the code for the decoder block looks like:

```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```
```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsampled_layer = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([upsampled_layer, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(output_layer, filters)
    return output_layer
```

I've also made use of *skip connections* allowing the network to use information from multiple resolution scales resulting on more precise segmentation decisions.


### FCN Model

To find the right model size, after completing the main FCN building blocks I've moved into designing the complete network architecture. My strategy to design a reasonable network was to start with a fairly simple (shallow) model and incrementally making it more complex by adding more layers.

I've tried several configurations observing the learning curves patterns and results for each and then settled for the network configuration that was just right for the dataset at hand.


#### One layer FCN model

My fist attempt was to simply use a single encoder block followed by a single decoder block.

![alt text][image3]
###### FCN Model - 1 Encoder layer, 1x1 Convolution and 1 Decoder layer

```python
def fcn_model(inputs, num_classes):
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_block_1 = encoder_block(inputs, filters = 8, strides = 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv1 = conv2d_batchnorm(encoder_block_1, filters = 16, kernel_size = 1, strides = 1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_block_1 = decoder_block(small_ip_layer = conv1, large_ip_layer = inputs, filters = num_classes)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation = 'softmax', padding = 'same')(decoder_block_1)
```


The network performs poorly getting a score of 17%.


#### Two layer FCN model

I’ve then added one more pair of encoder/decoder blocks.

![alt text][image4]
###### FCN Model - 2 Encoder layers, 1x1 Convolution and 2 Decoder layers

```python
def fcn_model(inputs, num_classes):
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_block_1 = encoder_block(inputs, filters = 32, strides = 2)
    encoder_block_2 = encoder_block(encoder_block_1, filters = 64, strides = 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv1 = conv2d_batchnorm(encoder_block_2, filters = 128, kernel_size = 1, strides = 1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_block_1 = decoder_block(small_ip_layer = conv1, large_ip_layer = encoder_block_1, filters = 32)
    decoder_block_2 = decoder_block(small_ip_layer = decoder_block_2, large_ip_layer = inputs, filters = num_classes)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation = 'softmax', padding = 'same')(decoder_block_2)
```


The network performs poorly getting a score of 17%.


#### Three layer FCN model

Once again, for my third attempt I’ve added yet another pair of encoder/decoder blocks.

![alt text][image5]
###### FCN Model - 3 Encoder layers, 1x1 Convolution and 3 Decoder layers


```python
def fcn_model(inputs, num_classes):
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_block_1 = encoder_block(inputs, filters = 32, strides = 2)
    encoder_block_2 = encoder_block(encoder_block_1, filters = 64, strides = 2)
    encoder_block_3 = encoder_block(encoder_block_2, filters = 128, strides = 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv1 = conv2d_batchnorm(encoder_block_3, filters = 128, kernel_size = 1, strides = 1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_block_1 = decoder_block(small_ip_layer = conv1, large_ip_layer = encoder_block_2, filters = 64)
    decoder_block_2 = decoder_block(small_ip_layer = decoder_block_1, large_ip_layer = encoder_block_1, filters = 32)
    decoder_block_3 = decoder_block(small_ip_layer = decoder_block_2, large_ip_layer = inputs, filters = num_classes)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation = 'softmax', padding = 'same')(decoder_block_3)
```


The network performs poorly getting a score of 17%.


#### Four layer FCN model

I've tried a 4th attempt added another pair of encoder/decoder blocks.

![alt text][image5]
###### FCN Model - 4 Encoder layers, 1x1 Convolution and 4 Decoder layers


```python
def fcn_model(inputs, num_classes):
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_block_1 = encoder_block(inputs, filters = 32, strides = 2)
    encoder_block_2 = encoder_block(encoder_block_1, filters = 64, strides = 2)
    encoder_block_3 = encoder_block(encoder_block_2, filters = 128, strides = 2)
    encoder_block_4 = encoder_block(encoder_block_3, filters = 256, strides = 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv1 = conv2d_batchnorm(encoder_block_4, filters = 256, kernel_size = 1, strides = 1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_block_1 = decoder_block(small_ip_layer = conv1, large_ip_layer = encoder_block_3, filters = 128)
    decoder_block_2 = decoder_block(small_ip_layer = decoder_block_1, large_ip_layer = encoder_block_2, filters = 64)
    decoder_block_3 = decoder_block(small_ip_layer = decoder_block_2, large_ip_layer = encoder_block_1, filters = 32)
    decoder_block_4 = decoder_block(small_ip_layer = decoder_block_3, large_ip_layer = inputs, filters = num_classes)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation = 'softmax', padding = 'same')(decoder_block_4)
```


The network performs poorly getting a score of 17%.


### Data Recording

I didn't record any data from simulator, I was able to do all required steps using the provided Training, Validation, and Sample Evaluation Data.

<4table><tbody>
    <tr><th align="center" colspan="3"> Dataset </td></tr>
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

This was the most laborious part of this project. My strategy was to first start with arbitrary values and then later tweak them one-by-one hoping to get to a passing score.

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


The whole idea of SGD is to estimate the error function (and its derivative) by randomly sampling a subset of the training data. This process thus avoids the prohibitive cost of calculating the actual error that requires processing the entire dataset.

I assume that the lower the batch size is, the noisier the training signal is going to be. On the flip side a higher batch size, it will take longer to compute the gradient for each step.

The training signal seems to be stable enough and a batch size of 1 seems to be a suitable value for this dataset. Before moving forward, I've decided to check for larger values. since larger mini-batch sizes can potentially have performance advantage due to GPU speed-up of matrix-matrix products over matrix-vector products. I've then found that the batch size of 16 halved the time to process one epoch from 80s to 40s with a better final score.


### Observations, possible improvements, things used

