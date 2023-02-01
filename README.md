# MNIST-classification-using-Convolutional-Neural-Network

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits used for training various image processing systems in deep learning. The dataset was imported directly from torchvision - a Pytorch library collection of datasets with some insights from Mike X Cohen, (sincxpress.com).

## Objective

Create a Convolutional Neural Network (CNN) model to classify MNIST digits.

## Scope

- Deep learning
- PyTorch
- CNN


## Methodology

The data and labels (train and test) have been imported directly from torchvision. The data is reshaped to 2D and converted to tensor as required by pytorch. Tensors are conveted to PyTorch datasets and then translated to dataloader objects. Batch size = 32.

The CNN architecture consists of:

- Image (1x28x28)
- Convolution layer, conv1 (10x26x26)
- Max Pool layer, maxpool1 (10x13x13)
- Convolution layer, conv2 (20x11x11)
- Max Pool layer, maxpool2 (20x5x5)
- Fully connected layer, fc1 (1x50)
- Output (1x10)

                                            CNN Architecture
![CNN architecture](https://user-images.githubusercontent.com/78843321/215979934-5219fcde-52e2-4c51-98cd-f6e96382355a.jpg)



First convolution layer, conv1, composes of 10 feature maps/ filter kernels to be created and learned by back propagation. conv1 removes one pixel from the top, bottom, left and right of the original image (28x28 to 26x26). 

conv1 parmeters: kernel_size = 5, stride = 1, padding = 1

Nh = floor((Mh + 2p - k)/Sh) + 1, where:
  Nh = no of pixels in current layer
  Mh = no of pixels in previous layer
  sh = stride
  p = padding
  k = no of pixels in kernel
  h (subscript) = height

A two-by-two max pooling, maxpol1, is applied to the output of conv1, giving a 13x13 image, hence reducing dimensionality and increasing receptive field size.conv2 contains 20 feature maps, more so, shaving off one pixel from the boundary of previous output. maxpool2 follows thereafter. 
  
conv2 parmeters: no of channels = 10, no of feature maps = 20, kernel_size = 5, stride = 1, padding = 1

Consequently, a fully connected layer, fc1 is applied. A vector of 50 units is obtained by linearizing the max pool output and then through a standard feed forward network to give output (1x10). 

The architecture ensures simultaneous reduction of image resoluton and increase in layer width with a total of 30,840 trainable parameters.

                            Model Summary
<img width="528" alt="Summary" src="https://user-images.githubusercontent.com/78843321/215980263-557c85cc-43df-46a5-bf83-de8068b7cc37.PNG">



The [model](https://github.com/gregoryoffodum/MNIST-classification-using-Convolutional-Neural-Network/blob/main/Classification%20of%20MNIST%20digits%20using%20CNN.ipynb) is run with 10 epochs iterating over batches with forward propagation, loss computation, back propagation and accuracy calculated.


<img width="468" alt="Loss" src="https://user-images.githubusercontent.com/78843321/215980671-24e33e24-1162-45a8-870f-a8aa7c4d2cf5.PNG">


<img width="448" alt="Accuracy" src="https://user-images.githubusercontent.com/78843321/215980690-78a2c0c5-609a-45f8-a1cf-db8b98272ece.PNG">

Recommendation: Although the model performs sufficiently well, a few hyperparameters can be tuned - no of epochs; convolution; pool layers, to compare loss and accuracy.




