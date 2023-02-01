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

<mage>

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

<image>

The model <ipynb> is run with 10 epochs iterating over batches with forward propagation, loss computation, back propagation and accuracy calculated.




## Recommendations
 Comaparing accuracy  Comment out the "conv2" layers in the mode definition. What else
   needs to be changed in the code for this to work with one convolution layer? Once you get it working, how does the
   accuracy compare between one and two conv layers? (hint: perhaps try adding some more training epochs)

   Your observation here is actually the main reason why MNIST isn't very useful for evaluating developments in DL:
   MNIST is way too easy! Very simple models do very well, so there is little room for variability. In fact, we'll
   stop using MNIST pretty soon...

   Final note about MNIST: You probably won't get much higher than 98% with this small dataset. These kinds of CNNs 
   can get >99% test-accuracy with the full dataset (60k samples instead of 18k).




