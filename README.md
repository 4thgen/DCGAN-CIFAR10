# DCGAN-CIFAR10
A implementation of DCGAN (Deep Convolutional Generative Adversarial Networks) for CIFAR10 image. 
this code is base on hwalsuklee/tensorflow-generative-model-collections (https://github.com/hwalsuklee/tensorflow-generative-model-collections), including some modification.
very thanks to @hwalsuklee

# Background
I tried to make a model for GAN with cifar10, it was different with MNIST image set. so the result was not good enough less than expected. maybe color dimension is more complicated factor for generation. 
A deep convolutional model for cifar10 made by own routine, according to many other implementatons, websites and blogs. and adding some significant tips.

# Coding Condition
- python 3.5.2
- tensorflow 1.3.0
- cifa10 python image set (https://www.cs.toronto.edu/~kriz/cifar.html)

# Building Model
## Discriminator
- D: (100, 32, 32, 3) // input image shape with batch size 100
- D: (100, 16, 16, 64) // after conv2d 5x5 stride 2
- D: (100, 8, 8, 128) // after conv2d 5x5 stride 2
- D: (100, 4, 4, 256) // after conv2d 5x5 stride 2
- D: (100, 2, 2, 512) // after conv2d 5x5 stride 2
- D: (100, 2048) // flatten
- D: (100, 1) // sigmoid out

## Generator
- G: (100, 100) // noise vector with batch size 100
- G: (100, 2048) // after linear
- G: (100, 2, 2, 512) // reshape
- G: (100, 4, 4, 256) // after deconv2d 5x5 stride 2
- G: (100, 8, 8, 128) // after deconv2d 5x5 stride 2
- G: (100, 16, 16, 64) // after deconv2d 5x5 stride 2
- G: (100, 32, 32, 3) // after deconv2d 5x5 stride 2 and tanh out

# TIPs
1. using leaky relu in descriminator all layer except last one.
2. using different learning rate for dicriminator(1e-3) and generator(1e-4)
3. using batch normalization with axis="channel index"
4. using tanh output in generator
5. not using batch normalization at first and last layer
6. train discriminator once and generator twice

# Result sample
generating image result at epoch 100
![epoch100](https://github.com/4thgen/DCGAN-CIFAR10/blob/master/result/GAN_epoch100_test_all_classes.png)

epoch result image movies (epoch 0~500)  
[![Youtude:DCGAN Epoch 000~499 Results](https://img.youtube.com/vi/FQJfQIec70E/0.jpg)](https://www.youtube.com/watch?v=FQJfQIec70E)

evolving generator (epoch 0~500)  
[![Evolution of DCGAN Generator (faster)](https://img.youtube.com/vi/_REVVMWa9aE/0.jpg)](https://www.youtube.com/watch?v=_REVVMWa9aE)
