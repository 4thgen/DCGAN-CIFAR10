# DCGAN-CIFAR10
A implementation of DCGAN (Deep Convolutional Generative Adversarial Networks) for CIFAR10 image 
this code is base on hwalsuklee/tensorflow-generative-model-collections (https://github.com/hwalsuklee/tensorflow-generative-model-collections), including some modification.
very thanks to @hwalsuklee

# Background
I tried to make a model for GAN with cifar10, it was different with mnist image set. so the result was not expected enough. maybe color dimension is more complicated factor for generation. 
A model for cifar10 made by own routine, according to many other implementatons, websites and blogs. and adding some tips.

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
1. using reaky relu in descriminator all layer except last one.
2. using different learning rate for dicriminator(1e-3) and generator(1e-4)
3. using batch normalization with axis=1
4. using tanh output in generator
5. not using batch normalization at first and last layer
6. train generator twice

# Result sample
