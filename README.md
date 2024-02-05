# Fashion GAN: Generative Adversarial Neural Network for Fashion Images

## Introduction
This project implements a Generative Adversarial Neural Network (GAN) using TensorFlow to generate fashion images. The GAN consists of a generator and a discriminator, trained in an adversarial manner to create images resembling the Fashion MNIST dataset.

## Required Dependencies
- Tensorflow
- Matplotlib

## Data
The data used in this project was Fashion MNIST dataset.

## Approaches
- Data was loaded, then preprocessed by scaling the images and transforming them to enhance training performance. The images are scaled to a range of [0, 1], and a TensorFlow dataset pipeline is established for efficient processing. 

- For the neural network architecture, the generator is responsible for creating synthetic images. It is a deep neural network with upsampling and convolutional layers.The discriminator distinguishes between real and generated images. It is a convolutional neural network with dropout layers.

- The training loop consists of training both the generator and discriminator in an adversarial manner. The losses are calculated using binary cross-entropy, and the Adam optimizer is utilized.

- A subclassed model, `FashionGAN`, is created to encapsulate the generator and discriminator. It includes a custom training step method to train both components simultaneously.

- A custom callback, `ModelMonitor`, is implemented to save generated images during training at each epoch. This callback is used to monitor the progress of the GAN.

- Loss curves for both the generator and discriminator are plotted to visualize the training progress in order to review the peformance of the model

- The trained generator is loaded, and sample images are generated and displayed for visual inspection.

- The trained generator and discriminator models are saved for future use.

## Conclusion
This project demonstrates the implementation of a GAN for generating fashion images using TensorFlow. The training loop, subclassed model, and monitoring callback provide a comprehensive overview of the GAN training process. 
Further improvements and fine-tuning can be explored to enhance the quality of generated images. I'll appreciate comments and contributions to enhance the performance of this model.
