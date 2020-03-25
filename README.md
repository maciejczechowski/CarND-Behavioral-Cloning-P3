# **Behavioral Cloning** 
**Behavioral Cloning Project**

This projects show how a convolution neural network can be used to clone driving behaviour and autonomously drive a car.

It uses the [Udacity driving simulator](https://github.com/udacity/self-driving-car-sim).

I have used my local machine, using builds for MacOS that can be find [here](https://github.com/endymioncheung/CarND-MacCatalinaSimulator).


The project uses Keras for creating neural network. It will use PlaidML backend if that is available, so training can also be done on AMD GPUs.

---
### Project files

The project consists of the following files:
* `model.py` Script to create and train the model
* `drive.py` Script for driving the car in autonomous mode in Udacity simulator
* `model.h5` Trained model
* `readme.md` This file - results

#### Training the model 
Training the model can be done by running the `model.py` script. You should put all the training images from the simulator (together with csv file) into the `Records` directory.

#### Driving the car
After training model, the car can be driven by using `drive.py` script. 
```sh
python drive.py model.h5
```

Script is currently set to drive @25MPH. 

### Model Architecture and Training Strategy

#### Network architecture
he model is based on the network described in the paper *End to End Learning for Self-Driving Cars* by NVIDIA (Available [here](https://arxiv.org/pdf/1604.07316v1.pdf))

That paper describes the case exactly as the one that had to be implemented here (except that their implementation used real-world images), so it was a great starting point.

It consists of the following layers:

Layer | Description
---| ---
Lambda | Normalization
Crop2D | Crops image to region of interest
Conv2D | 24 Filters, filter size: 5, stride: 2x2, RELU
Conv2D | 36 Filters, filter size: 5, stride: 2x2, RELU
Conv2D | 48 Filters, filter size: 5, stride: 2x2, RELU
Conv2D | 64 Filters, filter size: 3
Conv2D | 64 Filters, filter size: 3
Flatten |
Dropout | 50%
Dense | 50 outputs
Dropout | 50%
Dense  | 10 outputs
Dropout | 50%
Dense | 1 output

To introduce nonlinearity, all convolutional layers include RELU activation. I've tried to also include it on Dense layers, and while it improved accuracy numerically, the actual driving behaviour was worse.
The model had a tendency to overfit, therefore a dropout of 50% is included before each Dense layer.

The input data is normalized using Lambda layer (`x = x / 255.0 - 0.5`). To reduce the influence of the objects outside the road, a Crop2D layer is included.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### Training and validation data

Training and validation data were obtained by using Simulator. Validation data extracted, with a ratio of 20%.

The data consisted of the following:
* Udacity test data
* Driving track one
* Driving track one reversed
* Driving track two
* Driving track two reversed
* Driving the recovery scenarios on both track one and two

The data was further augmented (described below)

#### Data augmentation
Although I gathered a lot of test images (including reversed driving), the model had a tendency to overfit and generally fail to present a good driving skills.
I decided to further augment the data:
1. Each image is flipped and added to training set (with reversed steering angle). This allow to better generalize for equal left-right turning.

2. Left and right camera images are added with a 0.2 modifier to the steering angle. Inclusion of this images had a big impact on the final performance. It greately improved hadnling curves, but decreased the quality of driving straight. The car starts to do a small left-right turns on straights segments. I have finally decided to include this data, as it was highly beneficial for handling track 2.

#### Image preparation
Before passing image to model it is converted to YUV color space to initially decouple the data. R, G, B channels contain a lot of entangled information, and it seems that separate luma and chroma channels help network to make better decisions.
I have experimented with other transormations like:
* different color spaces (HSL, HSV, LAB)
* histogram equalization
* luma-negatives (negating luma channel, leaving chroma)
* providing 2-channel image with gray data and Canny transform
 
These did not have a significant effect of final result. They increased processing time a lot, so I have decided not to include them. 

Images are further cropped by network to focus just on road not on the side objects.

#### Training the model
Because of the size of the input data, model is trained using Generators which feed the process with batches. 
Data is shuffled prior to batching.

The final model was trained using 10 Epochs.
Eearly stopping and checkpoinst were included to cancel non-promising experiments.

### The results
The final model can effectively drive the both tracks without falling off the road. Track two can be driven with a speed of 25MPH, while track one can use maximum value (which is just over 30 on my simulator build).

The effect can be seen here: [Track one](run1.mp4), [Track two](run2.mp4)     
 
#### Further work
While this is was not checked in the project I think it is very worth analyzing how much using Computer Vision techniques to preprocess the data would improve the network.
Promising idea is to use warp-transform to convert image to birds view, although as with many other techniques in deep learning, the final result is unknown without trying.