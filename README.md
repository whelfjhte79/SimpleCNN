# SimpleCNN: A Simple Convolutional Neural Network Library
Convolutional Neural Network (CNN) Library in C++
SimpleCNN is a C++ library for building and training convolutional neural networks (CNNs) with ease. 
It provides a simple interface to create and train CNN models for various tasks, such as image classification.

## Getting Started

### Prerequisites

- C++ compiler
- OpenCV library (Optional); Only for reading data.

### Installation

1. Clone the SimpleCNN repository to your local machine:

   ```shell
   git clone https://github.com/yourusername/SimpleCNN.git


Build the library using your preferred C++ build system
Include the necessary headers and link against the SimpleCNN library in your C++ project.

## Example Usage
Here's a simple example of how to use SimpleCNN to create and train a CNN model for image classification:


```
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include "SimpleCNN.h"

using namespace cv;
using namespace std;
using namespace img_read;
using namespace cnn;

int main(void) {
    // Initialize data and labels
    Directory* dirent = new Directory("path/to/image/folder", img_read::FILENAME_EXTENSION::JPG);
    FileRead* label = new FileRead("path/to/label/file.csv");
    
    // Create a CNN model
    CNN* cnn = new CNN(dirent->getImageSet(), label->getLabel());
    
    // Add layers to the model
    cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::ReLU));
	  cnn->add(new Pooling(POOLING::Max));
	  cnn->add(new Padding(1));
	  cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::ReLU));
 	  cnn->add(new Pooling(POOLING::Max));
	  cnn->add(new Padding(1));
	  cnn->add(new Pooling(POOLING::Max));
	  cnn->add(new Flatten());
	  cnn->add(new FullyConnected(256, ACTIVATION::ReLU));
	  cnn->add(new FullyConnected(128, ACTIVATION::ReLU));
	  cnn->add(new FullyConnected(64, ACTIVATION::ReLU));
    // Add more layers as needed
    
    // Compile the model
    cnn->compile(OPTIMIZER::Mini_SGD, LOSS::CategoricalCrossentropy);
    
    // Train the model
    cnn->fit(epochs, batch_size);
    
    // Make predictions
    cnn->predict(image, true_label);

    cnn->accuracy();
    
    return 0;
}
```
Please replace "path/to/image/folder" and "path/to/label/file.csv" with the actual paths to your image dataset and label file.


