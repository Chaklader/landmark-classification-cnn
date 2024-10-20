Course Overview
–––––––––––––––

1. Introduction to CNNs and basic concepts

* Use Multi-Layer Perceptrons (MLPs) for image classification
* Understand the limitations of MLPs for images and how CNNs overcome them
* Learn basic concepts of CNNs and what makes them so powerful for image tasks

2. CNNs in more depth

* Learn all the basic layers that make up a CNN
* Put all the basic layers together to build a CNN from scratch
* Classify images using CNNs
* Use various methods to improve CNN performance
* Export models for production

3. Transfer learning

* Understand key innovative CNN architectures
* Implement transfer learning using a pre-trained network to classify different sets of images
* Fine-tune a pre-trained network on a new dataset

4. Autoencoders

* Explain the functionality of autoencoders for data compression, image denoising, and dimensionality reduction
* Build a simple autoencoder out of linear layers to perform anomaly detection
* Build CNN autoencoders to perform anomaly detection and image denoising

5. Object detection and segmentation

* Describe object detection, object localization, and image segmentation.
* Train and evaluate a one-stage object detection model to detect multiple objects in an image.
* Train and evaluate a semantic segmentation model to classify every pixel of an image.


## Flattening

Suppose we want to use an MLP to classify our image. The problem is, the network takes a 1d array as input, while we have 
images that are 28x28 matrices. The obvious solution is to flatten the matrix, i.e., to stack all the rows of the matrix 
in one long 1D vector, as in the image below.



## Loss Function

The loss function quantifies how far we are from the ideal state where the network does not make any mistakes and has 
perfect confidence in its answers.

Depending on the task and other considerations we might pick different loss functions. For image classification the most 
typical loss function is the Categorical Cross-Entropy (CCE) loss, defined as:


Definition
The CCE loss is defined as:
$$
\text{CCE} = -\sum_{i=1}^{n_\text{classes}} y_i \log(\hat{p}_i)
$$

Where:

$n_\text{classes}$ is the number of classes (10 for MNIST digits)
<br>
$y_i$ is the true label (ground truth) as a one-hot encoded vector
<br>
$\hat{p}_i$ is the predicted probability for class $i$

where:

1. The sum is taken over the classes (10 in our case)
2. yi is the ground truth, i.e., a one-hot encoded vector(opens in a new tab) of length 10
3. pi is the probability predicted by the network


# Loss Function: Categorical Cross-Entropy (CCE)

In our MNIST digit classification task, we use the Categorical Cross-Entropy (CCE) loss function. This choice is typical for multi-class classification problems where each sample belongs to exactly one class.

## Definition

The CCE loss is defined as:

$$
\text{CCE} = -\sum_{i=1}^{n_\text{classes}} y_i \log(\hat{p}_i)
$$

Where:
- $n_\text{classes}$ is the number of classes (10 for MNIST digits)
- $y_i$ is the true label (ground truth) as a one-hot encoded vector
- $\hat{p}_i$ is the predicted probability for class $i$


<br>

## Interpretation

1. The loss quantifies the difference between the predicted probability distribution and the true distribution (one-hot encoded ground truth).
2. A perfect prediction would result in a loss of 0, while incorrect predictions increase the loss value.
3. The logarithm heavily penalizes confident misclassifications, encouraging the model to be cautious with its predictions.

## Implementation in PyTorch

In PyTorch, we use `nn.CrossEntropyLoss()`, which combines a softmax activation and the CCE loss in one operation, improving numerical stability.

```textmate
criterion = nn.CrossEntropyLoss()
```

This loss function is well-suited for our MNIST task because:
1. It naturally handles multi-class problems.
2. It encourages the model to output well-calibrated probabilities.
3. It's differentiable, allowing for effective backpropagation during training.

By minimizing this loss during training, we push our model to make increasingly accurate predictions on the digit classification task.



<br>

# Sequential Neural Networks and Alternatives

## Sequential Neural Networks

A sequential neural network is a linear stack of layers where the output of one layer becomes the input to the next layer. This is the simplest and most common type of neural network architecture.

### Characteristics:
1. Layers are arranged in a straight line.
2. Data flows through the network in a single, forward direction.
3. Easy to conceptualize and implement.

### Example in PyTorch:


```textmate
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

## Non-Sequential Neural Networks

It is absolutely possible, and often necessary, to have neural networks that are not sequential. These are sometimes called non-sequential or non-linear neural networks.

### Types of Non-Sequential Architectures:

1. **Branching Networks**: Where the data flow splits and merges.
2. **Residual Networks (ResNets)**: Include skip connections that bypass one or more layers.
3. **Recurrent Neural Networks (RNNs)**: Have feedback loops, allowing information to persist.
4. **Graph Neural Networks**: Operate on graph-structured data.
5. **Multi-Input or Multi-Output Networks**: Accept or produce multiple tensors.

### Example of a Non-Sequential Network in PyTorch:

```textmate
class NonSequentialNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

## When to Use Non-Sequential Networks

1. **Complex Data Dependencies**: When your data has complex relationships that can't be captured by a simple linear flow.
2. **Performance Improvement**: Techniques like skip connections in ResNets can help with training very deep networks.
3. **Task-Specific Requirements**: Some tasks inherently require non-sequential processing, like image segmentation or machine translation.
4. **Multi-Modal Data**: When working with multiple types of input data simultaneously.

Non-sequential architectures offer more flexibility and can often capture more complex patterns in data, but they can also be more challenging to design and train.


ReLU Activation Function


The purpose of an activation function is to scale the outputs of a layer so that they are consistent, small values. Much 
like normalizing input values, this step ensures that our model trains efficiently!

A ReLU activation function stands for "Rectified Linear Unit" and is one of the most commonly used activation functions 
for hidden layers. It is an activation function, simply defined as the positive part of the input, x. So, for an input 
image with any negative pixel values, this would turn all those values to 0, black. You may hear this referred to as 
"clipping" the values to zero; meaning that is the lower bound.




## Design of an MLP (Multi-Layer Perceptron)


When designing an MLP you have a lot of different possibilities, and it is sometimes hard to know where to start. Unfortunately 
there are no strict rules, and experimentation is key. However, here are some guidelines to help you get started with an initial 
architecture that makes sense, from which you can start experimenting.

The number of inputs input_dim is fixed (in the case of MNIST images for example it is 28 x 28 = 784), so the first layer 
must be a fully-connected layer (Linear in PyTorch) with input_dim as input dimension.

Also the number of outputs is fixed (it is determined by the desired outputs). For a classification problem it is the number 
of classes n_classes, and for a regression problem it is 1 (or the number of continuous values to predict). So the output 
layer is a Linear layer with n_classes (in case of classification).

What remains to be decided is the number of hidden layers and their size. Typically you want to start from only one hidden 
layer, with a number of neurons between the input and the output dimension. Sometimes adding a second hidden layer helps, 
and in rare cases you might need to add more than one. But one is a good starting point.

As for the number of neurons in the hidden layers, a decent starting point is usually the mean between the input and the 
output dimension. Then you can start experimenting with increasing or decreasing, and observe the performances you get. 
If you see overfitting(opens in a new tab), start by adding regularization (dropout(opens in a new tab) and weight decay) 
instead of decreasing the number of neurons, and see if that fixes it. A larger network with a bit of drop-out learns 
multiple ways to arrive to the right answer, so it is more robust than a smaller network without dropout. If this doesn't 
address the overfitting, then decrease the number of neurons. If you see underfitting(opens in a new tab), add more neurons. 
You can start by approximating up to the closest power of 2. Keep in mind that the number of neurons also depends on the 
size of your training dataset: a larger network is more powerful but it needs more data to avoid overfitting.


```textmate
import torch
import torch.nn as nn

class MyModel(nn.Module):

  def __init__(self):

    super().__init__()

    # Create layers. In this case just a standard MLP
    self.model = nn.Sequential(
      # Input layer. The input is obviously 784. For
      # the output (which is the input to the hidden layer)
      # we take the mean between network input and output:
      # (784 + 10) / 2 = 397 which we round to 400
      nn.Linear(784, 400),
      nn.Dropout(0.5),  # Combat overfitting
      nn.ReLU(),
      # Hidden layer
      nn.Linear(400, 400),
      nn.Dropout(0.5),  # Combat overfitting
      nn.ReLU(),
      # Output layer, must receive the output of the
      # hidden layer and return the number of classes
      nn.Linear(400, 10)
    )

  def forward(self, x):

    # nn.Sequential will call the layers 
    # in the order they have been inserted
    return self.model(x)
```

<br>

# Training a Neural Network in PyTorch

## 1. Loss Function
- Specifies what the optimizer will minimize
- For classification tasks, use Cross Entropy Loss
- In PyTorch: `nn.CrossEntropyLoss()`
- Combines softmax and negative log likelihood loss

## 2. Optimizer
- Changes network parameters to minimize loss
- Specify which parameters to modify: typically `model.parameters()`
- Set learning rate and other parameters (e.g., weight decay)
- Common optimizers: SGD, Adam
- Example: `torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)`

## 3. Training Loop
1. Set number of epochs
2. Set model to training mode: `model.train()`
3. For each epoch:
   a. Loop through batches in training data loader
   b. Clear gradients: `optimizer.zero_grad()`
   c. Forward pass: `output = model(data)`
   d. Calculate loss: `loss = criterion(output, target)`
   e. Backward pass: `loss.backward()`
   f. Optimize: `optimizer.step()`
   g. Update running loss

## 4. Important Notes
- CrossEntropyLoss applies softmax internally
- Model output should be unnormalized class scores, not probabilities
- Alternative: Use `F.log_softmax()` in model's forward method and `nn.NLLLoss()`
- Consider adding softmax to model's forward method for inference after training

## 5. Monitoring Progress
- Use tqdm for progress bars during training
- Calculate and print average loss per epoch

## 6. Validation
- Perform validation after each training epoch
- Set model to evaluation mode for validation: `model.eval()`

Remember: The goal is to minimize the loss function by adjusting the model's parameters through multiple epochs of training.



## Model Validation


Validation Set: Takeaways

We create a validation set to:

1. Measure how well a model generalizes, during training
2. Tell us when to stop training a model; when the validation loss stops decreasing (and especially when the validation loss 
starts increasing and the training loss is still decreasing) we should stop training. It is actually more practical to train 
for a longer time than we should, but save the weights of the model at the minimum of the validation set, and then just throw 
away the epochs after the validation loss minimum.


<br>

![image info](images/train.png)

<br>


### Validation Loop

Once we have performed an epoch of training we can evaluate the model against the validation set to see how it is doing. 
This is accomplished with the validation loop:


```textmate
# Tell pytorch to stop computing gradients for the moment
# by using the torch.no_grad() context manager
with torch.no_grad():

  # set the model to evaluation mode
  # This changes the behavior of some layers like
  # Dropout with respect to their behavior during
  # training
  model.eval()

  # Keep track of the validation loss
  valid_loss = 0.0

  # Loop over the batches of validation data
  # (here we have removed the progress bar display that is
  # accomplished using tqdm in the video, for clarity)
  for batch_idx, (data, target) in enumerate(valid_dataloader):

    # 1. forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)

    # 2. calculate the loss
    loss_value = criterion(output, target)

    # Calculate average validation loss
    valid_loss = valid_loss + (
      (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
    )

  # Print the losses 
  print(f"Epoch {epoch+1}: training loss {train_loss:.5f}, valid loss {valid_loss:.5f}")

```

It is usually a good idea to wrap the validation loop in a function so you can return the validation loss for each epoch, 
and you can check whether the current epoch has the lowest loss so far. In that case, you save the weights of the model. 
We will see in one of the future exercises how to do that.


### The Test Loop

The test loop is identical to the validation loop, but we of course iterate over the test dataloader instead of the validation 
dataloader.


To visually summarize what we have discussed so far, here is a typical workflow for an image classification task:

<br>

![image info](images/visualize.png)

<br>


Typical Workflow for an image classification task:

1. DataLoaders with preprocessing: The process starts with data loaders that include preprocessing steps for the images.

2. Visualize: The preprocessed data is visualized to understand the input.

3. Model Definition: A neural network model is defined based on the task requirements.

4. Define Loss: The loss function is defined, typically using cross-entropy for classification tasks.

5. Define Optimizer: An optimizer is chosen, usually Adam or SGD (Stochastic Gradient Descent).

6. Train: The model is trained using the defined loss and optimizer. Training continues until the validation loss starts 
   increasing, indicating potential overfitting.

7. Experiment: This is an iterative process where the model design and parameters are adjusted based on the training results.

8. Select Best Model: After multiple experiments, the best performing model is selected.

9. Evaluate on Test Data: The chosen model is then evaluated on a separate test dataset to assess its generalization 
   performance.


This workflow is cyclical, with the "Experiment" step potentially leading back to adjusting the model definition, loss 
function, or optimizer. The goal is to iterate and improve until a satisfactory model is achieved, which is then finally 
tested on unseen data.



### Classifier Performance (MLP and CNN)


The MNIST dataset is very clean and is one of the few datasets where MLPs and Convolutional Neural Networks perform at a 
similar level of accuracy. However, all of the top-scoring architectures for MNIST(opens in a new tab) are CNNs (although 
their performance difference compared to MLPs is small).

In most cases, CNNs are vastly superior to MLPs, both in terms of accuracy and in terms of network size when dealing with 
images.

As we will see, the main reason for the superiority of CNNs is that MLPs have to flatten the input image, and therefore 
initially ignore most of the spatial information, which is very important in an image. Also, among other things, they are 
not invariant for translation. This means that they need to learn to recognize the same image all over again if we translate 
even slightly the objects in it.

CNNs instead don't need to flatten the image and can therefore immediately exploit the spatial structure. As we will see, 
through the use of convolution and pooling they also have approximate translation invariance, making them much better 
choices for image tasks.



# Multilayer Perceptrons (MLPs) vs Convolutional Neural Networks (CNNs)

## Multilayer Perceptrons (MLPs)

1. Structure:
   - Consist of fully connected layers
   - Each neuron connected to every neuron in the previous and next layer

2. Input handling:
   - Typically work with 1D input (flattened data)
   - Lose spatial information when dealing with 2D or 3D data

3. Parameter efficiency:
   - Large number of parameters, especially for high-dimensional input
   - Prone to overfitting on image data

4. Feature extraction:
   - Learn global patterns
   - No built-in understanding of spatial hierarchies

5. Invariance:
   - No built-in translation invariance

6. Applications:
   - Suitable for tabular data, simple pattern recognition
   - Less effective for complex spatial data like images

## Convolutional Neural Networks (CNNs)

1. Structure:
   - Consist of convolutional layers, pooling layers, and fully connected layers
   - Local connectivity in convolutional layers

2. Input handling:
   - Designed to work with grid-like topologies (e.g., 2D images, 3D videos)
   - Preserve spatial information

3. Parameter efficiency:
   - Fewer parameters due to parameter sharing in convolutional layers
   - Better at generalizing for image-like data

4. Feature extraction:
   - Learn local patterns and build up to more complex features
   - Hierarchical feature learning

5. Invariance:
   - Built-in translation invariance due to convolutional operations

6. Applications:
   - Excellent for image-related tasks (classification, segmentation, detection)
   - Effective for any data with spatial or temporal structure

## Key Differences

1. Connectivity: MLPs use full connectivity, while CNNs use local connectivity.
2. Spatial awareness: CNNs preserve and utilize spatial information, MLPs do not.
3. Parameter efficiency: CNNs are more parameter-efficient for image-like data.
4. Feature learning: CNNs learn hierarchical features, MLPs learn global patterns.
5. Invariance: CNNs have built-in translation invariance, MLPs do not.

## When to Use

- Use MLPs for:
  - Tabular data
  - Simple pattern recognition tasks
  - When input features don't have spatial or temporal relationships

- Use CNNs for:
  - Image processing tasks
  - Data with spatial or temporal structure
  - When you need to preserve and utilize spatial information



### Locally-Connected Layers


Convolutional Neural Networks are characterized by locally-connected layers, i.e., layers where neurons are connected to 
only a limited numbers of input pixels (instead of all the pixels like in fully-connected layers). Moreover, these neurons 
share their weights, which drastically reduces the number of parameters in the network with respect to MLPs. The idea behind 
this weight-sharing is that the network should be able to recognize the same pattern anywhere in the image.


### The Convolution Operation

CNNs can preserve spatial information, and the key to this capability is called the Convolution operation: it makes the 
network capable of extracting spatial and color patterns that characterize different objects.

CNNs use filters (also known as "kernels") to "extract" the features of an object (for example, edges). By using multiple 
different filters the network can learn to recognize complex shapes and objects.



### Image Filters

Image filters are a traditional concept in computer vision. They are small matrices that can be used to transform the input 
image in specific ways, for example, highlighting edges of objects in the image.

An edge of an object is a place in an image where the intensity changes significantly.

To detect these changes in intensity within an image, you can create specific image filters that look at groups of pixels 
and react to alternating patterns of dark/light pixels. These filters produce an output that shows edges of objects and 
differing textures.

We will see that CNNs can learn the most useful filters needed to, for example, classify an image. But before doing that, 
let's look at some specific filters that we can create manually to understand how they work.



# Kernel Convolution in Convolutional Neural Networks

Kernel convolution is a fundamental operation in Convolutional Neural Networks (CNNs) that enables the network to detect features in images.

## How Convolution Works

1. **Kernel (Filter)**: A small matrix of weights, typically 3x3 or 5x5.
2. **Sliding Window**: The kernel slides over the input image.
3. **Element-wise Multiplication**: At each position, the kernel is multiplied element-wise with the overlapping image patch.
4. **Summation**: The products are summed to produce a single output value.

## Example (based on the image):

- Kernel (3x3):
  ```
  [ 0  -1   0]
  [-1   4  -1]
  [ 0  -1   0]
  ```
- Image Patch (3x3):
  ```
  [140 120 120]
  [225 220 205]
  [255 250 230]
  ```
- Calculation:
  (0*140) + (-1*120) + (0*120) +
  (-1*225) + (4*220) + (-1*205) +
  (0*255) + (-1*250) + (0*230) = 60

The result (60) becomes the value in the output feature map for this position.

## Edge Handling

When the kernel reaches the edges or corners of the image, special handling is required:

1. **Padding**: Add a border of 0's (black pixels) around the image.
   - Maintains the original image size in the output.
   - Common in practice.

2. **Cropping**: Skip pixels that would require values from beyond the edge.
   - Results in a smaller output image.
   - Loses information at the edges.

3. **Extension**: Extend border pixels as needed.
   - Corner pixels: Extended in 90° wedges.
   - Edge pixels: Extended in lines.
   - Maintains original image size without introducing new values.

## Importance in CNNs

- Convolution allows CNNs to detect local patterns and features.
- Different kernels can detect various features (edges, textures, etc.).
- As the network deepens, it can recognize more complex patterns.
- The kernels' weights are learned during training, allowing the network to automatically discover important features for the task at hand.


### Edge Handling

Kernel convolution relies on centering a pixel and looking at its surrounding neighbors. So, what do you do if there are 
no surrounding pixels like on an image corner or edge? Well, there are a number of ways to process the edges, which are 
listed below. It’s most common to use padding, cropping, or extension. In extension, the border pixels of an image are 
copied and extended far enough to result in a filtered image of the same size as the original image.

1. Padding - The image is padded with a border of 0's, black pixels.

2. Cropping - Any pixel in the output image which would require values from beyond the edge is skipped. This method can result 
in the output image being smaller then the input image, with the edges having been cropped.

3. Extension - The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution. 
Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.

<br>

![image info](images/kernel_convolution.png)

<br>



#### Question 1:

Of the four kernels pictured above, which would be best for finding and enhancing horizontal edges and lines in an image?
If needed, use the bottom images and go through the math of applying the filters to those images. Which filter gives you a 
horizontal line in the output?

The correct answer is d. Let's examine why:

Kernel d:
```
-1 -2 -1
 0  0  0
 1  2  1
```

This kernel is designed to detect horizontal edges and lines because:

1. It has a strong horizontal structure.
2. The top row is negative, the middle row is zero, and the bottom row is positive.
3. When this kernel slides over a horizontal edge (dark above, light below), it will produce a strong positive response.

When applied to the horizontal edge image (the bottom-left image in the first figure), this kernel would highlight the horizontal line between the black and white regions.


####  Question 2:
Of the four kernels pictured above, which would be best for finding and enhancing vertical edges and lines in an image?
If needed, use the example images as in the previous question.

The correct answer is b. Let's examine why:

Kernel b:
```
-1  0  1
-2  0  2
-1  0  1
```

This kernel is designed to detect vertical edges and lines because:

1. It has a strong vertical structure.
2. The left column is negative, the middle column is zero, and the right column is positive.
3. When this kernel slides over a vertical edge (dark on the left, light on the right), it will produce a strong positive response.

When applied to the vertical edge image (the bottom-right image in the first figure), this kernel would highlight the vertical line between the black and white regions.

Key points to remember:
1. Edge detection kernels typically have opposite signs on opposite sides of the kernel.
2. The direction of the edge detection (horizontal or vertical) is perpendicular to the direction of the sign change in the kernel.
3. These kernels are examples of Sobel filters, which are commonly used for edge detection in image processing and computer vision tasks.

By understanding how these kernels operate, you can see how convolutional neural networks can automatically learn to detect various features in images, starting from simple edges and progressing to more complex patterns in deeper layers.

<br>

![image info](images/horizontal_vertical_edge.png)

<br>

The two filters we have looked at above are called Sobel filters. They are well-known filters used to isolate edges.


# Convolutional Neural Networks (CNNs) - Advanced Concepts

## 1. Kernel Convolution

Kernel convolution is the fundamental operation in CNNs that enables feature detection in images.

### Process:
1. A small matrix (kernel) slides over the input image
2. Element-wise multiplication of kernel with image patch
3. Sum of products becomes the output for that position

### Edge Handling:
- Padding: Add border of zeros
- Cropping: Skip edge pixels
- Extension: Extend border pixels

### Types of Kernels:
a. Horizontal edge detection:


```textmate
-1 -2 -1
 0  0  0
 1  2  1
```

b. Vertical edge detection:

```textmate
-1  0  1
-2  0  2
-1  0  1
```

## 2. Pooling

Pooling compresses information from a layer by summarizing areas of the feature maps.

### Process:
1. Slide a window over each feature map
2. Compute a summary statistic for each window

### Types:
- Max Pooling: Takes the maximum value in each window
- Average Pooling: Computes the average of values in each window

### Benefits:
1. Reduces spatial dimensions of feature maps
2. Introduces translation invariance
3. Reduces computational load for subsequent layers

## 3. CNN Architecture

A typical CNN block consists of:
1. Convolutional Layer
2. Activation Function (e.g., ReLU)
3. Pooling Layer

### Concept Abstraction:
- Stacking multiple CNN blocks allows the network to learn increasingly complex features
- Early layers detect simple features (edges, textures)
- Deeper layers combine these to detect more complex patterns (shapes, objects)

### Translation Invariance:
- Multiple CNN blocks enable the network to recognize objects regardless of their position in the image
- This is crucial for robust object recognition in various scenarios

## 4. Importance in Image Processing

- CNNs excel at tasks like image classification, object detection, and segmentation
- The combination of convolution and pooling allows for efficient feature learning and dimensionality reduction
- Learned features are often more effective than hand-crafted features

## 5. Training Process

1. Forward pass: Apply convolutions and pooling to input image
2. Compare output to ground truth (using a loss function)
3. Backward pass: Compute gradients and update kernel weights
4. Repeat process with many images to optimize kernel weights

By leveraging these concepts, CNNs have revolutionized computer vision tasks, achieving state-of-the-art performance in 
numerous applications.

### Pooling

Pooling is a mechanism often used in CNNs (and in neural networks in general). Pooling compresses information from a layer 
by summarizing areas of the feature maps produced in that layer. It works by sliding a window over each feature map, just 
like convolution, but instead of applying a kernel we compute a summary statistic (for example the maximum or the mean). 
If we take the maximum within each window, then we call this Max Pooling.

<br>

![image info](images/pooling.png)

<br>




### Concept Abstraction and Translation Variance


A block consisting of a convolutional layer followed by a max pooling layer (and an activation function) is the typical 
building block of a CNN.

By combining multiple such blocks, the network learns to extract more and more complex information from the image.

Moreover, combining multiple blocks allows the network to achieve translation invariance, meaning it will be able to 
recognize the presence of an object wherever that object is translated within the image.




### Effective Receptive Fields in CNNs


The Effective Receptive Field (ERF) in Convolutional Neural Networks refers to the area of the input image that influences 
a particular neuron in a deeper layer of the network. While the theoretical receptive field might be large, the effective 
area that significantly impacts the neuron's output is often smaller and has a Gaussian-like distribution of influence.


Key points:

1. As we go deeper into the network, each neuron indirectly sees a larger portion of the input image.
2. Not all pixels in the theoretical receptive field contribute equally to the neuron's output.
3. Central pixels typically have more influence than those at the edges of the receptive field.
4. The shape of the ERF is often Gaussian-like, with influence decreasing from the center outwards.
5. ERFs evolve during training, potentially becoming more focused or spread out.


## I. Introduction to Receptive Fields

A. Definition: The region in the input space that influences a particular CNN feature.

B. Theoretical vs. Effective Receptive Field
   1. Theoretical: The entire input area that could potentially influence the feature
   2. Effective: The area that actually has significant influence on the feature

## II. Characteristics of Effective Receptive Fields

A. Shape and Distribution
   1. Gaussian-like distribution of influence
   2. Center pixels have more impact than edge pixels

B. Size
   1. Grows larger in deeper layers of the network
   2. Affected by kernel size, stride, and network depth

C. Dynamic Nature
   1. ERFs evolve during network training
   2. Can become more focused or spread out based on the task

## III. Calculation and Visualization

A. Methods for calculating ERF
   1. Gradient-based approaches
   2. Deconvolution techniques

B. Visualization techniques
   1. Heat maps showing pixel influence
   2. Overlays on input images to demonstrate ERF size and shape

## IV. Importance in CNN Architecture

A. Feature hierarchies
   1. Shallow layers: small ERFs, local features
   2. Deep layers: large ERFs, global features

B. Network design considerations
   1. Balancing ERF size with computational efficiency
   2. Ensuring appropriate ERF growth through the network

## V. Impact on CNN Performance

A. Object detection and localization
   1. ERFs affect the network's ability to capture context
   2. Influence on the scale of objects that can be detected

B. Image classification
   1. Role in capturing both local and global features
   2. Importance for handling different scales of input

## VI. Advanced Concepts

A. Dilated/Atrous convolutions
   1. Technique to increase ERF without increasing parameters
   2. Applications in semantic segmentation

B. Attention mechanisms
   1. Dynamic adjustment of ERFs
   2. Allowing the network to focus on relevant parts of the input


By understanding Effective Receptive Fields, we gain insights into how CNNs process information and how to design more 
effective architectures for various computer vision tasks.


The concept of receptive field is that a pixel in the feature map of a deep layer is computed using information that originates 
from a large area of the input image, although it is mediated by other layers:

<br>

![image info](images/effective.png)

<br>




In practice things are a bit more complicated. When we compute the effective receptive field, instead of considering just 
whether the information contained in a given pixel is used or not by a pixel in a deeper layer, we can consider how many 
times that pixel is used. In other words, how many times that pixel was part of a convolution that ended up in a result 
used by the pixel in the deeper layer. Of course, pixels on the border of the input image are used during fewer convolutions 
than pixels in the center of the image. We can take this even further and ask how much a given pixel in the input image 
influences the pixel in a feature map deeper in the network. This means, if we change the value of the input pixel slightly, 
how much does the pixel in the deep layer change. If we take this into account, we end up with receptive fields that are more 
Gaussian-like, instead of flat as we have simplified them in the video, and they also evolve as we train the network. Visualization 
of an effective receptive field showing varying pixel influence in a neural network feature map. 



### CNN Architecture Blueprint


<br>

![image info](images/architecture.png)

<br>


–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

### Transformers 

–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project!
In this project, you will learn how to build a pipeline to process real-world, user-supplied images and to put your model into an app.
Given an image, your app will predict the most likely locations where the image was taken.

By completing this lab, you demonstrate your understanding of the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. 

Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.

### Why We're Here

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernable landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.

In this project, you will take the first steps towards addressing this problem by building a CNN-powered app to automatically predict the location of the image based on any landmarks depicted in the image. At the end of this project, your app will accept any user-supplied image as input and suggest the top k most relevant landmarks from 50 possible landmarks from across the world.


## Project Instructions

### Getting started

You have two choices for completing this project. You can work locally on your machine (NVIDIA GPU highly recommended), or you can work in the provided Udacity workspace that you can find in your classroom.

#### Setting up in the Udacity Project Workspace
You can find the Udacity Project Workspace in your Udacity classroom, in the Project section.

1. Start the workspace by clicking on `Project Workspace` in the left menu in the page
2. When prompted on whether you want a GPU or not, please ANSWER YES (the GPU is going to make everything several times faster)

The environment is already setup for you, including the starter code, so you can jump right into building the project!

#### Setting up locally

This setup requires a bit of familiarity with creating a working deep learning environment. While things should work out of the box, in case of problems you might have to do operations on your system (like installing new NVIDIA drivers) that are not covered in the class. Please do this if you are at least a bit familiar with these subjects, otherwise please consider using the provided Udacity workspace that you find in the classroom.

1. Open a terminal and clone the repository, then navigate to the downloaded folder:
	
	```	
		git clone https://github.com/udacity/cd1821-CNN-project-starter.git
		cd cd1821-CNN-project-starter
	```
    
2. Create a new conda environment with python 3.7.6:

    ```
        conda create --name udacity_cnn_project -y python=3.7.6
        conda activate udacity_cnn_project
    ```
    
    NOTE: you will have to execute `conda activate udacity_cnn_project` for every new terminal session.
    
3. Install the requirements of the project:

    ```
        pip install -r requirements.txt
    ```

4. Install and open Jupyter lab:
	
	```
        pip install jupyterlab
		jupyter lab
	```

### Developing your project

Now that you have a working environment, execute the following steps:

>**Note:** Complete the following notebooks in order, do not move to the next step if you didn't complete the previous one.

1. Open the `cnn_from_scratch.ipynb` notebook and follow the instructions there
2. Open `transfer_learning.ipynb` and follow the instructions
3. Open `app.ipynb` and follow the instructions there

## Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project rubric.  Review this rubric thoroughly and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.

## Project Submission

Your submission should consist of the github link to your repository.  Your repository should contain:
- The `landmark.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.

Please do __NOT__ include any of the project data sets provided in the `landmark_images/` folder.

### Ready to submit your project?

Click on the "Submit Project" button in the classroom and follow the instructions to submit!

## Dataset Info

The landmark images are a subset of the Google Landmarks Dataset v2.
