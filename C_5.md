# C-5: Object Detection and Segmentation

<br>
<br>

1. **Computer Vision Task Hierarchy**
    - Image classification vs. localization vs. detection vs. segmentation
2. **Object Localization**
    - Multi-head architecture
    - Loss functions
    - Bounding box representations
3. **Object Detection Fundamentals**
    - One-stage vs. two-stage detection
    - Anchor-based approaches
    - RetinaNet architecture
    - Feature Pyramid Networks
    - Focal Loss
4. **Object Detection Evaluation**
    - Precision and Recall
    - Intersection over Union (IoU)
    - Mean Average Precision (mAP)
    - Mean Average Recall (mAR)
5. **Semantic Segmentation**
    - UNet architecture
    - Skip connections
    - Dice Loss

#### Computer Vision Task Hierarchy

Computer vision encompasses a spectrum of increasingly complex tasks, each building upon the capabilities of simpler
ones while adding new dimensions of spatial understanding. This hierarchy of tasks forms a progression from basic
image-level understanding to detailed pixel-wise comprehension, with each level requiring more sophisticated
architectural approaches and presenting unique challenges.

##### Image Classification

At the foundation of computer vision lies image classification—the task of assigning one or more labels to an entire
image based on its visual content. This represents the most basic form of image understanding.

###### Key Characteristics

In image classification, the model answers a fundamental question: "What is in this image?" The output is a set of class
probabilities or scores indicating the likelihood that the image contains instances of different categories. For
example, an image might be classified as containing a "dog" with 95% confidence and a "cat" with 2% confidence.

The architectural approach typically involves:

- A convolutional neural network backbone that extracts hierarchical features
- A global pooling layer that aggregates spatial information
- One or more fully connected layers that map these features to class scores

###### Limitations

While powerful, image classification has significant limitations:

- It provides no spatial information about object locations
- It cannot differentiate between multiple instances of the same class
- It struggles with images containing multiple distinct objects

These limitations motivate the progression to more spatially-aware vision tasks.

##### Object Localization

Object localization extends classification by not only identifying what is in an image but also where the primary object
is located. It bridges the gap between image-level and object-level understanding.

###### Key Characteristics

In object localization, the model answers both "what" and "where" questions but focuses solely on the most prominent
object in the image. The output includes:

- A class label (or class probabilities)
- A bounding box defined by four coordinates that enclose the object

This task usually employs a multi-head architecture where:

- One head handles classification (identical to image classification)
- A parallel head predicts the bounding box coordinates

The loss function combines classification loss (typically cross-entropy) with a regression loss for bounding box
prediction (such as mean squared error or smooth L1 loss).

###### Limitations

While more spatially aware than classification, object localization still has constraints:

- It typically handles only a single dominant object per image
- It cannot address scenes with multiple objects of interest
- It provides only a rectangular approximation of the object's location, not its exact shape

##### Object Detection

Object detection represents a significant leap in complexity by identifying and localizing all objects of interest
within an image, regardless of their number or class.

###### Key Characteristics

In object detection, the model must simultaneously:

- Identify all instances of objects from known classes
- Predict accurate bounding boxes for each instance
- Handle variable numbers of objects across different images

The output consists of a list of detected objects, each with:

- A class label
- A confidence score
- A bounding box

Object detection architectures generally fall into two categories:

- Two-stage detectors like R-CNN variants that first propose regions of interest, then classify them
- One-stage detectors like YOLO and RetinaNet that predict classes and bounding boxes in a single forward pass

Both approaches must address the challenge of predicting objects at different scales and aspect ratios, typically
through techniques like:

- Anchor boxes (predefined boxes of different shapes)
- Feature pyramid networks (multi-scale feature representations)
- Non-maximum suppression (removing redundant detections)

###### Limitations

While powerful, object detection still has constraints:

- It provides only rectangular bounding boxes, not precise object shapes
- It cannot distinguish between touching or overlapping instances of the same class
- It operates at the object level rather than the pixel level

##### Semantic Segmentation

Semantic segmentation elevates vision understanding to the pixel level by classifying every pixel in an image according
to the object or region it belongs to.

###### Key Characteristics

In semantic segmentation, the model creates a dense, pixel-wise classification map where:

- Each pixel is assigned to exactly one class
- The output has the same spatial dimensions as the input image
- The prediction preserves the exact shape and boundaries of objects

The typical architecture follows an encoder-decoder structure:

- The encoder progressively reduces spatial dimensions while increasing feature depth
- The decoder recovers spatial information while reducing feature depth
- Skip connections often link corresponding encoder and decoder levels to preserve fine details

Common architectures include U-Net, FCN (Fully Convolutional Networks), and DeepLab variants.

###### Limitations

While offering pixel-perfect classification, semantic segmentation cannot distinguish between separate instances of the
same class—all "person" pixels are labeled identically, regardless of whether they belong to one person or many.

##### Instance Segmentation

At the apex of the computer vision task hierarchy, instance segmentation combines the instance-awareness of object
detection with the pixel-precision of semantic segmentation.

###### Key Characteristics

Instance segmentation requires the model to:

- Identify each individual object instance
- Classify each instance according to its class
- Precisely delineate the pixels belonging to each instance

This allows separate treatment of distinct objects from the same class—identifying not just "three people" but "person
1, person 2, and person 3," each with their exact pixel mask.

Popular architectures include:

- Mask R-CNN, which extends Faster R-CNN with a parallel branch for mask prediction
- YOLACT, which combines one-stage detection with prototype masks

Instance segmentation represents the most complete form of scene understanding among these tasks, offering both what and
where information at the finest possible granularity.

##### Progression of Complexity

The hierarchy of computer vision tasks represents a progression along multiple dimensions:

1. **Spatial granularity**: From image-level (classification) to object-level (detection) to pixel-level (segmentation)
2. **Output complexity**: From simple class labels to multiple coordinates to dense pixel maps
3. **Instance awareness**: From class-only to individual object instances
4. **Computational demands**: Increasing computational requirements at each level

Understanding this hierarchy helps in selecting the appropriate technique for a given application based on the required
level of detail and available computational resources. It also illustrates how advances in computer vision have
progressively enabled machines to perceive images with increasingly human-like understanding—from merely recognizing
content to precisely locating and delineating objects in complex scenes.

#### Object Localization

Object localization extends basic image classification by not only identifying what appears in an image but also
precisely where it is located. This spatial awareness represents a fundamental step toward more comprehensive scene
understanding, bridging the gap between simple classification and the more complex tasks of detection and segmentation.

<p align="center">
<img src="images/obj-loc.png" alt="Object Localization Architecture" width="600" height=auto>
</p>
<p align="center">figure: Multi-head architecture for object localization showing backbone and two heads</p>

##### Multi-Head Architecture

Object localization employs a distinctive multi-head neural network design that elegantly handles the dual tasks of
classification and spatial localization.

###### Shared Backbone

The foundation of an object localization network is a shared feature extraction backbone, typically a convolutional
neural network such as ResNet, EfficientNet, or VGG. This backbone serves a critical role:

1. It processes the raw input image through multiple convolutional layers
2. It progressively extracts increasingly abstract visual features
3. It creates rich feature representations that contain both semantic and spatial information

The backbone's ability to preserve some degree of spatial information while extracting meaningful features makes it
ideal for the dual requirements of localization and classification.

###### Classification Head

Branching from the shared backbone, the classification head focuses exclusively on determining the object's category:

1. It typically consists of one or more fully-connected layers
2. It often includes global pooling to aggregate spatial information
3. It terminates in a layer with neurons corresponding to possible classes
4. It produces a vector of class scores or probabilities

This head functions almost identically to the final layers in a standard classification network but shares its initial
feature extraction with the localization task.

###### Localization Head

In parallel with the classification head, the localization head specializes in predicting the object's spatial extent:

1. It consists of one or more fully-connected layers
2. It maintains access to the spatial information in the backbone's features
3. It terminates in a layer with typically four neurons
4. It outputs coordinates defining the object's bounding box

The localization head learns to map the spatial patterns in the feature maps to precise coordinate predictions,
essentially translating abstract representations back into image-space coordinates.

<p align="center">
<img src="images/slide.png" alt="Sliding Window Approach" width="600" height=auto>
</p>
<p align="center">figure: Sliding window approach for object detection</p>

###### Forward Pass Flow

During inference, the object localization network processes an image through a streamlined flow:

1. The input image passes through the shared backbone, generating feature maps
2. These feature maps simultaneously feed into both the classification and localization heads
3. The classification head produces class probabilities
4. The localization head generates bounding box coordinates
5. The network returns both outputs as a combined result

This parallel processing allows efficient computation of both "what" and "where" information in a single forward pass.

##### Loss Functions

Training an object localization network requires careful consideration of how to quantify errors in both classification
and bounding box prediction.

###### Combined Loss Formulation

The overall loss function for object localization typically takes the form of a weighted sum:

$$L = L_{cls} + \alpha \cdot L_{loc}$$

Where:

- $L_{cls}$ represents the classification loss
- $L_{loc}$ represents the localization loss
- $\alpha$ is a weighting hyperparameter that balances the two components

This formulation allows joint optimization of both tasks while controlling their relative importance.

###### Classification Loss

For the classification component, cross-entropy loss is the standard choice:

$$L_{cls} = -\log(\hat{p}_y)$$

Where $\hat{p}_y$ is the predicted probability for the true class $y$.

This loss penalizes the network when it assigns low probability to the correct class, encouraging confident and accurate
classification.

###### Localization Loss

For bounding box regression, mean squared error (MSE) is commonly used:

$$L_{loc} = \frac{1}{4}\sum_{i=1}^{4}(b_i - \hat{b}_i)^2$$

Where:

- $b_i$ represents the four ground truth bounding box coordinates
- $\hat{b}_i$ represents the corresponding predicted coordinates

This loss measures the squared difference between predicted and actual coordinates, penalizing spatial inaccuracy.

###### Loss Balancing

The hyperparameter $\alpha$ plays a crucial role in training dynamics:

1. If $\alpha$ is too small, the network might excel at classification but produce poor bounding boxes
2. If $\alpha$ is too large, bounding box prediction might improve at the expense of classification accuracy
3. The optimal value depends on the relative scales of the losses and the dataset characteristics

Finding the right balance often requires experimentation, though values between 0.5 and 5 serve as common starting
points.

###### Advanced Loss Variants

Beyond basic MSE, several specialized losses have been developed for bounding box regression:

1. **Smooth L1 Loss**: Combines the stability of L2 loss for small errors with the robustness of L1 loss for large
   errors
2. **IoU Loss**: Directly optimizes the Intersection over Union metric
3. **GIoU/DIoU/CIoU Losses**: Refinements that address limitations of the basic IoU loss

These advanced losses often provide better convergence and more accurate localization.

##### Bounding Box Representations

The way bounding boxes are represented significantly impacts network training and prediction quality.

###### Coordinate Formats

Two primary formats exist for representing bounding boxes:

1. **Corner format**: Specifies the top-left and bottom-right corners $$[x_{min}, y_{min}, x_{max}, y_{max}]$$

    This format directly describes the box boundaries but can be sensitive to scale.

2. **Center format**: Specifies the center point, width, and height $$[x_{center}, y_{center}, width, height]$$

    This format often provides more stable gradients during training.

Each format has advantages in different contexts, with center format generally preferred for regression tasks.

###### Coordinate Normalization

Raw pixel coordinates can cause training instability due to their potentially large range. To address this, coordinates
are typically normalized:

1. **Image-relative normalization**: All coordinates are divided by image dimensions, resulting in values between 0 and
   1 $$[x/W, y/H, w/W, h/H]$$
2. **Feature map normalization**: Coordinates are expressed relative to the feature map from which they're predicted

Normalization ensures that coordinate values stay within a reasonable range regardless of image size, improving training
stability.

###### Anchor-Based Prediction

Advanced localization systems often predict bounding boxes as offsets from predefined reference boxes called anchors:

1. Instead of direct coordinates, the network predicts transformation parameters
2. These parameters are applied to canonical anchor boxes to produce final predictions
3. This approach simplifies the regression problem, especially for multiple objects

For a center format box with anchor $(x_a, y_a, w_a, h_a)$, the network might predict offsets $(t_x, t_y, t_w, t_h)$
that are transformed:

$$x = x_a + t_x \cdot w_a$$ $$y = y_a + t_y \cdot h_a$$ $$w = w_a \cdot e^{t_w}$$ $$h = h_a \cdot e^{t_h}$$

This parameterization ensures that predictions stay reasonably close to realistic object proportions.

###### Regression Targets

During training, ground truth boxes must be converted to regression targets compatible with the network's output format:

1. For direct coordinate prediction, this might involve simple normalization
2. For anchor-based approaches, targets are the transform parameters that would convert anchors to ground truth boxes

Proper target generation is crucial for effective training, as it translates the human-annotated boxes into the
mathematical space in which the network operates.

Object localization serves as both a valuable capability in its own right and a foundational component of more complex
vision tasks like object detection. By understanding the multi-head architecture, loss function design, and bounding box
representation, we gain insight into how deep learning bridges the gap between recognizing objects and understanding
their spatial presence in the visual world.

#### Object Detection Fundamentals

Object detection extends beyond simple localization by identifying and localizing multiple objects of potentially
different classes within a single image. This capability forms the foundation for numerous applications from autonomous
driving to medical imaging. Understanding the core principles and architectures of object detection reveals how deep
learning models can effectively process complex visual scenes with multiple subjects of interest.

##### One-Stage vs. Two-Stage Detection

The object detection landscape is primarily divided into two architectural paradigms, each with distinct approaches to
the detection process.

###### Two-Stage Detection Framework

Two-stage detectors decompose the detection problem into sequential steps:

1. **Region Proposal**: The first stage generates candidate regions that might contain objects
    - Traditional methods used selective search or edge box algorithms
    - Modern approaches like Region Proposal Networks (RPNs) learn to propose regions
    - Typically generates 1,000-2,000 region proposals per image
2. **Classification and Refinement**: The second stage processes each proposal
    - Extracts features from each region using ROI pooling or similar techniques
    - Classifies the region contents (including a "background" class)
    - Refines the bounding box coordinates for better localization
    - May add additional tasks like mask prediction (in Mask R-CNN)

This family includes influential architectures like R-CNN, Fast R-CNN, Faster R-CNN, and Mask R-CNN, each representing
evolutionary improvements to the two-stage paradigm.

###### One-Stage Detection Framework

One-stage detectors tackle classification and localization simultaneously in a single forward pass:

1. They divide the image into a grid or use predefined anchor points
2. For each grid cell or anchor, they predict:
    - Class probabilities (including background)
    - Bounding box coordinates or offsets
    - Confidence scores indicating object presence
3. They process the entire image in a single network pass without intermediate region selection

Popular one-stage detectors include YOLO (You Only Look Once), SSD (Single Shot MultiBox Detector), and RetinaNet.

###### Comparative Analysis

Each approach offers distinct advantages and limitations:

**Two-Stage Strengths**:

- Generally higher detection accuracy, especially for small objects
- More flexible region processing
- Better handling of object overlaps

**Two-Stage Limitations**:

- Slower inference speed due to sequential processing
- More complex architecture and training process
- Higher computational requirements

**One-Stage Strengths**:

- Faster inference speeds, often suitable for real-time applications
- Simpler architectural design
- End-to-end training without intermediate steps

**One-Stage Limitations**:

- Historically lower accuracy (though gap has narrowed with advances like RetinaNet)
- Class imbalance challenges with background vs. foreground examples
- Fixed grid or anchor design may not optimally fit all object shapes

The choice between paradigms often involves balancing speed requirements against accuracy needs for specific
applications.

##### Anchor-Based Approaches

Anchors serve as reference boxes that enable detectors to handle objects of varying scales and aspect ratios
efficiently.

###### Anchor Fundamentals

Anchors are predefined bounding boxes with specific:

- Positions across the image
- Scales (sizes)
- Aspect ratios (height-to-width proportions)

They function as spatial hypotheses from which the network predicts adjustments, rather than predicting box coordinates
from scratch.

###### Anchor Generation Process

The process for generating an anchor set typically involves:

1. **Grid Definition**: The input image or feature maps are divided into a grid
2. **Anchor Placement**: Anchors are centered at each grid cell or regularly spaced positions
3. **Scale Variation**: Multiple scales are used at each position to capture objects of different sizes
4. **Aspect Ratio Variation**: Different height-width ratios accommodate various object shapes

A typical configuration might include 3 scales and 3 aspect ratios, resulting in 9 anchors per grid position.

###### Prediction Mechanism

For each anchor, the network predicts:

1. **Classification scores**: Probabilities for each object class plus background
2. **Bounding box refinements**: Offsets that transform the anchor into a precise object boundary

The transformation from anchor to predicted box typically follows:

$$x = x_a + w_a \cdot t_x$$ $$y = y_a + h_a \cdot t_y$$ $$w = w_a \cdot e^{t_w}$$ $$h = h_a \cdot e^{t_h}$$

Where $(x_a, y_a, w_a, h_a)$ are anchor parameters and $(t_x, t_y, t_w, t_h)$ are predicted transformations.

###### Training Considerations

During training, anchors are assigned to ground truth objects based on Intersection over Union (IoU):

- Anchors with high IoU to a ground truth box become positive examples
- Anchors with low IoU to all ground truth boxes become negative (background) examples
- Anchors in between are typically ignored during training

This assignment process creates the training targets for both classification and regression.

###### Anchor Design Impact

The design of the anchor set significantly influences detector performance:

- Too few anchors may miss objects with unusual shapes
- Too many anchors increase computational cost and can destabilize training
- Appropriate anchor distribution across scales is critical for detecting both large and small objects

Modern approaches often use dataset analysis to determine optimal anchor configurations.

##### RetinaNet Architecture

RetinaNet represents a breakthrough one-stage detector that combines architectural innovations with a novel loss
function to achieve state-of-the-art performance.

###### Architectural Overview

RetinaNet follows a clear architectural structure:

1. **Backbone**: Typically ResNet with Feature Pyramid Network integration
2. **Classification Subnet**: Predicts class probabilities for each anchor
3. **Box Regression Subnet**: Predicts bounding box refinements for each anchor

This design maintains the speed advantages of one-stage detectors while addressing their historical accuracy
limitations.

<p align="center">
<img src="images/retina_net.png" alt="RetinaNet Architecture" width="600" height=auto>
</p>
<p align="center">figure: RetinaNet architecture combining FPN with focal loss for object detection</p>

###### Classification Subnet

The classification subnet processes each level of the feature pyramid:

1. Four 3×3 convolutional layers with 256 filters, each followed by ReLU
2. A final 3×3 convolutional layer with $K \cdot A$ filters, where:
    - $K$ is the number of classes
    - $A$ is the number of anchors per location

This subnet is applied to each pyramid level with shared weights, producing class predictions for all anchors.

###### Box Regression Subnet

The box regression subnet mirrors the classification subnet's structure:

1. Four 3×3 convolutional layers with 256 filters, each followed by ReLU
2. A final 3×3 convolutional layer with $4 \cdot A$ filters (4 coordinates for each anchor)

Like the classification subnet, its weights are shared across pyramid levels for consistent feature interpretation.

###### Multi-Scale Detection

RetinaNet predicts objects at multiple scales through its feature pyramid integration:

- Larger objects are detected in higher pyramid levels (with larger receptive fields)
- Smaller objects are detected in lower pyramid levels (with finer spatial resolution)
- Anchors of different scales are assigned to appropriate pyramid levels

This multi-scale approach enables effective detection across a wide range of object sizes.

##### Feature Pyramid Networks

Feature Pyramid Networks (FPN) provide a structured approach to multi-scale feature extraction, addressing the challenge
of detecting objects across different scales.

###### Traditional Feature Pyramids

Before FPN, multi-scale detection typically used one of two approaches:

1. **Image pyramid**: Running the detector on multiple rescaled versions of the input image
    - Effective but computationally expensive
    - Requires multiple forward passes
2. **Feature hierarchy**: Using different layers from the CNN backbone
    - Efficient but limited by semantic gaps between shallow and deep features
    - Lower layers had fine resolution but weak semantics

###### FPN Architecture

FPN combines the best aspects of both approaches through a two-pathway structure:

1. **Bottom-up pathway**: The traditional CNN forward pass
    - Progressively reduces spatial dimensions
    - Increases semantic information
    - Creates a hierarchy of feature maps at different scales
2. **Top-down pathway**: Upsampling of higher-level features
    - Starts from the semantically strongest features at the top
    - Progressively upsamples to recover spatial resolution
    - Creates feature maps matching the scales of the bottom-up pathway
3. **Lateral connections**: Connect corresponding levels between pathways
    - Feature maps from the bottom-up pathway are combined with upsampled features
    - Typically implemented as 1×1 convolutions followed by addition
    - Enriches upsampled features with spatial information from earlier layers

<p align="center">
<img src="images/fpn.png" alt="Feature Pyramid Network" width="600" height=auto>
</p>
<p align="center">figure: Feature Pyramid Network (FPN) architecture for multi-scale feature extraction</p>

###### Feature Map Generation

The combined process generates a set of feature maps ${P_i}$ with the following characteristics:

1. Each level has the same channel dimension (typically 256)
2. Each level has a spatial resolution corresponding to its hierarchy level
3. Each level contains both high-level semantic information and appropriate spatial detail

These properties make the resulting feature maps ideal for detecting objects at their corresponding scales.

###### Implementation Details

A typical FPN implementation includes:

1. Selection of backbone feature maps from different stages (often C2, C3, C4, C5 in ResNet)
2. 1×1 convolutions on these maps to create lateral connections
3. Upsampling of higher-level features, typically using nearest-neighbor interpolation
4. Element-wise addition of lateral and upsampled features
5. 3×3 convolutions to create the final feature maps, reducing aliasing effects

The resulting feature pyramid provides strong multi-scale features with relatively low computational overhead.

##### Focal Loss

Focal Loss addresses a fundamental challenge in one-stage object detection: the extreme class imbalance between
foreground and background examples.

###### Class Imbalance Problem

One-stage detectors face a severe imbalance challenge:

1. They typically evaluate 100,000+ candidate locations per image
2. Only a tiny fraction of these contain objects (often <100)
3. The vast majority are easy-to-classify background examples
4. This imbalance can overwhelm the loss from rare positive examples

Standard cross-entropy loss gives equal weight to all examples, allowing easy negatives to dominate the gradient and
destabilize training.

<p align="center">
<img src="images/focal_loss.png" alt="Focal Loss" width="600" height=auto>
</p>
<p align="center">figure: Focal Loss function for addressing class imbalance in object detection</p>

###### Cross-Entropy Limitations

For a binary classification problem, the standard cross-entropy loss is:

$$\text{CE}(p, y) = -y \log(p) - (1-y) \log(1-p)$$

Where:

- $y \in {0, 1}$ is the ground truth class
- $p \in [0, 1]$ is the model's estimated probability for class 1

For well-classified examples (high $p$ for positive examples, low $p$ for negative examples), this loss still assigns
non-negligible values, allowing easy examples to dominate during training.

###### Focal Loss Formulation

Focal Loss modifies cross-entropy by adding a modulating factor:

$$\text{FL}(p_t) = -(1-p_t)^\gamma \log(p_t)$$

Where:

- $p_t$ is $p$ for positive examples ($y=1$) and $1-p$ for negative examples ($y=0$)
- $\gamma \geq 0$ is the focusing parameter that adjusts the rate at which easy examples are downweighted

This formulation has several key properties:

1. When $\gamma = 0$, Focal Loss is equivalent to standard cross-entropy
2. As $\gamma$ increases, the relative loss for well-classified examples is reduced
3. For hard examples where $p_t$ is small, the loss remains largely unchanged
4. For easy examples where $p_t$ is close to 1, the loss is significantly down-weighted

###### Effect on Training Dynamics

Focal Loss transforms training dynamics in several ways:

1. It effectively focuses training on hard examples that the model struggles with
2. It prevents easy background examples from overwhelming the loss
3. It eliminates the need for hard negative mining or similar sampling strategies
4. It allows the use of a dense set of anchors without sampling or reweighting

###### Implementation Details

In practice, an alpha-balanced version is often used:

$$\text{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

Where $\alpha_t$ is a weighting factor that can further address class imbalance.

Typical values include:

- $\gamma = 2$ for the focusing parameter
- $\alpha = 0.25$ for positive examples, $1-\alpha$ for negative examples

These settings effectively balance the contribution of positive and negative examples while focusing on hard examples.

The combination of anchor-based approaches, Feature Pyramid Networks, and Focal Loss in RetinaNet exemplifies how modern
object detection architectures address the fundamental challenges of localizing and classifying multiple objects across
varying scales, positions, and classes—all while maintaining computational efficiency. These innovations have
collectively transformed object detection from a challenging research problem into a practical technology deployed
across numerous real-world applications.

#### Object Detection Evaluation

Evaluating object detection models requires specialized metrics that assess both localization accuracy and
classification performance. Unlike simpler tasks like image classification where accuracy suffices, object detection
evaluation must account for the spatial component of predictions and handle the complexities of multiple objects per
image. A robust evaluation framework helps researchers and practitioners compare different detection approaches
objectively and identify areas for improvement.

##### Precision and Recall Fundamentals

Precision and recall form the foundational metrics for evaluating object detection performance, adapted from their
origins in information retrieval and binary classification.

###### Defining Detection Outcomes

In object detection, the four possible outcomes for each predicted box are:

1. **True Positive (TP)**: A detection that correctly identifies an object with sufficient overlap with the ground truth
   box
2. **False Positive (FP)**: A detection that either identifies the wrong object class or identifies an object where none
   exists
3. **False Negative (FN)**: A ground truth object that the detector fails to identify
4. **True Negative (TN)**: Correctly not detecting an object where none exists (rarely used in object detection
   evaluation)

Unlike in classification, determining whether a detection is correct requires spatial analysis through Intersection over
Union (IoU) thresholds, as well as class verification.

###### Precision Calculation

Precision measures the accuracy of positive predictions, answering: "Of all the objects the model detected, what
fraction were actually correct?"

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

High precision indicates that when the model predicts an object, it's likely to be correct—the model makes few false
alarms.

###### Recall Calculation

Recall measures the completeness of positive predictions, answering: "Of all the actual objects in the images, what
fraction did the model detect?"

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

High recall indicates that the model finds most of the actual objects—it misses few ground truth instances.

###### The Precision-Recall Tradeoff

Precision and recall typically exhibit an inverse relationship:

1. Increasing detection confidence threshold:
    - Improves precision (fewer false positives)
    - Reduces recall (more missed detections)
2. Decreasing detection confidence threshold:
    - Reduces precision (more false positives)
    - Improves recall (fewer missed detections)

This tradeoff is visualized through precision-recall curves, which plot precision against recall at various confidence
thresholds.

###### Application-Specific Considerations

Different applications prioritize these metrics differently:

1. **Safety-critical systems** (automated driving): May prioritize recall to ensure no objects are missed
2. **User-facing applications** (photo organization): May prioritize precision to avoid annoying false positives
3. **Balanced applications** (surveillance): May seek an optimal balance through F1-score or other combined metrics

Understanding this tradeoff helps practitioners select appropriate operating points for their specific use cases.

##### Intersection over Union (IoU)

Intersection over Union provides the spatial evaluation component unique to object detection, measuring how well
predicted bounding boxes align with ground truth.

###### Definition and Calculation

IoU quantifies the overlap between two bounding boxes as the ratio of their intersection area to their union area:

$$\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$

For two rectangular boxes $A$ and $B$:

1. Calculate intersection:
   $I = \max(0, \min(A_{x2}, B_{x2}) - \max(A_{x1}, B_{x1})) \times \max(0, \min(A_{y2}, B_{y2}) - \max(A_{y1}, B_{y1}))$
2. Calculate areas: $A_{area} = (A_{x2} - A_{x1}) \times (A_{y2} - A_{y1})$ and
   $B_{area} = (B_{x2} - B_{x1}) \times (B_{y2} - B_{y1})$
3. Calculate union: $U = A_{area} + B_{area} - I$
4. Compute IoU: $\text{IoU} = I / U$

The resulting value ranges from 0 (no overlap) to 1 (perfect overlap).

<p align="center">
<img src="images/bird.png" alt="Bird Detection Example" width="600" height=auto>
</p>
<p align="center">figure: Example of bird detection showing bounding box and IoU calculation</p>

###### IoU Thresholds

In object detection evaluation, an IoU threshold determines when a detection is considered a match to a ground truth
object:

1. A common threshold is 0.5, meaning boxes must overlap by at least 50% to be considered a match
2. More stringent evaluations use higher thresholds like 0.75 or 0.9
3. Modern evaluation protocols often report metrics at multiple IoU thresholds

The choice of threshold significantly impacts reported performance metrics—higher thresholds demand more precise
localization.

###### Matching Process

In images with multiple objects, determining matches involves:

1. Sorting detections by confidence score
2. Processing detections from highest to lowest confidence
3. Assigning each detection to at most one ground truth based on IoU
4. Once a ground truth is matched, it cannot be matched again

This process ensures that each ground truth object is matched with at most one detection, typically the one with highest
confidence that meets the IoU threshold.

###### IoU Limitations and Extensions

While effective, standard IoU has limitations:

1. It treats all spatial misalignments equally, regardless of direction
2. It doesn't account for the distance between non-overlapping boxes
3. It's sensitive to small errors for tiny objects

These limitations have motivated extensions like:

1. **Generalized IoU (GIoU)**: Accounts for the enclosing box of both prediction and ground truth
2. **Distance IoU (DIoU)**: Incorporates the distance between box centers
3. **Complete IoU (CIoU)**: Adds consideration of aspect ratio similarity

These refined metrics can provide more nuanced evaluation of localization quality.

##### Mean Average Precision (mAP)

Mean Average Precision serves as the primary summary metric for object detection performance, addressing both the
precision-recall tradeoff and multiclass evaluation.

###### Average Precision Calculation

Average Precision (AP) computes the area under the precision-recall curve for a specific class:

1. Generate the precision-recall curve by varying the confidence threshold
2. Apply interpolation to smooth the curve (addressing zigzag patterns)
3. Calculate the area under this interpolated curve

Mathematically, this is approximated as:

$$\text{AP} = \sum_{i} (r_{i+1} - r_i) \times p_{\text{interp}}(r_{i+1})$$

Where $p_{\text{interp}}(r)$ is the interpolated precision at recall level $r$.

###### Interpolation Methods

Different detection benchmarks use varying interpolation approaches:

1. **Pascal VOC (2007)**: Uses 11-point interpolation, sampling recall at [0, 0.1, ..., 1.0]
2. **Pascal VOC (2010-2012)**: Uses all points interpolation
3. **COCO**: Uses 101-point interpolation, sampling recall more densely

Modern implementations typically use all-point interpolation for greater precision.

###### Mean AP Across Classes

The mean Average Precision (mAP) extends AP to multiclass detection scenarios:

$$\text{mAP} = \frac{1}{N} \sum_{c=1}^{N} \text{AP}_c$$

Where:

- $N$ is the number of classes
- $\text{AP}_c$ is the Average Precision for class $c$

This averaging ensures that each class contributes equally to the final metric, regardless of its frequency in the
dataset.

###### IoU Variations in mAP

Modern object detection challenges report mAP across multiple IoU thresholds:

1. **mAP@0.5**: Uses a single IoU threshold of 0.5 (traditional Pascal VOC metric)
2. **mAP@0.75**: Uses a more stringent IoU threshold of 0.75
3. **mAP@[.5:.95]**: Averages mAP over multiple IoU thresholds from 0.5 to 0.95 in steps of 0.05 (COCO primary metric)

The last approach rewards detectors that produce highly accurate bounding boxes while still accounting for reasonable
localization.

###### Interpreting mAP

When analyzing mAP values:

1. Higher values indicate better overall detection performance
2. Gaps between mAP@0.5 and mAP@0.75 reveal localization precision
3. Class-specific AP highlights per-class performance variations
4. Compare different models using the same mAP definition, as variations in calculation can significantly affect
   reported numbers

A comprehensive evaluation often reports both overall mAP and class-specific AP to provide a complete performance
picture.

##### Mean Average Recall (mAR)

While less commonly used as a primary metric, Mean Average Recall provides valuable complementary information about a
detector's ability to find objects.

###### Average Recall Calculation

Average Recall (AR) measures the recall averaged over a range of IoU thresholds:

$$\text{AR} = \frac{2}{0.5} \int_{0.5}^{1.0} \text{Recall(IoU)} , d\text{IoU}$$

In practice, this integral is approximated by sampling at discrete IoU thresholds.

The factor of 2 normalizes the result to [0,1], since the integration range is [0.5,1.0] rather than [0,1].

###### AR Variations

AR is often computed with various constraints:

1. **AR@k**: Maximum of k detections per image (e.g., AR@1, AR@10, AR@100)
2. **AR-small/medium/large**: AR for objects of different size categories

These variations help evaluate recall performance under different operational constraints or for objects of different
scales.

###### Mean AR Across Classes

Similar to mAP, mean Average Recall (mAR) averages AR across all classes:

$$\text{mAR} = \frac{1}{N} \sum_{c=1}^{N} \text{AR}_c$$

Where:

- $N$ is the number of classes
- $\text{AR}_c$ is the Average Recall for class $c$

###### AR as a Complementary Metric

mAR serves several valuable roles in comprehensive evaluation:

1. It indicates a detector's theoretical maximum performance if its confidence ranking were perfect
2. It helps identify whether performance limitations stem from poor localization or poor confidence estimation
3. It's particularly valuable for applications where finding all instances is critical, regardless of confidence

The combination of mAP and mAR provides a more complete picture than either metric alone.

##### Practical Evaluation Considerations

Beyond the core metrics, several practical considerations affect meaningful object detection evaluation.

###### Benchmark Standards

Major object detection benchmarks have established standard evaluation protocols:

1. **PASCAL VOC**: Pioneered mAP@0.5 evaluation for 20 common object classes
2. **COCO**: Expanded to 80 classes with more sophisticated metrics including mAP@[.5:.95]
3. **Open Images**: Features hierarchical class relationships and instance segmentation evaluation

Following these established protocols enables fair comparison with published methods.

###### Scale-Specific Evaluation

Object scale significantly impacts detection difficulty, leading to scale-stratified evaluation:

1. **Small objects**: Area < 32² pixels
2. **Medium objects**: 32² to 96² pixels
3. **Large objects**: Area > 96² pixels

Reporting performance by scale category helps identify specific strengths and weaknesses of different detection
approaches.

###### Other Important Factors

Comprehensive evaluation should consider:

1. **Inference speed**: Often measured in frames per second (FPS) or milliseconds per image
2. **Model size**: Parameter count and memory requirements
3. **Hardware requirements**: GPU memory needs and computational complexity
4. **Edge cases**: Performance under occlusion, unusual viewpoints, or poor lighting

These practical considerations often determine a detector's suitability for real-world deployment beyond benchmark
performance.

###### Error Analysis

Breaking down detection errors provides valuable insights:

1. **Classification errors**: Correct localization but wrong class
2. **Localization errors**: Correct class but insufficient IoU
3. **Duplicate detection errors**: Multiple detections of the same object
4. **Background errors**: False positives in background regions
5. **Missed detection errors**: False negatives

Identifying the dominant error types helps guide focused improvement efforts.

Thorough evaluation using precision, recall, IoU, mAP, and mAR provides a comprehensive assessment of object detection
performance. These metrics collectively capture a detector's ability to accurately locate objects, classify them
correctly, and maintain high confidence in its predictions. By understanding these evaluation approaches, practitioners
can both select appropriate models for their applications and systematically improve detection performance.

#### Semantic Segmentation

Semantic segmentation represents the pinnacle of dense prediction tasks in computer vision, assigning a class label to
every pixel in an image. Unlike object detection which surrounds objects with bounding boxes, semantic segmentation
provides precise object boundaries and can identify amorphous regions like sky, road, or vegetation. This pixel-perfect
understanding enables applications from autonomous driving to medical image analysis, where exact boundaries matter
critically.

##### UNet Architecture

The UNet architecture has emerged as one of the most influential and widely-adopted frameworks for semantic
segmentation, particularly in medical imaging but increasingly across diverse domains.

###### Architectural Overview

UNet's distinctive U-shaped architecture consists of three major components:

1. **Contracting Path (Encoder)**: A series of convolutional blocks followed by downsampling operations that
   progressively reduce spatial dimensions while increasing feature depth.
2. **Bottleneck**: The lowest resolution section connecting the contracting and expansive paths, containing the most
   abstract representations.
3. **Expansive Path (Decoder)**: A series of upsampling operations followed by convolutional blocks that progressively
   increase spatial dimensions while decreasing feature depth.

This symmetric design creates a U-shaped profile when visualized, giving the architecture its name.

<p align="center">
<img src="images/unet.png" alt="UNet Architecture" width="600" height=auto>
</p>
<p align="center">figure: UNet architecture for semantic segmentation with encoder-decoder structure</p>

###### Contracting Path Details

The encoder follows a typical CNN pattern:

1. Each level consists of two or more 3×3 convolutional layers followed by non-linear activations (typically ReLU)
2. Feature maps double in number at each downsampling step (e.g., 64→128→256→512)
3. Downsampling occurs via 2×2 max pooling operations with stride 2, halving the spatial dimensions
4. The progressive reduction in resolution creates a hierarchical feature representation

This contracting process extracts increasingly abstract features while reducing spatial dimensions, effectively
compressing the image information.

###### Expansive Path Details

The decoder mirrors the encoder but works in reverse:

1. Each level begins with an upsampling operation (transposed convolution or interpolation followed by convolution)
2. Upsampled features are combined with corresponding encoder features via skip connections
3. After feature combination, two or more 3×3 convolutional layers with non-linear activations refine the features
4. Feature maps halve in number at each upsampling step (e.g., 512→256→128→64)

This expansive process gradually recovers spatial resolution while incorporating both high-level semantic information
and low-level spatial details.

###### Final Classification Layer

The network concludes with a 1×1 convolutional layer that maps the feature channels to the number of classes:

1. For binary segmentation, a single output channel with sigmoid activation
2. For multi-class segmentation, C output channels (where C is the number of classes) with softmax activation

This final layer makes the per-pixel classification decision based on the rich feature representation built through the
network.

##### Skip Connections

Skip connections represent the key innovation in UNet, enabling precise localization while preserving contextual
information.

###### The Localization Challenge

A fundamental challenge in segmentation architectures is balancing two competing needs:

1. **Context understanding**: Requires large receptive fields and deep networks, typically achieved through downsampling
2. **Precise localization**: Requires high-resolution feature maps to delineate exact boundaries

Without skip connections, upsampling alone struggles to recover the precise spatial details lost during downsampling,
resulting in coarse segmentation boundaries.

###### Skip Connection Mechanism

UNet's skip connections directly link corresponding layers between the contracting and expansive paths:

1. Feature maps from an encoder level are copied and concatenated with the upsampled feature maps in the corresponding
   decoder level
2. The concatenation occurs along the channel dimension, effectively combining features at the same resolution
3. This creates a composite feature map that contains both semantic context from deeper layers and spatial detail from
   earlier layers

These connections form information bridges across the architecture, allowing high-resolution details to flow directly to
the decoder.

###### Implementation Details

In practical UNet implementations, skip connections involve:

1. Storing feature maps from each encoder level before max pooling
2. After each upsampling step in the decoder, concatenating these stored features with the upsampled ones
3. Adjusting channel dimensions in the subsequent convolutions to process the expanded feature set

The concatenation operation preserves all information from both paths rather than using addition or other merging
operations.

###### Benefits of Skip Connections

Skip connections provide several critical advantages:

1. **Detail preservation**: Fine details lost during downsampling can be recovered during upsampling
2. **Gradient flow**: Direct paths from later to earlier layers improve gradient flow during training
3. **Feature reuse**: Early low-level features (edges, textures) remain directly accessible to later layers
4. **Multi-scale awareness**: The network can leverage information from multiple scales simultaneously

These benefits collectively enable UNet to produce segmentations with sharp, accurate boundaries—a critical requirement
for applications like medical imaging where precision directly impacts diagnostic accuracy.

###### Variations in Skip Connection Design

Since UNet's introduction, various skip connection designs have emerged:

1. **Feature selection**: Applying attention mechanisms to selectively emphasize important features
2. **Residual skip connections**: Adding residual connections within each level for improved gradient flow
3. **Dense skip connections**: Connecting each decoder level to all previous encoder levels
4. **Feature transformation**: Applying additional convolutions to encoder features before concatenation

These variations retain the core concept while addressing specific limitations or enhancing particular aspects of the
original design.

##### Dice Loss

The Dice Loss function addresses key challenges in segmentation training, particularly for imbalanced class
distributions common in medical and natural images.

###### Class Imbalance Problem

Semantic segmentation often faces extreme class imbalance:

1. In medical images, pathological regions might occupy <1% of the image
2. In natural scenes, certain classes (e.g., rare objects) may appear infrequently
3. With standard losses like cross-entropy, majority classes can dominate the gradient

This imbalance can lead models to predict only the dominant class, achieving high accuracy but failing on the critical
minority regions.

###### Dice Coefficient Fundamentals

The Dice coefficient (also known as the F1 score for binary cases) measures overlap between two sets:

$$\text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}$$

Where:

- X and Y are the two sets (predicted and ground truth segmentation)
- |X| denotes the size of set X
- |X ∩ Y| denotes the size of the intersection between X and Y

This coefficient ranges from 0 (no overlap) to 1 (perfect overlap).

###### Soft Dice Coefficient

To make the Dice coefficient differentiable for neural network training, a "soft" version is used:

$$\text{Dice}*{\text{soft}} = \frac{2\sum*{i=1}^{N} p_i g_i}{\sum_{i=1}^{N} p_i + \sum_{i=1}^{N} g_i}$$

Where:

- $p_i$ is the predicted probability for pixel i
- $g_i$ is the ground truth for pixel i (typically 0 or 1)
- N is the total number of pixels

This soft version accepts probability predictions rather than requiring hard binary decisions.

###### Dice Loss Formulation

The Dice Loss is simply defined as:

$$\text{Dice}*{\text{loss}} = 1 - \text{Dice}*{\text{soft}}$$

This creates a loss function that:

1. Equals 0 for perfect prediction
2. Approaches 1 for poor prediction
3. Can be used directly with gradient-based optimization

###### Relationship to F1 Score

The Dice coefficient can be interpreted in terms of precision and recall:

$$\text{Dice} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}}$$

Where:

- TP = True Positives
- FP = False Positives
- FN = False Negatives

This connection explains why Dice loss naturally balances precision and recall, unlike metrics like accuracy which can
be misleading with imbalanced classes.

###### Implementation Considerations

When implementing Dice Loss, several practical considerations improve stability:

1. **Smoothing factor**: Adding a small ε to both numerator and denominator prevents division by zero and stabilizes
   gradients:

$$\text{Dice}*{\text{loss}} = 1 - \frac{2\sum*{i=1}^{N} p_i g_i + \epsilon}{\sum_{i=1}^{N} p_i + \sum_{i=1}^{N} g_i + \epsilon}$$

1. **Multi-class adaptation**: For multiple classes, the Dice loss is typically calculated per class and then averaged:

$$\text{Dice}*{\text{multi}} = 1 - \frac{1}{C}\sum*{c=1}^{C}\text{Dice}_{c}$$

1. **Weighting strategies**: Classes can be weighted differently to further address imbalance

###### Dice Loss Advantages

In comparison to standard cross-entropy loss, Dice loss offers several benefits:

1. **Class balance**: Inherently addresses class imbalance without explicit class weighting
2. **Boundary emphasis**: Naturally emphasizes boundary regions where precision/recall metrics are most affected
3. **Direct optimization**: Optimizes the same metric that's often used for evaluation
4. **Scale invariance**: Not affected by the absolute number of pixels in each class

These advantages make Dice loss particularly well-suited for medical image segmentation, where precise delineation of
small structures is often critical.

###### Dice Loss Limitations

Despite its advantages, Dice loss has some limitations:

1. It can be unstable early in training when predictions are nearly random
2. It may struggle with very small structures where even slight misalignments drastically affect the coefficient
3. It doesn't account for spatial relationships between misclassified pixels

These limitations have led to hybrid approaches that combine Dice loss with cross-entropy or other loss functions.

The combination of UNet architecture with skip connections and Dice loss has revolutionized semantic segmentation,
particularly in medical imaging but increasingly across domains. This approach effectively balances the competing
requirements of contextual understanding and precise localization, while training reliably even with imbalanced class
distributions. Understanding these components provides insight into not just how semantic segmentation works, but why
certain architectural choices prove so effective for pixel-wise prediction tasks.
