# Common Test 1 - Multi-Class Classification

## Test Details
### Task:
Build a model for classifying the images into lenses using PyTorch or Keras.

### Dataset Description:
The Dataset consists of three classes, strong lensing images with no substructure, subhalo substructure, and vortex substructure.

### Evaluation Metric:
ROC curve (Receiver Operating Characteristic curve) and AUC score (Area Under the ROC Curve) 

## My Approach

### Files
1. CommonTest-1.ipynb (The Notebook for the task, includes all the training and evaluation scripts)
2. CommonTest-1_ROC-Curve.png (The ROC Curve image)

### Data Loading and Pre-Processing
I followed the standard data loading and pre-processing pipeline of Image Classification in PyTorch. I only used the AdvancedBlur augmentation from albumentations as my Data Augmentation (as other augmentations were giving worse results).

### Model
I have used the timm library to get the backbone architecture for the model. I experimented with several models but 'efficientnet_b3' gave me the best results. The model is initialized with the pretrained weights and as the images are grayscale images so 'in_chans=1' handles that part.

I have customized the model by adding few layers on top of the backbone model. I have created a Convolutional block to increase the model's representational power and improve its ability to capture important features in the data. The output from the backbone model is passed through this block. Then I decided to further add a Residual Connection in the network, this helps to improve the flow of information through the network and prevent the vanishing gradient problem, which can occur in very deep neural networks The output of the convolutional block is added to the output of the residual block. After this the output is passed through ReLU activation function, then flattened it and then finally pass it through the fully connected classifier to generate the final output. The final output is a vector of length 3, representing the model's prediction for the input image's class probabilities.

I used PReLU activation function in the convolutional and fully connected blocks. It is similar to the ReLU function, but with a small slope for negative inputs, which can help alleviate the "dying ReLU" problem where some neurons can become stuck at zero and stop learning. The dropout layers are also added to reduce overfitting.

### Training
#### Loss Function - 
When I trained the model with the normal Cross Entropy loss I found that my model was overfitting a lot. So I learnt about Label Smoothing technique (https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06) and it reduced the overfitting to some extent. 

In standard cross entropy, the true label is assigned a probability of 1, and all other labels are assigned a probability of 0. This can lead to overfitting in some cases, especially when the model is confident in its predictions. Label smoothing is a technique that addresses this problem by replacing the hard targets (0s and 1s) with soft targets that assign some probability to incorrect labels.

The label smoothing cross entropy loss function penalizes the model less harshly for making confident but incorrect predictions, which can improve the model's generalization performance. The amount of smoothing is controlled by a hyperparameter called the smoothing factor, which determines the extent to which the model should trust its predictions.

#### Training Trick - 
For the first few epochs i trained the model using the normal CrossEntropy Loss to make the model more confident. After that use LabelSmoothing so that it works as a Regularization method. This gave me the best results out of all the experiments I performed

#### Optimizer - 
I found the AdamW optimizer to give the best results. I used it because my model was overfitting with other optimizers and the AdamW optimizer is a variation of the popular Adam optimizer, which incorporates weight decay regularization into the optimization process to prevent overfitting. 

I also used Cosine Annealing with Warm Restarts as my learning rate scheduler. It is a learning rate scheduler that reduces the learning rate of the optimizer at specific intervals during training.

### Results - 
On Test data I got micro-average area = 0.985 , macro-average area = 0.984 , AUC for no substructure class = 0.984 , AUC for vortex class = 0.990, AUC for spherical class = 0.978.

#### ROC Curve -
![ROC Curve](https://github.com/Krishnav-Rajbangshi/ML4SCI-DeepLense-Tests/blob/main/CommonTest-1/CommonTest-1_ROC-Curve.png)
