# Specific Test 2 - Lens Finding

## Test Details
### Task:
Build a model for classifying the images into lenses using PyTorch or Keras.

### Dataset Description:
A data set comprising images with and without strong lenses. Also has an additional CSV file.

### Evaluation Metric:
ROC curve (Receiver Operating Characteristic curve) and AUC score (Area Under the ROC Curve) 


## My Approach

### Data Loading and Pre-Processing
I used both the images data and the numerical features data for my approach and it gave me the best results. 

There are 3 numerical features in the DataFrame, where all the values are not in scale. So I created a normalize function where I used the min-max scaling method to normalize each feature separately.

X_norm = (X - X_min) / (X_max - X_min)

I used max normalization to normalize the images

For Data Augmentation, I have defined two different augmentations for the train and test sets. For the training set, I used horizontal and vertical flips, an advanced blur with various limits on blur and noise, and set the images to be transformed to tensors using ToTensorV2(). For the test set, I simply transformed the images to tensors using ToTensorV2().

For creating the dataset I used the Pytorch standard DataLoader function, which creates PyTorch data loaders for the training, validation, and test datasets, which can be used to load batches of data for training or testing machine learning models.

### Model
I implemented a custom model by starting with a pre-trained ResNet50 backbone from the timm package. I experimented with various model architectures like efficient net, mobilenet, convnext etc. but ResNet50 gave me best results with my approach. The backbone was modified to remove the original fully connected layer and I added a few additional layers, including a 1D convolutional layer with 512 filters, a batch normalization layer, a PReLU activation function, another 1D convolutional layer with 256 filters, a batch normalization layer, and a Mish activation function. 

I used PReLU instead of traditional ReLU because it helps to alleviate the "dying ReLU" problem which can occur during training. I also used Mish activation function instead of other activation functions because it has been shown to perform better in terms of accuracy and convergence speed in some cases. It has a smooth gradient and is non-monotonic, which helps in better optimization and reduces the likelihood of vanishing gradients.

To capture spatial information more effectively, I added a custom spatial attention module to the model, which learns to selectively focus on certain spatial locations in the feature map. The spatial attention block includes two convolutional layers followed by a PReLU activation function and a sigmoid activation function. The sigmoid activation function produces a spatial attention map, which is then multiplied with the input feature map to obtain the attended feature map.

In addition to the image data, this model also takes in numerical features as input. To handle this, I added two fully connected layers to process the numerical features before concatenating them with the output from the custom blocks. The concatenated features are then passed through a final fully connected layer with the required number of output classes.

### Training
The training and validation follows the standard PyTorch training pipeline. Here I am using BCEWithLogitsLoss as my loss function.

I found the AdamW optimizer to give the best results. It is a variation of the popular Adam optimizer, which incorporates weight decay regularization into the optimization process to prevent overfitting. 

I also used Cosine Annealing with Warm Restarts as my learning rate scheduler. It is a learning rate scheduler that reduces the learning rate of the optimizer at specific intervals during training.

### Results
