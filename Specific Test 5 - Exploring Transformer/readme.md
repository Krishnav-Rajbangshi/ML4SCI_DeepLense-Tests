# Specific Test 5 - Exploring Transformers

## Test Details

### Task:
Use a vision transformer method of your choice to build a robust and efficient model for binary classification or unsupervised anomaly detection on the provided dataset.

### Dataset Description:
 A set of simulated strong gravitational lensing images with and without substructure. 

### Evaluation Metric:
ROC curve (Receiver Operating Characteristic curve) and AUC score (Area Under the ROC Curve)


## My Approach

### Data Loading and Pre-Processing
The data contains images of gravitational lenses with and without substructure. I created lists of images and labels with no_sub -> 0 and sub -> 1 as the labels and converted the lists into a pandas DataFrame. 

The images were normalized using the max normaliztion method (img = img / np.max(img))

For Data Augmentations, I have defined two different augmentations for the train and test sets. For the training set, I used horizontal and vertical flips, an advanced blur with various limits on blur and noise, and set the images to be transformed to tensors using ToTensorV2().  I also used CenterCrop which reduces the image size to (100, 100, 3), I used this to remove the redundant pixels. For the test set, I used the same CenterCrop and transformed the images to tensors using ToTensorV2().

For creating the dataset I used the Pytorch standard DataLoader function, which creates PyTorch data loaders for the training, validation, and test datasets, which can be used to load batches of data for training or testing machine learning models.

### Model

I created a custom model class named "CustomMobileViT". The backbone of the model is loaded from the timm library using the "create_model" function with the "mobilevitv2_150" vision transformer architecture and pretrained weights. The backbone model is used to extract features from the input image. I used this particular model because it didn't have any fixed input size and also it gave the best results.

I added some custom blocks to the backbone model in the "custom_blocks" sequential container. The first custom block is a convolutional block consisting of a convolutional layer, batch normalization, and PReLU activation function. The second custom block is the squeeze and excitation block, which helps the model to adaptively recalibrate the feature responses.

After the squeeze and excitation block, dropout regularization is applied with a rate of "dropout_rate". Then, another convolutional block with the same structure as the previous block is added. Another squeeze and excitation block is added after this block.

Then, adaptive average pooling is used to reduce the spatial dimensions of the feature map to 1x1. Then, the output is flattened and passed through a dropout layer. Finally, a linear layer is used to map the features to the desired output size.

I used the squeeze and excitation block (SE) because, the SE block learns a channel-wise weighting scheme that can be used to selectively amplify or attenuate the importance of each channel in the feature map, based on its relevance to the task at hand. This allows the network to focus on the most informative channels and ignore the less informative ones, thus improving its ability to capture discriminative features and reduce noise. 

In summary, the model consists of a backbone network that extracts features from the input image, followed by custom blocks that refine the features, and finally a linear layer that maps the refined features to the desired output size.

### Training

The training and validation follows the standard PyTorch training pipeline. Here I am using BCEWithLogitsLoss as my loss function.

I found the AdamW optimizer to give the best results. It is a variation of the popular Adam optimizer, which incorporates weight decay regularization into the optimization process to prevent overfitting.

I also used Cosine Annealing as my learning rate scheduler. It is a type of learning rate schedule that has the effect of starting with a large learning rate that is relatively rapidly decreased to a minimum value before being increased rapidly again.

### Results
