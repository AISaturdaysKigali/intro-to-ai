# Smile Classification with CNNs (PyTorch)

## Learning 

By the end of this classwork, you should be able to:
- Explain the role of **CNNs** in image classification.
- Implement a **data pipeline** in PyTorch for image datasets.
- Build and train a **CNN model** to classify smiling vs. not smiling faces.
- Apply **data augmentation** to improve generalization.
- Evaluate the model’s performance and reflect on possible improvements.

## Problem Context

We want to classify whether a person is **smiling** or **not smiling** using face images from the **CelebA** dataset.

- Dataset: CelebA (202,599 images, with 40 attributes).
- Attribute of interest: smile (binary: smiling or not).
- For this exercise: use a subset of **16,000 training images** to keep training fast.

## Tasks

### 1. Setup & Dataset

CelebA data comes in three partitions: a training dataset, a validation dataset, and a test dataset.

- Load CelebA dataset (hint: `torchvision.datasets.CelebA`).
- Use `target_type='attr'` to get **attribute labels**.
- Split into **train, validation, test** (hint: the `split` argument).
- Use the `transform` argument to apply data augmentation. (**After completing section 2**)
- Use the `target_transform` argument to extract the smile label. (**After completing section 3**)

### 2. Data Augmentation

Use `torchvision.transforms` to apply augmentation (rotation, flipping, shifting, etc.) to the images 

- Create separate transforms for training and validation/test sets. Specifically, compose (`torchvision.transforms.Compose()`) the following transformation in order.

    **Train Transforms**

    1. Random Crop to size `178` (hint: `RandomCrop()`)
    2. Random Horizontal Flip (hint: `RandomHorizontalFlip()`)
    3. Resize to size `64` (hint: `Resize()`)
    4. Convert to Tensor. (hint: `ToTensor()`)

    **Valid/Test Transforms**
    
    1. Center Crop to size `178` (hint: `CenterCrop()`)
    3. Resize to size `64`
    4. Convert to Tensor.

**Guiding Questions:**

- Why is data augmentation important here?
- Why do we use separate transformation for training and validation/testing?
- Why do we use the same transformation for validation and testing?

### 3. Extract the smile label

Since we want to classify whether a person in an image is smiling or not, we will only use the **Smiling** attribute. Therefore, we will extract the smile label from the 'attributes' list.

- Print the list of attribute names (hint: `train_set.attr_names`)
- Get the smiling index (hint: `train_set.attr_names.index("Smiling")`)
- Create a lambda function `get_smile = lambda attr: attr[smiling_idx]` to use inside the `target_transform` of `torchvision.datasets.CelebA()`. 
**Try to understand what this does**

**SANITY CHECK**:

At this point you should have the necessary variables to fill the `split`, `target_type`, `transform`, `target_transform` arguments of `torchvision.datasets.CelebA()`.

### 4. Training on a Reduced Subset
Instead of using all the available training and validation data, we will take a subset of **16,000 training examples** and **1,000 examples for validation**, as our goal here is to intentionally train our model with a small dataset:

- Create a subset of the training data of size **16000** (hint: `torch.utils.data.Subset()`)
- Create a subset of the validation data of size **1000** (hint: `torch.utils.data.Subset()`)

**SANITY CHECK**:

At this point,,

- `print(train_dataset)` should be **16000**
- `print(valid_dataset)` should be **1000**
- `print(test_dataset)` should be **19962**

### 5. (Optional): visualize the split/label distributions

- Plot a histogram of split sizes and a bar chart of smiling vs not smiling per split.

### 6. Dataloaders

Use `torch.utils.data.DataLoader` to build your `train` `valid` and `test` dataloaders.

- Use batch size 32 or 64.
- Shuffle only the **train** loader.

**SANITY CHECK**:

At this point if you do `x_batch, y_batch = next(iter(train_loader))`,

- `x_batch.shape` should be [batchsize×3×64×64]
- `y_batch.shape` should be [batchsize]

**Guiding Questions:**

- Why are these the expected shapes?

### 7. Build the CNN

- Define a **CNN architecture** in PyTorch (`nn.Module`).
- Keep it **simple first** (few conv + pooling layers).
- Add a final output layer for **binary classification**.

**Guiding Questions:**

- Why do we use convolutional layers instead of fully connected layers for images?

### 8. Training the Model

- Define a **loss function** (hint: `nn.BCEWithLogitsLoss`).
- Choose an **optimizer** (hint: `Adam` or `SGD`).
- Train for several epochs, recording/saving your training and validation accuracy.

**Guiding Questions:**
- How do we avoid overfitting with such a small dataset?
- How does the choice of optimizer affect training speed?

### 9. Evaluate & Visualize

- Plot training vs. validation accuracy/loss curves.
- **Evaluate on the test set**.
- Show some correctly vs. incorrectly classified images.

**Guiding Questions:**

- Where does the model perform poorly?
- Do misclassified examples share some pattern?

### 10. Extensions (Optional Challenges 🚀)

- Try **deeper CNNs**.
- Use **transfer learning** with a pretrained model (e.g., `ResNet18`).
- Experiment with **different hyperparameters** (e.g learning rate)
- Experiment with **different augmentation strategies**.
- Explore **class imbalance handling** if needed.

## Reflection Prompts

At the end, each student should reflect on:

1. **Model Understanding**: What features might the CNN be learning to detect a smile?
2. **Data Augmentation**: How did augmentation impact your results?
3. **Generalization**: What could improve performance further if we had more time/resources?