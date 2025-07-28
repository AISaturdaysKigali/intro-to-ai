# Training an MLP on MNIST with NumPy

In this classwork, you'll implement and train a Multi-Layer Perceptron (MLP) on the MNIST digit classification dataset. The task is broken down into manageable components, with **instructions** and **TODOs** guiding your coding efforts. This exercise builds your understanding of forward/backward passes, loss computation, parameter updates, and training loops — all **without using deep learning frameworks** like PyTorch or TensorFlow.

## Summary of Student Tasks

<img src="images/11_06.png" alt="Drawing"/>
<p style="text-align:center">Figure 1: The NN architecture for labeling handwritten digits</p>
<br>

You need to complete the following **TODOs**:

| **Section** | **Task Description** |
| ----------- | -------------------- |
| `Train-Validation-Test` | Split the data into Train-Validation-Test |
| `Linear`    | Initialize weights/bias and implement forward pass |
| `Sigmoid`   | Implement sigmoid activation function |
| `MSELoss`   | Compute mean squared error |
| `MLP`	      | Assemble network layers and implement forward pass |

Be sure to **test your implementations** as you go — syntax errors or incorrect dimensions will cause the training to crash. **ASK FOR HELP IF NEEDED!**

## 1. Load and Prepare the Dataset

We begin by loading the **MNIST** dataset using `fetch_openml`. It contains 70,000 grayscale images of handwritten digits (0-9), each of size 28x28.

```python
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X.values # This returns the array values
y = y.astype(int).values # This returns the array values
```

### Preprocessing: Normalize the data

Pixel values are originally in the range [0, 255]. We rescale them to [-1, 1] to help our MLP train more effectively.

```python
X = ((X / 255.) - .5) * 2
```

### Visualize sample digits

We plot one example of each digit (0 through 9) to understand the dataset visually.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()
for i in range(10):
    img = X[y == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
```

## 2. Train-Validation-Test Split

We split the dataset into:

- **Train set**: `55000` samples. For learning the model.
- **Validation set**: `5000` samples. For monitoring generalization during training.
- **Test set**: `10000` samples. For evaluating final performance.

**Hint:** Use `train_test_split` **twice** to do two sets of splits.

**NOTE**: Set `random_state=123` in each `train_test_split` function for **reproducibility**.

```python
from sklearn.model_selection import train_test_split

# Splits: 
# - 60k train+valid, 10k test
# - 55k train, 5k valid
```

## 3. Create Mini-batches

We implement a simple generator to yield mini-batches of data, which are used in training loops.

```python
def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]
```

## 4. Build Core Components of MLP

You will now implement various components needed for a simple neural network.

### A. Linear Layer (Fully Connected Layer)

**TODOs**:

- Initialize `weight` with small random values
- Initialize `bias` with zeros
- Initialize `grad_weight` with zeros. Same shape as `weight`
- Initialize `grad_bias` with zeros. Same shape as `bias`
- Implement `forward()`

```python
class Linear:
    def __init__(self, in_features, out_features):
        # TODO
        self.weight = ... # Initialize the weight randomly.
        self.bias = ... # Initialize the bias to all zeros.

        self.grad_weight = ... # Initialize the weight gradient to zeros. The shape should same as self.weight
        self.grad_bias = ... # Initialize the bias gradient to zeros. The shape should same as self.bias

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.input = x # Do not remove. Needed for backward computation
        
        output = ... # TODO: Compute the net input based in the linear equation.
        return output

    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weight.T)

        self.grad_weight[...] = np.dot(self.input.T, grad_output)
        self.grad_bias[...] = np.mean(grad_output, axis=0)
        return grad_input

    def parameters(self):
        """Returns the parameters of your linear layer"""
        return [self.weight, self.bias]

    def gradients(self):
        """Returns the gradients for the parameters of your linear layer"""
        return [self.grad_weight, self.grad_bias]
```

### B. Sigmoid Activation Function

**TODO:**

- Implement the sigmoid function in `forward`:

$$
\sigma(z) = \frac{1}{1 + e^{−z}​}
$$

```python
class Sigmoid:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.output = ... # TODO: Compute the sigmoid equation
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)
```

### C. Mean Squared Error Loss (MSELoss)

Even though MSE is not ideal for classification, we'll use it for simplicity, and because it's derivative is easy to understand and compute.

**TODO:**

- Implement `forward` pass:

$$
    MSE = \frac{1}{N}\Sigma(pred − target)^2
$$

```python
class MSELoss:
    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        self.pred = pred # Do not remove. Needed for backward computation
        self.target = target # Do not remove. Needed for backward computation
        error = ... # TODO: Compute the mean square error function
        return error

    def backward(self):
        return 2. * (self.pred - self.target) / self.target.shape[0]
```

## 5. Gradient Descent Optimizer

This `SGD` class performs parameter updates using gradients computed during backpropagation.

```python
class SGD:
    def __init__(self, params, grads, lr=0.1):
        self.params = params
        self.grads = grads
        self.lr = lr

    def step(self):
        """Update the parameters (weight and bias) of your netork using the Gradient Descent algorithm."""
        for p, g in zip(self.params, self.grads):
            p -= self.lr * g # gradient descent

    def zero_grad(self):
        """Zeros out the gradient of your network"""
        for g in self.grads:
            g.fill(0.0)
```

## 6. Assemble the MLP

An MLP with:

- **Input layer**: 784 neurons (28×28)
- **Hidden layer**: 50 neurons with Sigmoid activation
- **Output layer**: 10 neurons (one per digit) with Sigmoid activation

**TODOs:**

- In `__init__`: Create two linear layers and two activations
- In `forward`: Pass input through layers in sequence
- e.g. `x → linear1 → sigmoid → linear2 → sigmoid`

```python
class MLP:
    def __init__(self, in_features, hidden_size, out_features):
        # TODO Initialize your network

        # Do not change this variable names. The are needed for backward computation
        self.linear1 = ...
        self.act1 = ...
        self.linear2 = ...
        self.act2 = ...

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Forward propagation through your network"""
        # TODO Pass x (input) to your network and get y (output)
        return ...

    def backward(self, grad_output):
        """Backpropagation through your network"""
        grad_z2 = self.act2.backward(grad_output)
        grad_a1 = self.linear2.backward(grad_z2)

        grad_z1 = self.act1.backward(grad_a1)
        _ = self.linear1.backward(grad_z1)

    def parameters(self):
        """Return the parameters of your entire model/network"""
        return self.linear1.parameters() + self.linear2.parameters()

    def gradients(self):
        """Return the gradients of your entire model/network"""
        return self.linear1.gradients() + self.linear2.gradients()
```

## 7. Training and Evaluation

You are provided with the training loop and evaluation functions.

### Training Function

- Performs **forward** and **backward** passes for each mini-batch
- Updates weights using `SGD`

```python
def train(model, train_loader, optimizer, criterion):
    total_loss = 0.0
    num_samples = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        preds = model(X_batch)
        loss = criterion(preds, int_to_onehot(y_batch))

        grad_output = criterion.backward() # Gradient of the loss w.r.t output

        model.backward(grad_output)
        optimizer.step()

        total_loss += loss
        num_samples += len(y_batch)

    return total_loss / num_samples
```
### Validation Function

- Computes loss and accuracy on the validation set

```python
def valid(model, valid_loader, criterion):
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for X_batch, y_batch in valid_loader:
        preds = model(X_batch)
        loss = criterion(preds, int_to_onehot(y_batch))

        total_loss += loss

        predicted_labels = np.argmax(preds, axis=1)
        correct_predictions += (predicted_labels == y_batch).sum()
        total_predictions += len(y_batch)

    total_loss /= total_predictions
    accuracy = (correct_predictions / total_predictions)*100
    return total_loss, accuracy
```

### Convert Targets

Convert integer labels (0–9) to one-hot vectors:

```python
def int_to_onehot(y, num_labels=10):
      return np.eye(num_labels)[y]
```

## 8. Instantiate Model, Loss Function, and Optimizer

Before training, we must initialize our network and supporting components.

```python
in_features = 28 * 28      # Each image is 28x28 pixels
hidden_size = 50           # Size of the hidden layer (can experiment)
out_features = 10          # One output per digit class (0-9)
learning_rate = 0.1        # Step size for SGD updates

np.random.seed(123)        # For reproducible results

model = MLP(in_features=in_features, hidden_size=hidden_size, out_features=out_features)
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), model.gradients(), lr=learning_rate)
```

## 9. Run the Training Loop

Now that everything is in place, we can train the MLP using mini-batch SGD:

```python
num_epochs = 50
batch_size = 100

train_losses, valid_losses = [], []
for epoch in range(num_epochs):
    # Usually in the PyTorch way, dataloader SHOULD NOT be in the epoch loop
    train_loader = minibatch_generator(X_train, y_train, minibatch_size=batch_size)
    valid_loader = minibatch_generator(X_valid, y_valid, minibatch_size=batch_size)

    train_loss = train(model, train_loader, optimizer, criterion=loss_fn)
    valid_loss, valid_acc = valid(model, valid_loader, criterion=loss_fn)

    # Collect losses for plotting
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f"Epoch: {epoch+1:02d}/{num_epochs} | Train MSE: {train_loss:.4f} | Valid MSE: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%")
```

### Plot the Loss Curves

After training completes, plot both the training and validation loss on **a single graph** to visually compare them.

- Use `matplotlib` to plot the two loss curves on the same figure.
- Label your axes and add a legend.

```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
```

## 10. Final Evaluation on Test Set

Measure final generalization accuracy using:

```python
def test(model, test_loader):
    correct_predictions = 0
    total_predictions = 0
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)

        predicted_labels = np.argmax(preds, axis=1)
        correct_predictions += (predicted_labels == y_batch).sum()
        total_predictions += len(y_batch)

    accuracy = (correct_predictions / total_predictions)*100
    return accuracy


test_loader = minibatch_generator(X_test, y_test, minibatch_size=100)
test_acc = test(model, test_loader)
print(f'Test accuracy: {test_acc:.2f}%')
```

## Optional (for Curious Minds)

- Replace `MSELoss` with **cross-entropy** for better performance
- Implement **ReLU** activation (for hidden layers) and compare with sigmoid
- Add support for more than one hidden layer