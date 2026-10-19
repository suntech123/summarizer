# Chapter 6: Using a neural network to fit the data

This chapter covers building artificial neural networks to fit non-linear datasets. It explores non-linear activation functions, multi-layer architectures, and PyTorch's `nn` module primitives.

## 6.1 Artificial neurons

At the heart of deep learning are artificial neurons, mathematically represented as vector products followed by scalar bias additions and activation functions. The learning process relies on backpropagation to adjust parameters based on loss optimization. Figure 6.1 shows a high-level representation of a neuron performing linear transformation and non-linear activation.

[[img-01]]
Figure 6.1 - An artificial neuron: inputs are multiplied by weights, summed with a bias, and passed through an activation function to produce an output.

### 6.1.1 Composing a multilayer network

Single linear units cannot capture non-linear relationships. Multi-layer networks combine linear units with activation functions to approximate complex functions. Figure 6.2 shows the architecture of a multi-layer feedforward neural network.

[[img-02]]
Figure 6.2 - A multilayer feedforward neural network, consisting of an input layer, one or more hidden layers, and an output layer.

### 6.1.2 Understanding the error surface

Linear regression problems present convex loss surfaces with a single global minimum. Multi-layer neural networks generate non-convex error surfaces featuring multiple local minima, saddle points, and plateaus. Optimization in non-convex surfaces depends heavily on learning rates and initialization.

### 6.1.3 Activation functions

Activation functions introduce non-linearity, enabling networks to fit complex functions. Without non-linear activation functions, multi-layer networks collapse into equivalent single-layer linear transformations.

#### THE SIGMOID ACTIVATION

The sigmoid function maps real values to the range $(0, 1)$, bounded by horizontal asymptotes.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Figure 6.3 displays the plot of the sigmoid function along with its derivative.

[[img-03]]
Figure 6.3 - The sigmoid function and its derivative.

#### THE TANH ACTIVATION

The hyperbolic tangent function maps inputs to the range $(-1, 1)$, providing zero-centered outputs.

$$\operatorname{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Figure 6.4 illustrates the plot of the $\operatorname{tanh}$ activation function.

[[img-04]]
Figure 6.4 - The hyperbolic tangent ($\operatorname{tanh}$) function.

#### THE RELU ACTIVATION

The Rectified Linear Unit (ReLU) sets negative values to zero while passing positive values directly.

$$\operatorname{ReLU}(x) = \max(0, x)$$

Figure 6.5 presents the plot of the ReLU activation function.

[[img-05]]
Figure 6.5 - The Rectified Linear Unit (ReLU) function.

### 6.1.4 What activation functions do

Activation functions modulate outputs across hidden units, enabling selective neuron activation. In hidden layers, activations reshape input spaces into linearly separable representations. In output layers, activations restrict outputs to desired ranges or probability distributions.

Figure 6.6 demonstrates how linear transformations combined with non-linear activations shape output curves.

[[img-06]]
Figure 6.6 - Combining linear functions and activations allows constructing complex non-linear curves.

Figure 6.7 shows the step-by-step composition of multi-layer unit outputs.

[[img-07]]
Figure 6.7 - Output composition of two linear units passed through activation functions.

## 6.2 The PyTorch nn module

PyTorch provides the `torch.nn` module to build complex neural network architectures easily. The fundamental building block in `nn` is `nn.Module`, which manages parameter state and subcommand executions.

### 6.2.1 Using Linear as a replacement for polynomial

`nn.Linear` applies a linear transformation to incoming data: $y = x W^T + b$. It manages its own internal weight and bias parameters automatically.

```python
# In [2]:
import torch
import torch.nn as nn

linear_model = nn.Linear(1, 1)
linear_model(val_t)

```

Inspecting module parameters reveals automatically created weights and biases.

```python
# In [3]:
linear_model.weight
# Out [3]:
Parameter containing:
tensor([[-0.4869]], requires_grad=True)

```

```python
# In [4]:
linear_model.bias
# Out [4]:
Parameter containing:
tensor([0.4303], requires_grad=True)

```

### 6.2.2 A batch of inputs

PyTorch modules expect input tensors to have a batch dimension as the first dimension. Shape reshaping converts $1\text{D}$ inputs into $2\text{D}$ batch matrices of size $B \times N$. Figure 6.8 illustrates batching indexing over multiple data samples.

[[img-08]]
Figure 6.8 - Unbatched versus batched tensor inputs in PyTorch modules.

```python
# In [5]:
x = torch.val_t.unsqueeze(1)
linear_model(x)

```

### 6.2.3 Optimization with PyTorch nn

Modules expose parameters through the `parameters()` method, which can be passed directly to optimizers.

```python
# In [6]:
optimizer = torch.optim.SGD(
    linear_model.parameters(),
    lr=1e-2)

```

Loss functions are also provided as `nn` modules, such as `nn.MSELoss`.

```python
# In [7]:
loss_fn = nn.MSELoss()

```

The refactored training loop uses `nn.Linear` and `nn.MSELoss`.

```python
# In [8]:
def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val,
                  t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)
        
        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val, t_c_val)
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  f" Validation loss {loss_val.item():.4f}")

```

Executing the refactored training loop shows model parameter updates over epochs.

```python
# In [9]:
linear_model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(linear_model.parameters(), lr=1e-2)

training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    model = linear_model,
    loss_fn = loss_fn,
    t_u_train = t_un_train,
    t_u_val = t_un_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val)

print()
print(linear_model.weight)
print(linear_model.bias)

```

## 6.3 Finally a neural network

Replacing the single linear layer with a multi-layer neural network enables fitting non-linear data patterns.

### 6.3.1 Replacing the linear model

`nn.Sequential` chains multiple `nn.Module` layers together sequentially.

```python
# In [10]:
seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1))
seq_model

```

### 6.3.2 Inspecting parameters

Printing `seq_model.parameters()` lists parameter shapes for all contained submodules.

```python
# In [11]:
[param.shape for param in seq_model.parameters()]
# Out [11]:
[torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]

```

Named parameters facilitate parameter inspection across modules.

```python
# In [12]:
for name, param in seq_model.named_parameters():
    print(name, param.shape)

```

Custom names can be assigned using `OrderedDict`.

```python
# In [13]:
from collections import OrderedDict

seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
]))
seq_model

```

### 6.3.3 Comparing to the linear model

Training the multi-layer neural network shows lower training loss compared to the single linear model.

```python
# In [14]:
optimizer = torch.optim.SGD(seq_model.parameters(), lr=1e-3)

training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_un_train,
    t_u_val = t_un_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val)

print('output', seq_model(t_un_val))
print('answer', t_c_val)
print('hidden', seq_model.hidden_linear.weight.grad)

```

Evaluating model predictions against continuous ranges illustrates non-linear fitting capabilities. Figure 6.9 compares linear and multi-layer neural network fits on the temperature dataset.

[[img-09]]
Figure 6.9 - Neural network fitting non-linear temperature data compared to linear regression.

```python
# In [15]:
from matplotlib import pyplot as plt

t_range = torch.arange(20.0, 90.0).unsqueeze(1)

plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u_train.numpy() * 10 + 35, t_c_train.numpy(), 'o')
plt.plot(t_range.numpy(), seq_model(t_range * 0.1 - 3.5).detach().numpy(), 'c-')
plt.plot(t_range.numpy(), t_range.numpy() * 0.5 - 17.3, 'k--')

```

## 6.4 Exercises

1. Experiment with the number of hidden neurons in the neural network model, as well as the learning rate.
a. What changes does this produce in the output of the model?
b. Can you get the model to produce a line that overfits the data points?
2. Load the 3D images dataset from chapter 4 (CT scans).
a. Build a model using `nn.Sequential` with `nn.Linear` layers that takes the 3D volume as input and outputs a single float value.
b. How many parameters does this model have?

## 6.5 Summary

* Neural networks can be trained using gradient descent to fit non-linear relationships in data.
* PyTorch provides a rich library of building blocks in the `nn` module, including layer types and loss functions.
* `nn.Module` is the base class for all neural network components; submodules can contain other modules and parameters.
* Non-linear activation functions allow multi-layer networks to approximate complex mathematical functions.
* `nn.Sequential` provides an easy way to stack layers in a feedforward network.