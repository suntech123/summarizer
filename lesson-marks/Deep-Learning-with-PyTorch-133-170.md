# Chapter 5: The mechanics of learning

This chapter covers:

* Understanding how algorithms can learn from data
* Reframing learning as parameter estimation, using differentiation and gradient descent
* Walking through a simple learning algorithm
* How PyTorch supports learning with autograd

With the blooming of machine learning over the last decade, machines learning from experience have become mainstream. A learning algorithm is presented with input data paired with desired outputs, enabling it to produce correct outputs when fed new data similar to its training set. In deep learning, this process works even when inputs and desired outputs come from different domains.

## 5.1 A timeless lesson in modeling

Building models to explain input/output relationships dates back centuries, such as when Johannes Kepler formulated his three laws of planetary motion in the early 1600s based on data from Tycho Brahe. Lacking Newton's law of gravitation, Kepler extrapolated the simplest geometric model that fit the observations, taking six years of iteration. We can see this process in figure 5.1.

[[img-01]]
Figure 5.1 Johannes Kepler considers multiple candidate models that might fit the data at hand, settling on an ellipse.

Kepler's first law states that planet orbits are ellipses with the Sun at one focus, allowing him to estimate shape and size parameters from data to predict positions. His second law enabled predicting when a planet would reach a specific point in space over time. Without modern computing tools, Kepler tried different shapes, fit curves to subset observations, calculated predicted positions, and checked agreement with remaining observations.

Kepler's historical methodology outlines key steps of data science: gathering quality data, visualizing it, choosing a simple candidate model, splitting data for independent validation, iterating on parameter values, and validating on held-out data. Deviating from these steps often leads to failure in modeling tasks. Fitting data or making an algorithm learn involves estimating unknown parameters of a model function from observations.

Learning from data presumes an underlying model capable of approximating a wide family of functions rather than being engineered for a single specific problem. While a neural network could fit planetary trajectories without requiring Kepler's specific geometric insights, deriving underlying universal laws from generic models is more challenging. PyTorch simplifies automatic general function-fitting by making it easy to express analytical derivatives of fitting errors with respect to parameters.

## 5.2 Learning is just parameter estimation

To learn parameter estimation, we take noisy data, select a model, and adjust parameters for accurate predictions on new data. Figure 5.2 shows the high-level overview of what we'll implement by the end of the chapter. Given inputs, ground truth outputs, and initial weights, the model evaluates a forward pass to measure error against ground truth, then computes parameter gradients via the chain rule during a backward pass to update weights until error is minimized.

[[img-02]]
Figure 5.2 Our mental model of the learning process

Building a learning algorithm starting from first principles by hand clarifies core concepts before leveraging PyTorch automation. Covering these essentials establishes the foundational mechanics for training deep neural networks, even using simple non-neural models.

### 5.2.1 A hot problem

After acquiring an analog thermometer lacking unit markings, we build a dataset of paired readings and standard temperature measurements. By iteratively adjusting model weights to minimize reading errors, we calibrate the device to output interpretable temperature values.

### 5.2.2 Gathering some data

We record paired measurements in Celsius alongside unknown thermometer units over two weeks. Both sets of data contain measurement noise from the physical devices and human reading approximations.

```python
# In [2]:
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0,  4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

```

### 5.2.3 Visualizing the data

A quick plot of our data in figure 5.3 tells us that it's noisy, but we think there's a pattern here.

[[img-03]]
Figure 5.3 Our unknown data just might follow a linear model.

### 5.2.4 Choosing a linear model as a first try

We assume a linear relationship between unknown units and Celsius values, scaled by weight parameter $w$ and offset by bias parameter $b$.

$$t\_c = w \times t\_u + b$$

Estimating parameters $w$ and $b$ requires minimizing the discrepancy between predicted model outputs and actual measured Celsius values. This parameter optimization process relies on defining a loss function that outputs high values for large errors and reaches its minimum when predictions match ground truth.

## 5.3 Less loss is what we want

A loss function computes a single scalar value measuring difference between target outputs and predicted outputs. For predicted temperatures $t\_p$ and targets $t\_c$, raw errors are represented as $t\_p - t\_c$.

To keep loss positive regardless of error sign, standard choices include absolute difference $\vert{}t\_p - t\_c\vert{}$ or squared difference $(t\_p - t\_c)^2$. Both function choices are convex with clear minimums at zero, though squared difference produces smooth derivatives near zero while absolute difference has an undefined derivative at convergence.

[[img-04]]
Figure 5.4 Absolute difference versus difference squared

Squared error penalizes large outliers more heavily than absolute difference, prioritizing the correction of large errors. While neural network loss functions are non-convex, squared differences provide well-behaved optimization properties for linear models.

### 5.3.1 From problem back to PyTorch

Implementing the model in PyTorch involves defining a Python function that computes predictions using broadcasted scalar parameters.

```python
# In [3]:
def model(t_u, w, b):
    return w * t_u + b

```

The mean squared error loss function averages element-wise squared differences across the input tensor.

```python
# In [4]:
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

```

Initializing parameters to default values allows evaluating initial predictions and observing starting loss values.

```python
# In [5]:
w = torch.ones(())
b = torch.zeros(())

t_p = model(t_u, w, b)
t_p

```

```python
# Out [5]:
tensor([35.7000, 55.9000, 58.2000, 81.9000, 56.3000, 48.9000, 33.9000, 21.8000,
        48.4000, 60.4000, 68.4000])

```

```python
# In [6]:
loss = loss_fn(t_p, t_c)
loss

```

```python
# Out [6]:
tensor(1763.8846)

```

Broadcasting rules match tensor dimensions during binary operations when operand shapes differ. If dimensions match, size is 1, or one tensor has fewer dimensions, PyTorch automatically expands elements across missing axes.

```python
# In [7]:
x = torch.ones(())
y = torch.ones(3, 1)
z = torch.ones(1, 3)
a = torch.ones(2, 1, 1)
print(f"shapes: x: {x.shape}, y: {y.shape}")
print(f"        z: {z.shape}, a: {a.shape}")
print("x * y:", (x * y).shape)
print("y * z:", (y * z).shape)
print("y * z * a:", (y * z * a).shape)

```

```python
# Out [7]:
shapes: x: torch.Size([]), y: torch.Size([3, 1])
        z: torch.Size([1, 3]), a: torch.Size([2, 1, 1])
x * y: torch.Size([3, 1])
y * z: torch.Size([3, 3])
y * z * a: torch.Size([2, 3, 3])

```

## 5.4 Down along the gradient

Gradient descent optimizes parameters by estimating the rate of change of loss with respect to each parameter and updating them in the direction of decrease. Figure 5.5 illustrates this process as adjusting control knobs to minimize loss.

[[img-05]]
Figure 5.5 A cartoon depiction of the optimization process, where a person with knobs for w and b searches for the direction to turn the knobs that makes the loss decrease

### 5.4.1 Decreasing loss

We estimate local rate of change by adding a small value $\delta$ to a parameter and measuring loss difference over $2\delta$.

```python
# In [8]:
delta = 0.1

loss_rate_of_change_w = \
    (loss_fn(model(t_u, w + delta, b), t_c) - 
     loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)

```

Updating parameters proportionally to their loss rate of change scaled by a small `learning_rate` minimizes loss over iterations.

```python
# In [9]:
learning_rate = 1e-2

w = w - learning_rate * loss_rate_of_change_w

```

The same rate-of-change estimation and update step applies to bias parameter $b$.

```python
# In [10]:
loss_rate_of_change_b = \
    (loss_fn(model(t_u, w, b + delta), t_c) - 
     loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta)

b = b - learning_rate * loss_rate_of_change_b

```

### 5.4.2 Getting analytical

Finite-difference approximation requires multiple model evaluations and does not scale well to high-dimensional parameter spaces. Figure 5.6 shows how analytical derivatives evaluate infinitesimally small neighborhoods directly.

[[img-06]]
Figure 5.6 Differences in the estimated directions for descent when evaluating them at discrete locations versus analytically

Applying the chain rule decomposes the loss derivative into derivative of loss with respect to model output times model derivative with respect to parameter.

```python
# In [11]:
def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs

```

Model derivatives with respect to weight $w$ and bias $b$ are derived analytically from the linear function.

```python
# In [12]:
def dmodel_dw(t_u, w, b):
    return t_u

```

```python
# In [13]:
def dmodel_db(t_u, w, b):
    return 1.0

```

Combining these components yields a gradient function returning partial derivatives averaged over all input data points. Figure 5.7 illustrates this breakdown in mathematical notation.

```python
# In [14]:
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

```

[[img-07]]
Figure 5.7 The derivative of the loss function with respect to the weights

### 5.4.3 Iterating to fit the model

An epoch defines a complete training iteration during which parameters are updated across all training samples.

```python
# In [15]:
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)
        
        params = params - learning_rate * grad
        
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
    return params

```

Running the training loop with `learning_rate = 1e-2` causes parameter updates to explode, producing infinite loss.

```python
# In [17]:
training_loop(
    n_epochs = 100,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_u,
    t_c = t_c)

```

```python
# Out [17]:
Epoch 1, Loss 1763.884644
Params: tensor([-44.1730,  -0.8260])
Grad:   tensor([4517.2969,   82.6000])
Epoch 2, Loss 5802485.500000
Params: tensor([2568.4014,   45.1637])
Grad:   tensor([-261257.4219,   -4598.9712])
Epoch 3, Loss 19408035840.000000
Params: tensor([-148527.7344,   -2616.3933])
Grad:   tensor([15109614.0000,    266155.7188])
Epoch 10, Loss 90901154706620645225508955521810432.000000
Params: tensor([3.2144e+17, 5.6621e+15])
Grad:   tensor([-3.2700e+19, -5.7600e+17])
Epoch 11, Loss inf
Params: tensor([-1.8590e+19, -3.2746e+17])
Grad:   tensor([1.8912e+21, 3.3313e+19])
tensor([-1.8590e+19, -3.2746e+17])

```

Large parameter updates cause oscillations back and forth, diverging instead of converging. Figure 5.8 compares divergent optimization steps against stable converging steps.

[[img-08]]
Figure 5.8 Top: Diverging optimization on a convex function (parabola-like) due to large steps. Bottom: Converging optimization with small steps.

Reducing the learning rate to `1e-4` stabilizes training, though parameter updates become extremely slow.

```python
# In [18]:
training_loop(
    n_epochs = 100,
    learning_rate = 1e-4,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_u,
    t_c = t_c)

```

```python
# Out [18]:
Epoch 1, Loss 1763.884644
Params: tensor([ 0.5483, -0.0083])
Grad:   tensor([4517.2969,   82.6000])
Epoch 2, Loss 323.090546
Params: tensor([ 0.3623, -0.0118])
Grad:   tensor([1859.5493,   35.7843])
Epoch 3, Loss 78.929634
Params: tensor([ 0.2858, -0.0135])
Grad:   tensor([765.4667,  16.5122])
Epoch 10, Loss 29.105242
Params: tensor([ 0.2324, -0.0166])
Grad:   tensor([1.4803, 3.0544])
Epoch 11, Loss 29.104168
Params: tensor([ 0.2323, -0.0169])
Grad:   tensor([0.5781, 3.0384])
Epoch 99, Loss 29.023582
Params: tensor([ 0.2327, -0.0435])
Grad:   tensor([-0.0533,  3.0226])
Epoch 100, Loss 29.022669
Params: tensor([ 0.2327, -0.0438])
Grad:   tensor([-0.0532,  3.0226])
tensor([ 0.2327, -0.0438])

```

### 5.4.4 Normalizing inputs

Weight gradients are roughly 50 times larger than bias gradients because inputs exist on a larger scale. Scaling input tensor `t_u` by `0.1` brings gradient magnitudes into similar ranges, enabling stable updates with `learning_rate = 1e-2`.

```python
# In [19]:
t_un = 0.1 * t_u

```

```python
# In [20]:
training_loop(
    n_epochs = 100,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_un,
    t_c = t_c)

```

```python
# Out [20]:
Epoch 1, Loss 80.364342
Params: tensor([1.7761, 0.1064])
Grad:   tensor([-77.6140, -10.6400])
Epoch 2, Loss 37.574917
Params: tensor([2.0848, 0.1303])
Grad:   tensor([-30.8623,  -2.3864])
Epoch 3, Loss 30.871077
Params: tensor([2.2094, 0.1217])
Grad:   tensor([-12.4631,   0.8587])
Epoch 10, Loss 29.030487
Params: tensor([ 2.3232, -0.0710])
Grad:   tensor([-0.5355,  2.9295])
Epoch 11, Loss 28.941875
Params: tensor([ 2.3284, -0.1003])
Grad:   tensor([-0.5240,  2.9264])
Epoch 99, Loss 22.214186
Params: tensor([ 2.7508, -2.4910])
Grad:   tensor([-0.4453,  2.5208])
Epoch 100, Loss 22.148710
Params: tensor([ 2.7553, -2.5162])
Grad:   tensor([-0.4446,  2.5165])
tensor([ 2.7553, -2.5162])

```

Training for 5,000 epochs yields parameters corresponding closely to Fahrenheit-to-Celsius conversion constants.

```python
# In [21]:
params = training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_un,
    t_c = t_c,
    print_params = False)

params

```

```python
# Out [21]:
Epoch 1, Loss 80.364342
Epoch 2, Loss 37.574917
Epoch 3, Loss 30.871077
Epoch 10, Loss 29.030487
Epoch 11, Loss 28.941875
Epoch 99, Loss 22.214186
Epoch 100, Loss 22.148710
Epoch 4000, Loss 2.927680
Epoch 5000, Loss 2.927648
tensor([ 5.3671, -17.3012])

```

### 5.4.5 Visualizing (again)

Plotting the fitted linear model against original raw data demonstrates a strong linear fit despite reading noise.

```python
# In [22]:
%matplotlib inline
from matplotlib import pyplot as plt

t_p = model(t_un, *params)

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')

```

[[img-09]]
Figure 5.9 The plot of our linear-fit model (solid line) versus our input data (circles)

## 5.5 PyTorch's autograd: Backpropagating all things

Backpropagation calculates gradients of composite function graphs backward using the chain rule.

### 5.5.1 Computing the gradient automatically

PyTorch autograd tracks operation history on tensors to automate derivative computations.

```python
# In [3]:
def model(t_u, w, b):
    return w * t_u + b

# In [4]:
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

```

Constructing tensors with `requires_grad=True` instructs PyTorch to track operations for automatic gradient evaluation.

```python
# In [5]:
params = torch.tensor([1.0, 0.0], requires_grad=True)

```

Unpopulated gradient attributes evaluate to `None` until `backward()` is called on a target scalar loss.

```python
# In [6]:
params.grad is None

```

```python
# Out [6]:
True

```

Calling `loss.backward()` populates derivative values inside parameter `.grad` attributes. Figure 5.10 shows the forward operation graph and reverse gradient traversal.

```python
# In [7]:
loss = loss_fn(model(t_u, *params), t_c)
loss.backward()

params.grad

```

```python
# Out [7]:
tensor([4517.2969,   82.6000])

```

[[img-10]]
Figure 5.10 The forward graph and backward graph of the model as computed with autograd

Gradients accumulate into leaf node `.grad` attributes on repeated `backward()` calls, requiring explicit zeroing using `.zero_()`.

```python
# In [8]:
if params.grad is not None:
    params.grad.zero_()

```

Updating parameters in an autograd training loop requires encapsulating modifications inside a `torch.no_grad()` context block.

```python
# In [9]:
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_()
            
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
        
        with torch.no_grad():
            params -= learning_rate * params.grad
            
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            
    return params

```

Executing the autograd training loop reproduces identical parameters to manual analytical gradient computation.

```python
# In [10]:
training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0], requires_grad=True),
    t_u = t_un,
    t_c = t_c)

```

```python
# Out [10]:
Epoch 500, Loss 7.860116
Epoch 1000, Loss 3.828538
Epoch 1500, Loss 3.092191
Epoch 2000, Loss 2.957697
Epoch 2500, Loss 2.933134
Epoch 3000, Loss 2.928648
Epoch 3500, Loss 2.927830
Epoch 4000, Loss 2.927679
Epoch 4500, Loss 2.927652
Epoch 5000, Loss 2.927647
tensor([ 5.3671, -17.3012], requires_grad=True)

```

### 5.5.2 Optimizers a la carte

The `torch.optim` module provides optimization algorithms that automate parameter updates.

```python
# In [5]:
import torch.optim as optim

dir(optim)

```

```python
# Out [5]:
['ASGD',
 'Adadelta',
 'Adagrad',
 'Adam',
 'Adamax',
 'LBFGS',
 'Optimizer',
 'RMSprop',
 'Rprop',
 'SGD',
 'SparseAdam']

```

Optimizers hold references to parameter tensors, exposing `zero_grad()` to reset gradients and `step()` to update values. Figure 5.11 illustrates parameter registration and update steps.

[[img-11]]
Figure 5.11 (A) Conceptual representation of how an optimizer holds a reference to parameters. (B) After a loss is computed from inputs, (C) a call to backward leads to .grad being populated on parameters. (D) At that point, the optimizer can access .grad and compute the parameter updates.

Instantiating an SGD optimizer automates gradient subtraction during step execution.

```python
# In [6]:
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5
optimizer = optim.SGD([params], lr=learning_rate)

```

```python
# In [7]:
t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)
loss.backward()

optimizer.step()

params

```

```python
# Out [7]:
tensor([9.5483e-01, -8.2600e-04], requires_grad=True)

```

Placing `optimizer.zero_grad()` before `loss.backward()` prevents gradient accumulation across steps.

```python
# In [8]:
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)

optimizer.zero_grad()
loss.backward()
optimizer.step()

params

```

```python
# Out [8]:
tensor([1.7761, 0.1064], requires_grad=True)

```

Integrating the optimizer into `training_loop` streamlines parameter management.

```python
# In [9]:
def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            
    return params

```

```python
# In [10]:
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    params = params,
    t_u = t_un,
    t_c = t_c)

```

```python
# Out [10]:
Epoch 500, Loss 7.860118
Epoch 1000, Loss 3.828538
Epoch 1500, Loss 3.092191
Epoch 2000, Loss 2.957697
Epoch 2500, Loss 2.933134
Epoch 3000, Loss 2.928648
Epoch 3500, Loss 2.927830
Epoch 4000, Loss 2.927680
Epoch 4500, Loss 2.927651
Epoch 5000, Loss 2.927648
tensor([ 5.3671, -17.3012], requires_grad=True)

```

Testing adaptive optimizers like Adam demonstrates insensitivity to input scaling, operating stably on unnormalized data.

```python
# In [11]:
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate)

training_loop(
    n_epochs = 2000,
    optimizer = optimizer,
    params = params,
    t_u = t_u,
    t_c = t_c)

```

```python
# Out [11]:
Epoch 500, Loss 7.612903
Epoch 1000, Loss 3.086700
Epoch 1500, Loss 2.928578
Epoch 2000, Loss 2.927646
tensor([ 0.5367, -17.3021], requires_grad=True)

```

### 5.5.3 Training, validation, and overfitting

Evaluating models on independent validation data detects overfitting where models memorize training points without generalizing. Figure 5.12 depicts splitting data into distinct training and validation sets.

[[img-12]]
Figure 5.12 Conceptual representation of a data-producing process and the collection and use of training data and independent validation data

High training loss indicates insufficient model capacity or uninformative features. Divergence between training and validation loss signals overfitting. Figure 5.13 illustrates severe overfitting from over-parameterization.

[[img-13]]
Figure 5.13 Rather extreme example of overfitting

Mitigating overfitting involves gathering more data, adding regularization penalties, augmenting inputs with noise, or choosing simpler models. Shuffling data indices with `torch.randperm` splits datasets into validation and training tensors.

```python
# In [12]:
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_indices, val_indices

```

```python
# Out [12]:
(tensor([9, 6, 5, 8, 4, 7, 0, 1, 3]), tensor([ 2, 10]))

```

Training and validation subsets are sliced using permuted index tensors.

```python
# In [13]:
train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u

```

Evaluating validation loss alongside training loss inside the main loop tracks generalization behavior.

```python
# In [14]:
def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u,
                  train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
        
        val_t_p = model(val_t_u, *params)
        val_loss = loss_fn(val_t_p, val_t_c)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")
            
    return params

```

```python
# In [15]:
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    params = params,
    train_t_u = train_t_un,
    val_t_u = val_t_un,
    train_t_c = train_t_c,
    val_t_c = val_t_c)

```

```python
# Out [15]:
Epoch 1, Training loss 66.5811, Validation loss 142.3890
Epoch 2, Training loss 38.8626, Validation loss 64.0434
Epoch 3, Training loss 33.3475, Validation loss 39.4590
Epoch 500, Training loss 7.1454, Validation loss 9.1252
Epoch 1000, Training loss 3.5940, Validation loss 5.3110
Epoch 1500, Training loss 3.0942, Validation loss 4.1611
Epoch 2000, Training loss 3.0238, Validation loss 3.7693
Epoch 2500, Training loss 3.0139, Validation loss 3.6279
Epoch 3000, Training loss 3.0125, Validation loss 3.5756
tensor([ 5.1964, -16.7512], requires_grad=True)

```

Figure 5.14 summarizes training versus validation loss curves under various overfitting conditions.

[[img-14]]
Figure 5.14 Overfitting scenarios when looking at the training (solid line) and validation (dotted line) losses. (A) Training and validation losses do not decrease; the model is not learning due to no information in the data or insufficient capacity of the model. (B) Training loss decreases while validation loss increases: overfitting. (C) Training and validation losses decrease exactly in tandem. Performance may be improved further as the model is not at the limit of overfitting. (D) Training and validation losses have different absolute values but similar trends: overfitting is under control.

### 5.5.4 Autograd nits and switching it off

Evaluating training and validation passes creates separate computation graphs sharing only leaf parameter tensors. Figure 5.15 shows graph branching and backpropagation routing when `backward()` is called.

[[img-15]]
Figure 5.15 Diagram showing how gradients propagate through a graph with two losses when .backward is called on one of them

Disabling autograd construction during validation passes via `torch.no_grad()` avoids unnecessary graph building overhead.

```python
# In [16]:
def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u,
                  train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
        
        with torch.no_grad():
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad == False
            
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

```

Dynamic graph creation can be toggled conditionally using `torch.set_grad_enabled(is_train)` to share forward pass implementations.

```python
# In [17]:
def calc_forward(t_u, t_c, is_train):
    with torch.set_grad_enabled(is_train):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        return loss

```

## 5.6 Conclusion

We started this chapter with a big question: how is it that a machine can learn from examples? We spent the rest of the chapter describing the mechanism with which a model can be optimized to fit data. We chose to stick with a simple model in order to see all the moving parts without unneeded complications.

Now that we've had our fill of appetizers, in chapter 6 we'll finally get to the main course: using a neural network to fit our data. We'll work on solving the same thermometer problem, but with the more powerful tools provided by the torch.nn module. We'll adopt the same spirit of using this small problem to illustrate the larger uses of PyTorch. The problem doesn't need a neural network to reach a solution, but it will allow us to develop a simpler understanding of what's required to train a neural network.

## 5.7 Exercise

1 Redefine the model to be $w2 * t\_u2 + w1 * t\_u + b$.

* a What parts of the training loop, and so on, need to change to accommodate this redefinition?
* b What parts are agnostic to swapping out the model?
* c Is the resulting loss higher or lower after training?
* d Is the actual result better or worse?

## 5.8 Summary

* Linear models are the simplest reasonable model to use to fit data.
* Convex optimization techniques can be used for linear models, but they do not generalize to neural networks, so we focus on stochastic gradient descent for parameter estimation.
* Deep learning can be used for generic models that are not engineered for solving a specific task, but instead can be automatically adapted to specialize themselves on the problem at hand.
* Learning algorithms amount to optimizing parameters of models based on observations. A loss function is a measure of the error in carrying out a task, such as the error between predicted outputs and measured values. The goal is to get the loss function as low as possible.
* The rate of change of the loss function with respect to the model parameters can be used to update the same parameters in the direction of decreasing loss.
* The optim module in PyTorch provides a collection of ready-to-use optimizers for updating parameters and minimizing loss functions.
* Optimizers use the autograd feature of PyTorch to compute the gradient for each parameter, depending on how that parameter contributes to the final output. This allows users to rely on the dynamic computation graph during complex forward passes.
* Context managers like with torch.no_grad(): can be used to control autograd's behavior.
* Data is often split into separate sets of training samples and validation samples. This lets us evaluate a model on data it was not trained on.
* Overfitting a model happens when the model's performance continues to improve on the training set but degrades on the validation set. This is usually due to the model not generalizing, and instead memorizing the desired outputs for the training set.