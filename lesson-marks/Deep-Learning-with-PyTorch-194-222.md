# Chapter 7: Telling birds from airplanes: Learning from images

The previous chapter covered the mechanics of gradient descent optimization using a simple one-input, one-output regression model. This chapter expands neural network foundations by addressing image recognition tasks. Image recognition historically demonstrated the powerful capabilities of deep learning methods.

## 7.1 A dataset of tiny images

Image recognition is best approached step by step using a dataset of small images. While MNIST contains handwritten digits, CIFAR-10 provides a classic alternative consisting of 60,000 $32\times32$ RGB images divided into 10 distinct classes. The `torchvision` module enables downloading and loading CIFAR-10 as PyTorch tensors.

[[img-01]]
Figure 7.1 - Image samples from all CIFAR-10 classes

### 7.1.1 Downloading CIFAR-10

The `torchvision.datasets` module is imported to download the training and validation subsets of CIFAR-10.

```python
# In [2]:
from torchvision import datasets
data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

```

Datasets downloaded via `torchvision.datasets` inherit directly from `torch.utils.data.Dataset`.

```python
# In [4]:
type(cifar10).__mro__
# Out [4]:
(torchvision.datasets.cifar.CIFAR10,
 torchvision.datasets.vision.VisionDataset,
 torch.utils.data.dataset.Dataset,
 object)

```

### 7.1.2 The Dataset class

A PyTorch `Dataset` object implements `__len__` and `__getitem__` methods to provide uniform access to data samples. Looking at figure 7.2, we see what PyTorch Dataset is all about. The `__len__` method returns the total sample count, while `__getitem__` returns a sample alongside its integer class label.

[[img-02]]
Figure 7.2 - Concept of a PyTorch Dataset object: it doesn't necessarily hold the data, but it provides uniform access to it through len and getitem

Passing the dataset object to `len` returns the total number of items.

```python
# In [5]:
len(cifar10)
# Out [5]:
50000

```

Subscript indexing invokes `__getitem__`, returning a PIL image and its integer label index.

```python
# In [6]:
img, label = cifar10[99]
img, label, class_names[label]
# Out [6]:
(<PIL.Image.Image image mode=RGB size=32x32 at 0x7FB383657390>,
 1,
 'automobile')

```

The returned PIL image can be plotted directly to inspect the data sample.

```python
# In [7]:
plt.imshow(img)
plt.show()

```

This produces the output shown in figure 7.3.

[[img-03]]
Figure 7.3 - The 99th image from the CIFAR-10 dataset: an automobile

### 7.1.3 Dataset transforms

The `torchvision.transforms` module provides composable operations to turn PIL images into PyTorch tensors.

```python
# In [8]:
from torchvision import transforms
dir(transforms)
# Out [8]:
['CenterCrop',
 'ColorJitter',
 'Normalize',
 'Pad',
 'RandomAffine',
 'RandomResizedCrop',
 'RandomRotation',
 'RandomSizedCrop',
 'TenCrop',
 'ToPILImage',
 'ToTensor',
 ...]

```

`ToTensor` converts PIL images or NumPy arrays into tensors arranged as $C\times H\times W$.

```python
# In [9]:
from torchvision import transforms
to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
img_t.shape
# Out [9]:
torch.Size([3, 32, 32])

```

Passing `transform=transforms.ToTensor()` directly to `CIFAR10` ensures dataset indexing returns tensors instead of PIL images.

```python
# In [10]:
tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=False,
                                  transform=transforms.ToTensor())

```

Accessing elements from the dataset returns tensor instances directly.

```python
# In [11]:
img_t, _ = tensor_cifar10[99]
type(img_t)
# Out [11]:
torch.Tensor

```

The tensor channels are placed in the first dimension with 32-bit floating-point precision.

```python
# In [12]:
img_t.shape, img_t.dtype
# Out [12]:
(torch.Size([3, 32, 32]), torch.float32)

```

`ToTensor` rescales standard 8-bit image values from $[0, 255]$ down to $[0.0, 1.0]$.

```python
# In [13]:
img_t.min(), img_t.max()
# Out [13]:
(tensor(0.), tensor(1.))

```

To display the tensor using Matplotlib, the dimensions must be permuted from $C\times H\times W$ to $H\times W\times C$.

```python
# In [14]:
plt.imshow(img_t.permute(1, 2, 0))
plt.show()
# Out [14]:
<Figure size 432x288 with 1 Axes>

```

As we can see in figure 7.4, we get the same output as before.

[[img-04]]
Figure 7.4 - We've seen this one already.

### 7.1.4 Normalizing data

Transforms can be chained using `transforms.Compose` to perform normalization directly during data loading. Normalizing channels to zero mean and unit variance prevents vanishing gradients across activation functions.

Channel-wise mean and standard deviation are calculated across all images in the training set.

```python
# In [15]:
imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
imgs.shape
# Out [15]:
torch.Size([3, 32, 32, 50000])

```

The mean value for each channel is computed across all flattened spatial dimensions and samples.

```python
# In [16]:
imgs.view(3, -1).mean(dim=1)
# Out [16]:
tensor([0.4915, 0.4823, 0.4468])

```

Similarly, the standard deviation is computed for each channel across all elements.

```python
# In [17]:
imgs.view(3, -1).std(dim=1)
# Out [17]:
tensor([0.2470, 0.2435, 0.2616])

```

These computed metrics are passed to initialize the `Normalize` transform module.

```python
# In [18]:
transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
# Out [18]:
Normalize(mean=(0.4915, 0.4823, 0.4468), std=(0.247, 0.2435, 0.2616))

```

The `Normalize` transform is chained immediately after `ToTensor` inside `Compose`.

```python
# In [19]:
transformed_cifar10 = datasets.CIFAR10(
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

```

Visualizing normalized images directly renders dark or distorted regions because pixel values shift outside the $[0.0, 1.0]$ bounds.

```python
# In [21]:
img_t, _ = transformed_cifar10[99]
plt.imshow(img_t.permute(1, 2, 0))
plt.show()

```

The renormalized red car we get is shown in figure 7.5.

[[img-05]]
Figure 7.5 - Our random CIFAR-10 image after normalization

## 7.2 Distinguishing birds from airplanes

An automated filter system is needed to classify images and discard airplane photos from a bird-watching blog stream. Instead of manually deleting, she needs a neural network—an AI if we're into fancy marketing speak—to throw away the airplanes right away. An automated network like that shown in figure 7.6 can perform binary classification between airplane and bird samples extracted from CIFAR-10.

[[img-06]]
Figure 7.6 - The problem at hand: we're going to help our friend tell birds from airplanes for her blog, by training a neural network to do the job.

### 7.2.1 Building the dataset

The binary dataset is constructed by filtering CIFAR-10 samples and remapping labels 0 (airplane) and 2 (bird) to contiguous integers 0 and 1.

```python
# In [5]:
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label])
          for img, label in cifar10
          if label in [0, 2]]
cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val
              if label in [0, 2]]

```

### 7.2.2 A fully connected model

Images are converted to 1D vectors containing $32\times32\times3 = 3072$ input features. This is what figure 7.7 illustrates.

[[img-07]]
Figure 7.7 - Treating our image as a 1D vector of values and training a fully connected classifier on it

A fully connected network maps these input features through a hidden linear layer, a non-linear activation, and a final linear layer outputting two class scores.

```python
# In [6]:
import torch.nn as nn
n_out = 2
model = nn.Sequential(
    nn.Linear(
        3072,
        512,
    ),
    nn.Tanh(),
    nn.Linear(
        512,
        n_out,
    )
)

```

### 7.2.3 Output of a classifier

Classification requires categorical outputs represented using one-hot encoding vectors. The target output components are interpreted as probabilities corresponding to each output category. Probability representations require each element to lie within $[0.0, 1.0]$ and sum to $1.0$.

### 7.2.4 Representing the output as probabilities

Softmax normalizes vector elements by dividing each elementwise exponential by the sum of exponentials. The expression for softmax is shown in figure 7.8.

[[img-08]]
Figure 7.8 - Handwritten softmax

```python
# In [7]:
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()

```

Passing a sample input vector through `softmax` produces a valid probability distribution.

```python
# In [8]:
x = torch.tensor([1.0, 2.0, 3.0])
softmax(x)
# Out [8]:
tensor([0.0900, 0.2447, 0.6652])

```

Summing the outputs of `softmax` verifies that the elements sum to 1.0.

```python
# In [9]:
softmax(x).sum()
# Out [9]:
tensor(1.)

```

PyTorch provides `nn.Softmax`, which requires specifying the dimension along which probabilities are evaluated.

```python
# In [10]:
softmax = nn.Softmax(dim=1)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [1.0, 2.0, 3.0]])
softmax(x)
# Out [10]:
tensor([[0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652]])

```

Appending `nn.Softmax` to the sequential model guarantees probability-constrained outputs.

```python
# In [11]:
model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.Softmax(dim=1)
)

```

An un-trained model is evaluated on a single normalized bird image from the dataset (figure 7.9).

```python
# In [12]:
img, _ = cifar2[0]
plt.imshow(img.permute(1, 2, 0))
plt.show()

```

[[img-09]]
Figure 7.9 - A random bird from the CIFAR-10 dataset (after normalization)

The image tensor is unrolled into a 1D vector and given a batch dimension at index 0.

```python
# In [13]:
img_batch = img.view(-1).unsqueeze(0)

```

Passing the batched tensor through the model yields initial probability outputs.

```python
# In [14]:
out = model(img_batch)
out
# Out [14]:
tensor([[0.4784, 0.5216]], grad_fn=<SoftmaxBackward>)

```

Class predictions are extracted by finding the index corresponding to the maximum probability along dimension 1.

```python
# In [15]:
index = torch.max(out, dim=1)
index
# Out [15]:
tensor([1])

```

### 7.2.5 A loss for classifying

Maximizing the target class probability is equivalent to maximizing the likelihood. The negative log likelihood (NLL) penalizes incorrect predictions severely as predicted target probabilities approach zero.

$$\operatorname{NLL} = -\sum \log(\text{out}\_i[c\_i])$$

Let's take a look at figure 7.10, which shows the NLL as a function of predicted probability.

[[img-10]]
Figure 7.10 - The NLL loss as a function of the predicted probabilities

When low probabilities are assigned to the data, the NLL grows to infinity, whereas it decreases at a rather shallow rate when probabilities are greater than 0.5.

PyTorch provides `nn.NLLLoss`, which expects log-probabilities as inputs rather than raw probabilities. `nn.LogSoftmax` computes log-probabilities directly to maintain numerical stability.

```python
model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1))

```

Instantiating `nn.NLLLoss` allows evaluation against target class labels.

```python
loss = nn.NLLLoss()

```

The loss is evaluated on output log-probabilities alongside ground truth class indices.

```python
img, label = cifar2[0]
out = model(img.view(-1).unsqueeze(0))
loss(out, torch.tensor([label]))

```

In figure 7.11, we see that the cross-entropy loss has some slope when the prediction is off target (in the low-loss corner, the correct class is assigned a predicted probability of 99.97%), while the MSE we dismissed at the beginning saturates much earlier and crucially also for very wrong predictions.

[[img-11]]
Figure 7.11 - The cross entropy (left) and MSE between predicted probabilities and the target probability vector (right) as functions of the predicted scores—that is, before the (log-) softmax

### 7.2.6 Training the classifier

The training execution flow across individual samples, minibatches, and full datasets is illustrated in figure 7.12.

[[img-12]]
Figure 7.12 - Training loops: (A) averaging updates over the whole dataset; (B) updating the model at each sample; (C) averaging updates over minibatches

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1))

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()
n_epochs = 100

for epoch in range(n_epochs):
    for img, label in cifar2:
        out = model(img.view(-1).unsqueeze(0))
        loss = loss_fn(out, torch.tensor([label]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

```

Evaluating gradients over minibatches introduces stochasticity, aiding convergence and escaping local minima. As depicted in figure 7.13, gradients from minibatches are randomly off the ideal trajectory, which is part of the reason why we want to use a reasonably small learning rate.

[[img-13]]
Figure 7.13 - Gradient descent averaged over the whole dataset (light path) versus stochastic gradient descent, where the gradient is estimated on randomly picked minibatches

`DataLoader` shuffles data items and batches them into minibatches for model iteration. Figure 7.14 shows the data loader shuffling the indices it gets from the Dataset.

[[img-14]]
Figure 7.14 - A data loader dispensing minibatches by using a dataset to sample individual data items

`DataLoader` is instantiated with a batch size and shuffling enabled.

```python
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

```

The inner training loop iterates over minibatches delivered by the data loader.

```python
import torch
import torch.nn as nn

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1))

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()
n_epochs = 100

for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

```

Validation accuracy is evaluated without calculating gradients across the validation loader.

```python
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

print("Accuracy: %f" % (correct / total))

```

Model capacity can be expanded by chaining additional intermediate linear layers.

```python
model = nn.Sequential(
    nn.Linear(3072, 1024),
    nn.Tanh(),
    nn.Linear(1024, 512),
    nn.Tanh(),
    nn.Linear(512, 128),
    nn.Tanh(),
    nn.Linear(128, 2),
    nn.LogSoftmax(dim=1))

```

`nn.CrossEntropyLoss` combines `nn.LogSoftmax` and `nn.NLLLoss` into a single module taking unnormalized scores.

```python
model = nn.Sequential(
    nn.Linear(3072, 1024),
    nn.Tanh(),
    nn.Linear(1024, 512),
    nn.Tanh(),
    nn.Linear(512, 128),
    nn.Tanh(),
    nn.Linear(128, 2))
loss_fn = nn.CrossEntropyLoss()

```

Counting tensor elements via `numel()` computes the total trainable parameter count.

```python
# In [7]:
numel_list = [p.numel()
              for p in connected_model.parameters()
              if p.requires_grad == True]
sum(numel_list), numel_list
# Out [7]:
(3737474, [3145728, 1024, 524288, 512, 65536, 128, 256, 2])

```

Evaluating parameter sizes in the smaller network demonstrates significant memory overhead in the initial linear layer.

```python
# In [9]:
numel_list = [p.numel() for p in first_model.parameters()]
sum(numel_list), numel_list
# Out [9]:
(1574402, [1572864, 512, 1024, 2])

```

Inspecting parameter weight and bias shapes confirms matrix dimensions equal output by input sizes.

```python
# In [10]:
linear = nn.Linear(3072, 1024)
linear.weight.shape, linear.bias.shape
# Out [10]:
(torch.Size([1024, 3072]), torch.Size([1024]))

```

### 7.2.7 The limits of going fully connected

Fully connected layers compute linear combinations across all input pixels simultaneously. Figure 7.15 shows what is going on.

[[img-15]]
Figure 7.15 - Using a fully connected module with an input image: every input pixel is combined with every other to produce each element in the output.

Fully connected architectures lack translation invariance, forcing parameters to relearn features at every spatial position. This is illustrated in the top half of figure 7.16.

[[img-16]]
Figure 7.16 - Translation invariance, or the lack thereof, with fully connected layers

Data augmentation can compensate for spatial shifts, but requires expanding network capacity to memorize all translated variations.

## 7.3 Conclusion

This chapter demonstrated end-to-end image classification using datasets, models, loss functions, and optimization loops. Treating 2D image structures as 1D vectors creates parameter inefficiency and lacks spatial translation invariance.

## 7.4 Exercises

1 Use `torchvision` to implement random cropping of the data.
a How are the resulting images different from the uncropped originals?
b What happens when you request the same image a second time?
c What is the result of training using randomly cropped images?
2 Switch loss functions (perhaps MSE).
a Does the training behavior change?
3 Is it possible to reduce the capacity of the network enough that it stops overfitting?
a How does the model perform on the validation set when doing so?

## 7.5 Summary

* Computer vision is one of the most extensive applications of deep learning.
* Several datasets of annotated images are publicly available; many of them can be accessed via `torchvision`.
* `Datasets` and `DataLoaders` provide a simple yet effective abstraction for loading and sampling datasets.
* For a classification task, using the softmax function on the output of a network produces values that satisfy the requirements for being interpreted as probabilities. The ideal loss function for classification in this case is obtained by using the output of softmax as the input of a non-negative log likelihood function. The combination of softmax and such loss is called cross entropy in PyTorch.
* Nothing prevents us from treating images as vectors of pixel values, dealing with them using a fully connected network, just like any other numerical data. However, doing so makes it much harder to take advantage of the spatial relationships in the data.
* Simple models can be created using `nn.Sequential`.