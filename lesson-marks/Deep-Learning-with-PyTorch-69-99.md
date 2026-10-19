# Chapter 3: It starts with a tensor

This chapter covers understanding tensors, the basic data structure in PyTorch, as well as indexing and operating on tensors to represent multidimensional data. It also introduces how to interoperate with NumPy multidimensional arrays, move computations to the GPU for speed, and save tensors to disk.

Deep learning frameworks process inputs by converting them into floating-point representation, performing mathematical operations, and learning from data. These collections of floating-point numbers and the computations that created them are key building blocks in PyTorch.

## 3.1 The world as floating-point numbers

Floating-point numbers are the primary numerical representation used in deep learning models. Neural networks transform these floating-point representations through successive operations to extract meaningful patterns or solve specific tasks.

By organizing floating-point numbers into multidimensional arrays, deep learning models can process complex inputs like images, audio, text, or tabular data.

## 3.2 Tensors: Multidimensional arrays

In Python, collections of numbers are typically stored in lists. However, Python lists are one-dimensional and inefficient for the mathematical operations required by deep learning.

Tensors provide a contiguous block of memory allocated for numerical data, allowing efficient execution of vector and matrix operations.

### 3.2.1 From Python lists to PyTorch tensors

Let's construct our first PyTorch tensor and compare it with standard Python lists (`code/p1ch3/1_tensors.ipynb`).

```python
# In[2]:
a = [1.0, 2.0, 1.0]
# In[3]:
a[0]
# Out[3]:
1.0
# In[4]:
a[2] = 3.0
a
# Out[4]:
[1.0, 2.0, 3.0]

```

Python lists store collections of pointers to individually allocated Python objects, as shown in figure 3.1.

[[img-01]]
Figure 3.1 - Python list of floats (top) and 1D PyTorch tensor (bottom).

PyTorch tensors store contiguous blocks of unboxed floating-point numbers in memory rather than references to Python objects.

### 3.2.2 Constructing our first tensors

Now let's create a PyTorch tensor containing the same values:

```python
# In[5]:
import torch
a = torch.ones(3)
a
# Out[5]:
tensor([1., 1., 1.])
# In[6]:
a[1]
# Out[6]:
tensor(1.)
# In[7]:
float(a[1])
# Out[7]:
1.0
# In[8]:
a[2] = 2.0
a
# Out[8]:
tensor([1., 1., 2.])

```

We can also create a tensor filled with specific values by passing a list to the `torch.tensor` constructor:

```python
# In[9]:
points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
points
# Out[9]:
tensor([4., 1., 5., 3., 2., 1.])

```

We can pass 2D coordinates as a list of lists or reshape a 1D tensor into a 2D tensor using `view`:

```python
# In[10]:
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points
# Out[10]:
tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])

```

Checking the shape of the tensor using `shape` gives us the size across each dimension:

```python
# In[11]:
points.shape
# Out[11]:
torch.Size([3, 2])

```

We can also use `zeros` or `ones` to initialize tensors with zeros or ones, respectively:

```python
# In[12]:
points = torch.zeros(3, 2)
points
# Out[12]:
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])

```

### 3.2.3 Indexing tensors

We can access elements in a multidimensional tensor using 2D indexing syntax `[i, j]` or sequential 1D indexing:

```python
# In[13]:
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points[0, 0]
# Out[13]:
tensor(4.)
# In[14]:
points[0]
# Out[14]:
tensor([4., 1.])

```

The output is another tensor that presents a different view of the same underlying storage.

## 3.3 Named tensors

PyTorch allows dimensions to be assigned explicit names to avoid confusing index positions in multidimensional tensors.

### 3.3.1 Dimension names

To make things concrete, imagine that we have a 3D tensor like `img_t` from section 2.1. We can pass a tuple of dimension names to the `names` argument when creating a tensor:

```python
# In[15]:
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
weights_named
# Out[15]:
tensor([0.2126, 0.7152, 0.0722], names=('channels',))

```

When we already have a tensor and want to add names without altering the data, we can call `refine_names`:

```python
# In[16]:
img_t = torch.randn(3, 5, 5) # shape [channels, rows, columns]
weights = torch.tensor([0.2126, 0.7152, 0.0722])
img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
img_named.shape, img_named.names
# Out[16]:
(torch.Size([3, 5, 5]), ('channels', 'rows', 'columns'))

```

### 3.3.2 Operating on named tensors

Operations with named tensors check dimension alignment automatically. If we want to combine tensors with different named dimensions, we use `align_as`:

```python
# In[17]:
weights_aligned = weights_named.align_as(img_named)
weights_aligned.shape, weights_aligned.names
# Out[17]:
(torch.Size([3, 1, 1]), ('channels', 'rows', 'columns'))

```

Functions accepting dimension arguments can take named dimensions:

```python
# In[18]:
gray_named = (img_named * weights_aligned).sum('channels')
gray_named.shape, gray_named.names
# Out[18]:
(torch.Size([5, 5]), ('rows', 'columns'))

```

If we want to drop named dimensions and return to un-named tensors, we use `rename(None)`:

```python
# In[19]:
gray_unnamed = gray_named.rename(None)
gray_unnamed.shape, gray_unnamed.names
# Out[19]:
(torch.Size([5, 5]), None)

```

## 3.4 Tensor element types

Tensors can hold different numerical data types, specified via the `dtype` parameter.

### 3.4.1 Specifying the numeric type with `dtype`

The default data type for tensors is 32-bit floating-point. We can specify other types using the `dtype` argument:

```python
# In[20]:
double_points = torch.zeros(10, 2, dtype=torch.float64)
short_points = torch.ones(10, 2, dtype=torch.int16)

```

### 3.4.2 Managing a tensor's `dtype` attribute

We can inspect the `dtype` attribute of a tensor:

```python
# In[21]:
double_points.dtype, short_points.dtype
# Out[21]:
(torch.float64, torch.int16)

```

We can convert a tensor to a different data type using casting methods or the `to` method:

```python
# In[22]:
double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()

```

```python
# In[23]:
double_points = torch.zeros(10, 2).to(torch.float64)
short_points = torch.ones(10, 2).to(dtype=torch.int16)

```

Under the hood, `to` checks whether conversion is necessary and returns the existing tensor if the target type matches.

## 3.5 The Tensor API

PyTorch provides a vast range of operations on tensors, which can be categorized into creation, manipulation, math, and utility operations.

Creation ops include functions for constructing a tensor, like `ones` or `from_numpy`. Manipulation ops include functions for changing shape or layout, like `transpose` or `view`. Math ops perform elementwise or reduction operations, like `abs`, `mean`, or `sum`.

## 3.6 Tensors: Scenic views of storage

Values in a tensor are stored in a contiguous block of memory managed by a `torch.Storage` instance.

### 3.6.1 Indexing into storage

We can access the underlying storage of a tensor using the `storage` method:

```python
# In[24]:
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storage()
# Out[24]:
 4.0
 1.0
 5.0
 3.0
 2.0
 1.0
[torch.FloatStorage of size 6]

```

Figure 3.2 shows how storage relates to tensors indexing into the same memory.

[[img-02]]
Figure 3.2 - Tensor storage and layout.

Modifying the storage directly changes the values in any tensor pointing to that storage:

```python
# In[25]:
points_storage = points.storage()
points_storage[0] = 2.0
points
# Out[25]:
tensor([[2., 1.],
        [5., 3.],
        [2., 1.]])

```

### 3.6.2 Modifying stored values in-place

Operations that end with an underscore operate in-place, modifying the target tensor directly rather than returning a new tensor:

```python
# In[26]:
a = torch.ones(3, 2)
a.zero_()
a
# Out[26]:
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])

```

## 3.7 Tensor metadata: Size, offset, and stride

To index into storage efficiently, tensors rely on three metadata properties: size (or shape), storage offset, and stride.

Figure 3.3 displays how offset, size, and stride define the sub-tensor views.

[[img-03]]
Figure 3.3 - Relationships among a tensor's size, storage offset, and stride.

### 3.7.1 Views of storage

The storage offset is the index in the storage corresponding to the first element in the tensor. The size is a tuple specifying the number of elements along each dimension. The stride is the number of elements in storage that must be skipped to move to the next element along each dimension.

```python
# In[27]:
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point.storage_offset()
# Out[27]:
2
# In[28]:
second_point.size()
# Out[28]:
torch.Size([2])
# In[29]:
second_point.shape
# Out[29]:
torch.Size([2])
# In[30]:
points.stride()
# Out[30]:
(2, 1)

```

### 3.7.2 Transposing without copying

Transposing a tensor changes its stride and shape without altering or reallocating the underlying storage:

```python
# In[31]:
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points
# Out[31]:
tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])
# In[32]:
points_t = points.t()
points_t
# Out[32]:
tensor([[4., 5., 2.],
        [1., 3., 1.]])

```

We can verify that two tensors share the same storage by checking `storage().data_ptr()`:

```python
# In[33]:
id(points.storage()) == id(points_t.storage())
# Out[33]:
True

```

```python
# In[34]:
points.stride()
# Out[34]:
(2, 1)
# In[35]:
points_t.stride()
# Out[35]:
(1, 2)

```

### 3.7.3 Transposing in higher dimensions

Transposing in higher dimensions is generalized using the `transpose` method, where we pass the two dimension indices to swap:

```python
# In[36]:
some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
some_t.shape
# Out[36]:
torch.Size([3, 4, 5])
# In[37]:
transpose_t.shape
# Out[37]:
torch.Size([5, 4, 3])
# In[38]:
some_t.stride()
# Out[38]:
(20, 5, 1)
# In[39]:
transpose_t.stride()
# Out[39]:
(1, 5, 20)

```

### 3.7.4 Contiguous tensors

A tensor is contiguous if its elements are laid out sequentially in memory from rightmost to leftmost dimension.

```python
# In[40]:
points.is_contiguous()
# Out[40]:
True
# In[41]:
points_t.is_contiguous()
# Out[41]:
False

```

We can create a new contiguous tensor from a non-contiguous one using `contiguous()`:

```python
# In[42]:
points_t_cont = points_t.contiguous()
points_t_cont.is_contiguous()
# Out[42]:
True
# In[43]:
points_t_cont.stride()
# Out[43]:
(3, 1)

```

## 3.8 Moving tensors to the GPU

PyTorch tensors can be stored on different hardware devices, such as the CPU or NVIDIA GPUs using CUDA.

### 3.8.1 Managing a tensor's `device` attribute

We can create a tensor directly on the GPU by passing `device='cuda'` to the constructor:

```python
# In[44]:
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')

```

Alternatively, we can move an existing CPU tensor to the GPU using the `to` method:

```python
# In[45]:
points_gpu = points.to(device='cuda')

```

If we have multiple GPUs, we can specify the index using `cuda:0`, `cuda:1`, and so on:

```python
# In[46]:
points_gpu = points.to(device='cuda:0')

```

Operations performed on GPU tensors are executed on the GPU hardware:

```python
# In[47]:
points_p = points_gpu * 0.2

```

Moving a tensor back to the CPU is done using `cpu()` or passing `device='cpu'` to `to`:

```python
# In[48]:
points_cpu = points_gpu.cpu()

```

We can also use shorthand methods `cuda()` and `cpu()`:

```python
# In[49]:
points_gpu = points.cuda()
points_gpu = points.cuda(0)
points_cpu = points_gpu.cpu()

```

## 3.9 NumPy interoperability

PyTorch tensors can be converted to NumPy arrays and vice versa with zero-copy memory sharing.

To convert a PyTorch tensor to a NumPy array, use `numpy()`:

```python
# In[50]:
points = torch.ones(3, 4)
points_np = points.numpy()
points_np
# Out[50]:
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]], dtype=float32)

```

To convert a NumPy array to a PyTorch tensor, use `from_numpy()`:

```python
# In[51]:
points = torch.from_numpy(points_np)

```

## 3.10 Generalized tensors are tensors, too

PyTorch provides tensor implementations beyond standard dense tensors, such as sparse tensors and dispatching logic for varied storage layouts.

Figure 3.4 shows the dispatching mechanism PyTorch uses to execute tensor functions across different devices and layouts.

[[img-04]]
Figure 3.4 - High-level diagram of PyTorch dispatching mechanism for tensor operations.

## 3.11 Serializing tensors

PyTorch provides utility functions to save and load tensors to and from disk.

### 3.11.1 Saving and loading tensors with `torch.save`

We can serialize a tensor to a file using `torch.save`:

```python
# In[52]:
torch.save(points, '../data/p1ch3/ourPoints.t')

```

Alternatively, we can pass a file descriptor:

```python
# In[53]:
with open('../data/p1ch3/ourPoints.t', 'wb') as f:
    torch.save(points, f)

```

To load the tensor back from disk, use `torch.load`:

```python
# In[54]:
points = torch.load('../data/p1ch3/ourPoints.t')

```

```python
# In[55]:
with open('../data/p1ch3/ourPoints.t', 'rb') as f:
    points = torch.load(f)

```

### 3.11.2 Serializing to HDF5 with h5py

For interoperability with other libraries, we can save tensors in HDF5 format using `h5py`:

```python
# In[56]:
import h5py
f = h5py.File('../data/p1ch3/ourPoints.hdf5', 'w')
dset = f.create_dataset('coords', data=points.numpy())
f.close()

```

We can read specific slices back from an HDF5 file without loading the whole dataset into memory:

```python
# In[57]:
f = h5py.File('../data/p1ch3/ourPoints.hdf5', 'r')
dset = f['coords']
last_points = dset[-2:]

```

```python
# In[58]:
last_points = torch.from_numpy(dset[-2:])
f.close()

```

## 3.12 Conclusion

Tensors form the core data structure in PyTorch. Understanding tensor memory layouts, indexing, device management, and serialization allows us to handle complex datasets effectively.

## 3.13 Exercises

1. Create a tensor `a` from `list(range(9))`. Predict and then check what the size, offset, and stride are.
a. Create a new tensor `b = a.view(3, 3)`. What is the value of `b[1, 1]`?
b. Verify that `a` and `b` share the same storage.
2. Create a tensor `a = torch.randn(3, 2, 2)`.
a. Find the mathematical operations that convert `a` to a tensor of shape `(3, 4)`.
b. Which ops operate in-place?

## 3.14 Summary

* PyTorch has a comprehensive standard library for tensor creation, manipulation, and mathematical operations.
* Tensors are multidimensional arrays that store data of uniform types in contiguous memory buffers.
* Tensors provide flexible indexing and view manipulation through stride, size, and storage offset metadata without copying underlying data.
* Tensor operations can be seamlessly performed on CPUs or GPUs.
* PyTorch tensors share memory directly with NumPy arrays for efficient data exchange.
* Tensors can be easily serialized and saved to disk using standard PyTorch formats or external libraries like HDF5.