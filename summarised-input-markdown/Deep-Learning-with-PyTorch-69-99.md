# **Chapter 3: It starts with a tensor**

In the previous chapter, we took a tour of some of the many applications that deep learning enables. They invariably consisted of taking data in some form, like images or text, and producing data in another form, like labels, numbers, or more images or text. Viewed from this angle, deep learning really consists of building a system that can transform data from one representation to another. This transformation is driven by extracting commonalities from a series of examples that demonstrate the desired mapping, such as combining dog shapes and golden retriever colors to correctly map images. The resulting system can consume broad swaths of similar inputs and produce meaningful output for those inputs.

The process begins by converting our input into floating-point numbers. We will cover converting image pixels to numbers, as we see in the first step of figure 3.1, in chapter 4 (along with many other types of data). But before we can get to that, in this chapter, we learn how to deal with all the floating-point numbers in PyTorch by using tensors.

Figure 3.1 A deep neural network learns how to transform an input representation to an output representation. (Note: The numbers of neurons and outputs are not to scale.)

---

### **3.1 The world as floating-point numbers**

Since floating-point numbers are the way a network deals with information, we need a way to encode real-world data into something digestible by a network and then decode the output back to something we can understand and use for our purpose. Neural networks learn data transformations in stages, creating a sequence of intermediate representations where early stages capture simple edges or textures and deeper stages capture complex structures like eyes, ears, or noses.

Intermediate representations are collections of floating-point numbers that characterize the input and capture its structure in a way that is instrumental for describing how inputs are mapped to the outputs of the neural network. Characterized specifically for the task at hand and learned from relevant examples, these collections of floating-point numbers and their manipulation form the heart of modern AI.

It’s important to keep in mind that these intermediate representations (like those shown in the second step of figure 3.1) are the results of combining the input with the weights of the previous layer of neurons. Each intermediate representation is unique to the inputs that preceeded it.

Before we can begin the process of converting our data to floating-point input, we must first have a solid understanding of how PyTorch handles and stores data—as input, as intermediate representations, and as output. This chapter will be devoted to precisely that.

To this end, PyTorch introduces a fundamental data structure: the tensor. In the context of deep learning, tensors refer to the generalization of vectors and matrices to an arbitrary number of dimensions, as we can see in figure 3.2. Another name for the same concept is multidimensional array, where the dimensionality of a tensor coincides with the number of indexes used to refer to scalar values within the tensor.

Figure 3.2 Tensors are the building blocks for representing data in PyTorch.

PyTorch is not the only library that deals with multidimensional arrays. NumPy is by far the most popular multidimensional array library, to the point that it has now arguably become the lingua franca of data science. PyTorch features seamless interoperability with NumPy, which brings with it first-class integration with the rest of the scientific libraries in Python, such as SciPy, Scikit-learn, and Pandas.

Compared to NumPy arrays, PyTorch tensors have a few superpowers, such as the ability to perform very fast operations on graphical processing units (GPUs), distribute operations on multiple devices or machines, and keep track of the graph of computations that created them. These are all important features when implementing a modern deep learning library.

We’ll start this chapter by introducing PyTorch tensors, covering the basics in order to set things in motion for our work in the rest of the book. First and foremost, we’ll learn how to manipulate tensors using the PyTorch tensor library. This includes things like how the data is stored in memory, how certain operations can be performed on arbitrarily large tensors in constant time, and the aforementioned NumPy interoperability and GPU acceleration. Understanding the capabilities and API of tensors is important if they’re to become go-to tools in our programming toolbox. In the next chapter, we’ll put this knowledge to good use and learn how to represent several different kinds of data in a way that enables learning with neural networks.

---

### **3.2 Tensors: Multidimensional arrays**

We have already learned that tensors are the fundamental data structure in PyTorch. A tensor is an array: that is, a data structure that stores a collection of numbers that are accessible individually using an index, and that can be indexed with multiple indices.

#### **3.2.1 From Python lists to PyTorch tensors**

Let’s see list indexing in action so we can compare it to tensor indexing. Take a list of three numbers in Python:

```python
a = [1.0, 2.0, 1.0]
```

We can access the first element of the list using the corresponding zero-based index or reassign its values:

```python
a
# Out: 1.0

a = 3.0
a
# Out: [1.0, 2.0, 3.0]
```

It is not unusual for simple Python programs dealing with vectors of numbers, such as the coordinates of a 2D line, to use Python lists to store the vectors. As we will see in the following chapter, using the more efficient tensor data structure, many types of data—from images to time series, and even sentences—can be represented. By defining operations over tensors, some of which we’ll explore in this chapter, we can slice and manipulate data expressively and efficiently at the same time, even from a high-level (and not particularly fast) language such as Python.

#### **3.2.2 Constructing our first tensors**

Let’s construct our first PyTorch tensor and see what it looks like:

```python
import torch
a = torch.ones(3)
a
# Out: tensor([1., 1., 1.])

a
# Out: tensor(1.)

float(a)
# Out: 1.0

a = 2.0
a
# Out: tensor([1., 1., 2.])
```

After importing the torch module, we call a function that creates a (one-dimensional) tensor of size 3 filled with the value 1.0. We can access an element using its zero-based index or assign a new value to it. Although on the surface this example doesn’t differ much from a list of number objects, under the hood things are completely different.

#### **3.2.3 The essence of tensors**

Python lists or tuples of numbers are collections of Python objects that are individually allocated in memory, as shown on the left in figure 3.3. PyTorch tensors or NumPy arrays, on the other hand, are views over (typically) contiguous memory blocks containing unboxed C numeric types rather than Python objects. Each element is a 32-bit (4-byte) float in this case, as we can see on the right side of figure 3.3. This means storing a 1D tensor of 1,000,000 float numbers will require exactly 4,000,000 contiguous bytes, plus a small overhead for the metadata (such as dimensions and numeric type).

Figure 3.3 Python object (boxed) numeric values versus tensor (unboxed array) numeric values

Say we have a list of coordinates we’d like to use to represent a geometrical object: perhaps a 2D triangle with vertices at coordinates (4, 1), (5, 3), and (2, 1). Instead of having coordinates as numbers in a Python list, as we did earlier, we can use a one-dimensional tensor by storing Xs in the even indices and Ys in the odd indices:

```python
points = torch.zeros(6)
points = 4.0
points = 1.0
points = 5.0
points = 3.0
points = 2.0
points = 1.0
```

We can also pass a Python list to the constructor, to the same effect:

```python
points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
points
# Out: tensor([4., 1., 5., 3., 2., 1.])
```

To get the coordinates of the first point, we do the following:

```python
float(points), float(points)
# Out: (4.0, 1.0)
```

This is OK, although it would be practical to have the first index refer to individual 2D points rather than point coordinates. For this, we can use a 2D tensor:

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points
# Out: tensor([[4., 1.],
#              [5., 3.],
#              [2., 1.]])
```

Here, we pass a list of lists to the constructor. We can ask the tensor about its shape:

```python
points.shape
# Out: torch.Size()
```

This informs us about the size of the tensor along each dimension. We could also use zeros or ones to initialize the tensor, providing the size as a tuple:

```python
points = torch.zeros(3, 2)
points
# Out: tensor([[0., 0.],
#              [0., 0.],
#              [0., 0.]])
```

Now we can access an individual element in the tensor using two indices:

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points
# Out: tensor(1.)
```

This returns the Y-coordinate of the zeroth point in our dataset. We can also access the first element in the tensor as we did before to get the 2D coordinates of the first point:

```python
points
# Out: tensor([4., 1.])
```

The output is another tensor that presents a different view of the same underlying data. The new tensor is a 1D tensor of size 2, referencing the values of the first row in the points tensor. Creating this new view avoids memory allocation or value copying, as explored in section 3.7.

---

### **3.3 Indexing tensors**

What if we need to obtain a tensor containing all points but the first? That’s easy using range indexing notation, which also applies to standard Python lists:

```python
some_list = list(range(6))
some_list[:]
some_list[1:4]
some_list[1:]
some_list[:4]
some_list[:-1]
some_list[1:4:2]
```

To achieve our goal, we can use the same notation for PyTorch tensors, with the added benefit that, just as in NumPy and other Python scientific libraries, we can use range indexing for each of the tensor’s dimensions:

```python
points[1:]
points[1:, :]
points[1:, 0]
points[None]
```

In addition to using ranges, PyTorch features a powerful form of indexing, called advanced indexing, which we will look at in the next chapter.

---

### **3.4 Named tensors**

The dimensions (or axes) of our tensors usually index something like pixel locations or color channels. This means when we want to index into a tensor, we need to remember the ordering of the dimensions and write our indexing accordingly. As data is transformed through multiple tensors, keeping track of which dimension contains what data can be error-prone.

To make things concrete, imagine that we have a 3D tensor like img_t from section 2.1.4 (we will use dummy data for simplicity here), and we want to convert it to grayscale. We looked up typical weights for the colors to derive a single brightness value:

```python
img_t = torch.randn(3, 5, 5) # shape [channels, rows, columns]
weights = torch.tensor([0.2126, 0.7152, 0.0722])
```

We also often want our code to generalize—for example, from grayscale images represented as 2D tensors with height and width dimensions to color images adding a third channel dimension (as in RGB), or from a single image to a batch of images. In section 2.1.4, we introduced an additional batch dimension in batch_t; here we pretend to have a batch of 2:

```python
batch_t = torch.randn(2, 3, 5, 5) # shape [batch, channels, rows, columns]
```

So sometimes the RGB channels are in dimension 0, and sometimes they are in dimension 1. But we can generalize by counting from the end: they are always in dimension -3, the third from the end. The lazy, unweighted mean can thus be written as follows:

```python
img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)
img_gray_naive.shape, batch_gray_naive.shape
# Out: (torch.Size(), torch.Size())
```

But now we have the weight, too. PyTorch will allow us to multiply things that are the same shape, as well as shapes where one operand is of size 1 in a given dimension. It also appends leading dimensions of size 1 automatically through broadcasting. batch_t of shape (2, 3, 5, 5) is multiplied by unsqueezed_weights of shape (3, 1, 1), resulting in a tensor of shape (2, 3, 5, 5), from which we can then sum the third dimension from the end (the three channels):

```python
unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)
img_weights = (img_t * unsqueezed_weights)
batch_weights = (batch_t * unsqueezed_weights)
img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)
batch_weights.shape, batch_t.shape, unsqueezed_weights.shape
# Out: (torch.Size(), torch.Size(), torch.Size())
```

Because this gets messy quickly—and for the sake of efficiency—the PyTorch function einsum (adapted from NumPy) specifies an indexing mini-language giving index names to dimensions for sums of such products. As often in Python, broadcasting—a form of summarizing unnamed things—is done using three dots '...'; but don’t worry too much about einsum, because we will not use it in the following:

```python
img_gray_weighted_fancy = torch.einsum('...chw,c->...hw', img_t, weights)
batch_gray_weighted_fancy = torch.einsum('...chw,c->...hw', batch_t, weights)
batch_gray_weighted_fancy.shape
# Out: torch.Size()
```

As we can see, there is quite a lot of bookkeeping involved. This is error-prone, especially when the locations where tensors are created and used are far apart in our code. This has caught the eye of practitioners, and so it has been suggested that the dimension be given a name instead.

PyTorch 1.3 added named tensors as an experimental feature. Tensor factory functions such as tensor and rand take a names argument. The names should be a sequence of strings:

```python
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
weights_named
# Out: tensor([0.2126, 0.7152, 0.0722], names=('channels',))
```

When we already have a tensor and want to add names (but not change existing ones), we can call the method refine_names on it. Similar to indexing, the ellipsis (...) allows you to leave out any number of dimensions. With the rename sibling method, you can also overwrite or drop (by passing in None) existing names:

```python
img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')
print("img named:", img_named.shape, img_named.names)
print("batch named:", batch_named.shape, batch_named.names)
# Out: img named: torch.Size() ('channels', 'rows', 'columns')
#      batch named: torch.Size() (None, 'channels', 'rows', 'columns')
```

For operations with two inputs, in addition to the usual dimension checks—whether sizes are the same, or if one is 1 and can be broadcast to the other—PyTorch will now check the names for us. So far, it does not automatically align dimensions, so we need to do this explicitly. The method align_as returns a tensor with missing dimensions added and existing ones permuted to the right order:

```python
weights_aligned = weights_named.align_as(img_named)
weights_aligned.shape, weights_aligned.names
# Out: (torch.Size(), ('channels', 'rows', 'columns'))

gray_named = (img_named * weights_aligned).sum('channels')
gray_named.shape, gray_named.names
# Out: (torch.Size(), ('rows', 'columns'))
```

If we try to combine dimensions with different names, we get an error:

```python
gray_named = (img_named[..., :3] * weights_named).sum('channels')
# RuntimeError: Error when attempting to broadcast dims ['channels', 'rows', 'columns'] and dims ['channels']: dim 'columns' and dim 'channels' are at the same position from the right but do not match.
```

If we want to use tensors outside functions that operate on named tensors, we need to drop the names by renaming them to None. The following gets us back into the world of unnamed dimensions:

```python
gray_plain = gray_named.rename(None)
gray_plain.shape, gray_plain.names
# Out: (torch.Size(), (None, None))
```

Given the experimental nature of this feature at the time of writing, and to avoid mucking around with indexing and alignment, we will stick to unnamed in the remainder of the book. Named tensors have the potential to eliminate many sources of alignment errors, which—if the PyTorch forum is any indication—can be a source of headaches. It will be interesting to see how widely they will be adopted.

---

### **3.5 Tensor element types**

So far, we have covered the basics of how tensors work, but we have not yet touched on what kinds of numeric types we can store in a Tensor. As we hinted at in section 3.2, using the standard Python numeric types can be suboptimal for several reasons:

- Numbers in Python are objects, where boxing converts floating-point numbers into full Python objects with reference counting, making allocating millions inefficient.
- Lists in Python are meant for sequential collections of objects without efficient mathematical operations like dot products or optimized memory layouts.
- The Python interpreter is slow compared to optimized, compiled C code when performing operations on large collections of numerical data.

For these reasons, data science libraries rely on NumPy or introduce dedicated data structures like PyTorch tensors, which provide efficient low-level implementations of numerical data structures and related operations on them, wrapped in a convenient high-level API. To enable this, the objects within a tensor must all be numbers of the same type, and PyTorch must keep track of this numeric type.

#### **3.5.1 Specifying the numeric type with dtype**

The dtype argument to constructor functions specifies the data type of the numbers stored in the tensor, defining possible values and byte sizes. The available dtypes include:

- torch.float32 or torch.float: 32-bit floating-point
- torch.float64 or torch.double: 64-bit, double-precision floating-point
- torch.float16 or torch.half: 16-bit, half-precision floating-point
- torch.int8: signed 8-bit integers
- torch.uint8: unsigned 8-bit integers
- torch.int16 or torch.short: signed 16-bit integers
- torch.int32 or torch.int: signed 32-bit integers
- torch.int64 or torch.long: signed 64-bit integers
- torch.bool: Boolean

The default data type for tensors is 32-bit floating-point.

#### **3.5.2 A dtype for every occasion**

Neural network computations are typically executed with 32-bit floating-point precision. Higher precision, like 64-bit, doesn't improve model accuracy and increases memory footprint and computation time. Half-precision 16-bit floats decrease GPU memory usage with minimal accuracy impact.

Tensors can be used as indexes in other tensors. In this case, PyTorch expects indexing tensors to have a 64-bit integer data type. Creating a tensor with integers as arguments, such as using torch.tensor(), will create a 64-bit integer tensor by default. As such, we’ll spend most of our time dealing with float32 and int64.

Finally, predicates on tensors, such as points > 1.0, produce bool tensors indicating whether each individual element satisfies the condition. These are the numeric types in a nutshell.

#### **3.5.3 Managing a tensor’s dtype attribute**

In order to allocate a tensor of the right numeric type, we can specify the proper dtype as an argument to the constructor. For example:

```python
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([,], dtype=torch.short)
```

We can find out about the dtype for a tensor by accessing the corresponding attribute:

```python
short_points.dtype
# Out: torch.int16
```

We can also cast the output of a tensor creation function to the right type using the corresponding casting method, such as:

```python
double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()
```

or the more convenient to method:

```python
double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)
```

Under the hood, to checks whether the conversion is necessary and, if so, does it. The dtype-named casting methods like float are shorthands for to, but the to method can take additional arguments that we’ll discuss in section 3.9.

When mixing input types in operations, the inputs are converted to the larger type automatically. Thus, if we want 32-bit computation, we need to make sure all our inputs are (at most) 32-bit:

```python
points_64 = torch.rand(5, dtype=torch.double)
points_short = points_64.to(torch.short)
points_64 * points_short
# Out: tensor([0., 0., 0., 0., 0.], dtype=torch.float64)
```

---

### **3.6 The tensor API**

At this point, we know what PyTorch tensors are and how they work under the hood. Before we wrap up, it is worth taking a look at the tensor operations that PyTorch offers. It would be of little use to list them all here. Instead, we’re going to get a general feel for the API and establish a few directions on where to find things in the online documentation at http://pytorch.org/docs.

First, the vast majority of operations on and between tensors are available in the torch module and can also be called as methods of a tensor object. For instance, the transpose function we encountered earlier can be used from the torch module:

```python
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a.shape, a_t.shape
# Out: (torch.Size(), torch.Size())
```

or as a method of the a tensor:

```python
a = torch.ones(3, 2)
a_t = a.transpose(0, 1)
a.shape, a_t.shape
# Out: (torch.Size(), torch.Size())
```

There is no difference between the two forms; they can be used interchangeably. The online docs are exhaustive and well organized, with the tensor operations divided into groups:

- Creation ops—Functions for constructing a tensor, like ones and from_numpy
- Indexing, slicing, joining, mutating ops—Functions for changing the shape, stride, or content of a tensor, like transpose
- Math ops—Functions for manipulating the content of the tensor through computations:
  - Pointwise ops—Functions for obtaining a new tensor by applying a function to each element independently, like abs and cos
  - Reduction ops—Functions for computing aggregate values by iterating through tensors, like mean, std, and norm
  - Comparison ops—Functions for evaluating numerical predicates over tensors, like equal and max
  - Spectral ops—Functions for transforming in and operating in the frequency domain, like stft and hamming_window
  - Other operations—Special functions operating on vectors, like cross, or matrices, like trace
  - BLAS and LAPACK operations—Functions following the Basic Linear Algebra Subprograms (BLAS) specification for scalar, vector-vector, matrix-vector, and matrix-matrix operations
- Random sampling—Functions for generating values by drawing randomly from probability distributions, like randn and normal
- Serialization—Functions for saving and loading tensors, like load and save
- Parallelism—Functions for controlling the number of threads for parallel CPU execution, like set_num_threads

Take some time to play with the general tensor API. This chapter has provided all the prerequisites to enable this kind of interactive exploration. We will also encounter several of the tensor operations as we proceed with the book, starting in the next chapter.

---

### **3.7 Tensors: Scenic views of storage**

Values in tensors are allocated in contiguous chunks of memory managed by torch.Storage instances. A storage is a one-dimensional array of numerical data: that is, a contiguous block of memory containing numbers of a given type, such as float or int64. A PyTorch Tensor instance is a view of such a Storage instance that is capable of indexing into that storage using an offset and per-dimension strides.

Multiple tensors can index the same storage even if they index into the data differently. We can see an example of this in figure 3.4. In fact, when we requested points in section 3.2, what we got back is another tensor that indexes the same storage as the points tensor—just not all of it, and with different dimensionality (1D versus 2D). The underlying memory is allocated only once, however, so creating alternate tensor-views of the data can be done quickly regardless of the size of the data managed by the Storage instance.

Figure 3.4 Tensors are views of a Storage instance.

#### **3.7.1 Indexing into storage**

Let’s see how indexing into the storage works in practice with our 2D points. The storage for a given tensor is accessible using the .storage property:

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storage()
# Out: 4.0 1.0 5.0 3.0 2.0 1.0
# [torch.FloatStorage of size 6]
```

Even though the tensor reports itself as having three rows and two columns, the storage under the hood is a contiguous array of size 6. In this sense, the tensor just knows how to translate a pair of indices into a location in the storage.

We can also index into a storage manually:

```python
points_storage = points.storage()
points_storage
# Out: 4.0

points.storage()
# Out: 1.0
```

We can’t index a storage of a 2D tensor using two indices. The layout of a storage is always one-dimensional, regardless of the dimensionality of any and all tensors that might refer to it.

At this point, it shouldn’t come as a surprise that changing the value of a storage leads to changing the content of its referring tensor:

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storage = points.storage()
points_storage = 2.0
points
# Out: tensor([[2., 1.],
#              [5., 3.],
#              [2., 1.]])
```

#### **3.7.2 Modifying stored values: In-place operations**

In addition to the operations on tensors introduced in the previous section, a small number of operations exist only as methods of the Tensor object. They are recognizable from a trailing underscore in their name, like zero_, which indicates that the method operates in place by modifying the input instead of creating a new output tensor and returning it. For instance, the zero_ method zeros out all the elements of the input. Any method without the trailing underscore leaves the source tensor unchanged and instead returns a new tensor:

```python
a = torch.ones(3, 2)
a.zero_()
a
# Out: tensor([[0., 0.],
#              [0., 0.],
#              [0., 0.]])
```

---

### **3.8 Tensor metadata: Size, offset, and stride**

In order to index into a storage, tensors rely on a few pieces of information that, together with their storage, unequivocally define them: size, offset, and stride. How these interact is shown in figure 3.5. The size (or shape, in NumPy parlance) is a tuple indicating how many elements across each dimension the tensor represents. The storage offset is the index in the storage corresponding to the first element in the tensor. The stride is the number of elements in the storage that need to be skipped over to obtain the next element along each dimension.

Figure 3.5 Relationship between a tensor’s offset, size, and stride. Here the tensor is a view of a larger storage, like one that might have been allocated when creating a larger tensor.

#### **3.8.1 Views of another tensor’s storage**

We can get the second point in the tensor by providing the corresponding index:

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points
second_point.storage_offset()
# Out: 2

second_point.size()
# Out: torch.Size()
```

The resulting tensor has offset 2 in the storage (since we need to skip the first point, which has two items), and the size is an instance of the Size class containing one element, since the tensor is one-dimensional. It’s important to note that this is the same information contained in the shape property of tensor objects:

```python
second_point.shape
# Out: torch.Size()
```

The stride is a tuple indicating the number of elements in the storage that have to be skipped when the index is increased by 1 in each dimension. For instance, our points tensor has a stride of (2, 1):

```python
points.stride()
# Out: (2, 1)
```

Accessing an element i, j in a 2D tensor results in accessing the storage_offset + stride * i + stride * j element in the storage. The offset will usually be zero; if this tensor is a view of a storage created to hold a larger tensor, the offset might be a positive value.

This indirection between Tensor and Storage makes some operations inexpensive, like transposing a tensor or extracting a subtensor, because they do not lead to memory reallocations. Instead, they consist of allocating a new Tensor object with a different value for size, storage offset, or stride.

We already extracted a subtensor when we indexed a specific point and saw the storage offset increasing. Let’s see what happens to the size and stride as well:

```python
second_point = points
second_point.size()
# Out: torch.Size()

second_point.storage_offset()
# Out: 2

second_point.stride()
# Out: (1,)
```

The bottom line is that the subtensor has one less dimension, as we would expect, while still indexing the same storage as the original points tensor. This also means changing the subtensor will have a side effect on the original tensor:

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points
second_point = 10.0
points
# Out: tensor([[ 4.,  1.],
#              [10.,  3.],
#              [ 2.,  1.]])
```

This might not always be desirable, so we can eventually clone the subtensor into a new tensor:

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points.clone()
second_point = 10.0
points
# Out: tensor([[ 4.,  1.],
#              [ 5.,  3.],
#              [ 2.,  1.]])
```

#### **3.8.2 Transposing without copying**

Let’s try transposing now. Let’s take our points tensor, which has individual points in the rows and X and Y coordinates in the columns, and turn it around so that individual points are in the columns. We take this opportunity to introduce the t function, a shorthand alternative to transpose for two-dimensional tensors:

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points
# Out: tensor([[4., 1.],
#              [5., 3.],
#              [2., 1.]])

points_t = points.t()
points_t
# Out: tensor([[4., 5., 2.],
#              [1., 3., 1.]])
```

We can easily verify that the two tensors share the same storage:

```python
id(points.storage()) == id(points_t.storage())
# Out: True
```

and that they differ only in shape and stride:

```python
points.stride()
# Out: (2, 1)

points_t.stride()
# Out: (1, 2)
```

This tells us that increasing the first index by one in points—for example, going from points to points—will skip along the storage by two elements, while increasing the second index—from points to points—will skip along the storage by one. In other words, the storage holds the elements in the tensor sequentially row by row.

Figure 3.6 Transpose operation applied to a tensor

#### **3.8.3 Transposing in higher dimensions**

Transposing in PyTorch is not limited to matrices. We can transpose a multidimensional array by specifying the two dimensions along which transposing (flipping shape and stride) should occur:

```python
some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
some_t.shape
# Out: torch.Size()

transpose_t.shape
# Out: torch.Size()

some_t.stride()
# Out: (20, 5, 1)

transpose_t.stride()
# Out: (1, 5, 20)
```

A tensor whose values are laid out in the storage starting from the rightmost dimension onward (that is, moving along rows for a 2D tensor) is defined as contiguous. Contiguous tensors are convenient because we can visit them efficiently in order without jumping around in the storage (improving data locality improves performance because of the way memory access works on modern CPUs).

#### **3.8.4 Contiguous tensors**

Some tensor operations in PyTorch only work on contiguous tensors, such as view, which we’ll encounter in the next chapter. In that case, PyTorch will throw an informative exception and require us to call contiguous explicitly. It’s worth noting that calling contiguous will do nothing (and will not hurt performance) if the tensor is already contiguous.

In our case, points is contiguous, while its transpose is not:

```python
points.is_contiguous()
# Out: True

points_t.is_contiguous()
# Out: False
```

We can obtain a new contiguous tensor from a non-contiguous one using the contiguous method. The content of the tensor will be the same, but the stride will change, as will the storage:

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_t = points.t()
points_t
# Out: tensor([[4., 5., 2.],
#              [1., 3., 1.]])

points_t.storage()
# Out: 4.0 1.0 5.0 3.0 2.0 1.0
# [torch.FloatStorage of size 6]

points_t.stride()
# Out: (1, 2)

points_t_cont = points_t.contiguous()
points_t_cont
# Out: tensor([[4., 5., 2.],
#              [1., 3., 1.]])

points_t_cont.stride()
# Out: (3, 1)

points_t_cont.storage()
# Out: 4.0 5.0 2.0 1.0 3.0 1.0
# [torch.FloatStorage of size 6]
```

Notice that the storage has been reshuffled in order for elements to be laid out row-by-row in the new storage. The stride has been changed to reflect the new layout.

Figure 3.7 Relationship between a tensor’s offset, size, and stride. Here the tensor is a view of a larger storage, like one that might have been allocated when creating a larger tensor.

---

### **3.9 Moving tensors to the GPU**

So far in this chapter, when we’ve talked about storage, we’ve meant memory on the CPU. PyTorch tensors also can be stored on a different kind of processor: a graphics processing unit (GPU). Every PyTorch tensor can be transferred to (one of) the GPU(s) in order to perform massively parallel, fast computations. All operations that will be performed on the tensor will be carried out using GPU-specific routines that come with PyTorch.

#### **3.9.1 Managing a tensor’s device attribute**

In addition to dtype, a PyTorch Tensor also has the notion of device, which is where on the computer the tensor data is placed. Here is how we can create a tensor on the GPU by specifying the corresponding argument to the constructor:

```python
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
```

We could instead copy a tensor created on the CPU onto the GPU using the to method:

```python
points_gpu = points.to(device='cuda')
```

Doing so returns a new tensor that has the same numerical data, but stored in the RAM of the GPU, rather than in regular system RAM. Now that the data is stored locally on the GPU, we’ll start to see the speedups mentioned earlier when performing mathematical operations on the tensor. In almost all cases, CPU- and GPU-based tensors expose the same user-facing API, making it much easier to write code that is agnostic to where, exactly, the heavy number crunching is running.

If our machine has more than one GPU, we can also decide on which GPU we allocate the tensor by passing a zero-based integer identifying the GPU on the machine, such as:

```python
points_gpu = points.to(device='cuda:0')
```

At this point, any operation performed on the tensor, such as multiplying all elements by a constant, is carried out on the GPU:

```python
points = 2 * points
points_gpu = 2 * points.to(device='cuda')
```

Note that the points_gpu tensor is not brought back to the CPU once the result has been computed. Performing additional operations like adding a constant continues on the GPU:

```python
points_gpu = points_gpu + 4
```

In order to move the tensor back to the CPU, we need to provide a cpu argument to the to method, or use shorthand methods:

```python
points_cpu = points_gpu.to(device='cpu')

points_gpu = points.cuda()
points_gpu = points.cuda(0)
points_cpu = points_gpu.cpu()
```

---

### **3.10 NumPy interoperability**

PyTorch tensors can be converted to NumPy arrays and vice versa very efficiently. By doing so, we can take advantage of the huge swath of functionality in the wider Python ecosystem that has built up around the NumPy array type. This zero-copy interoperability with NumPy arrays is due to the storage system working with the Python buffer protocol.

To get a NumPy array out of our points tensor, we call:

```python
points = torch.ones(3, 4)
points_np = points.numpy()
points_np
# Out: array([[1., 1., 1., 1.],
#        [1., 1., 1., 1.],
#        [1., 1., 1., 1.]], dtype=float32)
```

which returns a NumPy multidimensional array sharing the same underlying buffer with the tensor storage on CPU RAM. Modifying the NumPy array leads to changes in the originating tensor.

Conversely, we obtain a PyTorch tensor from a NumPy array using `torch.from_numpy()`:

```python
points = torch.from_numpy(points_np)
```

---

### **3.11 Generalized tensors are tensors, too**

For the purposes of this book, tensors are multidimensional arrays. How data is stored under the hood is separate from the tensor API, allowing any implementation meeting the API contract to be considered a tensor.

PyTorch dispatches computation functions based on hardware devices (CPUs, GPUs, TPUs) or memory layout strategies (dense, sparse, quantized), as shown in figure 3.8.

Figure 3.8 The dispatcher in PyTorch is one of its key infrastructure bits.

Quantized tensors and sparse tensors utilize specialized backends to optimize computation and memory footprint.

---

### **3.12 Serializing tensors**

PyTorch uses `pickle` under the hood alongside a dedicated storage serialization format to save tensors to disk and reload them:

```python
torch.save(points, '../data/p1ch3/ourpoints.t')
```

```python
with open('../data/p1ch3/ourpoints.t','wb') as f:
    torch.save(points, f)
```

```python
points = torch.load('../data/p1ch3/ourpoints.t')
```

```python
with open('../data/p1ch3/ourpoints.t','rb') as f:
    points = torch.load(f)
```

#### **3.12.1 Serializing to HDF5 with h5py**

For interoperable serialization with other software, tensors can be saved in the portable HDF5 format using `h5py`.

`h5py` is installed via conda:

```bash
$ conda install h5py
```

Converting tensors to NumPy arrays allows saving into HDF5 datasets:

```python
import h5py
f = h5py.File('../data/p1ch3/ourpoints.hdf5', 'w')
dset = f.create_dataset('coords', data=points.numpy())
f.close()
```

HDF5 datasets can be indexed directly on disk to load specific slices into memory:

```python
f = h5py.File('../data/p1ch3/ourpoints.hdf5', 'r')
dset = f['coords']
last_points = dset[-2:]
```

Passing the dataset slice to `torch.from_numpy()` creates a PyTorch tensor:

```python
last_points = torch.from_numpy(dset[-2:])
f.close()
```

---

### **3.13 Conclusion**

Now we have covered everything we need to get started with representing everything in floats. We’ll cover other aspects of tensors—such as creating views of tensors; indexing tensors with other tensors; and broadcasting, which simplifies performing element-wise operations between tensors of different sizes or shapes—as needed along the way.

In chapter 4, we will learn how to represent real-world data in PyTorch. We will start with simple tabular data and move on to something more elaborate. In the process, we will get to know more about tensors.

---

### **3.14 Exercises**

1 Create a tensor a from list(range(9)). Predict and then check the size, offset, and stride.
   a Create a new tensor using b = a.view(3, 3). What does view do? Check that a and b share the same storage.
   b Create a tensor c = b[1:,1:]. Predict and then check the size, offset, and stride.
2 Pick a mathematical operation like cosine or square root. Can you find a corresponding function in the torch library?
   a Apply the function element-wise to a. Why does it return an error?
   b What operation is required to make the function work?
   c Is there a version of your function that operates in place?

---

### **3.15 Summary**

- Neural networks transform floating-point representations into other floating-point representations. The starting and ending representations are typically human interpretable, but the intermediate representations are less so.
- These floating-point representations are stored in tensors.
- Tensors are multidimensional arrays; they are the basic data structure in PyTorch.
- PyTorch has a comprehensive standard library for tensor creation, manipulation, and mathematical operations.
- Tensors can be serialized to disk and loaded back.
- All tensor operations in PyTorch can execute on the CPU as well as on the GPU, with no change in the code.
- PyTorch uses a trailing underscore to indicate that a function operates in place on a tensor (for example, Tensor.sqrt_).

---