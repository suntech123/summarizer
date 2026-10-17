# Chapter 4: Real-world data representation using tensors

In the previous chapter, we learned that tensors are building blocks for data in PyTorch, functioning as inputs, outputs, and parameters like weights and biases. Knowing how to perform operations on tensors and index them effectively is central to using PyTorch successfully. This chapter covers representing real-world data as PyTorch tensors, working with various data types, loading data from files, converting data to tensors, and shaping tensors to serve as inputs for neural network models.

We explore how to take raw data such as videos or text and represent them as tensors appropriate for training deep learning models. Each section covers a specific data type and dataset, starting with image data, moving to volumetric medical data, tabular wine data, time-series bike-sharing data, and text data from Jane Austen. In every section, we stop right before feeding the data into a model to prepare excellent training material for subsequent chapters.

## 4.1 Working with images
[[content - 4.1 Working with images]]

### 4.1.1 Adding color channels
[[content - 4.1 Working with images/4.1.1 Adding color channels]]

### 4.1.2 Loading an image file
[[content - 4.1 Working with images/4.1.2 Loading an image file]]

### 4.1.3 Working with images
[[content - 4.1 Working with images/4.1.3 Working with images]]

### 4.1.4 Normalizing the data
[[content - 4.1 Working with images/4.1.4 Normalizing the data]]

## 4.2 3D images: Volumetric data
[[content - 4.2 3D images: Volumetric data]]

### 4.2.1 Loading a specialized format
[[content - 4.2 3D images: Volumetric data/4.2.1 Loading a specialized format]]

## 4.3 Representing tabular data
[[content - 4.3 Representing tabular data]]

### 4.3.1 Using a real-world dataset
[[content - 4.3 Representing tabular data/4.3.1 Using a real-world dataset]]

### 4.3.2 Loading a wine data tensor
[[content - 4.3 Representing tabular data/4.3.2 Loading a wine data tensor]]

### 4.3.3 Representing scores
[[content - 4.3 Representing tabular data/4.3.3 Representing scores]]

### 4.3.4 One-hot encoding
[[content - 4.3 Representing tabular data/4.3.4 One-hot encoding]]

### 4.3.5 When to categorize
[[content - 4.3 Representing tabular data/4.3.5 When to categorize]]

### 4.3.6 Finding thresholds
[[content - 4.3 Representing tabular data/4.3.6 Finding thresholds]]

## 4.4 Working with time series
[[content - 4.4 Working with time series]]

### 4.4.1 Adding a time dimension
[[content - 4.4 Working with time series/4.4.1 Adding a time dimension]]

### 4.4.2 Shaping the data by time period
[[content - 4.4 Working with time series/4.4.2 Shaping the data by time period]]

### 4.4.3 Ready for training
[[content - 4.4 Working with time series/4.4.3 Ready for training]]

## 4.5 Representing text
[[content - 4.5 Representing text]]

### 4.5.1 Converting text to numbers
[[content - 4.5 Representing text/4.5.1 Converting text to numbers]]

### 4.5.2 One-hot-encoding characters
[[content - 4.5 Representing text/4.5.2 One-hot-encoding characters]]

### 4.5.3 One-hot encoding whole words
[[content - 4.5 Representing text/4.5.3 One-hot encoding whole words]]

### 4.5.4 Text embeddings
[[content - 4.5 Representing text/4.5.4 Text embeddings]]

### 4.5.5 Text embeddings as a blueprint
[[content - 4.5 Representing text/4.5.5 Text embeddings as a blueprint]]

## 4.6 Conclusion

We have explored loading and shaping various real-world data types into tensor structures for neural network ingestion, laying the groundwork for training deep learning models in subsequent chapters.

## 4.7 Exercises

**Problem 4.1** Take several pictures of red, blue, and green items with your phone or other digital camera (or download some from the internet, if a camera isn't available). a Load each image, and convert it to a tensor. b For each image tensor, use the mean() method to get a sense of how bright the image is. c Take the mean of each channel of your images. Can you identify the red, green, and blue items from only the channel averages?

**Problem 4.2** Select a relatively large file containing Python source code. a Build an index of all the words in the source file (feel free to make your tokenization as simple or as complex as you like; we suggest starting with replacing r"[^a-zA-Z0-9_]+" with spaces). b Compare your index with the one we made for Pride and Prejudice. Which is larger? c Create the one-hot encoding for the source code file. d What information is lost with this encoding? How does that information compare to what's lost in the Pride and Prejudice encoding?

## 4.8 Summary

Neural networks require data to be represented as multidimensional numerical tensors, often 32-bit floating-point. In general, PyTorch expects data to be laid out along specific dimensions according to the model architecture—for example, convolutional versus recurrent. We can reshape data effectively with the PyTorch tensor API. Thanks to how the PyTorch libraries interact with the Python standard library and surrounding ecosystem, loading the most common types of data and converting them to PyTorch tensors is convenient. Images can have one or many channels. The most common are the red-green-blue channels of typical digital photos. Many images have a per-channel bit depth of 8, though 12 and 16 bits per channel are not uncommon. These bit depths can all be stored in a 32-bit floating-point number without loss of precision. Single-channel data formats sometimes omit an explicit channel dimension. Volumetric data is similar to 2D image data, with the exception of adding a third dimension (depth). Converting spreadsheets to tensors can be very straightforward. Categorical- and ordinal-valued columns should be handled differently from interval-valued columns. Text or categorical data can be encoded to a one-hot representation through the use of dictionaries. Very often, embeddings give good, efficient representations.