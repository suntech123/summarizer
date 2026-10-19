# Chapter 4: Real-world data representation using tensors

In the previous chapter, we learned that tensors are the building blocks for data in PyTorch. Neural networks take tensors as input and produce tensors as outputs. All operations within a neural network and during optimization are operations between tensors, and all parameters in a neural network are tensors. Having a good sense of how to perform operations on tensors and index them effectively is central to using tools like PyTorch successfully.

Now that you know the basics of tensors, your dexterity with them will grow as you make your way through the book. We’ll learn how to take a piece of data, a video, or a line of text, and represent it with a tensor in a way that is appropriate for training a deep learning model. We’ll cover different types of data with a focus on the types relevant to this book and show how to represent that data as tensors. Then we’ll learn how to load the data from the most common on-disk formats and get a feel for those data types’ structure so we can see how to prepare them for training a neural network.

Each section in this chapter will describe a data type, and each will come with its own dataset. We’ll start with image data, then demonstrate working with a three-dimensional array using medical data representing patient anatomy as a volume. Next, we’ll work with tabular data about wines, move to ordered tabular data with a time-series dataset from a bike-sharing program, and finally dip into text data from Jane Austen. In every section, we will stop right before feeding the data to a model.

## 4.1 Working with images

The introduction of convolutional neural networks revolutionized computer vision, and image-based systems have since acquired a whole new set of capabilities. Problems that required complex pipelines of highly tuned algorithmic building blocks are now solvable at unprecedented levels of performance by training end-to-end networks using paired input-and-desired-output examples. In order to participate in this revolution, we need to be able to load an image from common image formats and then transform the data into a tensor representation that has the various parts of the image arranged in the way PyTorch expects.

An image is represented as a collection of scalars arranged in a regular grid with a height and a width (in pixels). We might have a single scalar per grid point (the pixel) representing a grayscale image, or multiple scalars per grid point representing different colors or features like depth from a depth camera. Scalars representing values at individual pixels are often encoded using 8-bit integers, as in consumer cameras. In medical, scientific, and industrial applications, higher numerical precision such as 12-bit or 16-bit allows a wider range or increased sensitivity when encoding physical properties like bone density, temperature, or depth.

### 4.1.1 Adding color channels

There are several ways to encode colors into numbers. The most common is RGB, where a color is defined by three numbers representing the intensity of red, green, and blue. We can think of a color channel as a grayscale intensity map of only the color in question. Figure 4.1 shows a rainbow, where each of the RGB channels captures a certain portion of the spectrum (the figure is simplified, in that it elides things like the orange and yellow bands being represented as a combination of red and green).

[[img-01]]
Figure 4.1 - A rainbow, broken into red, green, and blue channels.

The red band of the rainbow is brightest in the red channel of the image, while the blue channel has both the blue band of the rainbow and the sky as high-intensity. Note also that the white clouds are high-intensity in all three channels.

### 4.1.2 Loading an image file

Images come in several different file formats, but luckily there are plenty of ways to load images in Python. Let’s start by loading a PNG image using the `imageio` module (`code/p1ch4/1_image_dog.ipynb`).

```python
# In[2]:
import imageio
img_arr = imageio.imread('../data/p1ch4/image-dog/bobby.jpg')
img_arr.shape
# Out[2]:
(720, 1280, 3)

```

At this point, `img` is a NumPy array-like object with three dimensions: two spatial dimensions, width and height; and a third dimension corresponding to the red, green, and blue channels. Any library that outputs a NumPy array will suffice to obtain a PyTorch tensor. The only thing to watch out for is the layout of the dimensions, as PyTorch modules dealing with image data require tensors to be laid out as $C \times H \times W$: channels, height, and width, respectively.

### 4.1.3 Changing the layout

We can use the tensor’s `permute` method with the old dimensions for each new dimension to get to an appropriate layout. Given an input tensor $H \times W \times C$ as obtained previously, we get a proper layout by having channel 2 first and then channels 0 and 1:

```python
# In[3]:
img = torch.from_numpy(img_arr)
out = img.permute(2, 0, 1)

```

We’ve seen this previously, but note that this operation does not make a copy of the tensor data. Instead, `out` uses the same underlying storage as `img` and only plays with the size and stride information at the tensor level. Changing a pixel in `img` will lead to a change in `out`.

Other deep learning frameworks use different layouts, such as TensorFlow originally keeping the channel dimension last, resulting in an $H \times W \times C$ layout. So far, we have described a single image. To create a dataset of multiple images to use as an input for our neural networks, we store the images in a batch along the first dimension to obtain an $N \times C \times H \times W$ tensor.

As a slightly more efficient alternative to using `stack` to build up the tensor, we can preallocate a tensor of appropriate size and fill it with images loaded from a directory, like so:

```python
# In[4]:
batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

```

This indicates that our batch will consist of three RGB images 256 pixels in height and 256 pixels in width. We can now load all PNG images from an input directory and store them in the tensor:

```python
# In[5]:
import os
data_dir = '../data/p1ch4/image-cats/'
filenames = [name for name in os.listdir(data_dir)
if os.path.splitext(name)[-1] == '.png']
for i, filename in enumerate(filenames):
img_arr = imageio.imread(os.path.join(data_dir, filename))
img_t = torch.from_numpy(img_arr)
img_t = img_t.permute(2, 0, 1)
img_t = img_t[:3]
batch[i] = img_t

```

### 4.1.4 Normalizing the data

Neural networks usually work with floating-point tensors as their input and exhibit the best training performance when the input data ranges roughly from 0 to 1, or from -1 to 1. So a typical thing we’ll want to do is cast a tensor to floating-point and normalize the values of the pixels. One possibility is to just divide the values of the pixels by 255 (the maximum representable number in 8-bit unsigned):

```python
# In[6]:
batch = batch.float()
batch /= 255.0

```

Another possibility is to compute the mean and standard deviation of the input data and scale it so that the output has zero mean and unit standard deviation across each channel:

```python
# In[7]:
n_channels = batch.shape[1]
for c in range(n_channels):
mean = torch.mean(batch[:, c])
std = torch.std(batch[:, c])
batch[:, c] = (batch[:, c] - mean) / std

```

We can perform several other operations on inputs, such as geometric transformations like rotations, scaling, and cropping. These may help with training or may be required to make an arbitrary input conform to the input requirements of a network, like the size of the image.

## 4.2 3D images: Volumetric data

We’ve learned how to load and represent 2D images, like the ones we take with a camera. In medical imaging applications involving CT (computed tomography) scans, we typically deal with sequences of images stacked along the head-to-foot axis, each corresponding to a slice across the human body. In CT scans, the intensity represents the density of different parts of the body mapped from dark to bright when displayed on a clinical workstation.

CTs have only a single intensity channel, similar to a grayscale image. By stacking individual 2D slices into a 3D tensor, we can build volumetric data representing the 3D anatomy of a subject. Unlike what we saw in figure 4.1, the extra dimension in figure 4.2 represents an offset in physical space, rather than a particular band of the visible spectrum.

[[img-02]]
Figure 4.2 - Slices of a CT scan, from the top of the head to the jawline.

There’s no fundamental difference between a tensor storing volumetric data versus image data. We just have an extra dimension, depth, after the channel dimension, leading to a 5D tensor of shape $N \times C \times D \times H \times W$.

### 4.2.1 Loading a specialized format

Let’s load a sample CT scan using the `volread` function in the `imageio` module, which takes a directory as an argument and assembles all Digital Imaging and Communications in Medicine (DICOM) files in a series in a NumPy 3D array (`code/p1ch4/2_volumetric_ct.ipynb`).

```python
# In[2]:
import imageio
dir_path = "../data/p1ch4/volumetric-dicom/2-LUNG 3.0 B70f-04083"
vol_arr = imageio.volread(dir_path, 'DICOM')
vol_arr.shape
# Out[2]:
Reading DICOM (examining files): 1/99 files (1.0%99/99 files (100.0%)
Found 1 correct series.
Reading DICOM (loading data): 31/99 (31.392/99 (92.999/99 (100.0%)
(99, 512, 512)

```

As was true in section 4.1.3, the layout is different from what PyTorch expects, due to having no channel information. So we’ll have to make room for the channel dimension using `unsqueeze`:

```python
# In[3]:
vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol, 0)
vol.shape
# Out[3]:
torch.Size([1, 99, 512, 512])

```

At this point we could assemble a 5D dataset by stacking multiple volumes along the batch direction, just as we did in the previous section.

## 4.3 Representing tabular data

The simplest form of data we’ll encounter on a machine learning job is sitting in a spreadsheet, CSV file, or database. It is a table containing one row per sample (or record), where columns contain one piece of information about our sample.

We assume there’s no meaning to the order in which samples appear in the table: such a table is a collection of independent samples. Columns may contain numerical values or categorical labels, meaning tabular data is typically not homogeneous. PyTorch tensors, on the other hand, are homogeneous numerical entities that take real numbers as inputs and produce real numbers as output.

### 4.3.1 Using a real-world dataset

Our first job as deep learning practitioners is to encode heterogeneous, real-world data into a tensor of floating-point numbers, ready for consumption by a neural network. The Wine Quality dataset is a freely available table containing chemical characterizations of samples of *vinho verde*, together with a sensory quality score.

The file contains a comma-separated collection of values organized in 12 columns preceded by a header line containing the column names. The first 11 columns contain values of chemical variables, and the last column contains the sensory quality score from 0 (very bad) to 10 (excellent). These are the column names in the order they appear in the dataset:

fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
quality

A possible machine learning task on this dataset is predicting the quality score from chemical characterization alone. As we can see in figure 4.3, we’re hoping to find a relationship between one of the chemical columns in our data and the quality column. Here, we’re expecting to see quality increase as sulfur decreases.

[[img-03]]
Figure 4.3 - The (we hope) relationship between sulfur and quality in wine.

### 4.3.2 Loading a wine data tensor

Python offers several options for quickly loading a CSV file, such as the `csv` module, NumPy, and Pandas. Since PyTorch has excellent NumPy interoperability, we’ll load our file using NumPy and turn the resulting array into a PyTorch tensor (`code/p1ch4/3_tabular_wine.ipynb`).

```python
# In[2]:
import csv
wine_path = "../data/p1ch4/tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";",
skiprows=1)
wineq_numpy
# Out[2]:
array([[ 7. , 0.27, 0.36, ..., 0.45, 8.8 , 6. ],
[ 6.3 , 0.3 , 0.34, ..., 0.49, 9.5 , 6. ],
[ 8.1 , 0.28, 0.4 , ..., 0.44, 10.1 , 6. ],
...,
[ 6.5 , 0.24, 0.19, ..., 0.46, 9.4 , 6. ],
[ 5.5 , 0.29, 0.3 , ..., 0.38, 12.8 , 7. ],
[ 6. , 0.21, 0.38, ..., 0.32, 11.8 , 6. ]], dtype=float32)

```

Here we specify 32-bit floating-point types, the semicolon delimiter, and skip the header row. Let’s check that all the data has been read:

```python
# In[3]:
col_list = next(csv.reader(open(wine_path), delimiter=';'))
wineq_numpy.shape, col_list
# Out[3]:
((4898, 12),
['fixed acidity',
'volatile acidity',
'citric acid',
'residual sugar',
'chlorides',
'free sulfur dioxide',
'total sulfur dioxide',
'density',
'pH',
'sulphates',
'alcohol',
'quality'])

```

We then proceed to convert the NumPy array to a PyTorch tensor:

```python
# In[4]:
wineq = torch.from_numpy(wineq_numpy)
wineq.shape, wineq.dtype
# Out[4]:
(torch.Size([4898, 12]), torch.float32)

```

At this point, we have a floating-point `torch.Tensor` containing all the columns, including the last, which refers to the quality score.

### 4.3.3 Representing scores

We could treat the score as a continuous variable or as a label in a classification task. In both approaches, we will typically remove the score from the tensor of input data and keep it in a separate tensor:

```python
# In[5]:
data = wineq[:, :-1]
data, data.shape
# Out[5]:
(tensor([[ 7.00, 0.27, ..., 0.45, 8.80],
[ 6.30, 0.30, ..., 0.49, 9.50],
...,
[ 5.50, 0.29, ..., 0.38, 12.80],
[ 6.00, 0.21, ..., 0.32, 11.80]]), torch.Size([4898, 11]))
# In[6]:
target = wineq[:, -1]
target, target.shape
# Out[6]:
(tensor([6., 6., ..., 7., 6.]), torch.Size([4898]))

```

If we want to transform the target tensor into a tensor of labels, one option is simply to treat labels as an integer vector of scores:

```python
# In[7]:
target = wineq[:, -1].long()
target
# Out[7]:
tensor([6, 6, ..., 7, 6])

```

### 4.3.4 One-hot encoding

The other approach is to build a one-hot encoding of the scores: that is, encode each of the 10 scores in a vector of 10 elements, with all elements set to 0 but one. We can achieve one-hot encoding using the `scatter_` method, which fills the tensor with values from a source tensor along the indices provided as arguments:

```python
# In[8]:
target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
# Out[8]:
tensor([[0., 0., ..., 0., 0.],
[0., 0., ..., 0., 0.],
...,
[0., 0., ..., 0., 0.],
[0., 0., ..., 0., 0.]])

```

The underscore ending on `scatter_` indicates an in-place operation in PyTorch. Its arguments are:

* The dimension along which the following two arguments are specified
* A column tensor indicating the indices of the elements to scatter
* A tensor containing the elements to scatter or a single scalar to scatter (1, in this case)

The second argument requires the same number of dimensions as the target tensor, so we add a dummy dimension using `unsqueeze`:

```python
# In[9]:
target_unsqueezed = target.unsqueeze(1)
target_unsqueezed
# Out[9]:
tensor([[6],
[6],
...,
[7],
[6]])

```

### 4.3.5 When to categorize

Ordinal data can be treated as categorical or continuous depending on whether ordering is a priority. We summarize our data mapping in a small flow chart in figure 4.4.

[[img-04]]
Figure 4.4 - How to treat columns with continuous, ordinal, and categorical data.

We can use the functions in the PyTorch Tensor API to obtain the mean and standard deviations for each column:

```python
# In[10]:
data_mean = torch.mean(data, dim=0)
data_mean
# Out[10]:
tensor([6.85e+00, 2.78e-01, 3.34e-01, 6.39e+00, 4.58e-02, 3.53e+01,
1.38e+02, 9.94e-01, 3.19e+00, 4.90e-01, 1.05e+01])
# In[11]:
data_var = torch.var(data, dim=0)
data_var
# Out[11]:
tensor([7.12e-01, 1.02e-02, 1.46e-02, 2.57e+01, 4.77e-04, 2.89e+02,
1.81e+03, 8.95e-06, 2.28e-02, 1.30e-02, 1.51e+00])

```

We normalize the data by subtracting the mean and dividing by the standard deviation:

```python
# In[12]:
data_normalized = (data - data_mean) / torch.sqrt(data_var)
data_normalized
# Out[12]:
tensor([[ 1.72e-01, -8.18e-02, ..., -3.49e-01, -1.39e+00],
[-6.57e-01, 2.16e-01, ..., 1.35e-03, -8.24e-01],
...,
[-1.61e+00, 1.17e-01, ..., -9.63e-01, 1.86e+00],
[-1.01e+00, -6.77e-01, ..., -1.49e+00, 1.04e+00]])

```

### 4.3.6 Finding thresholds

First, we’re going to determine which rows in `target` correspond to a score less than or equal to 3:

```python
# In[13]:
bad_indexes = target <= 3
bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum()
# Out[13]:
(torch.Size([4898]), torch.bool, tensor(20))

```

By using advanced indexing, we can use a boolean tensor to filter data rows corresponding to `True`:

```python
# In[14]:
bad_data = data[bad_indexes]
bad_data.shape
# Out[14]:
torch.Size([20, 11])

```

Now we can get information about wines grouped into good, middling, and bad categories by taking the mean of each column:

```python
# In[15]:
bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target >= 7]
bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)
for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))
# Out[15]:
0 fixed acidity 7.60 6.89 6.73
1 volatile acidity 0.33 0.28 0.27
2 citric acid 0.34 0.34 0.33
3 residual sugar 6.39 6.71 5.26
4 chlorides 0.05 0.05 0.04
5 free sulfur dioxide 53.33 35.42 34.55
6 total sulfur dioxide 170.60 141.83 125.25
7 density 0.99 0.99 0.99
8 pH 3.19 3.18 3.22
9 sulphates 0.47 0.49 0.50
10 alcohol 10.34 10.26 11.42

```

Let’s get the indexes where the total sulfur dioxide column is below the midpoint we calculated earlier:

```python
# In[16]:
total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum()
# Out[16]:
(torch.Size([4898]), torch.bool, tensor(2727))

```

Next, we’ll need to get the indexes of the actually good wines:

```python
# In[17]:
actual_indexes = target > 5
actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum()
# Out[17]:
(torch.Size([4898]), torch.bool, tensor(3258))

```

We will perform a logical “and” between our prediction indexes and the actual good indexes to see how well we did:

```python
# In[18]:
n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
n_matches, n_matches / n_predicted, n_matches / n_actual
# Out[18]:
(2018, 0.74000733406674, 0.6193984039287906)

```

## 4.4 Working with time series

In the previous section, every row in the table was independent from the others. We’ll switch to a dataset from a Washington, D.C., bike-sharing system reporting the hourly count of rental bikes in 2011–2012 along with weather and seasonal information. Our goal will be to take a flat, 2D dataset and transform it into a 3D one, as shown in figure 4.5.

[[img-05]]
Figure 4.5 - Transforming a 1D, multichannel dataset into a 2D, multichannel dataset by separating the date and hour of each sample into separate axes.

### 4.4.1 Adding a time dimension

We want to change the row-per-hour organization so that we have one axis for the day, another axis for the hour of the day, and a third axis for our different columns of data. Let’s load the data (`code/p1ch4/4_time_series_bikes.ipynb`):

```python
# In[2]:
bikes_numpy = np.loadtxt(
"../data/p1ch4/bike-sharing-dataset/hour-fixed.csv",
dtype=np.float32,
delimiter=",",
skiprows=1,
converters={1: lambda x: float(x[8:10])})
bikes = torch.from_numpy(bikes_numpy)
bikes
# Out[2]:
tensor([[1.0000e+00, 1.0000e+00, ..., 1.3000e+01, 1.6000e+01],
[2.0000e+00, 1.0000e+00, ..., 3.2000e+01, 4.0000e+01],
...,
[1.7378e+04, 3.1000e+01, ..., 4.8000e+01, 6.1000e+01],
[1.7379e+04, 3.1000e+01, ..., 3.7000e+01, 4.9000e+01]])

```

For every hour, the dataset reports the following variables:

* Index of record: `instant`
* Day of month: `day`
* Season: `season` (1: spring, 2: summer, 3: fall, 4: winter)
* Year: `yr` (0: 2011, 1: 2012)
* Month: `mnth` (1 to 12)
* Hour: `hr` (0 to 23)
* Holiday status: `holiday`
* Day of the week: `weekday`
* Working day status: `workingday`
* Weather situation: `weathersit` (1: clear, 2:mist, 3: light rain/snow, 4: heavy rain/snow)
* Temperature in °C: `temp`
* Perceived temperature in °C: `atemp`
* Humidity: `hum`
* Wind speed: `windspeed`
* Number of casual users: `casual`
* Number of registered users: `registered`
* Count of rental bikes: `cnt`

The neural network model will need to see sequences of values for each quantity: $N$ parallel sequences of size $C$, where $N$ represents the time axis.

### 4.4.2 Shaping the data by time period

To break up the two-year dataset into daily observation periods, our time series dataset would be a tensor of dimension 3 and shape $N \times C \times L$, where $C$ is 17 channels and $L$ is 24 hours. Let's take a look at the shape and strides of our `bikes` tensor:

```python
# In[3]:
bikes.shape, bikes.stride()
# Out[3]:
(torch.Size([17520, 17]), (17, 1))

```

We reshape the data to have 3 axes—day, hour, and columns:

```python
# In[4]:
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
daily_bikes.shape, daily_bikes.stride()
# Out[4]:
(torch.Size([730, 24, 17]), (408, 17, 1))

```

To get to our desired $N \times C \times L$ ordering, we transpose the tensor:

```python
# In[5]:
daily_bikes = daily_bikes.transpose(1, 2)
daily_bikes.shape, daily_bikes.stride()
# Out[5]:
(torch.Size([730, 17, 24]), (408, 1, 17))

```

### 4.4.3 Ready for training

Limiting ourselves to the first day, we initialize a zero-filled matrix and check the weather values:

```python
# In[6]:
first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
first_day[:,9]
# Out[6]:
tensor([1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 2, 2,
2, 2])

```

Then we scatter ones into our matrix according to the corresponding level at each row:

```python
# In[7]:
weather_onehot.scatter_(
dim=1,
index=first_day[:,9].unsqueeze(1).long() - 1,
value=1.0)
# Out[7]:
tensor([[1., 0., 0., 0.],
[1., 0., 0., 0.],
...,
[0., 1., 0., 0.],
[0., 1., 0., 0.]])

```

Last, we concatenate our matrix to our original dataset using the `cat` function:

```python
# In[8]:
torch.cat((bikes[:24], weather_onehot), 1)[:1]
# Out[8]:
tensor([[ 1.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000,
6.0000, 0.0000, 1.0000, 0.2400, 0.2879, 0.8100, 0.0000,
3.0000, 13.0000, 16.0000, 1.0000, 0.0000, 0.0000, 0.0000]])

```

We could have done the same with the reshaped `daily_bikes` tensor:

```python
# In[9]:
daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4,
daily_bikes.shape[2])
daily_weather_onehot.shape
# Out[9]:
torch.Size([730, 4, 24])

```

We scatter the one-hot encoding and concatenate along the $C$ dimension:

```python
# In[10]:
daily_weather_onehot.scatter_(
1, daily_bikes[:,9,:].long().unsqueeze(1) - 1, 1.0)
daily_weather_onehot.shape
# Out[10]:
torch.Size([730, 4, 24])

```

```python
# In[11]:
daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)

```

Alternatively, we can rescale the weather variable to the interval $[0.0, 1.0]$:

```python
# In[12]:
daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0

```

We can also map temperature to $[0.0, 1.0]$:

```python
# In[13]:
temp = daily_bikes[:, 10, :]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - temp_min)
/ (temp_max - temp_min))

```

Or subtract the mean and divide by the standard deviation:

```python
# In[14]:
temp = daily_bikes[:, 10, :]
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - torch.mean(temp))
/ torch.std(temp))

```

## 4.5 Representing text

Recurrent neural networks (RNNs) and transformers have revolutionized natural language processing (NLP) by enabling end-to-end network training on large corpora directly. Our goal is to convert text into numerical tensors suitable for network input.

### 4.5.1 Converting text to numbers

Networks can operate at character level or word level using one-hot encoding. We load *Pride and Prejudice* from Project Gutenberg (`code/p1ch4/5_text_jane_austen.ipynb`):

```python
# In[2]:
with open('../data/p1ch4/jane-austen/1342-0.txt', encoding='utf8') as f:
text = f.read()

```

### 4.5.2 One-hot-encoding characters

We restrict the encoding to ASCII characters. We split text into lines and select an arbitrary line:

```python
# In[3]:
lines = text.split('\n')
line = lines[200]
line
# Out[3]:
'“Impossible, Mr. Bennet, impossible, when I am not acquainted with him'

```

We initialize a tensor for the line:

```python
# In[4]:
letter_t = torch.zeros(len(line), 128)
letter_t.shape
# Out[4]:
torch.Size([70, 128])

```

We iterate over the line and populate ones:

```python
# In[5]:
for i, letter in enumerate(line.lower().strip()):
letter_index = ord(letter) if ord(letter) < 128 else 0
letter_t[i][letter_index] = 1

```

### 4.5.3 One-hot encoding whole words

We define `clean_words` to preprocess text:

```python
# In[6]:
def clean_words(input_str):
punctuation = '.,;:"!?”“_-'
word_list = input_str.lower().replace('\n',' ').split()
word_list = [word.strip(punctuation) for word in word_list]
return word_list
words_in_line = clean_words(line)
line, words_in_line
# Out[6]:
('“Impossible, Mr. Bennet, impossible, when I am not acquainted with him',
['impossible',
'mr',
'bennet',
'impossible',
'when',
'i',
'am',
'not',
'acquainted',
'with',
'him'])

```

We build a mapping of vocabulary to indices:

```python
# In[7]:
word_list = sorted(set(clean_words(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}
len(word2index_dict), word2index_dict['impossible']
# Out[7]:
(7261, 3394)

```

We encode the sentence into a tensor:

```python
# In[8]:
word_t = torch.zeros(len(words_in_line), len(word2index_dict))
for i, word in enumerate(words_in_line):
word_index = word2index_dict[word]
word_t[i][word_index] = 1
print('{:2} {:4} {}'.format(i, word_index, word))
print(word_t.shape)
# Out[8]:
0 3394 impossible
1 4305 mr
2 813 bennet
3 3394 impossible
4 7078 when
5 3315 i
6 415 am
7 4436 not
8 239 acquainted
9 7148 with
10 3215 him
torch.Size([11, 7261])

```

Figure 4.6 compares options for splitting text and using embeddings.

[[img-06]]
Figure 4.6 - Three ways to encode a word.

### 4.5.4 Text embeddings

Embeddings compress high-dimensional representations down to floating-point vectors of manageable size. Figure 4.7 illustrates manual embeddings in a 2D space.

[[img-07]]
Figure 4.7 - Our manual word embeddings.

### 4.5.5 Text embeddings as a blueprint

Embeddings provide an efficient alternative to one-hot encodings for processing categorical data across various domains.

## 4.6 Conclusion

We covered loading and shaping fundamental data types into tensors suitable for deep learning models.

## 4.7 Exercises

1. Take several pictures of red, blue, and green items with your phone or other digital camera (or download some from the internet, if a camera isn’t available).
a. Load each image, and convert it to a tensor.
b. For each image tensor, use the `.mean()` method to get a sense of how bright the image is.
c. Take the mean of each channel of your images. Can you identify the red, green, and blue items from only the channel averages?
2. Select a relatively large file containing Python source code.
a. Build an index of all the words in the source file (feel free to make your tokenization as simple or as complex as you like; we suggest starting with replacing `r"[^a-zA-Z0-9_]+"` with spaces).
b. Compare your index with the one we made for *Pride and Prejudice*. Which is larger?
c. Create the one-hot encoding for the source code file.
d. What information is lost with this encoding? How does that information compare to what’s lost in the *Pride and Prejudice* encoding?

## 4.8 Summary

* Neural networks require data to be represented as multidimensional numerical tensors, often 32-bit floating-point.
* In general, PyTorch expects data to be laid out along specific dimensions according to the model architecture—for example, convolutional versus recurrent. We can reshape data effectively with the PyTorch tensor API.
* Thanks to how the PyTorch libraries interact with the Python standard library and surrounding ecosystem, loading the most common types of data and converting them to PyTorch tensors is convenient.
* Images can have one or many channels. The most common are the red-green-blue channels of typical digital photos.
* Many images have a per-channel bit depth of 8, though 12 and 16 bits per channel are not uncommon. These bit depths can all be stored in a 32-bit floating-point number without loss of precision.
* Single-channel data formats sometimes omit an explicit channel dimension.
* Volumetric data is similar to 2D image data, with the exception of adding a third dimension (depth).
* Converting spreadsheets to tensors can be very straightforward. Categorical- and ordinal-valued columns should be handled differently from interval-valued columns.
* Text or categorical data can be encoded to a one-hot representation through the use of dictionaries. Very often, embeddings give good, efficient representations.