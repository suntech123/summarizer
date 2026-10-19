# Chapter 1: Introducing deep learning and the PyTorch Library

The poorly defined term artificial intelligence covers a set of disciplines that have been subjected to a tremendous amount of research, scrutiny, confusion, fantastical hype, and sci-fi fearmongering. Reality is, of course, far more sanguine. It would be disingenuous to assert that today’s machines are learning to “think” in any human sense of the word. Rather, we’ve discovered a general class of algorithms that are able to approximate complicated, nonlinear processes very, very effectively, which we can use to automate tasks that were previously limited to humans.

For example, at [https://talktotransformer.com](https://talktotransformer.com), a language model called GPT-2 can generate coherent paragraphs of text one word at a time. When we fed it this very paragraph, it produced the following:

Next we’re going to feed in a list of phrases from a corpus of email addresses, and see if the program can parse the lists as sentences. Again, this is much more complicated and far more complex than the search at the beginning of this post, but hopefully helps you understand the basics of constructing sentence structures in various programming languages.

That’s remarkably coherent for a machine, even if there isn’t a well-defined thesis behind the rambling. Even more impressively, the ability to perform these formerly human-only tasks is acquired through examples, rather than encoded by a human as a set of handcrafted rules. In a way, we’re learning that intelligence is a notion we often conflate with self-awareness, and self-awareness is definitely not required to successfully carry out these kinds of tasks. In the end, the question of computer intelligence might not even be important. Edsger W. Dijkstra found that the question of whether machines could think was “about as relevant as the question of whether Submarines Can Swim.”

That general class of algorithms we’re talking about falls under the AI subcategory of deep learning, which deals with training mathematical entities named deep neural networks by presenting instructive examples. Deep learning uses large amounts of data to approximate complex functions whose inputs and outputs are far apart, like an input image and, as output, a line of text describing the input; or a written script as input and a natural-sounding voice reciting the script as output; or, even more simply, associating an image of a golden retriever with a flag that tells us “Yes, a golden retriever is present.” This kind of capability allows us to create programs with functionality that was, until very recently, exclusively the domain of human beings.

## 1.1 The deep learning revolution

To appreciate the paradigm shift ushered in by this deep learning approach, let’s take a step back for a bit of perspective. Until the last decade, the broader class of systems that fell under the label machine learning relied heavily on feature engineering. Features are transformations on input data that facilitate a downstream algorithm, like a classifier, to produce correct outcomes on new data. Feature engineering consists of coming up with the right transformations so that the downstream algorithm can solve a task. For instance, in order to tell ones from zeros in images of handwritten digits, we would come up with a set of filters to estimate the direction of edges over the image, and then train a classifier to predict the correct digit given a distribution of edge directions. Another useful feature could be the number of enclosed holes, as seen in a zero, an eight, and, particularly, loopy twos.

Deep learning, on the other hand, deals with finding such representations automatically, from raw data, in order to successfully perform a task. In the ones versus zeros example, filters would be refined during training by iteratively looking at pairs of examples and target labels. This is not to say that feature engineering has no place with deep learning; we often need to inject some form of prior knowledge in a learning system. However, the ability of a neural network to ingest data and extract useful representations on the basis of examples is what makes deep learning so powerful. The focus of deep learning practitioners is not so much on handcrafting those representations, but on operating on a mathematical entity so that it discovers representations from the training data autonomously. Often, these automatically created features are better than those that are handcrafted! As with many disruptive technologies, this fact has led to a change in perspective.

On the left side of figure 1.1, we see a practitioner busy defining engineering features and feeding them to a learning algorithm; the results on the task will be as good as the features the practitioner engineers. On the right, with deep learning, the raw data is fed to an algorithm that extracts hierarchical features automatically, guided by the optimization of its own performance on the task; the results will be as good as the ability of the practitioner to drive the algorithm toward its goal.

[[img-01]]
Figure 1.1 - Deep learning exchanges the need to handcraft features for an increase in data and computational requirements.

Starting from the right side in figure 1.1, we already get a glimpse of what we need to execute successful deep learning:

* We need a way to ingest whatever data we have at hand.
* We somehow need to define the deep learning machine.
* We must have an automated way, training, to obtain useful representations and make the machine produce desired outputs.

This leaves us with taking a closer look at this training thing we keep talking about. During training, we use a criterion, a real-valued function of model outputs and reference data, to provide a numerical score for the discrepancy between the desired and actual output of our model (by convention, a lower score is typically better). Training consists of driving the criterion toward lower and lower scores by incrementally modifying our deep learning machine until it achieves low scores, even on data not seen during training.

## 1.2 PyTorch for deep learning

PyTorch is a library for Python programs that facilitates building deep learning projects. It emphasizes flexibility and allows deep learning models to be expressed in idiomatic Python. This approachability and ease of use found early adopters in the research community, and in the years since its first release, it has grown into one of the most prominent deep learning tools across a broad range of applications.

As Python does for programming, PyTorch provides an excellent introduction to deep learning. At the same time, PyTorch has been proven to be fully qualified for use in professional contexts for real-world, high-profile work. We believe that PyTorch’s clear syntax, streamlined API, and easy debugging make it an excellent choice for introducing deep learning. We highly recommend studying PyTorch for your first deep learning library. Whether it ought to be the last deep learning library you learn is a decision we leave up to you.

At its core, the deep learning machine in figure 1.1 is a rather complex mathematical function mapping inputs to an output. To facilitate expressing this function, PyTorch provides a core data structure, the tensor, which is a multidimensional array that shares many similarities with NumPy arrays. Around that foundation, PyTorch comes with features to perform accelerated mathematical operations on dedicated hardware, which makes it convenient to design neural network architectures and train them on individual machines or parallel computing resources.

This book is intended as a starting point for software engineers, data scientists, and motivated students fluent in Python to become comfortable using PyTorch to build deep learning projects. We want this book to be as accessible and useful as possible, and we expect that you will be able to take the concepts in this book and apply them to other domains. To that end, we use a hands-on approach and encourage you to keep your computer at the ready, so you can play with the examples and take them a step further. By the time we are through with the book, we expect you to be able to take a data source and build out a deep learning project with it, supported by the excellent official documentation.

Although we stress the practical aspects of building deep learning systems with PyTorch, we believe that providing an accessible introduction to a foundational deep learning tool is more than just a way to facilitate the acquisition of new technical skills. It is a step toward equipping a new generation of scientists, engineers, and practitioners from a wide range of disciplines with working knowledge that will be the backbone of many software projects during the decades to come.

In order to get the most out of this book, you will need two things:

* Some experience programming in Python. We’re not going to pull any punches on that one; you’ll need to be up on Python data types, classes, floating-point numbers, and the like.
* A willingness to dive in and get your hands dirty. We’ll be starting from the basics and building up our working knowledge, and it will be much easier for you to learn if you follow along with us.

*Deep Learning with PyTorch* is organized in three distinct parts. Part 1 covers the foundations, examining in detail the facilities PyTorch offers to put the sketch of deep learning in figure 1.1 into action with code. Part 2 walks you through an end-to-end project involving medical imaging: finding and classifying tumors in CT scans, building on the basic concepts introduced in part 1, and adding more advanced topics. The short part 3 rounds off the book with a tour of what PyTorch offers for deploying deep learning models to production.

Deep learning is a huge space. In this book, we will be covering a tiny part of that space: specifically, using PyTorch for smaller-scope classification and segmentation projects, with image processing of 2D and 3D datasets used for most of the motivating examples. This book focuses on practical PyTorch, with the aim of covering enough ground to allow you to solve real-world machine learning problems, such as in vision, with deep learning or explore new models as they pop up in research literature. Most, if not all, of the latest publications related to deep learning research can be found in the arXiv public preprint repository, hosted at [https://arxiv.org](https://arxiv.org).

## 1.3 Why PyTorch?

As we’ve said, deep learning allows us to carry out a very wide range of complicated tasks, like machine translation, playing strategy games, or identifying objects in cluttered scenes, by exposing our model to illustrative examples. In order to do so in practice, we need tools that are flexible, so they can be adapted to such a wide range of problems, and efficient, to allow training to occur over large amounts of data in reasonable times; and we need the trained model to perform correctly in the presence of variability in the inputs. Let’s take a look at some of the reasons we decided to use PyTorch.

PyTorch is easy to recommend because of its simplicity. Many researchers and practitioners find it easy to learn, use, extend, and debug. It’s Pythonic, and while like any complicated domain it has caveats and best practices, using the library generally feels familiar to developers who have used Python previously.

More concretely, programming the deep learning machine is very natural in PyTorch. PyTorch gives us a data type, the Tensor, to hold numbers, vectors, matrices, or arrays in general. In addition, it provides functions for operating on them. We can program with them incrementally and, if we want, interactively, just like we are used to from Python. If you know NumPy, this will be very familiar.

But PyTorch offers two things that make it particularly relevant for deep learning: first, it provides accelerated computation using graphical processing units (GPUs), often yielding speedups in the range of 50x over doing the same calculation on a CPU. Second, PyTorch provides facilities that support numerical optimization on generic mathematical expressions, which deep learning uses for training. Note that both features are useful for scientific computing in general, not exclusively for deep learning. In fact, we can safely characterize PyTorch as a high-performance library with optimization support for scientific computing in Python.

A design driver for PyTorch is expressivity, allowing a developer to implement complicated models without undue complexity being imposed by the library (it’s not a framework!). PyTorch arguably offers one of the most seamless translations of ideas into Python code in the deep learning landscape. For this reason, PyTorch has seen widespread adoption in research, as witnessed by the high citation counts at international conferences.

PyTorch also has a compelling story for the transition from research and development into production. While it was initially focused on research workflows, PyTorch has been equipped with a high-performance C++ runtime that can be used to deploy models for inference without relying on Python, and can be used for designing and training models in C++. It has also grown bindings to other languages and an interface for deploying to mobile devices. These features allow us to take advantage of PyTorch’s flexibility and at the same time take our applications where a full Python runtime would be hard to get or would impose expensive overhead.

Of course, claims of ease of use and high performance are trivial to make. We hope that by the time you are in the thick of this book, you’ll agree with us that our claims here are well founded.

### 1.3.1 The deep learning competitive landscape

While all analogies are flawed, it seems that the release of PyTorch 0.1 in January 2017 marked the transition from a Cambrian-explosion-like proliferation of deep learning libraries, wrappers, and data-exchange formats into an era of consolidation and unification.

At the time of PyTorch’s first beta release:

* Theano and TensorFlow were the premiere low-level libraries, working with a model that had the user define a computational graph and then execute it.
* Lasagne and Keras were high-level wrappers around Theano, with Keras wrapping TensorFlow and CNTK as well.
* Caffe, Chainer, DyNet, Torch (the Lua-based precursor to PyTorch), MXNet, CNTK, DL4J, and others filled various niches in the ecosystem.

In the roughly two years that followed, the landscape changed drastically. The community largely consolidated behind either PyTorch or TensorFlow, with the adoption of other libraries dwindling, except for those filling specific niches. In a nutshell:

* Theano, one of the first deep learning frameworks, has ceased active development.
* TensorFlow:
* Consumed Keras entirely, promoting it to a first-class API
* Provided an immediate-execution “eager mode” that is somewhat similar to how PyTorch approaches computation
* Released TF 2.0 with eager mode by default


* JAX, a library by# Chapter 1: Introducing deep learning and the PyTorch Library

Artificial intelligence spans disciplines that are often subject to intense research, hype, and speculation. In reality, today's machines do not think in a human sense, but rather utilize a general class of algorithms capable of effectively approximating complex, nonlinear processes to automate tasks previously limited to humans. For example, language models like GPT-2 can generate coherent text sequentially word by word.

These capabilities are acquired through learning from examples rather than relying on handcrafted rules encoded by humans. Self-awareness is not required to carry out complex tasks, rendering philosophical questions about computer intelligence largely practical. Deep learning trains deep neural networks on vast amounts of data to approximate functions where inputs and outputs are conceptually distant, enabling novel computer capabilities.

## 1.1 The deep learning revolution

Traditional machine learning workflows relied heavily on manual feature engineering to transform input data into representations that facilitated downstream classification or prediction algorithms. For instance, digit recognition systems relied on custom edge-detection filters or counts of enclosed loops to distinguish numbers.

Deep learning automates feature discovery by extracting hierarchical representations directly from raw data during training. While domain knowledge remains valuable, automatically learned features frequently outperform handcrafted counterparts, shifting the practitioner's focus to optimizing mathematical models rather than engineering input features.

On the left side of figure 1.1, we see a practitioner busy defining engineering features and feeding them to a learning algorithm; the results on the task will be as good as the features the practitioner engineers. On the right, with deep learning, the raw data is fed to an algorithm that extracts hierarchical features automatically, guided by the optimization of its own performance on the task; the results will be as good as the ability of the practitioner to drive the algorithm toward its goal.

[[img-01]]
Figure 1.1 - Deep learning exchanges the need to handcraft features for an increase in data and computational requirements.

Starting from the right side in figure 1.1, we already get a glimpse of what we need to execute successful deep learning:

* We need a way to ingest whatever data we have at hand.
* We somehow need to define the deep learning machine.
* We must have an automated way, training, to obtain useful representations and make the machine produce desired outputs.

Model training uses a loss criterion to quantify the error between model predictions and target labels. Optimization iteratively adjusts model parameters to minimize this error, allowing the system to generalize well to unseen data.

## 1.2 PyTorch for deep learning

PyTorch is a flexible Python library designed for building deep learning models using idiomatic, approachable syntax. Its user-friendly interface led to rapid adoption in research, evolving into a widely used ecosystem across academia and industry.

PyTorch serves as both an accessible entry point for beginners and a robust framework qualified for production-grade, high-profile applications. Its clear syntax, streamlined API, and straightforward debugging tools make it an ideal choice for deep learning practitioners.

At its core, the deep learning machine in figure 1.1 is a rather complex mathematical function mapping inputs to an output. To express these mathematical operations, PyTorch relies on the tensor—a multidimensional array structure similar to NumPy arrays—backed by hardware acceleration for efficient computation on GPUs and parallel systems.

The book targets software engineers, data scientists, and students fluent in Python seeking practical experience in building PyTorch projects. It covers foundational topics, 2D and 3D medical image segmentation projects, and production deployment strategies.

To maximize learning outcomes, readers should possess prior Python programming experience and a willingness to engage directly with hands-on code examples.

The book is structured into three parts: foundational PyTorch concepts, an end-to-end medical imaging project, and deployment techniques for production environments. Deep learning encompasses vast domain areas, and this text focuses primarily on practical classification and segmentation tasks.

## 1.3 Why PyTorch?

Deep learning models handle complex, varied tasks by learning directly from data examples. Practical implementation requires tools that balance flexible model specification with high computational performance.

PyTorch is widely recommended due to its simplicity, Pythonic design, and seamless debugging experience. It feels familiar to Python developers while offering high performance for complex computational tasks.

The framework provides multidimensional tensors and comprehensive mathematical functions that execute interactively or incrementally. Crucially, PyTorch offers GPU acceleration—often yielding $50\times$ speedups over CPU execution—and an autograd engine for automatic differentiation.

PyTorch emphasizes expressiveness, enabling developers to write clean, unencumbered Python code without framework overhead. This clarity has driven widespread adoption and high citation rates across top research conferences.

Transitioning models from research to production is supported by a C++ runtime, mobile bindings, and cross-language support. These deployment features allow high-performance inference without requiring a full Python execution environment.

### 1.3.1 The deep learning competitive landscape

The release of PyTorch 0.1 in early 2017 marked a transition from a fragmented landscape of deep learning tools toward ecosystem consolidation.

At the time of PyTorch’s first beta release:

* Theano and TensorFlow were the premiere low-level libraries, working with a model that had the user define a computational graph and then execute it.
* Lasagne and Keras were high-level wrappers around Theano, with Keras wrapping TensorFlow and CNTK as well.
* Caffe, Chainer, DyNet, Torch (the Lua-based precursor to PyTorch), MXNet, CNTK, DL4J, and others filled various niches in the ecosystem.

In subsequent years, framework adoption consolidated predominantly around TensorFlow and PyTorch.

* Theano, one of the first deep learning frameworks, has ceased active development.
* TensorFlow:
* Consumed Keras entirely, promoting it to a first-class API
* Provided an immediate-execution “eager mode” that is somewhat similar to how PyTorch approaches computation
* Released TF 2.0 with eager mode by default


* JAX, a library by Google that was developed independently from TensorFlow, has started gaining traction as a NumPy equivalent with GPU, autograd and JIT capabilities.
* PyTorch:
* Consumed Caffe2 for its backend
* Replaced most of the low-level code reused from the Lua-based Torch project
* Added support for ONNX, a vendor-neutral model description and exchange format
* Added a delayed-execution “graph mode” runtime called TorchScript
* Released version 1.0
* Replaced CNTK and Chainer as the framework of choice by their respective corporate sponsors



TensorFlow maintains a strong footprint in enterprise production, whereas PyTorch dominates research and education while expanding its production capabilities through features like TorchScript.

## 1.4 An overview of how PyTorch supports deep learning projects

While PyTorch provides a Python interface, its core computational backends are written in C++ and CUDA for GPU acceleration. Models can be developed in Python and optionally executed via C++ for low-latency production environments.

The main Python interface centres on the `torch` module, providing the `Tensor` object along with mathematical routines supporting CPU and GPU execution. Automatic differentiation is handled natively by the underlying `autograd` engine.

Beyond general scientific computing, PyTorch provides specialized deep learning components for network building, data ingestion, and optimization. Figure 1.2 shows a standard setup that loads data, trains a model, and then deploys that model to production.

The `torch.nn` module offers predefined neural network layers, loss functions, and architectural building blocks. Training requires combining these network blocks with a data source, an optimizer from `torch.optim`, and hardware acceleration resources.

[[img-02]]
Figure 1.2 - Basic, high-level structure of a PyTorch project, with data loading, training, and deployment to production

At left in figure 1.2, we see that quite a bit of data processing is needed before the training data even reaches our model. Custom datasets are wrapped using `torch.utils.data.Dataset` to convert raw inputs into standardized PyTorch tensors.

Because disk I/O and data preprocessing can create training bottlenecks, `torch.utils.data.DataLoader` parallelizes data loading across background child processes to assemble mini-batches efficiently.

With the mechanism for getting batches of samples in place, we can turn to the training loop itself at the center of figure 1.2. The model evaluates incoming batches, computes loss values against target outputs, and updates weights using optimizers from `torch.optim`.

For large-scale models or datasets, training can be distributed across multiple GPUs or cluster nodes using `torch.nn.parallel.DistributedDataParallel` and the `torch.distributed` submodule, as depicted in the bottom center of figure 1.2.

The outcome of training is an optimized model, shown to the right of the training loop in figure 1.2. Deployment involves serving the model on cloud infrastructure, embedding it in applications, or executing it on mobile hardware.

While PyTorch defaults to immediate eager-mode execution, models can be compiled ahead of time using TorchScript. TorchScript serializes neural network models into Python-independent intermediate representations compatible with C++ runtimes or open standards like ONNX.

## 1.5 Hardware and software requirements

Basic model execution and fine-tuning pretrained networks can run on standard CPUs or modern laptops. However, full model training on large datasets generally requires CUDA-capable GPUs to achieve acceptable training times.

Using a GPU accelerates deep learning training loops by $40\times$ to $50\times$ compared to CPU execution. Larger production models may require multi-GPU setups or cloud-based computing platforms like Google Colaboratory.

PyTorch supports Linux, macOS, and Windows operating systems. Note that precompiled macOS binaries are CPU-only due to the lack of CUDA support on current Apple hardware.

Official installation instructions and package dependencies are available on the PyTorch website and managed via Pip or Conda. Part 2 example projects require at least 200 GB of available disk storage for dataset extraction and caching.

### 1.5.1 Using Jupyter Notebooks

Code examples throughout the book rely on interactive Jupyter Notebooks, which execute Python code within browser-based cells. Notebooks preserve execution state across cells, allowing seamless integration of live code, outputs, and documentation text.

To access the project code, start the Jupyter Notebook server from the cloned repository root directory. The server will launch a browser window displaying the local notebook files.

NOTE Jupyter Notebooks are a powerful tool for expressing and investigating ideas through code. While we think that they make for a good fit for our use case with this book, they’re not for everyone. We would argue that it’s important to focus on removing friction and minimizing cognitive overhead, and that’s going to be different for everyone. Use what you like during your experimentation with PyTorch.

Full working code for all listings from the book can be found at the book’s website ([www.manning.com/books/deep-learning-with-pytorch](https://www.google.com/search?q=https%3A%2F%2Fwww.manning.com%2Fbooks%2Fdeep-learning-with-pytorch)) and in our repository on GitHub ([https://github.com/deep-learning-with-pytorch/dlwpt-code](https://github.com/deep-learning-with-pytorch/dlwpt-code)).

## 1.6 Exercises

1. Start Python to get an interactive prompt.
a. What Python version are you using? We hope it is at least 3.6!
b. Can you import `torch`? What version of PyTorch do you get?
c. What is the result of `torch.cuda.is_available()`? Does it match your expectation based on the hardware you’re using?
2. Start the Jupyter notebook server.
a. What version of Python is Jupyter using?
b. Is the location of the `torch` library used by Jupyter the same as the one you imported from the interactive prompt?

## 1.7 Summary

* Deep learning models automatically learn to associate inputs and desired outputs from examples.
* Libraries like PyTorch allow you to build and train neural network models efficiently.
* PyTorch minimizes cognitive overhead while focusing on flexibility and speed. It also defaults to immediate execution for operations.
* TorchScript allows us to precompile models and invoke them not only from Python but also from C++ programs and on mobile devices.
* Since the release of PyTorch in early 2017, the deep learning tooling ecosystem has consolidated significantly.
* PyTorch provides a number of utility libraries to facilitate deep learning projects.