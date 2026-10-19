# Chapter 15: Deploying to production

This chapter covers options for deploying PyTorch models, working with the PyTorch JIT, deploying a model server, exporting models, running exported and natively implemented models from C++, and running models on mobile. Part 1 provided foundational knowledge about models, and part 2 outlined a detailed approach to model creation for specific problems. To make models useful, they must be deployed to production environments where running large-scale inference efficiently impacts both system architecture and operational costs. Beginning with release 1.0, PyTorch introduced production-focused capabilities that transformed it from a research tool into an end-to-end framework.

Depending on the use case, deploying to production takes several forms, such as building network services using lightweight Python frameworks like Flask or Sanic. Models can also be exported to standard formats like Open Neural Network Exchange (ONNX) to run on specialized hardware, cloud platforms, or optimized processors. Integrating models into larger non-Python software environments is enabled through C++ execution, while mobile support allows running applications directly on edge devices like smartphones. Demonstrations use the nodule classifier from chapter 14 for initial serving, before switching to the CycleGAN horse-to-zebra model for the remaining deployment approaches.

## 15.1 Serving PyTorch models

Setting up network access for a model begins with the simplest possible server implementation before addressing its limitations.

### 15.1.1 Our model behind a Flask server

Flask is a popular Python web framework that can be installed via package managers.

```python
from flask import Flask
app = Flask(__name__)

@app.route("/hello")
def hello():
    return "Hello World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

```

The basic application listens on port 8000 and exposes a route returning text. It can be extended to handle model predictions by loading pre-trained weights and exposing a POST route. In this setup, a `/predict` endpoint receives input files containing binary pixel data and JSON metadata, and returns a JSON response containing the malignancy probability for a single sample. Data decoding requires converting JSON metadata, transforming binary buffers into NumPy arrays, and creating PyTorch tensors. Model inference runs in evaluation mode inside a `torch.no_grad()` block to disable gradient tracking.

```python
import numpy as np
import sys
import os
import torch
from flask import Flask, request, jsonify
import json
from p2ch13.model_cls import LunaModel

app = Flask(__name__)
model = LunaModel()
model.load_state_dict(torch.load(sys.argv[1], map_location='cpu')['model_state'])
model.eval()

def run_inference(in_tensor):
    with torch.no_grad():
        out_tensor = model(in_tensor.unsqueeze(0))[1].squeeze(0)
        probs = out_tensor.tolist()
        out = {'prob_malignant': probs[1]}
        return out

@app.route("/predict", methods=["POST"])
def predict():
    meta = json.load(request.files['meta'])
    blob = request.files['blob'].read()
    in_tensor = torch.from_numpy(np.frombuffer(blob, dtype=np.float32))
    in_tensor = in_tensor.view(*meta['shape'])
    out = run_inference(in_tensor)
    return jsonify(out)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

```

Running the server script with the saved model checkpoint allows a client script to send requests and obtain inference outputs. While functional, simple Flask serving leaves considerable room for improvement regarding scalability and protocol efficiency.

### 15.1.2 What we want from deployment

High-performance deployment requires modern asynchronous protocols, request batching for GPU efficiency, multi-threading capability that bypasses the Python Global Interpreter Lock (GIL), minimized data copying, and safe decoding mechanisms. Batching multiple client requests together significantly improves GPU throughput compared to processing individual requests sequentially or concurrently.

### 15.1.3 Request batching

Using Sanic enables asynchronous processing and request batching across parallel connections.

[[img-01]]
Figure 15.1 - Dataflow with request batching

Decoupling request handling from model execution allows incoming requests to enqueue work items, which a separate model runner worker gathers into batches based on queue size or timeout thresholds.

[[img-02]]
Figure 15.2 - Our asynchronous server consists of three blocks: request processor, model runner, and model execution. These blocks are a bit like functions, but the first two will yield to the event loop in between.

The request processing pipeline decodes inputs, places tasks onto a queue, waits asynchronously, and encodes outputs, while the model runner loop periodically extracts batches, executes inference in a separate thread, and signals completion.

```python
class ModelRunner:
    def __init__(self, model_name):
        self.model_name = model_name
        self.queue = []
        self.queue_lock = None
        self.model = get_pretrained_model(self.model_name, map_location=device)
        self.needs_processing = None
        self.needs_processing_timer = None

```

```python
async def process_input(self, input):
    our_task = {"done_event": asyncio.Event(loop=app.loop),
                "input": input,
                "time": app.loop.time()}
    async with self.queue_lock:
        if len(self.queue) >= MAX_QUEUE_SIZE:
            raise HandlingError("I'm too busy", code=503)
        self.queue.append(our_task)
        self.schedule_processing_if_needed()
    await our_task["done_event"].wait()
    return our_task["output"]

```

```python
async def model_runner(self):
    self.queue_lock = asyncio.Lock(loop=app.loop)
    self.needs_processing = asyncio.Event(loop=app.loop)
    while True:
        await self.needs_processing.wait()
        self.needs_processing.clear()
        if self.needs_processing_timer is not None:
            self.needs_processing_timer.cancel()
            self.needs_processing_timer = None
        async with self.queue_lock:
            to_process = self.queue[:MAX_BATCH_SIZE]
            del self.queue[:len(to_process)]
            self.schedule_processing_if_needed()
            batch = torch.stack([t["input"] for t in to_process], dim=0)
            result = await app.loop.run_in_executor(
                None, functools.partial(self.run_model, batch)
            )
            for t, r in zip(to_process, result):
                t["output"] = r
                t["done_event"].set()
            del to_process

```

Running Python model inference in a background thread remains constrained by the GIL unless compiled formats or JIT compilation are used.

## 15.2 Exporting models

Exporting models allows bypassing Python execution overhead, eliminating GIL bottlenecks, and running inference on non-Python environments or embedded platforms.

### 15.2.1 Interoperability beyond PyTorch with ONNX

Open Neural Network Exchange (ONNX) defines a standard format for representing neural network operations, enabling execution across diverse frameworks, specialized hardware, and edge runtimes. Models are exported by passing dummy input tensors to `torch.onnx.export`, which traces the execution graph.

```python
import onnxruntime
sess = onnxruntime.InferenceSession("seg_model.onnx")
input_name = sess.get_inputs()[0].name
pred_onnx = sess.run(None, {input_name: batch})

```

Unsupported native operations in ONNX standards cause export errors.

### 15.2.2 PyTorch's own export: Tracing

PyTorch provides its own intermediate format, TorchScript, which can be generated using `torch.jit.trace` by supplying example input tensors. All model parameters must have `requires_grad` set to `False` prior to tracing to ensure gradient computation is disabled during serialized execution.

```python
import torch
from p2ch13.model_seg import UNetWrapper

seg_dict = torch.load('data-unversioned/part2/models/p2ch13/seg_2019-10-20_15.57.21_none.best.state', map_location='cpu')
seg_model = UNetWrapper(in_channels=8, n_classes=1, depth=4, wf=3, padding=True, batch_norm=True, up_mode='upconv')
seg_model.load_state_dict(seg_dict['model_state'])
seg_model.eval()

for p in seg_model.parameters():
    p.requires_grad_(False)

dummy_input = torch.randn(1, 8, 512, 512)
traced_seg_model = torch.jit.trace(seg_model, dummy_input)

```

Traced models are saved using `torch.jit.save` and loaded independently using `torch.jit.load`. Source code files should always be preserved alongside exported models to facilitate future modifications.

### 15.2.3 Our server with a traced model

Replacing standard model loading with `torch.jit.load` in the asynchronous server enables multithreaded model inference released from Python GIL lock contention.

## 15.3 Interacting with the PyTorch JIT

The PyTorch JIT provides compiler optimizations and standalone execution capabilities for PyTorch models.

### 15.3.1 What to expect from moving beyond classic Python/PyTorch

While removing Python overhead yields modest speedups, eliminating GIL locks enables effective multithreaded concurrency. Additionally, JIT compilation allows global computation graph optimizations, such as kernel fusion.

[[img-03]]
Figure 15.3 - LSTM cell pointwise operations. From five inputs at the top, this block computes two outputs at the bottom. The boxes in between are intermediate results that vanilla PyTorch will store in memory but the JIT fuser will just keep in registers.

Fusing operations reduces GPU memory reads and writes, producing substantial speed Improvements during execution.

### 15.3.2 The dual nature of PyTorch as interface and backend

PyTorch separates user-level interfaces from underlying backend C++ libraries like ATen and LibTorch.

[[img-04]]
Figure 15.4 - Many ways of calling into PyTorch

TorchScript IR serves as a symbolic bridge connecting Python and C++ execution environments.

### 15.3.3 TorchScript

TorchScript graphs are created via tracing (recording executed operations on sample inputs) or scripting (compiling Python code directly into TorchScript IR). Unlike dynamic Python, TorchScript uses static typing.

```python
def myfn(x):
    y = x[0]
    for i in range(1, x.size(0)):
        y = y + x[i]
    return y

```

Tracing unrolls control flow for fixed tensor sizes, whereas scripting preserves dynamic loops.

```python
scripted_fn = torch.jit.script(myfn)
print(scripted_fn.code)

```

The `@torch.jit.script` decorator or `@torch.jit.export` annotation exports modules and specific functions.

### 15.3.4 Scripting the gaps of traceability

Dynamic control flow or indexing operations can be selectively scripted using `@torch.jit.script` to resolve tracer warnings.

```python
class UNetUpBlock(nn.Module):
    def center_crop(self, layer, target_size):
        layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        crop1 = self.center_crop(bridge, up.shape[2:])

```

```python
@torch.jit.script
def center_crop(layer, target):
    layer_height, layer_width = layer.size()
    target_height, target_width = target.size()
    diff_y = (layer_height - target_height) // 2
    diff_x = (layer_width - target_width) // 2
    return layer[:, :, diff_y: (diff_y + target_height), diff_x: (diff_x + target_width)]

class UNetUpBlock(nn.Module):
    def forward(self, x, bridge):
        crop1 = center_crop(bridge, up)

```

Combining tracing with scripted sub-components provides flexibility across complex architectures.

## 15.4 LibTorch: PyTorch in C++

LibTorch provides native C++ libraries for loading, executing, and defining PyTorch models.

### 15.4.1 Running JITed models from C++

Exported TorchScript models are loaded in C++ using `torch::jit::load`.

```cpp
#include "torch/script.h"
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;

int main(int argc, char **argv) {
    CImg<float> image(argv[2]);
    image.resize(227, 227);
    CImg<float> out_img(output.data_ptr<float>(), output.size(2), output.size(3), 1, output.size(1));
    out_img.save(argv[3]);
    return 0;
}

```

```cpp
auto input_ = torch::tensor(torch::ArrayRef<float>(image.data(), image.size()));
auto input = input_.reshape({1, 3, image.height(), image.width()}).div_(255);
auto module = torch::jit::load(argv[1]);
std::vector<torch::jit::IValue> inputs;
inputs.push_back(input);
auto output = module.forward(inputs).toTensor();
output.contiguous().mul_(255);

```

Input data must be wrapped inside `torch::jit::IValue` instances before invoking `forward`.

```cmake
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cyclegan-jit)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(cyclegan-jit cyclegan_jit.cpp)
target_link_libraries(cyclegan-jit pthread jpeg X11)
target_link_libraries(cyclegan-jit "${TORCH_LIBRARIES}")
set_property(TARGET cyclegan-jit PROPERTY CXX_STANDARD 14)

```

CMake configures the build setup and links necessary LibTorch dependencies.

### 15.4.2 C++ from the start: The C++ API

Models can also be built natively in C++ using the `torch::nn` modular API.

```cpp
#include <torch/torch.h>
#define cimg_use_jpeg
#include <CImg.h>
using torch::Tensor;

```

```cpp
struct ResNetBlock : torch::nn::Module {
    torch::nn::Sequential conv_block;
    ResNetBlock(int64_t dim)
        : conv_block(
            torch::nn::ReflectionPad2d(1),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 3)),
            torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(dim)),
            torch::nn::ReLU(true),
            torch::nn::ReflectionPad2d(1),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 3)),
            torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(dim))) {
        register_module("conv_block", conv_block);
    }
    Tensor forward(const Tensor &inp) {
        return inp + conv_block->forward(inp);
    }
};

```

Submodules must be explicitly initialized and registered via `register_module`. Options objects manage keyword configurations.

```cpp
struct ResNetGeneratorImpl : torch::nn::Module {
    torch::nn::Sequential model;
    ResNetGeneratorImpl(int64_t input_nc = 3, int64_t output_nc = 3, int64_t ngf = 64, int64_t n_blocks = 9) {
        TORCH_CHECK(n_blocks >= 0);
        model->push_back(torch::nn::ReflectionPad2d(3));
        model->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf_mult, ngf_mult * 2, 3).stride(2).padding(1)));
        register_module("model", model);
    }
    Tensor forward(const Tensor &inp) { return model->forward(inp); }
};
TORCH_MODULE(ResNetGenerator);

```

The `TORCH_MODULE` macro defines reference-counted wrapper classes for C++ module structures.

```cpp
ResNetGenerator model;
torch::load(model, argv[1]);
cimg_library::CImg<float> image(argv[2]);
image.resize(400, 400);
auto input_ = torch::tensor(torch::ArrayRef<float>(image.data(), image.size()));
auto input = input_.reshape({1, 3, image.height(), image.width()});
torch::NoGradGuard no_grad;
model->eval();
auto output = model->forward(input);
cimg_library::CImg<float> out_img(output.data_ptr<float>(), output.size(3), output.size(2), 1, output.size(1));
cimg_library::CImgDisplay disp(out_img, "See a C++ API zebra!");
while (!disp.is_closed()) {
    disp.wait();
}

```

Running inference uses `torch::NoGradGuard` to disable gradients during evaluation.

## 15.5 Going mobile

PyTorch Mobile brings TorchScript execution to Android and iOS runtimes.

```groovy
dependencies {
    implementation 'org.pytorch:pytorch_android:1.4.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.4.0'
}

```

Dependencies are configured inside Gradle configuration files.

```java
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class MainActivity extends AppCompatActivity {
    private org.pytorch.Module model;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        try {
            model = Module.load(assetFilePath(this, "traced_zebra_model.pt"));
        } catch (IOException e) {
            Log.e("Zebraify", "Error reading assets", e);
            finish();
        }
    }
}

```

Asset files are copied to local paths before being loaded by `Module.load`.

```java
@Override
protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
        Bitmap bitmap = (Bitmap) data.getExtras().get("data");
        final float[] means = new float[]{0.0f, 0.0f, 0.0f};
        final float[] stds = new float[]{1.0f, 1.0f, 1.0f};
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, means, stds);
        final Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
        Bitmap output_bitmap = tensorToBitmap(outputTensor, means, stds, Bitmap.Config.RGB_565);
        image_view.setImageBitmap(output_bitmap);
    }
}

```

`TensorImageUtils` converts Android Bitmaps to input Tensors, which are evaluated by the loaded mobile model.

[[img-05]]
Figure 15.5 - Our CycleGAN zebra app

### 15.5.1 Improving efficiency: Model design and quantization

Model size and compute requirements can be reduced using distillation or quantization. Quantization converts weights and activations from 32-bit floating-point numbers to 8-bit integers (`torch.quint8` or `torch.qint8`).

## 15.6 Emerging technology: Enterprise serving of PyTorch models

Production ecosystems continue evolving with dedicated serving tools such as RedisAI, TorchServe, MLflow, Cortex, and EuclidesDB.

## 15.7 Conclusion

PyTorch models can be deployed across network service architectures, C++ systems, and mobile devices using TorchScript and exported formats.

## 15.8 Exercises

1. Pick a project that sounds exciting to you. Kaggle is a great place to start looking. Dive in.

## 15.9 Summary

* PyTorch models can be served via Python web frameworks like Flask.
* JITed models bypass the GIL during concurrent Python serving.
* Asynchronous handling and request batching maximize hardware efficiency.
* ONNX allows exporting PyTorch models to broad cross-platform runtimes.
* The JIT executes arbitrary PyTorch code within C++ or mobile environments.
* Tracing exports static control flow, whereas scripting supports dynamic structures.
* C++ natively supports model creation, loading, and execution.
* PyTorch Mobile enables direct integration on Android and iOS devices.
* Model distillation and 8-bit quantization improve mobile deployment efficiency.
* Enterprise model serving ecosystems continue to evolve rapidly.