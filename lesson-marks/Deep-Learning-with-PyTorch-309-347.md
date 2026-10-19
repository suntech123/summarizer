# Chapter 11: Training a classification model to detect suspected tumors

This chapter covers using PyTorch to train a 3D convolutional neural network, constructing a training and validation loop using PyTorch modules, displaying performance metrics during training, and using TensorBoard to plot metrics and monitor training. Our primary goal is to take candidate locations identified in the previous chapter and classify whether each candidate represents an actual nodule or a non-nodule. Building a effective training script requires combining dataset handling, network architecture design, loss calculation, optimization, and metric logging into a structured application.

## 11.1 A foundational model and training loop

Before making complex optimizations, establishing a clean foundational model and training loop provides a baseline for future improvements.

[[img-01]]
Figure 11.1 - Overview of our project's application structure for training the classification model

Structuring our project around a main application class keeps command-line parsing, data loading, model initialization, and epoch execution cleanly organized.

## 11.2 Application structure and initialization

The main entry point for our training pipeline is implemented via the `LunaTrainingApp` class, which handles initialization, CLI configuration, and main loop execution.

[[img-02]]
Figure 11.2 - High-level execution path for `LunaTrainingApp`

Command-line parameters specify hyperparameters such as batch size, worker process count, number of epochs, and GPU execution settings.

```python
import argparse
import sys

from util.util import str2bool

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        parser.add_argument('--balanced',
            help="Balance the training data or not.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augmented',
            help="Augment the training data or not.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the images.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the images.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly scaling the images.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the images.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the images.",
            action='store_true',
            default=False,
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

```

### 11.2.1 Pretraining setup and initialization

During setup, we detect hardware availability, load training and validation datasets, initialize model weights, and set up the optimizer.

[[img-03]]
Figure 11.3 - Pre-training setup and initialization step details

If CUDA-compatible GPUs are present, model parameters and tensors are transferred to GPU memory for accelerated training.

```python
    def initModel(self):
        self.model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.9)

```

Using SGD is generally considered a safe place to start when training convolutional networks on image datasets.

### 11.2.2 Data loaders and batch creation

PyTorch `DataLoader` instances manage multi-threaded background data fetching, batching, and memory pinning.

[[img-04]]
Figure 11.4 - Dataset loading, batching, and worker process interaction

Using `pin_memory=True` speeds up host-to-GPU tensor transfers during batch processing.

```python
    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
            ratio_int=int(self.cli_args.balanced),
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

```

Validation data loaders use separate split parameters to ensure complete independence from training data.

## 11.3 Our first-pass neural network design

Our initial network architecture adapts 2D convolutional designs to 3D volumetric CT patches using `nn.Conv3d` and `nn.MaxPool3d` operations.

[[img-05]]
Figure 11.5 - Neural network architecture components: backbone, head, and output layers

The backbone extracts 3D spatial feature representations, which are flattened and processed by linear layers in the network head.

### 11.3.1 Module blocks and sub-networks

To keep the model clean and modular, we build a repeated `LunaBlock` module consisting of 3D convolutions, batch normalization, ReLU activations, and max pooling.

```python
import torch
import torch.nn as nn

class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=False,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=False,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_g):
        out = self.conv1(input_g)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool(out)
        return out

```

Batch normalization stabilizes intermediate activations across layers during training passes.

### 11.3.2 The full model

Combining four sequential `LunaBlock` sub-modules forms the feature extraction backbone of our complete 3D classifier network.

[[img-06]]
Figure 11.6 - The complete `LunaModel` 3D CNN architecture

The final linear layer maps compressed feature representations to two unnormalized output logits corresponding to negative (non-nodule) and positive (nodule) predictions.

```python
class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu',
                )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_g):
        bn_g = self.tail_batchnorm(input_g)
        out_g = self.block1(bn_g)
        out_g = self.block2(out_g)
        out_g = self.block3(out_g)
        out_g = self.block4(out_g)

        conv_flat = out_g.view(out_g.size(0), -1)
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)

```

The forward pass returns both raw logits (for loss computation) and softmax probabilities (for downstream inference metrics).

## 11.4 Training and validating the model

We calculate loss using `nn.CrossEntropyLoss`, which applies log-softmax internally for numerical stability.

```python
>>> import torch
>>> logits = torch.tensor([[2.718, 0.135], [0.135, 20.086]])
>>> probabilities = torch.softmax(logits, dim=1)
>>> probabilities
tensor([[0.9298, 0.0702],
        [0.0000, 1.0000]])

```

We'll use the logits when we calculate the `nn.CrossEntropyLoss`.

[[img-07]]
Figure 11.7 - Execution sequence of the training loop within `LunaTrainingApp`

The main training loop iterates through datasets epoch by epoch, updating weights during training and evaluating model performance during validation.

```python
    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

```

`doTraining` processes training mini-batches, computes loss, invokes backpropagation, and updates network weights.

```python
    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        for batch_ndx, batch_tup in enumerate(train_dl):
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
            )

            loss_var.backward()
            self.optimizer.step()

        return trnMetrics_g.to('cpu')

```

Batch losses are computed by passing candidate inputs through the model and comparing outputs against target binary labels.

```python
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_g = self.loss_fn(
            logits_g,
            label_g[:, 1],
        )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        return loss_g.mean()

```

By recording the label, prediction, and loss for each and every sample in our dataset, we build up a detailed record of how training progresses.

```python
    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            for batch_ndx, batch_tup in enumerate(val_dl):
                self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                )

        return valMetrics_g.to('cpu')

```

Calling `torch.no_grad()` during validation disables autograd history tracking, reducing memory consumption and speeding up inference execution.

## 11.5 Outputting performance metrics

Logging epoch statistics provides immediate feedback on model training trajectory and convergence.

[[img-08]]
Figure 11.8 - Performance logging step within the overall application workflow

Metrics logged after each epoch include overall cross-entropy loss, classification accuracy, precision, and recall.

```python
    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):
        neg_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        pos_mask = ~neg_mask

        neg_count = neg_mask.sum().item()
        pos_count = pos_mask.sum().item()

        neg_correct = (
            metrics_t[METRICS_PRED_NDX] <= classificationThreshold
        ) & neg_mask
        pos_correct = (
            metrics_t[METRICS_PRED_NDX] > classificationThreshold
        ) & pos_mask

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean().item()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX][neg_mask].mean().item()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX][pos_mask].mean().item()

        metrics_dict['correct/all'] = (
            (neg_correct.sum() + pos_correct.sum()).item()
            / metrics_t.shape[1] * 100
        )
        metrics_dict['correct/neg'] = (
            neg_correct.sum().item() / neg_count * 100
        )
        metrics_dict['correct/pos'] = (
            pos_correct.sum().item() / pos_count * 100
        )

```

Next, we use those masks to compute some per-label statistics and log summary strings to standard output.

```python
        log.info(
            ("E{} {:5} " +
             "{loss/all:.4f} loss, " +
             "{correct/all:-5.1f}% correct, " +
             "{correct/neg:-5.1f}% correct neg, " +
             "{correct/pos:-5.1f}% correct pos"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict
            )
        )

```

Detailed per-class correct output percentages highlight whether predictions are balanced across both classes.

## 11.6 Running the training script

If the first epoch seems to be taking a very long time (more than 10 or 15 minutes), check whether GPU acceleration is correctly active.

```bash
$ ls -lp data-unversioned/part2/luna/

```

We execute training by passing options directly to our script from the command line interface.

```python
>>> for i, _ in enumerateWithEstimate(list(range(234)), 'sleeping'): time.sleep(0.1)

```

Script utilities print progress updates and estimated completion times during data loading and training passes.

## 11.7 Evaluating the model: 99.7% correct means we're done, right?

Running our baseline model yields an overall validation accuracy of 99.7%, which initially appears like outstanding performance. However, because actual nodules comprise only ~0.2% of candidate samples in our raw dataset, a naive model predicting every candidate as non-nodule achieves 99.8% accuracy. Looking at per-class metrics reveals that our model predicts negative for every candidate, failing to detect a single actual nodule.

## 11.8 Graphing training metrics with TensorBoard

Tracking training progress using text logs becomes unwieldy across dozens of epochs. TensorBoard provides interactive visual charts for scalar values, image maps, and model graphs.

```bash
$ tensorboard --logdir runs

```

Launching the TensorBoard web service opens a browser dashboard displaying live updating metric plots.

[[img-09]]
Figure 11.9 - TensorBoard scalar dashboard showing loss curves across training epochs

Comparing training curves against validation curves helps spot subtle signs of overfitting early.

[[img-10]]
Figure 11.10 - Detailed TensorBoard scalar plots displaying training and validation metrics side by side

The numbers you get by mousing over the lines with the cursor show exact scalar values for any specific epoch.

[[img-11]]
Figure 11.11 - Comparative TensorBoard runs displaying different hyperparameter configurations

TensorBoard allows overlaying multiple training runs to compare hyperparameter choices directly.

## 11.9 Why isn't the model learning to detect nodules?

Our model fails to learn nodule features because class imbalance severely skews gradient updates. Because 99.8% of the samples are negative, the overall loss is heavily dominated by background candidates. Minimizing total loss causes the optimizer to quickly converge on predicting negative for all inputs. Solving this requires rebalancing our dataset and re-weighting class loss contributions during training.

## 11.10 Summary

* We will use PyTorch's `torch.optim.SGD` (stochastic gradient descent) optimizer with momentum to train our 3D CNN classifier.
* Constructing a clean application wrapper class encapsulates data loading, model forward passes, backpropagation, and metric logging.
* Overall accuracy is misleading on heavily imbalanced datasets; tracking per-class metrics is essential.
* TensorBoard provides real-time visualization of scalar loss and accuracy metrics during training.
* Severe class imbalance forces unweighted classifiers to learn trivial predictions that ignore rare positive samples.