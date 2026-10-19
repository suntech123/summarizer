# Chapter 13: Using segmentation to find suspected nodules

This chapter covers segmenting data with a pixel-to-pixel U-Net model, cropping input data into manageable sizes, designing a custom loss function for segmentation, using image output to debug model training, and assessing segmentation model performance. As explained in chapter 9, our project's goal is to detect lung cancer from CT scans. The second main step in our overall design is candidate generation. We will accomplish this using a segmentation model, which operates similarly to the classification model developed in part 2.

To build our candidate generation system, we need to adapt our approach across five major areas:

1. Change the model (step 2A). We will switch from a classification architecture to a segmentation network (specifically U-Net).
2. Change the dataset (step 2B). We need to load continuous slices or 3D sub-volumes instead of individual candidate patches.
3. Update the training script. We must adapt the loss function and metrics calculation to handle per-pixel outputs.
4. Convert segmentation outputs to candidates (step 2C). We will extract connected components from the output mask to form discrete nodule candidates.
5. Feed candidates into the classifier (step 2D). The output candidates will be passed directly into our classifier from chapter 12.

## 13.1 Semantic segmentation: Per-pixel classification

Semantic segmentation assigns a class label to every individual pixel in an input image. Unlike standard classification, which outputs a single label for an entire image, segmentation preserves spatial structure by producing an output tensor of identical spatial dimensions to the input.

NOTE As we go through the code examples in this chapter, keep in mind that we're working with 2D image slices rather than 3D volumes.

[[img-01]]
Figure 13.1 - Comparison of image classification, object detection, and semantic segmentation

Standard classification networks downsample inputs through successive pooling layers, discarding precise spatial location information. For segmentation, we need a network that can upsample feature maps back to the original image resolution while retaining fine-grained spatial details.

### 13.1.1 Understanding U-Net architecture

Convolutional operations reduce spatial dimensions while increasing channel depth. To produce dense pixel predictions, architectures must expand feature maps back to input dimensions.

[[img-02]]
Figure 13.2 - Spatial dimensions decrease while channel dimensions increase through successive convolutional layers.

Upsampling can be performed using simple interpolation or learnable transposed convolutions. Transposed convolutions reverse the spatial reduction of standard convolutions by padding feature maps and applying kernel filters.

[[img-03]]
Figure 13.3 - Transposed convolution upsampling a $2 \times 2$ input into a $4 \times 4$ output feature map

The U-Net architecture combines a contracting encoder path with an expanding decoder path. Skip connections directly pass high-resolution feature maps from encoder blocks to corresponding decoder blocks, preserving fine spatial details lost during downsampling.

[[img-04]]
Figure 13.4 - The U-Net architecture, showing contracting encoder blocks, expanding decoder blocks, and horizontal skip connections

Skip connections concatenation enables the decoder to utilize both high-level semantic context and precise low-level spatial boundaries.

## 13.2 Updating the model for segmentation

We construct a 2D U-Net implementation consisting of contracting blocks, expanding blocks, and a final $1 \times 1$ convolution layer to output per-pixel class logits.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, depth=5, wf=6, padding=True, batch_norm=True, up_mode='upconv'):
        super().__init__()
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in range(depth - 1):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + depth - 2 - i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + depth - 2 - i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != self.depth - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-1 - i])

        return self.last(x)

```

Each convolutional block contains two consecutive $3 \times 3$ convolutions, optional batch normalization, and ReLU activations.

```python
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super().__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

```

The upsampling block handles spatial expansion and concatenates incoming skip-connection tensors before applying convolutions.

```python
class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        return self.conv_block(out)

```

Batch normalization accelerates convergence and helps stabilize training across varying input intensity distributions.

## 13.3 Updating the dataset for segmentation

Segmentation requires datasets to output 2D CT image slices alongside corresponding per-pixel ground truth binary masks.

### 13.3.1 Slice-based training data extraction

Full 3D CT volumes are sliced into 2D image planes. Instead of feeding entire $512 \times 512$ slices, random square patches are cropped around target areas to reduce GPU memory footprint during training.

[[img-05]]
Figure 13.5 - Extracting 2D slices and sub-volume crops from 3D CT scans

Training datasets balance cropped patches containing positive nodule pixels against background tissue crops.

### 13.3.2 LIDC annotation processing

The LIDC dataset provides detailed polygonal nodule boundaries drawn by multiple expert radiologists.

[[img-06]]
Figure 13.6 - Overlapping annotations from multiple radiologists for a single nodule candidate

Individual annotations are combined into a consolidated 3D binary volume mask using a majority-voting consensus.

```python
class SegmentationDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None, contextSlices_count=3, fullSlice_bool=False):
        self.contextSlices_count = contextSlices_count
        self.fullSlice_bool = fullSlice_bool
        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCtRawCandidateFilter().keys())

        if val_stride > 0:
            if isValSet_bool:
                self.series_list = self.series_list[::val_stride]
            else:
                del self.series_list[::val_stride]

        self.sample_list = []
        for series_uid in self.series_list:
            index_rec = getCtRawCandidateFilter()[series_uid]
            for slice_ndx in range(index_rec.slice_count):
                self.sample_list.append((series_uid, slice_ndx))

        self.candidateInfo_list = getCandidateInfoList()

```

Each item returned by the dataset provides multi-slice contextual inputs paired with single-slice target ground truth masks.

[[img-07]]
Figure 13.7 - Slicing CT scans into 2D slices across spatial dimensions

Contextual adjacent slices are stacked along input channels to provide pseudo-3D information to the 2D network.

[[img-08]]
Figure 13.8 - Stacking adjacent 2D CT slices along input channels

The final target label represents a single $0/1$ mask for the central slice in the context window.

```python
    def __getitem__(self, ndx):
        sample_item = self.sample_list[ndx]
        if isinstance(sample_item, tuple):
            series_uid, slice_ndx = sample_item
            ct = getCt(series_uid)
            patient_a, crop_g = self.get_full_slice(ct, slice_ndx)
        else:
            candidateInfo_tup = sample_item
            patient_a, crop_g = self.get_candidate_crop(candidateInfo_tup)

        return patient_a, crop_g, candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz

```

Caching extracted 2D slice arrays to disk dramatically reduces repeat load times during training epochs.

[[img-09]]
Figure 13.9 - In-memory and disk caching pipeline for CT slice samples

The dataset handles random cropping and data augmentation transforms during online training execution.

```python
    def get_candidate_crop(self, candidateInfo_tup):
        ct = getCt(candidateInfo_tup.series_uid)
        slice_ndx = int(ct.change_coords_xyz_to_index(candidateInfo_tup.center_xyz)[2])
        crop_a = np.zeros((self.contextSlices_count * 2 + 1, 64, 64), dtype=np.float32)

        for i, context_ndx in enumerate(range(slice_ndx - self.contextSlices_count, slice_ndx + self.contextSlices_count + 1)):
            context_ndx = max(0, min(context_ndx, ct.hu_a.shape[0] - 1))
            crop_a[i] = ct.get_crop(candidateInfo_tup.center_xyz, (64, 64), context_ndx)

        mask_a = ct.build_mask(candidateInfo_tup.center_xyz, (64, 64), slice_ndx)
        return crop_a, mask_a

```

Proper sampling ensures that empty background slices do not dominate model training updates.

## 13.4 Updating the training script for segmentation

Training segmentation models requires custom loss functions, training loops, and validation metrics tailored to pixel-wise predictions.

[[img-10]]
Figure 13.10 - Complete system diagram highlighting the segmentation model training loop

We build a dedicated training script handling data loading, optimization, logging, and evaluation steps.

### 13.4.1 Sørensen-Dice coefficient and Dice loss

Standard cross-entropy loss performs poorly on heavily imbalanced segmentation tasks where target positive pixels comprise a tiny fraction of total image area. The Sørensen-Dice coefficient measures spatial overlap between predicted mask $P$ and ground truth mask $G$:

$$\text{Dice} = \frac{2 \vert{}P \cap G\vert{}}{\vert{}P\vert{} + \vert{}G\vert{}}$$

We convert the Dice coefficient into a differentiable loss function for training:

$$\text{DiceLoss} = 1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}$$

Where $p_i$ is the predicted probability for pixel $i$, $g_i$ is the ground truth binary label, and $\epsilon$ is a smooth factor preventing division by zero.

```python
class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        intersection = (prediction * target).sum(dim=(1, 2, 3))
        union = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        return 1.0 - dice.mean()

```

Combining Dice loss with standard Binary Cross Entropy (BCE) loss yields stable gradient updates during early epochs while optimizing for spatial overlap.

### 13.4.2 Training loop implementation

The main training application initializes the dataset, model, optimizer, and loss functions.

```python
class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        self.args = parse_args(sys_argv)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = UNet(in_channels=7, n_classes=1, depth=4, wf=3).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

```

The epoch loop alternates between training steps over cropped patches and validation steps over full CT slices.

```python
    def do_training(self, epoch_ndx, train_dl):
        self.model.train()
        for batch_ndx, batch_tup in enumerate(train_dl):
            self.optimizer.zero_grad()
            input_t, label_t, _, _ = batch_tup
            input_g = input_t.to(self.device, non_blocking=True)
            label_g = label_t.to(self.device, non_blocking=True)

            prediction_g = self.model(input_g)
            loss_dice = self.dice_loss(prediction_g, label_g)
            loss_bce = self.bce_loss(prediction_g, label_g)
            loss = loss_dice + loss_bce * 10

            loss.backward()
            self.optimizer.step()

```

Validation evaluates segmentation quality using per-pixel precision, recall, and Dice scores across complete slices.

```python
    def do_validation(self, epoch_ndx, val_dl):
        self.model.eval()
        with torch.no_grad():
            for batch_ndx, batch_tup in enumerate(val_dl):
                input_t, label_t, _, _ = batch_tup
                input_g = input_t.to(self.device, non_blocking=True)
                label_g = label_t.to(self.device, non_blocking=True)
                prediction_g = self.model(input_g)
                # Compute and log metrics...

```

### 13.4.3 TensorBoard logging and image visualization

TensorBoard integration logs scalar loss metrics along with visual overlays of predicted segmentation masks against ground truth target masks.

[[img-11]]
Figure 13.11 - TensorBoard visualization comparing ground truth positive labels against raw CT inputs and predicted probability masks

```python
    def log_images(self, epoch_ndx, mode_str, input_t, label_t, prediction_g):
        for i in range(min(input_t.shape[0], 4)):
            writer = self.trn_writer if mode_str == 'trn' else self.val_writer
            writer.add_image(f"{mode_str}/{i}_input", input_t[i, 3:4], epoch_ndx)
            writer.add_image(f"{mode_str}/{i}_label", label_t[i], epoch_ndx)
            writer.add_image(f"{mode_str}/{i}_pred", torch.sigmoid(prediction_g[i]), epoch_ndx)

```

Visualizing predicted masks side-by-side with ground truth labels accelerates diagnosis of training issues like over-segmentation or under-segmentation.

Our approach will cheat a little by adding together predicted positive pixels across image patches to generate visual summary logs.

This looks very similar to the writer.add_scalar calls from prior chapters, but handles 2D image arrays instead of single values.

When we add similar code to our classification training script in the next chapter, we can monitor output feature maps alongside training loss curves.

```python
state = {
    'sys_argv': sys.argv,
    'time': datetime.datetime.now().isoformat(),
    'model_state': self.model.state_dict(),
    'optimizer_state': self.optimizer.state_dict(),
    'epoch': epoch_ndx,
    'totalTrainingSamples_count': self.totalTrainingSamples_count,
}

```

Saving checkpoint dictionaries containing model and optimizer states ensures training can be resumed seamlessly.

## 13.5 Results

We will update our classification training script in the next chapter with a similar routine to evaluate overall end-to-end performance.

[[img-12]]
Figure 13.12 - Training and validation loss curves showing convergence over training epochs

Initial epochs show rapid reduction in Dice loss, with validation metrics stabilizing as the network learns precise nodule boundaries.

E15 trn 0.2226 loss, 0.6234 precision, 0.7102 recall tag: seg_pr/recall

[[img-13]]
Figure 13.13 - Validation recall metrics plateauing over late training epochs

TRAINING DATASET.

[[img-14]]
Figure 13.14 - Precision-Recall metrics displaying overfitting behavior when trained over small sample counts

Training on a limited subset of CT slices leads to over-fitting, emphasizing the need for comprehensive dataset coverage and data augmentation.

## 13.6 Conclusion

Semantic segmentation using U-Net successfully isolates candidate nodule regions directly from 2D CT scan slices. Combining Dice loss with context-aware slice loading provides robust candidate masks for downstream classification.

## 13.7 Exercises

1. Modify the U-Net architecture to include residual connections inside each convolutional block.
2. Experiment with different weights between Dice loss and BCE loss to see how it impacts validation precision and recall.
3. Extend the dataset implementation to support 3D convolutions over volumetric input blocks.

## 13.8 Summary

* It is possible to train a segmentation model on image crops while validating on whole CT slices.
* Semantic segmentation produces per-pixel probability maps matching input spatial dimensions.
* The U-Net architecture uses skip connections to preserve high-resolution spatial features in decoder layers.
* Sørensen-Dice loss optimizes spatial overlap for imbalanced binary segmentation tasks.
* Visualizing intermediate prediction masks in TensorBoard provides essential qualitative debugging signals.