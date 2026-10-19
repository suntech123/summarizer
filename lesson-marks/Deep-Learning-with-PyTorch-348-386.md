# Chapter 12: Improving training with metrics and augmentation

This chapter covers defining and computing precision and recall, balancing training data to prevent model bias, using TensorBoard to visualize metrics during training, and applying data augmentation to prevent overfitting. Our model from chapter 11 achieved 99.4% accuracy, but this metric is misleading because 99.8% of the candidates in our dataset are non-nodules. A model that simply predicts every sample as negative would achieve high accuracy while failing to detect actual nodules.

```
                  PREDICTED POSITIVE  PREDICTED NEGATIVE
ACTUAL POSITIVE   True Positive       False Negative
ACTUAL NEGATIVE   False Positive      True Negative

```

To accurately evaluate performance on imbalanced datasets, we must move beyond accuracy and measure specific diagnostic outcomes.

## 12.1 High-level plan for improvement

Improving our training pipeline requires four major steps. First, we need to implement precision and recall metrics to properly measure positive and negative predictions. Second, we must balance our training data so the model sees an equal proportion of positive and negative samples. Third, we will visualize these metrics in TensorBoard to monitor progress and catch overfitting early. Finally, we will use data augmentation to artificially expand our dataset and improve generalization.

## 12.2 Good dogs vs. bad guys: False positives and false negatives

Understanding classification outcomes requires categorizing predictions into four distinct groups. A true positive occurs when the model correctly identifies a positive sample, such as an actual nodule. A true negative occurs when the model correctly identifies a negative sample. A false positive occurs when a negative sample is incorrectly classified as positive. A false negative occurs when a positive sample is missed and classified as negative.

[[img-01]]
Figure 12.1 - The four possible classification outcomes illustrated through guard dogs detecting burglars versus birds

In medical imaging, false negatives are particularly dangerous because missing a malignant nodule delays crucial treatment. False positives lead to unnecessary follow-up procedures and patient anxiety, but are generally preferable to missed detections.

## 12.3 Graphing the positives and negatives

Evaluating model predictions across different decision thresholds provides deeper insight into diagnostic performance.

[[img-02]]
Figure 12.2 - Sorting prediction probabilities for actual positive and negative samples across varying decision thresholds

Adjusting the decision threshold changes the balance between false positives and false negatives.

[[img-03]]
Figure 12.3 - The effect of changing prediction thresholds on classification boundaries

Lowering the threshold catches more true positives at the cost of increasing false positives, while raising the threshold reduces false positives but increases false negatives.

[[img-04]]
Figure 12.4 - Comparing high and low threshold configurations on overall candidate classification

Tracking recall and precision across different operating thresholds allows us to pick the optimal operating point for clinical use.

### 12.3.1 Recall is Rex's tool

Recall measures the fraction of actual positive samples that were correctly identified by the model:

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

[[img-05]]
Figure 12.5 - Recall measures how effectively the model detects all true positive events.

A high-recall model minimizes false negatives, ensuring that very few actual nodules are missed.

[[img-06]]
Figure 12.6 - A low-threshold model achieves high recall by flagging almost all potential positive events.

In medical screening, high recall is prioritized during early detection stages to avoid missing critical diagnoses.

### 12.3.2 Precision is Preston's forte

Precision measures the fraction of positive predictions that were actually correct:

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

[[img-07]]
Figure 12.7 - Precision measures the proportion of positive predictions that are truly positive events.

A high-precision model minimizes false positives, ensuring that flagged candidates are highly likely to be actual nodules.

[[img-08]]
Figure 12.8 - A high-threshold model achieves high precision by only flagging candidates with very high confidence.

High precision prevents diagnostic systems from overwhelming clinicians with false alarms.

### 12.3.3 Combining precision and recall: The $F_1$ score

Precision and recall often trade off against each other as decision thresholds change.

[[img-09]]
Figure 12.9 - The trade-off between precision and recall across decision thresholds

The $F_1$ score harmonic mean summarizes this balance into a single scalar metric:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

[[img-10]]
Figure 12.10 - Harmonic mean ($F_1$ score) vs. arithmetic average across precision and recall values

Unlike a simple average, the harmonic mean heavily penalizes low performance in either individual metric.

[[img-11]]
Figure 12.11 - Comparing $F_1$ score contour maps against standard arithmetic averaging contours

The $F_1$ score reaches its maximum value of $1.0$ only when both precision and recall are perfect.

### 12.3.4 Updating our metrics calculation

We update our training loop to calculate precision, recall, and $F_1$ score alongside classification loss.

```python
def logMetrics(self, epoch_ndx, mode_str, metrics_t, classificationThreshold=0.5):
    pos_mask = metrics_t[METRICS_LABEL_NDX] > 0.5
    neg_mask = ~pos_mask

    pos_pred_mask = metrics_t[METRICS_PRED_NDX] > classificationThreshold
    neg_pred_mask = ~pos_pred_mask

    tp = (pos_mask & pos_pred_mask).sum().item()
    fp = (neg_mask & pos_pred_mask).sum().item()
    fn = (pos_mask & neg_pred_mask).sum().item()
    tn = (neg_mask & neg_pred_mask).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)

```

Executing the updated training script outputs detailed metric breakdowns for each epoch.

```bash
$ ../.venv/bin/python -m p2ch12.training
Starting LunaTrainingApp, Namespace(batch_size=32, comment='dwlpt', balanced=False, augmented=False, augment_flip=False, augment_offset=False, augment_scale=False, augment_rotate=False, augment_noise=False, num_workers=8, epochs=1)
Epoch 1/1:
  trn: Loss 0.0026, Accuracy 99.8%, Precision 0.0000, Recall 0.0000, F1 0.0000
  val: Loss 0.0021, Accuracy 99.8%, Precision 0.0000, Recall 0.0000, F1 0.0000

```

The metrics confirm that our unbalanced baseline model learns to predict negative for every sample, yielding zero recall and zero $F_1$ score despite 99.8% nominal accuracy.

## 12.4 What does an ideal dataset look like?

An ideal training dataset presents an equal number of positive and negative examples during gradient descent.

[[img-12]]
Figure 12.12 - Unbalanced training datasets cause models to ignore rare positive classes.

In our raw dataset, negative samples outnumber positive samples by roughly $500:1$.

[[img-13]]
Figure 12.13 - Constructing balanced mini-batches ensures equal gradient contribution from both positive and negative samples.

Constructing balanced batches ensures that loss updates treat both classes with equal weight.

### 12.4.1 Balancing our dataset

We modify `LunaDataset` to separate positive and negative candidate samples into distinct lists.

```python
class LunaDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None, sortby_str='random', ratio_int=0):
        self.ratio_int = ratio_int
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        if val_stride > 0:
            if isValSet_bool:
                self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            else:
                del self.candidateInfo_list[::val_stride]

        self.pos_list = [x for x in self.candidateInfo_list if x.isNodule_bool]
        self.neg_list = [x for x in self.candidateInfo_list if not x.isNodule_bool]

```

When ratio sampling is enabled, each dataset epoch samples negative candidates proportional to the available positive candidates.

```python
    def __len__(self):
        if self.ratio_int:
            return len(self.pos_list) * self.ratio_int
        else:
            return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            pos_ndx = ndx // self.ratio_int
            if ndx % self.ratio_int == 0:
                candidateInfo_tup = self.pos_list[pos_ndx % len(self.pos_list)]
            else:
                neg_ndx = ndx - pos_ndx - 1
                candidateInfo_tup = self.neg_list[neg_ndx % len(self.neg_list)]
        else:
            candidateInfo_tup = self.candidateInfo_list[ndx]

        return getCtCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            (32, 48, 48),
        ) + (candidateInfo_tup.isNodule_bool,)

```

Training with balanced mini-batches immediately forces the model to learn features that distinguish nodules from non-nodules.

```bash
$ ../.venv/bin/python -m p2ch12.training --balanced
E1 trn_neg Loss 0.0026, Acc 68.2%, Prec 0.0000, Rec 0.0000
E1 trn_pos Loss 0.8231, Acc 51.2%, Prec 0.5120, Rec 1.0000

```

```bash
E2 val Loss 0.0603, Acc 98.5%, Prec 0.1271, Rec 0.8456

```

Metric outputs show that balanced training dramatically improves recall, enabling the model to detect over 84% of validation nodules.

## 12.5 Revisiting the problem of overfitting

While balanced training improves candidate recall, tracking performance over multiple epochs reveals severe overfitting.

[[img-14]]
Figure 12.14 - TensorBoard loss curves for positive candidates showing diverging validation loss due to overfitting

Training loss steadily decreases toward zero, while validation loss plateaus and begins to rise.

[[img-15]]
Figure 12.15 - Negative loss curves showing validation performance degradation over extended training epochs

Overfitting occurs because the network rapidly memorizes the small number of unique positive samples present in the training set.

## 12.6 Preventing overfitting with data augmentation

Data augmentation transforms training samples dynamically on each pass, preventing the network from memorizing exact voxel patterns.

```python
class SegmentApp:
    def __init__(self, sys_argv=None):
        pass

```

Random spatial transforms artificially expand training variability without altering underlying medical labels.

### 12.6.1 Augmenting data

We implement random affine transformations—including flipping, shifting, scaling, and rotation—alongside additive Gaussian noise.

```python
def getCtAugmentedCandidate(
    augmentation_dict,
    series_uid,
    center_xyz,
    width_irc,
):
    ct_t, center_i = getCtRawCandidate(series_uid, center_xyz, width_irc)

    if augmentation_dict.get('flip', False):
        if random.random() > 0.5:
            ct_t = torch.flip(ct_t, dims=[1])
        if random.random() > 0.5:
            ct_t = torch.flip(ct_t, dims=[2])
        if random.random() > 0.5:
            ct_t = torch.flip(ct_t, dims=[3])

    return ct_t, center_i

```

Rotation is the first augmentation method applied to 3D sub-volumes.

[[img-16]]
Figure 13.16 - Random spatial rotation applied to 3D candidate crops

Rotating candidate volumes along random axes changes pixel arrangements while preserving physical geometry.

[[img-17]]
Figure 13.17 - Random spatial flipping applied across candidate dimensions

Axis flips mirror candidate volumes without changing nodule morphology.

```bash
$ ../.venv/bin/python -m p2ch12.training --balanced --augmented

```

Training with full data augmentation stabilizes validation metrics and significantly reduces overfitting across epochs.

[[img-18]]
Figure 12.18 - Validation precision and recall curves with data augmentation enabled

Data augmentation keeps validation loss aligned with training loss, resulting in robust overall $F_1$ performance.

## 12.7 Conclusion

Combining precision/recall tracking, balanced ratio sampling, TensorBoard visualization, and dynamic data augmentation yields a robust classification model capable of detecting lung nodule candidates accurately.

## 12.8 Exercises

1. The $F_1$ score can be generalized to support values other than 1.
a. Read about the $F_\beta$ score and explain how setting $\beta = 2$ or $\beta = 0.5$ shifts focus between precision and recall.
b. Implement an $F_\beta$ calculation inside `logMetrics` and evaluate model performance under different $\beta$ values.
2. Modify `LunaDataset` to support independent control over flipping along individual spatial axes (X, Y, Z).
3. Experiment with adding random intensity jitter (brightness/contrast adjustments) to `getCtAugmentedCandidate` and observe its impact on validation loss.

## 12.9 Summary

* A binary label and output threshold yield four prediction outcomes: true positive, false positive, true negative, and false negative.
* Precision measures the proportion of positive predictions that are correct, while recall measures the proportion of actual positives detected.
* The $F_1$ score combines precision and recall into a single harmonic mean.
* Unbalanced datasets cause classifiers to optimize accuracy by predicting the majority class exclusively.
* Ratio sampling balances training mini-batches, forcing the model to learn features for both positive and negative classes.
* Data augmentation—via rotation, flipping, scaling, and noise—prevents networks from memorizing training samples and mitigates overfitting.
* TensorBoard visualization allows tracking training vs. validation metrics over time to evaluate model generalization.