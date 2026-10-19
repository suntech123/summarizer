# Chapter 14: End-to-end nodule analysis, and where to go next

This chapter covers connecting segmentation and candidate classification into an end-to-end pipeline, predicting malignancy directly from CT scans, calculating quantitative diagnostic metrics, and reviewing techniques and directions for future performance improvements. Part 2 builds toward a complete end-to-end clinical diagnostic system. Achieving high performance requires connecting previously isolated pipeline stages into a unified workflow.

## 14.1 Towards the finish line

We can get a hint of the work remaining by looking at our system diagram from chapter 11.

[[img-01]]
Figure 14.1 - System diagram, showing the work we have completed in parts 1–3 and highlighting the final step covered in chapter 14

Connecting components into an end-to-end system enables evaluating end-to-end metrics on raw CT scans. Predicting malignancy directly from CT scans provides a final clinical assessment.

## 14.2 Quantitative performance metrics

An ideal system identifies all true nodules while eliminating all non-nodules. Diagnostic systems must balance sensitivity and specificity across varying decision thresholds.

### 14.2.1 Independence of the validation set

Using separate validation sets prevents data leakage during hyperparameter tuning. Data leakage occurs when validation feedback implicitly influences training choices. Maintaining strict separation ensures realistic real-world performance estimates.

### 14.2.2 ROC/AUC metrics

Receiver Operating Characteristic (ROC) curves plot True Positive Rate against False Positive Rate across all thresholds. The Area Under the Curve (AUC) summarizes overall classification power into a single scalar value. A random classifier yields an AUC of 0.5, whereas a perfect classifier achieves 1.0.

## 14.3 Bridging CT segmentation and nodule candidate classification

The end-to-end analysis pipeline connects raw CT scan processing to nodule candidate classification.

[[img-02]]
Figure 14.2 - Bridging segmentation and candidate classification

Combining segmentation and candidate classification allows running prediction directly on unannotated CT scans.

### 14.3.1 Segmentation

Segmentation generates 3D binary masks highlighting potential nodule candidate regions. Connected component analysis converts continuous mask regions into discrete candidate groupings. Center coordinates and bounding sizes are extracted for each detected candidate region.

### 14.3.2 Grouping candidate voxels into nodule candidates

Grouped voxels are converted into physical millimeter coordinates using CT spatial metadata. Candidates with volumes below minimum size thresholds are filtered out to reduce noise.

[[img-03]]
Figure 14.3 - Converting 2D slice masks into 3D candidates

Center coordinates, effective diameters, and confidence scores are packaged into structured candidate tuples.

### 14.3.3 Evaluating candidate classification

Candidate tuples are passed into the classification network to obtain final nodule probabilities.

```python
class NodnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=4,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--run-segmentation',
            help="Run candidate segmentation step.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--run-classification',
            help="Run candidate classification step.",
            action='store_true',
            default=False,
        )

        self.cli_args = parser.parse_args(sys_argv)

```

```python
    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        if self.cli_args.run_segmentation:
            self.segmentation_dataset = LunarSegmentationDataset(
                val_stride=10,
                isValSet_bool=True,
            )
            self.segmentation_dl = DataLoader(
                self.segmentation_dataset,
                batch_size=self.cli_args.batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=True,
            )
            self.seg_model = UNetWrapper(
                in_channels=1,
                n_classes=1,
                depth=4,
                wf=3,
                padding=True,
                batch_norm=True,
                up_mode='upconv',
            )
            self.load_model(self.cli_args.segmentation_path, self.seg_model)
            self.seg_model.eval()

            self.run_segmentation()

        if self.cli_args.run_classification:
            self.classification_dataset = LunarClassificationDataset(
                val_stride=10,
                isValSet_bool=True,
            )
            self.classification_dl = DataLoader(
                self.classification_dataset,
                batch_size=self.cli_args.batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=True,
            )
            self.cls_model = LunaModel()
            self.load_model(self.cli_args.classification_path, self.cls_model)
            self.cls_model.eval()

            self.run_classification()

```

```python
    def run_segmentation(self):
        self.candidate_info_list = []
        for batch_ndx, batch_tup in enumerate(self.segmentation_dl):
            input_t, label_t, series_uid_list, _ = batch_tup
            input_g = input_t.to(self.device)

            with torch.no_grad():
                output_g = self.seg_model(input_g)

            for i, series_uid in enumerate(series_uid_list):
                mask_a = output_g[i, 0].cpu().numpy() > 0.5
                candidate_list = self.group_segmentation_output(
                    mask_a, series_uid
                )
                self.candidate_info_list.extend(candidate_list)

```

```python
    def group_segmentation_output(self, mask_a, series_uid):
        clean_a = ndimage.binary_erosion(mask_a, iterations=1)
        clean_a = ndimage.binary_dilation(clean_a, iterations=2)

        labels_a, num_features = ndimage.label(clean_a)
        candidate_list = []

        ct = getCt(series_uid)
        for i in range(1, num_features + 1):
            candidate_mask = labels_a == i
            center_xyz = ndimage.center_of_mass(
                candidate_mask
            )
            center_xyz = ct.change_coords_index_to_xyz(center_xyz)
            candidate_list.append(CandidateInfoTuple(
                isNodule_bool=False,
                hasAnnotation_bool=False,
                isMalignant_bool=False,
                diameter_mm=0.0,
                series_uid=series_uid,
                center_xyz=center_xyz,
            ))
        return candidate_list

```

Running classification on extracted candidate locations filters false positives while retaining true nodule locations.

### 14.3.4 End-to-end pipeline execution

Executing the full pipeline processes raw CT series down to classified nodule predictions.

```python
    def run_classification(self):
        self.predictions_list = []
        for batch_ndx, batch_tup in enumerate(self.classification_dl):
            input_t, label_t, series_uid_list, center_xyz_list = batch_tup
            input_g = input_t.to(self.device)

            with torch.no_grad():
                output_g = self.cls_model(input_g)

            probs_g = torch.softmax(output_g, dim=1)

            for i in range(len(series_uid_list)):
                self.predictions_list.append((
                    probs_g[i, 1].item(),
                    series_uid_list[i],
                    center_xyz_list[i],
                ))

```

This candidate is assigned a 53% probability of being a nodule. End-to-end execution generates predictions across all series in the validation dataset.

## 14.4 Predicting malignancy

Malignancy prediction evaluates whether detected nodules are benign or cancerous.

We run the following, which should take half an hour to an hour when run on a GPU:

```bash
$ python -m p2ch14.dseg_standalone --run-segmentation --run-classification

```

Malignancy labels provided in LIDC data enable training classifiers to estimate cancer risk.

### 14.4.1 LIDC dataset annotations

LIDC annotations include subjective radiologist ratings for malignancy, sphericity, and margin sharpness.

The pylicd library gives us ready access to these annotations. Averaging radiologist ratings produces numerical ground truth labels for training.

### 14.4.2 Malignancy model architecture

Malignancy classification reuses the convolutional backbone while replacing the final linear header.

[[img-04]]
Figure 14.4 - System diagram focused on malignancy classification

Transfer learning initializes feature extraction weights from prior candidate classification training.

### 14.4.3 Diagnostic ROC evaluation

Evaluating diagnostic performance relies on ROC curves comparing true positives against false positives.

[[img-05]]
Figure 14.5 - FROC curve for nodule detection

A False Positive per Scan (FP/scan) rate measures detection specificity across CT volumes.

Here, we also call out two specific threshold values: diameters of 5.42 mm and 8.35 mm.

[[img-06]]
Figure 14.6 - Precision-Recall curve for nodule candidate detection

Precision-Recall curves provide complementary performance insight when class distributions are imbalanced.

To compute the area under this curve, we can use the `sklearn.metrics.roc_auc_score` function.

### 14.4.4 Malignancy training and fine-tuning

Fine-tuning adapts pre-trained candidate classification weights to malignancy assessment tasks.

Recall from chapter 8 that we could interpret the intermediate values as class probabilities. Fine-tuning with lower learning rates prevents destructive weight updates in early backbone layers.

[[img-07]]
Figure 14.7 - Transfer learning architecture for malignancy classification

Passing strict False allows loading partial model weights while updating modified classification heads.

```python
    def init_model(self):
        self.model = LunaModel()
        if self.cli_args.transfer_path:
            state = torch.load(self.cli_args.transfer_path)
            model_state = state['model_state']
            self.model.load_state_dict(model_state, strict=False)

        self.model.to(self.device)
        self.optimizer = self.init_optimizer()

```

Fine-tuning improves overall classification AUC on independent validation sets.

losses stagnating is qualitatively different from training loss going down while validation loss goes up.

[[img-08]]
Figure 14.8 - ROC curve comparing baseline and fine-tuned malignancy models

The fine-tuned model achieves superior AUC performance compared to baseline diameter-based heuristics.

### 14.4.5 TensorBoard integration

TensorBoard logs loss curves, metric distributions, and prediction visualizations during training.

When logging to TensorBoard, we can track detailed metrics across epochs.

```python
    def log_metrics(self, epoch_ndx, mode_str, metrics_t, classification_threshold=0.5):
        writer = self.trn_writer if mode_str == 'trn' else self.val_writer
        writer.add_scalar('loss/total', metrics_t[METRICS_LOSS_NDX].mean(), epoch_ndx)

        posLabel_mask = metrics_t[METRICS_LABEL_NDX] > 0.5
        negLabel_mask = ~posLabel_mask

        posPred_mask = metrics_t[METRICS_PRED_P_NDX] > classification_threshold
        negPred_mask = ~posPred_mask

        tp_count = (posLabel_mask & posPred_mask).sum().item()
        fp_count = (negLabel_mask & posPred_mask).sum().item()
        fn_count = (posLabel_mask & negPred_mask).sum().item()
        tn_count = (negLabel_mask & negPred_mask).sum().item()

        precision = tp_count / max(tp_count + fp_count, 1)
        recall = tp_count / max(tp_count + fn_count, 1)

        writer.add_scalar('metrics/precision', precision, epoch_ndx)
        writer.add_scalar('metrics/recall', recall, epoch_ndx)

```

Given this observation on the training performance, we can inspect misclassified examples.

fine-tuning. It probably would be good to pull up a few images of that type. TensorBoard logs provide visual verification of model focus areas during training.

## 14.5 What we see when we diagnose

Reviewing end-to-end diagnostic outputs reveals qualitative strengths and failure modes.

What we see when we diagnose CT scans highlights key challenges in automated processing.

Let's see what we get: Total | Complete Miss | Filtered Out | Pred. Benign | Pred. Malignant.

```python
def print_diagnosis(series_uid, candidate_info_list, predictions_list):
    print(f"Diagnosis for series {series_uid}:")
    for info, pred in zip(candidate_info_list, predictions_list):
        prob = pred[0]
        print(f"  Center: {info.center_xyz}, Malignancy Prob: {prob:.2f}")

```

The general message is that there are many steps in the pipeline where errors can accumulate. Segmentation misses small or low-contrast nodules before classification occurs. End-to-end evaluation highlights critical areas requiring future pipeline refinement.

## 14.6 What next? Additional sources of inspiration (and data)

Improving model performance relies on exploring advanced architectures, data augmentation, and external datasets.

[[img-09]]
Figure 14.9 - Overview of potential model improvements

Regularization, data augmentation, and ensembling consistently improve generalization performance.

In ensembling, we typically use predictions from multiple trained models to reduce variance. Advanced architectures like 3D ResNets capture richer spatial relationships.

other ways to do self-supervised learning leverage unlabeled CT data for feature pre-training. Self-supervised pre-training improves downstream classification when labeled data is scarce.

to 'easy' and 'hard' to classify nodules, focal loss reweights hard training examples. Hard negative mining focuses model capacity on ambiguous false positive candidates.

LUNA PAPERS The LUNA Grand Challenge has collected several results ([https://luna16.grand-challenge.org/Results/](https://luna16.grand-challenge.org/Results/)). Public leaderboards provide benchmarks and architectural inspiration for model development.

further every day, right now. It's an active field with rapid ongoing technological progress. Staying connected with literature ensures continuous integration of state-of-the-art techniques.

## 14.7 Conclusion

Combining segmentation and classification yields an end-to-end automated lung cancer diagnostic pipeline. Quantitative metrics, transfer learning, and diagnostic visualization confirm end-to-end system viability.

## 14.8 Exercises

1. Implement a test set for classification, or reuse the test set from chapter 10.
2. Experiment with different segmentation thresholds to observe impacts on candidate generation sensitivity.
3. Train a malignancy classifier using focal loss to address class imbalance.

## 14.9 Summary

* Connecting segmentation and classification enables end-to-end CT scan diagnosis.
* ROC curves and AUC scores provide threshold-independent classification evaluation.
* False Positives per Scan (FP/scan) metrics quantify clinical detection usability.
* Transfer learning adapts candidate classification backbone weights to malignancy prediction.
* End-to-end error propagation requires carefully tuning each pipeline stage.
* Ensembling, 3D architectures, and self-supervised learning further boost performance.
* TensorBoard allows us to write out arbitrary scalar, image, and histogram data for analysis.