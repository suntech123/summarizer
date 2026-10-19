# Chapter 9: Using PyTorch to fight cancer

This chapter covers breaking a large problem into smaller, easier-to-solve steps, organizing your code for a large-scale project, understanding CT scans, and downloading and preparing medical image data for deep learning. Machine learning applications in healthcare present unique technical challenges and high operational stakes.

## 9.1 Introduction to the use case

Our goal for part 2 of this book is to build an end-to-end lung cancer detection system using PyTorch. Automated medical diagnostic systems must detect subtle physiological anomalies across volumetric patient scans. Developing a functional system requires framing the medical task into well-defined machine learning operations.

## 9.2 Preparing for a large-scale project

Lung cancer is one of the most common and lethal forms of cancer worldwide, making early detection critical for patient survival rates. Because manual review of full CT scans by radiologists is time-consuming and prone to human oversight, automated screening systems offer significant practical value.

High-throughput medical imaging pipelines process large amounts of data, requiring careful computational resource planning. While small experiments run easily on consumer hardware, training complex 3D deep neural networks on full volumetric CT datasets typically requires dedicated GPU acceleration to complete within reasonable timeframes.

## 9.3 What is a CT scan, exactly?

Computed Tomography (CT) scans capture 3D volumetric images of internal human anatomy by taking multiple 2D X-ray measurements around a rotational axis.

[[img-01]]
Figure 9.1 - A 3D CT scan constructed from a stack of 2D axial slice images

Unlike standard RGB color images that contain three color channels, CT scans consist of single-channel volumetric scalar data representing radiodensity.

[[img-02]]
Figure 9.2 - 2D slices extracted along axial, sagittal, and coronal anatomical planes

NOTE CT scans actually measure radiodensity, which is expressed in quantitative units known as Hounsfield units (HU). Standard anatomical structures map to specific HU density ranges: air measures $-1000\text{ HU}$, water measures $0\text{ HU}$, soft tissue spans $+20\text{ HU}$ to $+80\text{ HU}$, and dense bone exceeds $+1000\text{ HU}$.

## 9.4 The project: An end-to-end detector for lung cancer

Our detection system takes raw 3D CT scans as input and outputs spatial coordinates and malignancy probability estimates for suspected lung nodules.

[[img-03]]
Figure 9.3 A patient inside a CT scanner, with an example output axial slice showing lung tissue and internal structures

Building a full diagnostic system directly from raw CT scans requires decomposing the pipeline into modular sub-tasks.

[[img-04]]
Figure 9.4 - End-to-end system workflow diagram showing the four main processing steps: data loading, candidate segmentation, candidate filtering/grouping, and nodule classification

The complete processing pipeline decomposes into four main sequential stages:

1. Load raw CT scan volumes and parse clinical annotation metadata files.
2. Identify voxels of potential tumors using a 2D/3D segmentation model.
3. Group connected candidate voxels into discrete spatial candidate regions.
4. Classify individual nodule candidates as benign or malignant tumors.

On the shoulders of giants We are standing on the shoulders of medical imaging researchers who have established standard datasets and benchmark tasks. Modular pipeline design allows updating individual model components without rearchitecting the entire system.

[[img-05]]
Figure 9.5 - Single axial CT slice displaying lung tissue boundaries, airway structures, and candidate nodule regions

Analyzing raw volumetric data reveals that candidate nodules represent an extremely tiny fraction of total lung volume.

[[img-06]]
Figure 9.6 - Anatomical cross-section showing detailed internal lung structures and tissue density variations

Isolating candidate regions before running deep classification models reduces overall computational requirements.

[[img-07]]
Figure 9.7 - Sequential axial slices showing a spherical lung nodule expanding across adjacent vertical slices

Examining 3D spatial continuity across adjacent slices helps distinguish actual spherical nodules from linear blood vessels.

[[img-08]]
Figure 10.8 - System architecture map highlighting data loading and candidate filtering steps within the overall project design

Separating dataset processing from model execution maintains clean code boundaries and simplifies pipeline debugging.

[[img-09]]
Figure 9.9 - Spatial visualization of false positive candidate locations distributed across a lung CT scan

Candidate generation prioritizes high recall to avoid missing true nodules, relying on downstream classifiers to filter out false positives.

### 9.4.3 Our data source: The LUNA Grand Challenge

We use the LUNA16 (LUng Nodule Analysis 2016) dataset, a widely recognized public benchmark dataset derived from the LIDC-IDRI collection. LUNA16 provides 888 anonymized thoracic CT scans along with expert radiologist annotations for nodule locations and sizes.

## 9.5 Downloading the LUNA16 data

Navigate to [https://luna16.grand-challenge.org/download/](https://www.google.com/search?q=https://luna16.grand-challenge.org/download/) to register and download the raw CT scan subsets and CSV annotation files. The dataset is partitioned into 10 subset archives (`subset0` through `subset9`), total approximately 120 GB, and should be extracted into a unified local data directory structure.

## 9.6 Summary

* Lung cancer detection from 3D CT scans presents a compelling real-world deep learning application.
* Computed Tomography produces 3D single-channel scalar volumes measured in quantitative Hounsfield units (HU).
* Decomposing the detection task into data loading, candidate segmentation, region grouping, and nodule classification makes large medical projects manageable.
* Public datasets like LUNA16 provide standardized, expert-annotated 3D CT scans for training and evaluation.