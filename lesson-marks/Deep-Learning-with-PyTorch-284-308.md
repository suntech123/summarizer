# Chapter 10: Combining data sources into a unified dataset

This chapter covers loading and parsing our dataset's raw annotations, converting CT scan data into PyTorch tensors, converting patient coordinates to array coordinates, constructing a PyTorch `Dataset` subclass, and visualizing data using Jupyter Notebook and Matplotlib.

[[img-01]]
Figure 10.1 - Project progress: Chapter 10 focuses on Step 1, data loading.

This is a crucial moment, when we begin writing the data-loading code that will feed our network.

## 10.1 Raw data source: LUNA618 annotations

The LUNA16 dataset provides raw 3D CT scans in MetaImage (`.mhd` and `.raw`) format accompanied by CSV annotation files.

```
annotations.csv
candidates.csv

```

The `annotations.csv` file contains verified nodule locations with X, Y, Z millimeter coordinates and diameters.

### 10.1.1 Parsing LUNA's annotation data

Parsing annotation files provides candidate candidate tuples consisting of series UIDs, center spatial coordinates, diameters, and binary nodule flags.

```python
import csv
from collections import namedtuple

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz'
)

```

```python
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open('data/part2/luna/annotations.csv', "r") as f:
        for row in csv.reader(f):
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

```

Locations provided in annotations match patient coordinate systems rather than raw array index spaces.

```python
    candidateInfo_list = []
    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in csv.reader(f):
            series_uid = row[0]

            if requireOnDisk_bool and series_uid not in presentOnDisk_set:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    if abs(candidateCenter_xyz[i] - annotationCenter_xyz[i]) > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

```

Now we'll build our full list of candidates using the information parsed from both CSV files.

```python
class LunaDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None):
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

```

Filtering dataset splits ensures valid evaluation sets isolated from training loops.

## 10.2 Loading individual CT scans

The native file format for CT scans is DICOM (www.dicomstandard.org), but LUNA16 provides combined MetaImage headers and raw binary arrays.

```python
import SimpleITK as sitk

def getCt(series_uid):
    mhd_path = glob.glob(
        'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
    )[0]
    ct_mhd = sitk.ReadImage(mhd_path)
    ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

    return Ct(series_uid, ct_a, ct_mhd)

```

SimpleITK loads `.mhd` files, converting 3D scalar fields into NumPy arrays.

## 10.3 Locating a nodule using the patient coordinate system

CT scans store scalar density values in Hounsfield units (HU), where air measures $-1000\text{ HU}$ and water measures $0\text{ HU}$. Values above $0\text{ HU}$ don't scale perfectly with density, but lung tissue typically measures between $-1000\text{ HU}$ and $-400\text{ HU}$.

[[img-02]]
Figure 10.2 - CT scan slice image transformation dataflow

Converting physical coordinates ($X, Y, Z$) to array index coordinates ($I, R, C$) requires applying origin offset, voxel spacing, and direction transformation matrices.

[[img-03]]
Figure 10.3 - Physical millimeter XYZ patient coordinate space versus 3D array IRC index space

Patient coordinate origin and voxel size properties determine transformation offsets.

[[img-04]]
Figure 10.4 - CT slice visualization showing 2D index axes overlaid with physical spatial dimensions

Voxel spacing accounts for non-uniform voxel grid dimensions across physical axes.

```python
class Ct:
    def __init__(self, series_uid, hu_a, ct_mhd):
        self.series_uid = series_uid
        self.hu_a = hu_a

        self.origin_xyz = ct_mhd.GetOrigin()
        self.vxSize_xyz = ct_mhd.GetSpacing()
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def change_coords_xyz_to_index(self, center_xyz):
        cri_a = (np.array(center_xyz) - self.origin_xyz) / self.vxSize_xyz
        cri_a = np.dot(np.linalg.inv(self.direction_a), cri_a)
        return tuple(int(round(x)) for x in reversed(cri_a))

```

To go back from XYZ to IRC, we need to perform matrix multiplication using the inverse direction matrix.

```python
    def change_coords_index_to_xyz(self, index_irc):
        cri_a = np.array(list(reversed(index_irc)))
        cri_a = np.dot(self.direction_a, cri_a)
        xyz_a = cri_a * self.vxSize_xyz + self.origin_xyz
        return tuple(xyz_a)

```

The `Ct` class now has all the data needed to convert back and forth between millimeter patient coordinates and array indices.

```python
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    center_irc = ct.change_coords_xyz_to_index(center_xyz)

    slice_list = []
    for axis, center_val in enumerate(center_irc):
        start_ndx = int(round(center_val - width_irc[axis] / 2))
        end_ndx = int(start_ndx + width_irc[axis])

        assert start_ndx >= 0, (series_uid, center_xyz, center_irc, axis, start_ndx)
        assert end_ndx <= ct.hu_a.shape[axis], (series_uid, center_xyz, center_irc, axis, end_ndx)

        slice_list.append(slice(start_ndx, end_ndx))

    ct_crop = ct.hu_a[tuple(slice_list)]

    return ct_crop, center_irc

```

The `getCtRawCandidate` function takes the center expressed in physical XYZ coordinates along with desired bounding crop sizes.

## 10.4 A straightforward dataset implementation

We encapsulate candidate loading within `LunaDataset`, a custom subclass of `torch.utils.data.Dataset`.

```python
import copy
import torch
from torch.utils.data import Dataset

class LunaDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None):
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

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor(
            [not candidateInfo_tup.isNodule_bool, candidateInfo_tup.isNodule_bool],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )

```

The implementation of our custom subclass will not be too complex, returning candidate tensors paired with target binary classification labels. We will get to those in a moment in sections 10.5.1 and 10.5.2.

```python
>>> ds = LunaDataset()
>>> candidate_t, pos_t, series_uid, center_irc = ds[0]
>>> candidate_t.shape, pos_t, series_uid, center_irc
(torch.Size([1, 32, 48, 48]), tensor([1, 0]), '1.3.6.1.4.1.14519.5.2.1.7014.4598.129202573212879025000539169642', tensor([91, 362, 339]))

```

Here we see the four items from our dataset output: 3D tensor input crop, classification tensor, series UID string, and index array location.

```python
class LunaDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None):
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

```

### 10.4.1 Constructing our dataset in LunaDataset.__init__

Initializing `LunaDataset` copies candidate tuples and applies validation splitting rules to prevent data contamination.

The other caveat regarding separation of validation data is ensuring patient scans do not overlap across training and validation splits.

### 10.4.2 Rendering the data

Visualizing dataset outputs verifies spatial coordinate transformations and crop boundaries before training models.

```python
import matplotlib.pyplot as plt

def renderCandidate(candidate_t):
    fig = plt.figure(figsize=(9, 9))
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(candidate_t[0, i * 3], cmap='gray')
    plt.show()

```

Again, either use `p2ch10_explore_data.ipynb` directly or start a new notebook to inspect rendered CT cross-sections.

## 10.5 Conclusion

Parsing LUNA annotations and establishing spatial transformations allows us to convert raw 3D CT volumes into PyTorch-ready training samples.

## 10.6 Exercises

1. Inspect candidate crops rendered across multiple axial slices in a Jupyter notebook.
2. Calculate total nodule count versus non-nodule candidate count in the LUNA dataset.

## 10.7 Summary

* LUNA16 annotations supply candidate coordinates in physical millimeter XYZ space.
* CT volume pixel densities are measured in Hounsfield units (HU).
* Transforming XYZ patient coordinates to IRC array indices requires origin offset, spacing, and direction matrices.
* `LunaDataset` subclasses PyTorch `Dataset` to return cropped 3D tensor samples paired with binary one-hot target vectors.
* Visualizing data slices using Matplotlib ensures processing pipeline correctness prior to model training.