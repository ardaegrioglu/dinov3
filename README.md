Here’s a ready-to-commit `README.md` you can copy-paste:

# DINOv3 Segmentation — Data Prep & Training

This repo trains a segmentation head on **ImageNet-S**, which is derived from **ImageNet-1k**. Follow the steps below to prepare datasets and launch training on Slurm.

## 1) Prerequisites

* Access to the **ImageNet-1k** dataset (requires registration and license from ImageNet).
* Python environment with this project’s dependencies installed.
* Slurm cluster access (the example uses `gpu-l40s` and account `krishna`; change to yours).

---

## 2) Prepare ImageNet-1k

Download or mount ImageNet-1k so its root looks like:

```
<ROOT>/test/ILSVRC2012_test_00000001.JPEG
<ROOT>/test/[..]
<ROOT>/test/ILSVRC2012_test_00100000.JPEG
<ROOT>/train/n01440764/n01440764_10026.JPEG
<ROOT>/train/[...]
<ROOT>/train/n15075141/n15075141_9993.JPEG
<ROOT>/val/n01440764/ILSVRC2012_val_00000293.JPEG
<ROOT>/val/[...]
<ROOT>/val/n15075141/ILSVRC2012_val_00049174.JPEG
<ROOT>/labels.txt
```

In addition, the implementation expects metadata files under an **extra** directory:

```
<EXTRA>/class-ids-TRAIN.npy
<EXTRA>/class-ids-VAL.npy
<EXTRA>/class-names-TRAIN.npy
<EXTRA>/class-names-VAL.npy
<EXTRA>/entries-TEST.npy
<EXTRA>/entries-TRAIN.npy
<EXTRA>/entries-VAL.npy
```

Generate them once with:

```python
from dinov3.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```

> Replace `<ROOT>` with your ImageNet root and `<EXTRA>` with the path where you want these `.npy` files written.

---

## 3) Prepare ImageNet-S

ImageNet-S is derived from ImageNet-1k. You can prepare it in **one command**:

```bash
cd datapreparation
bash data_preparation.sh [your imagenet path] [the path to save ImageNet-S datasets] [split: 50 300 919 all] [whether to copy new images: false, true]
```

### Prepare only parts (optional)

**Training sets:**

```bash
python datapreparation_train.py \
  --imagenet-dir [your imagenet path] \
  --save-dir [the path to save ImageNet-S datasets] \
  --mode [split: 50 300 919 all]
# Add --copy to copy files instead of making symlinks.
```

**Validation + Test sets:**

```bash
python datapreparation_val.py \
  --imagenet-dir [your imagenet path] \
  --save-dir [the path to save ImageNet-S datasets] \
  --mode [split: 50 300 919 all]
# This copies new images for val/test.
```

**Annotations (semantic segmentation masks):**

```bash
bash datapreparation_anno.sh [the path to save ImageNet-S datasets] [split: 50 300 919 all]
```

### Expected structure

```
├── imagenet-s
    ├── ImageNetS919
        ├── train-full                 # full ImageNet-1k training set (1000 classes)
        ├── train                      # ImageNet-S-919 train (919 classes)
        ├── train-semi                 # 10 annotated images/class
        ├── train-semi-segmentation    # segmentation masks for train-semi
        ├── validation                 # ImageNet-S-919 val
        ├── validation-segmentation    # masks for val
        └── test                       # ImageNet-S-919 test (masks on online eval server)
    ├── ImageNetS300        
        ├── train
        ├── train-semi
        ├── train-semi-segmentation
        ├── validation
        ├── validation-segmentation
        └── test
    └── ImageNetS50                            
        ├── train
        ├── train-semi
        ├── train-semi-segmentation
        ├── validation
        ├── validation-segmentation
        └── test
```

### Image counts (reference)

| Dataset          | Categories |     Train |    Val |   Test |
| ---------------- | ---------: | --------: | -----: | -----: |
| ImageNet-S_{50}  |         50 |    64,431 |    752 |  1,682 |
| ImageNet-S_{300} |        300 |   384,862 |  4,097 |  9,088 |
| ImageNet-S (919) |        919 | 1,183,322 | 12,419 | 27,423 |

We annotate **39,842** val/test images and **9,190** training images with precise pixel-level masks.

---

## 4) Launch Training

Set an output directory and run (adjust account/partition and paths as needed):

```bash

PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --slurm-account krishna \
  --slurm-partition gpu-l40s \
  --nodes 1 \
  --ngpus 4 \
  --config-file dinov3/configs/train/vitl_segmentation.yaml \
  output_dir=$OUTPUT_DIR \
  train.output_dir=$OUTPUT_DIR \
  train.dataset_path=ImageNet:split=TRAIN:Imagenet:extra=Imagenet \
  segmentation.enabled=true \
  gram.use_loss=false \
  gram.loss_weight=0.0 \
  gram.ema_teacher=true \
  gram.ckpt=null \
  gram.it_load_ema_teacher=0
```

**Notes:**

* `train.dataset_path` must point to your ImageNet **root** and **extra** metadata directory.
* If your cluster uses different GPU partitions/accounts, change `--slurm-partition` and `--slurm-account`.
* Use `-d` or config flags to adjust batch size, epochs, etc., in `dinov3/configs/train/vitl_segmentation.yaml`.

---

## 5) Tips

* Keep datasets and outputs out of Git; add large folders (e.g., `ImageNet-S/`, `output/`, `logs/`, `wandb/`, `eval_result*/`) to `.gitignore`.
* Verify masks exist for `train-semi` and `validation` splits before training.
* If you change default branch names or remotes, update tracking with `git push -u origin <branch>`.

---

## 6) Licenses

* **ImageNet-1k** is subject to its own license and usage policies.
* This code follows the license terms of DINOv3 and any third-party components it uses. Check the LICENSE files accordingly.
