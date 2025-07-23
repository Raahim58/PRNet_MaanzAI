# PRNet ChestXRay Training

Progressive Resolution–based Network (PRNet) for Chest X‑Ray disease classification, with support for various backbones, image resolutions, and training options.

## Requirements

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/PRNet.git
   cd PRNet
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
PRNet/
├── train.py           # CLI entrypoint
├── models/            # model & backbone implementations
├── data/              # your data: train/, val/, test/ folders
└── outputs/
    ├── curves/        # training curves (PNG)
    └── results/       # aggregated CSV results
```

## Usage

### Local / Server

```bash
python train.py \
  --backbone efficientnet_b5 \
  --image-size 512 \
  --progressive \
  --bias-softmax \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --patience 5 \
  --data-dir /path/to/data
```

### Kaggle

```bash
!git clone https://github.com/your-username/PRNet.git
%cd PRNet
!pip install -r requirements.txt
!python train.py \
  --backbone efficientnet_b5 \
  --image-size 512 \
  --progressive \
  --bias-softmax
```

### Colab

```bash
!git clone https://github.com/your-username/PRNet.git
%cd PRNet
!pip install -r requirements.txt
!python train.py \
  --backbone efficientnet_b5 \
  --image-size 512 \
  --progressive \
  --bias-softmax
```

## CLI Arguments

| Flag             | Type   | Default                | Description                                                 |
| ---------------- | ------ | ---------------------- | ----------------------------------------------------------- |
| `--backbone`     | string | **required**           | Backbone model name (e.g. `efficientnet_b0`, `resnet50`)    |
| `--image-size`   | int    | **required**           | Square input resolution                                     |
| `--progressive`  | —      | `False`                | Enable Progressive Resolution pipeline                      |
| `--bias-softmax` | —      | `False`                | Apply bias‑adjustable softmax                               |
| `--epochs`       | int    | `20`                   | Number of training epochs                                   |
| `--batch-size`   | int    | `16`                   | Batch size                                                  |
| `--lr`           | float  | `1e-4`                 | Initial learning rate                                       |
| `--patience`     | int    | `5`                    | Early‑stopping patience (epochs without improvement)        |
| `--data-dir`     | string | `/kaggle/working/data` | Root dir containing `train/`, `val/`, `test/` image folders |

## Outputs

* **outputs/curves/** – PNG files of training/validation curves
* **outputs/results/prnet\_results.csv** – CSV log of each run’s metrics

## Example

```bash
python train.py \
  --backbone resnet50 \
  --image-size 224 \
  --epochs 30 \
  --batch-size 32 \
  --lr 5e-5 \
  --patience 7 \
  --data-dir ./data
```

````

```text
# requirements.txt
torch>=1.12
torchvision
timm
pandas
numpy
matplotlib
albumentations
tqdm
````
