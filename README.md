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
├── scripts/
│   ├── train.py       # Training script
│   ├── eval.py        # Model evaluation script
│   └── predict.py     # Single image/batch prediction script
│   └── covid.jpeg     # custom image to test predict.py on
└── results/
    ├── curves/        # training loss and accuracy curves (PNG)
    └── models/        # best model checkpoints
    └── models/        # aggregated CSV results
└── PRNet_trained      # original colab/kaggle notebook
└── PRNet...           # original paper
```

## Usage

### Training

#### Local / Server

```bash
python scripts/train.py \
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

#### Kaggle

```bash
!git clone https://github.com/your-username/PRNet.git
%cd PRNet
!pip install -r requirements.txt
!python scripts/train.py \
  --backbone efficientnet_b5 \
  --image-size 512 \
  --progressive \
  --bias-softmax
```

#### Colab

```bash
!git clone https://github.com/your-username/PRNet.git
%cd PRNet
!pip install -r requirements.txt
!python scripts/train.py \
  --backbone efficientnet_b5 \
  --image-size 512 \
  --progressive \
  --bias-softmax
```

### Model Evaluation

Evaluate a trained model on the test dataset:

```bash
python scripts/eval.py \
  --model_path results/models/efficientnet_b5_stage3_512.pth \
  --backbone efficientnet_b5 \
  --image_size 640 \
  --bias_softmax \
  --data_dir data
```

#### Kaggle Example

```bash
%%bash
PYTHONUNBUFFERED=1 python -u scripts/eval.py \
  --model_path /kaggle/input/effnetb5_sz640_nbas_pr/pytorch/default/1/efficientnet_b5_SZ640_NBAS_PR_stage5_512.pth \
  --backbone efficientnet_b5 \
  --image_size 640 \
  --bias_softmax
```

### Single Image Prediction

Make predictions on individual images or batches:

```bash
python scripts/predict.py \
  --model_path results/models/efficientnet_b5_stage3_512.pth \
  --backbone efficientnet_b5 \
  --image_path sample_images/covid.jpeg \
  --image_size 512 \
  --bias_softmax \
  --topk 3
```

#### Kaggle Example

```bash
%%bash
PYTHONUNBUFFERED=1 python -u scripts/predict.py \
  --model_path /kaggle/input/effnetb5_sz640_nbas_pr/pytorch/default/1/efficientnet_b5_SZ640_NBAS_PR_stage5_512.pth \
  --backbone efficientnet_b5 \
  --image_path /kaggle/input/coviddd/covid.jpeg \
  --image_size 640 \
  --bias_softmax
```

## CLI Arguments

### train.py

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

### eval.py

| Flag             | Type   | Default | Description                                          |
| ---------------- | ------ | ------- | ---------------------------------------------------- |
| `--model_path`   | string | **required** | Path to trained model checkpoint                |
| `--backbone`     | string | **required** | Backbone model name                             |
| `--image_size`   | int    | **required** | Square input resolution                         |
| `--bias_softmax` | —      | `False` | Apply bias‑adjustable softmax                   |
| `--data_dir`     | string | `data`  | Base data dir containing train/val/test folders |
| `--batch_size`   | int    | `16`    | Batch size for evaluation                       |
| `--device`       | string | `None`  | Device to use (auto-detected if None)          |

### predict.py

| Flag             | Type   | Default | Description                                          |
| ---------------- | ------ | ------- | ---------------------------------------------------- |
| `--model_path`   | string | **required** | Path to trained model checkpoint                |
| `--backbone`     | string | **required** | Backbone model name                             |
| `--image_path`   | string | **required** | Path to a single image or folder               |
| `--image_size`   | int    | **required** | Square input resolution                         |
| `--bias_softmax` | —      | `False` | Apply bias‑adjustable softmax                   |
| `--topk`         | int    | `3`     | Number of top predictions to show               |
| `--data_dir`     | string | `data`  | Data dir to infer class names from data/train  |
| `--batch_size`   | int    | `16`    | Batch size for batch predictions                |
| `--device`       | string | `None`  | Device to use (auto-detected if None)          |

## Outputs

### Training
* **outputs/curves/** – PNG files of training/validation curves
* **outputs/results/prnet_results.csv** – CSV log of each run's metrics

### Evaluation
* **outputs/results/** – JSON evaluation reports with detailed metrics, per-class accuracy, and confusion matrices

### Prediction
* Console output with top-k predictions and confidence scores
* For batch predictions: results for each image in the specified folder

## Complete Workflow Example

1. **Train a model:**
   ```bash
   python scripts/train.py \
     --backbone efficientnet_b5 \
     --image-size 640 \
     --progressive \
     --bias-softmax \
     --epochs 30 \
     --batch-size 16 \
     --lr 1e-4 \
     --patience 7 \
     --data-dir ./data
   ```

2. **Evaluate the trained model:**
   ```bash
   python scripts/eval.py \
     --model_path outputs/models/efficientnet_b5_stage5_640.pth \
     --backbone efficientnet_b5 \
     --image_size 640 \
     --bias_softmax \
     --data_dir ./data
   ```

3. **Make predictions on new images:**
   ```bash
   python scripts/predict.py \
     --model_path outputs/models/efficientnet_b5_stage5_640.pth \
     --backbone efficientnet_b5 \
     --image_path sample_images/covid.jpeg \
     --image_size 640 \
     --bias_softmax \
     --topk 5
   ```

## Requirements

```text
torch>=1.12
torchvision
timm
pandas
numpy
matplotlib
albumentations
tqdm
```
