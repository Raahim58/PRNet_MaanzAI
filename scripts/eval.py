"""
Evaluate a trained EfficientNet model on the test split and optionally apply Bias-Adjustable Softmax (BAS).

Usage examples:
    python eval.py --model_path results/models/efficientnet_b5_stage3_512.pth \
                   --backbone efficientnet_b5 \
                   --image_size 640 \
                   --bias_softmax \
                   --data_dir data \

Output: prints summary, per-class accuracy and saves JSON report under outputs/results/
"""
import os, random, shutil
import json
import argparse
import time
from glob import glob
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import timm

# 0.1 fix random seeds for reproducibility
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_global_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 1.1 LOAD DATASET
import kagglehub
# Download latest version
path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
print("Path to dataset files:", path)

# set the base paths
base_dataset_path = os.path.join(path, "COVID-19_Radiography_Dataset")
output_base = "data"
splits = ['train', 'val', 'test']
split_ratio = [0.8, 0.1, 0.1]  # 80% train, 10% val, 10% test
classes = ['COVID', 'Normal', 'Viral Pneumonia', 'Lung_Opacity']

# create output directories
for split in splits:
    for class_name in classes:
        os.makedirs(os.path.join(output_base, split, class_name), exist_ok=True)

# function to split and copy images
for class_name in classes:
    source_dir = os.path.join(base_dataset_path, class_name, "images")
    all_images = os.listdir(source_dir)
    random.shuffle(all_images)
    n_total = len(all_images)
    n_train = int(split_ratio[0] * n_total)
    n_val = int(split_ratio[1] * n_total)
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]

    def copy_images(image_list, split_name):
        for image in tqdm(image_list, desc=f"{split_name} - {class_name}"):
            src = os.path.join(source_dir, image)
            dst = os.path.join(output_base, split_name, class_name, image)
            shutil.copyfile(src, dst)
    copy_images(train_images, "train")
    copy_images(val_images, "val")
    copy_images(test_images, "test")

# 1.2 CREATE DATASET
class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, class_names, transform=None):
        self.image_paths = []
        self.labels = []
        for idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(('.png','jpg','jpeg')):
                    self.image_paths.append(fpath)
                    self.labels.append(idx)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image) # PIL → H×W×C numpy array
        if self.transform:
            augmented = self.transform(image=image) # must pass as keyword
            image = augmented['image'] # grab the transformed tensor
        return image, label

# 1.3 TRANSFORMATIONS
def get_train_augmentations(image_size: int):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        # A.RandomBrightness(limit=0.2, p=0.3), # changes image brightness to mimic lighting variation # search for right import
        # A.RandomContrast(limit=0.2, p=0.3), # modifies contrast to handle visual differences
        A.Blur(blur_limit=3, p=0.2), # general softening of the image
        A.MedianBlur(blur_limit=3, p=0.2), # removes noise while keeping edges sharp
        A.GaussianBlur(blur_limit=(3,5), p=0.2), # natural smooth blur like out-of-focus camera
        A.MotionBlur(blur_limit=5, p=0.2), # simulates camera shake or patient movement
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3), # lens-like warping of image
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3), # distorts image with grid pattern
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3), # adjusts tint, saturation, brightness
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5), # shifts, zooms, rotates image slightly
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_augmentations(image_size: int):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

# 1.4 VISUALIZE IMAGES
def imshow(img_tensor, mean, std):
    """
    img_tensor: C×H×W torch Tensor, normalized
    mean, std: sequences of length C
    returns: H×W×C numpy array in [0,1]
    """
    # move to C×H×W numpy
    img = img_tensor.cpu().numpy()
    # unnormalize per channel
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    # transpose to H×W×C
    img = np.transpose(img, (1,2,0))
    # clip to valid range
    return np.clip(img, 0, 1)

def show_batch(dataset, class_names, num_samples=16):
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(loader))

    # constants – must match your Normalize()
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    n = int(num_samples**0.5)  # for a 4×4 grid if num_samples=16
    fig, axes = plt.subplots(n, n, figsize=(n*4, n*4))

    for ax, img_t, lab in zip(axes.flatten(), images, labels):
        img = imshow(img_t, mean, std)
        ax.imshow(img)
        ax.set_title(class_names[lab], fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# 2. MODEL
class EfficientNetClassifier(nn.Module):
    def __init__(self, name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=True, features_only=True)
        in_ch = self.backbone.feature_info[-1]["num_chs"]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_ch, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]  # take the deepest feature map only
        out = self.head(feats)
        return out

def create_effnet_backbone(name, num_classes):
    return EfficientNetClassifier(name, num_classes).to(device)

# 3. BAS
def bias_softmax(logits, exponents):
    """
    applies bias-adjustable softmax to modify class probabilities

    how it works:
    1. get normal softmax probabilities (0 to 1, sum to 1)
    2. raise each class probability to its own exponent
       - exponent > 1: makes high probs higher, low probs lower (sharpens)
       - exponent < 1: makes distribution more uniform (smooths)
       - exponent = 1: no change (normal softmax)
    3. renormalize so they still sum to 1

    example:
    - normal probs: [0.6, 0.3, 0.1]
    - exponents: [1.0, 0.5, 2.0]
    - after power: [0.6^1.0, 0.3^0.5, 0.1^2.0] = [0.6, 0.55, 0.01]
    - after normalize: [0.52, 0.47, 0.01] (approximately)

    args:
        logits: model outputs before softmax, shape (batch_size, num_classes)
        exponents: list of exponents for each class, length = num_classes

    returns:
        adjusted probabilities, same shape as input
    """
    # step 1: get normal softmax probabilities
    normal_probs = torch.softmax(logits, dim=-1)

    # step 2: raise each class to its exponent
    # convert exponents to tensor on same device as probabilities
    exp_tensor = torch.tensor(exponents, device=normal_probs.device)
    adjusted_probs = normal_probs ** exp_tensor

    # step 3: renormalize so each row sums to 1
    # sum across classes (dim=-1) and keep dimension for broadcasting
    normalized = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)

    return normalized

def search_best_exponents(validation_loader, trained_model, search_range=(0.1, 3.0), step_size=0.1):
    """
    finds the best exponent for each class using grid search

    the idea:
    - for each class, try different exponent values
    - pick the exponent that gives highest validation accuracy
    - do this one class at a time (greedy search)

    why this works:
    - if a class is being under-predicted, use exponent > 1 to boost it
    - if a class is being over-predicted, use exponent < 1 to dampen it
    - the paper found exponents [1.0, 0.4, 1.6] for covid/pneumonia/normal
      meaning pneumonia was over-predicted so they dampened it with 0.4

    args:
        validation_loader: dataloader for validation set
        trained_model: the model to tune (should be already trained)
        search_range: (min_exp, max_exp) to search
        step_size: how fine-grained the search is

    returns:
        list of best exponents for each class
    """
    print("collecting validation predictions for bias tuning...")

    # put model in evaluation mode
    trained_model.eval()

    # collect all validation data predictions in one go
    # this is more efficient than running inference multiple times
    all_logits = []  # model outputs (before softmax)
    all_true_labels = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            logits = trained_model(images)
            all_logits.append(logits.cpu()) # move back to CPU for storage (saves GPU memory)
            # labels might already be tensors, but ensure they're tensors
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            all_true_labels.append(labels)

    # combine all batches into single tensors
    all_logits = torch.cat(all_logits, dim=0)  # shape: (total_samples, num_classes)
    all_true_labels = torch.cat(all_true_labels, dim=0)  # shape: (total_samples,)
    num_classes = all_logits.size(1)
    print(f"collected {all_logits.size(0)} samples with {num_classes} classes")

    # start with default exponents (no bias adjustment)
    best_exponents = [1.0] * num_classes

    # optimize each class exponent one by one
    for class_idx in range(num_classes):
        print(f"tuning exponent for class {class_idx}...")

        best_accuracy = 0.0
        best_exponent = 1.0

        # try different exponent values for this class
        search_values = np.arange(search_range[0], search_range[1] + step_size/2, step_size)

        for exp_value in search_values:
            # create exponent list with only this class modified
            test_exponents = best_exponents.copy()
            test_exponents[class_idx] = exp_value

            # apply bias-adjustable softmax with these exponents
            adjusted_probs = bias_softmax(all_logits, test_exponents)

            # get predictions (highest probability class)
            predictions = adjusted_probs.argmax(dim=1)

            # calculate accuracy
            correct = (predictions == all_true_labels).float()
            accuracy = correct.mean().item()

            # keep track of best exponent for this class
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_exponent = exp_value

        # update the best exponent for this class
        best_exponents[class_idx] = best_exponent
        print(f"  class {class_idx}: best_exponent={best_exponent:.2f}, accuracy={best_accuracy:.4f}")

    print(f"final best exponents: {best_exponents}")
    return best_exponents


# 4. EVALUATE FUNCTION
def evaluate(model, dataloader, device, exponents, criterion=None, bias=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_classes = 4
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc='Evaluating'):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
            # apply BAS if needed
            if bias:
                logits = bias_softmax(logits, exponents)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            # if n_classes is None:
            #     n_classes = preds.size(1)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if (criterion is not None and total>0) else None
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # per-class accuracy
    per_class_acc = {}
    for c in range(n_classes):
        mask = (all_labels == c)
        if mask.sum().item() == 0:
            per_class_acc[c] = None
        else:
            per_class_acc[c] = (all_preds[mask] == all_labels[mask]).sum().item() / mask.sum().item()

    return {
        'accuracy': accuracy,
        'avg_loss': avg_loss,
        'per_class_accuracy': per_class_acc,
        'total_samples': int(total)
    }

# 5. MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--image_size', type=int, required=True)
    parser.add_argument("--bias_softmax",action="store_true")
    parser.add_argument('--data_dir', default='data', help='base data dir containing train/val/test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # discover classes from train folder
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'train')
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)
    print('Detected classes:', class_names)

    # build model and load weights
    model = EfficientNetClassifier(args.backbone, num_classes).to(device)
    state = torch.load(args.model_path, map_location=device)
    # If checkpoint contains 'state_dict' key, handle it
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    print(f'Loaded model weights from {args.model_path}')

    # bias-adjustable softmax tuning (only if requested)
    best_exps = [1.0] * num_classes  # default exponents
    if args.bias_softmax:
        print("tuning bias-adjustable softmax on validation set...")
        final_val_loader = DataLoader(
            ChestXRayDataset(f"{data_dir}/val", class_names, transform=get_val_augmentations(args.image_size)),
            batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        best_exps = search_best_exponents(final_val_loader, model)
        print(f"best bias exponents: {best_exps}")

    # test loader
    test_dir = os.path.join(data_dir, 'test')
    val_transform = get_val_augmentations(args.image_size)
    test_dataset = ChestXRayDataset(test_dir, class_names, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    t0 = time.time()
    report = evaluate(model, test_loader, device, exponents=best_exps, criterion=criterion, bias=args.bias_softmax)
    elapsed = time.time() - t0

    print('\nEvaluation summary:')
    print(f"  Total samples: {report['total_samples']}")
    if report['avg_loss'] is not None:
        print(f"  Avg test loss: {report['avg_loss']:.4f}")
    print(f"  Test accuracy: {report['accuracy']*100:.2f}%")
    print('  Per-class accuracy:')
    for idx, acc in report['per_class_accuracy'].items():
        name = class_names[idx]
        print(f"    {idx} - {name}: {acc if acc is None else f'{acc*100:.2f}%'}")
    print(f'Evaluation time: {elapsed:.1f}s')

    os.makedirs('outputs/results', exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.model_path))[0]
    outpath = f'outputs/results/eval_{basename}.json'

    meta = {
        'model': args.model_path,
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'bias-softmax': args.bias_softmax,
        'class_names': class_names,
        'report': report,
        'elapsed_s': elapsed
    }
    with open(outpath, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved evaluation report to {outpath}')


if __name__ == '__main__':
    main()
