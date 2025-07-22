PRNet: Progressive Resolution based Network for Radiograph Classification
=========================================================================

This repository contains the official implementation of **PRNet: Progressive Resolution based Network for Radiograph based disease classification**, which achieved **96.33% accuracy** and secured **2nd position** on the EE-RDS COVID-19/Pneumonia classification challenge leaderboard.

📋 Table of Contents
--------------------

*   [Overview](https://claude.ai/chat/ff17a220-5533-4d11-a3d3-4c5645c448e0#overview)
    
*   [Key Features](https://claude.ai/chat/ff17a220-5533-4d11-a3d3-4c5645c448e0#key-features)
    
*   [Architecture](https://claude.ai/chat/ff17a220-5533-4d11-a3d3-4c5645c448e0#architecture)
    
*   [Installation](https://claude.ai/chat/ff17a220-5533-4d11-a3d3-4c5645c448e0#installation)
    
*   [Dataset Setup](https://claude.ai/chat/ff17a220-5533-4d11-a3d3-4c5645c448e0#dataset-setup)
    
*   [Usage](https://claude.ai/chat/ff17a220-5533-4d11-a3d3-4c5645c448e0#usage)
    
*   [Training](https://claude.ai/chat/ff17a220-5533-4d11-a3d3-4c5645c448e0#training)
    
*   [Results](https://claude.ai/chat/ff17a220-5533-4d11-a3d3-4c5645c448e0#results)
    
*   [Citation](https://claude.ai/chat/ff17a220-5533-4d11-a3d3-4c5645c448e0#citation)
    
*   [Contributing](https://claude.ai/chat/ff17a220-5533-4d11-a3d3-4c5645c448e0#contributing)
    
*   [License](https://claude.ai/chat/ff17a220-5533-4d11-a3d3-4c5645c448e0#license)
    

🔍 Overview
-----------

PRNet introduces a novel progressive resolution training strategy combined with Bias-Adjustable Softmax for medical image classification. The method progressively trains models from low to high resolution images, allowing the network to learn from global features first, then gradually focus on detailed local features.

### Key Contributions:

*   **Progressive Resolution Training**: Multi-stage training pipeline that starts with 256×256 images and progressively increases to 640×640
    
*   **Bias-Adjustable Softmax**: Novel probability adjustment mechanism for handling class imbalance
    
*   **State-of-the-art Results**: 96.33% accuracy on COVID-19/Pneumonia classification
    

✨ Key Features
--------------

*   🎯 **Multi-class Classification**: COVID-19, Pneumonia, and Normal cases
    
*   🔄 **Progressive Training**: 5-stage resolution training (256→380→460→512→640)
    
*   ⚖️ **Bias Adjustment**: Custom softmax for imbalanced datasets
    
*   🚀 **EfficientNet Backbone**: Supports B0-B5 architectures
    
*   📊 **Comprehensive Evaluation**: 5-fold cross-validation with detailed metrics
    
*   💾 **Model Checkpointing**: Automatic best model saving with early stopping
    

🏗️ Architecture
----------------

### Progressive Resolution Training Pipeline

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Stage 1: 256×256 → Stage 2: 380×380 → Stage 3: 460×460 → Stage 4: 512×512 → Stage 5: 640×640     ↓              ↓              ↓              ↓              ↓  EfficientNet    Transfer      Transfer      Transfer      Transfer     B5          Weights       Weights       Weights       Weights   `

### Bias-Adjustable Softmax

Our novel softmax modification adjusts class probabilities at inference:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Bias_Adjustable_Softmax = (exp(x_i) / Σ(exp(x_j)))^p   `

Where p is a class-specific exponent determined by training data distribution.

🛠️ Installation
----------------

### Prerequisites

*   Python 3.8+
    
*   CUDA-capable GPU (recommended)
    
*   16GB+ RAM
    

### Clone Repository

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   git clone https://github.com/Raahim58/PRNet_MaanzAI.git  cd PRNet_MaanzAI   `

### Install Dependencies

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Create virtual environment  python -m venv prnet_env  source prnet_env/bin/activate  # On Windows: prnet_env\Scripts\activate  # Install PyTorch (adjust CUDA version as needed)  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Install other dependencies  pip install -r requirements.txt   `

### Requirements.txt

Create a requirements.txt file with:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   torch>=1.9.0  torchvision>=0.10.0  timm>=0.6.0  albumentations>=1.3.0  opencv-python>=4.5.0  pandas>=1.3.0  numpy>=1.21.0  tqdm>=4.62.0  matplotlib>=3.4.0  scikit-learn>=1.0.0   `

📂 Dataset Setup
----------------

### Directory Structure

Organize your dataset in the following structure:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data/  ├── train/  │   ├── COVID/  │   │   ├── image1.jpg  │   │   └── ...  │   ├── PNEUMONIA/  │   │   ├── image1.jpg  │   │   └── ...  │   └── NORMAL/  │       ├── image1.jpg  │       └── ...  ├── val/  │   ├── COVID/  │   ├── PNEUMONIA/  │   └── NORMAL/  └── test/      ├── COVID/      ├── PNEUMONIA/      └── NORMAL/   `

### Dataset Information

*   **Training**: 5000+ COVID, 4000+ Pneumonia, 7000+ Normal images
    
*   **Validation**: 1432 COVID, 1000 Pneumonia, 1000 Normal images
    
*   **Test**: Unlabeled evaluation set
    

🚀 Usage
--------

### Quick Start - Single Model Training

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python run_config.py \      --backbone efficientnet_b5 \      --image-size 640 \      --progressive \      --bias-softmax \      --epochs 20 \      --batch-size 16 \      --data-dir ./data   `

### Command Line Arguments

ArgumentTypeRequiredDefaultDescription--backbonestr✅-EfficientNet variant (b0-b5)--image-sizeint✅-Final image resolution--progressiveflag❌FalseEnable progressive training--bias-softmaxflag❌FalseEnable bias-adjustable softmax--epochsint❌20Total training epochs--batch-sizeint❌16Batch size--lrfloat❌1e-4Learning rate--patienceint❌5Early stopping patience--data-dirstr❌./dataDataset directory path

🏋️ Training
------------

### Progressive Training Pipeline

The training consists of 5 stages with increasing image resolutions:

1.  **Stage 1**: 256×256 - Learn global features
    
2.  **Stage 2**: 380×380 - Intermediate details
    
3.  **Stage 3**: 460×460 - Fine-grained features
    
4.  **Stage 4**: 512×512 - High resolution details
    
5.  **Stage 5**: 640×640 - Maximum detail preservation
    

Each stage transfers weights from the previous stage and trains until convergence.

### Training Features

*   **Automatic Mixed Precision (AMP)**: Faster training with lower memory usage
    
*   **Early Stopping**: Prevents overfitting with configurable patience
    
*   **Learning Rate Scheduling**: Warmup + Cosine Annealing
    
*   **Data Augmentation**: Comprehensive augmentation pipeline using Albumentations
    

### Training Different Configurations

#### EfficientNet-B0 with Progressive Training

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python run_config.py --backbone efficientnet_b0 --image-size 640 --progressive   `

#### EfficientNet-B5 with Bias-Adjustable Softmax

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python run_config.py --backbone efficientnet_b5 --image-size 512 --bias-softmax   `

#### Full Pipeline (Best Results)

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python run_config.py \      --backbone efficientnet_b5 \      --image-size 640 \      --progressive \      --bias-softmax \      --epochs 20 \      --batch-size 16   `

### Batch Training Script

For reproducing paper results, create train\_all\_configs.py:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import subprocess  import os  configs = [      ("efficientnet_b0", 512, False, False),      ("efficientnet_b0", 512, False, True),      ("efficientnet_b0", 640, True, False),      ("efficientnet_b0", 640, True, True),      ("efficientnet_b5", 512, False, False),      ("efficientnet_b5", 512, False, True),      ("efficientnet_b5", 640, True, False),      ("efficientnet_b5", 640, True, True),  # Best config  ]  for backbone, size, progressive, bias_softmax in configs:      cmd = [          "python", "run_config.py",          "--backbone", backbone,          "--image-size", str(size),          "--epochs", "20",          "--batch-size", "16"      ]      if progressive:          cmd.append("--progressive")      if bias_softmax:          cmd.append("--bias-softmax")      print(f"Running: {' '.join(cmd)}")      subprocess.run(cmd)   `

📊 Results
----------

### Performance Comparison

BackboneImage SizeProgressiveBias-Softmax5-Fold AccVal AccTest AccEfficientNet-B0512×512NoNo96.73%96.02%95.66%EfficientNet-B0512×512NoYes97.77%96.41%96.00%EfficientNet-B0640×640YesNo97.13%96.75%95.80%EfficientNet-B0640×640YesYes97.83%97.28%96.00%EfficientNet-B5512×512NoNo97.40%97.11%96.00%EfficientNet-B5512×512NoYes97.91%97.55%96.03%EfficientNet-B5640×640YesNo97.58%97.49%96.17%**EfficientNet-B5640×640YesYes97.99%97.73%96.33%**

### Key Findings

*   **Progressive Training** consistently improves performance across all configurations
    
*   **Bias-Adjustable Softmax** helps handle class imbalance effectively
    
*   **EfficientNet-B5** with full pipeline achieves best results: **96.33% test accuracy**
    

### Output Files

Training generates several output files:

*   outputs/results/prnet\_results.csv - Comprehensive results table
    
*   outputs/curves/\[model\_name\]\_training\_curves.png - Training/validation curves
    
*   \[backbone\]\_stage\[X\]\_\[resolution\].pth - Model checkpoints for each stage
    

🔧 Data Augmentation
--------------------

The training pipeline uses comprehensive augmentation from Albumentations:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   train_transforms = A.Compose([      A.Resize(image_size, image_size),      A.HorizontalFlip(p=0.5),      A.RandomBrightnessContrast(p=0.3),      A.OneOf([          A.Blur(blur_limit=3, p=1),          A.MedianBlur(blur_limit=3, p=1),          A.GaussianBlur(blur_limit=3, p=1),          A.MotionBlur(blur_limit=3, p=1),      ], p=0.3),      A.OneOf([          A.OpticalDistortion(p=1),          A.GridDistortion(p=1),      ], p=0.3),      A.HueSaturationValue(p=0.3),      A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),      ToTensorV2(),  ])   `

🎯 Inference
------------

### Single Image Prediction

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import torch  from model import create_effnet_backbone  from utils import get_val_augmentations, bias_softmax  # Load model  model = create_effnet_backbone('efficientnet_b5', num_classes=3)  model.load_state_dict(torch.load('best_model.pth'))  model.eval()  # Preprocess image  transform = get_val_augmentations(640)  image = transform(image=cv2.imread('chest_xray.jpg'))['image']  image = image.unsqueeze(0)  # Predict with bias-adjustable softmax  with torch.no_grad():      logits = model(image)      # Apply bias adjustment (p values from paper: [1.0, 0.4, 1.6])      probs = bias_softmax(logits, [1.0, 0.4, 1.6])      prediction = probs.argmax(1)  classes = ['COVID', 'PNEUMONIA', 'NORMAL']  print(f"Prediction: {classes[prediction.item()]}")   `

### Batch Inference

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # For batch inference on test set  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  predictions = []  with torch.no_grad():      for images, _ in test_loader:          logits = model(images.to(device))          probs = bias_softmax(logits, [1.0, 0.4, 1.6])          preds = probs.argmax(1)          predictions.extend(preds.cpu().numpy())   `

📈 Monitoring Training
----------------------

### Viewing Training Progress

Training outputs real-time progress bars and metrics:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   stage 1/5: resolution 256x256, batch size 16  stage 1 @256px - training epoch 1: 100%|██████████| 625/625 [02:15<00:00, acc: 0.892]  stage 1 @256px - validating epoch 1: 100%|██████████| 156/156 [00:30<00:00, val_acc: 0.885]  Epoch 1: train_acc=0.8920, val_acc=0.8850, train_loss=0.3245, val_loss=0.3456  new best val acc: 0.8850 - saved to efficientnet_b5_stage1_256.pth   `

### Training Curves

Automatic generation of training curves showing:

*   Training/Validation Loss
    
*   Training/Validation Accuracy
    
*   Learning Rate Schedule
    

🔍 Model Architecture Details
-----------------------------

### EfficientNet Backbone

PRNet supports multiple EfficientNet variants:

ModelParametersImage SizeTop-1 Acc (ImageNet)EfficientNet-B05.3M224×22477.3%EfficientNet-B17.8M240×24079.2%EfficientNet-B29.2M260×26080.3%EfficientNet-B312M300×30081.7%EfficientNet-B419M380×38083.0%EfficientNet-B530M456×45683.7%

### Custom Components

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Bias-Adjustable Softmax Implementation  def bias_softmax(logits, exponents):      softmax_probs = F.softmax(logits, dim=1)      adjusted_probs = torch.pow(softmax_probs, torch.tensor(exponents).to(logits.device))      return adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)  # Progressive Resolution Scheduler  class ProgressiveTrainer:      def __init__(self, stages=[256, 380, 460, 512, 640]):          self.stages = stages          self.current_stage = 0      def next_stage(self):          if self.current_stage < len(self.stages) - 1:              self.current_stage += 1              return True          return False   `

🐛 Troubleshooting
------------------

### Common Issues

#### CUDA Out of Memory

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Reduce batch size  python run_config.py --batch-size 8 --backbone efficientnet_b5  # Use gradient accumulation (modify code)  # Or use smaller model  python run_config.py --backbone efficientnet_b0 --image-size 512   `

#### Training Divergence

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Lower learning rate  python run_config.py --lr 5e-5  # Increase warmup epochs (modify code)  # Add more regularization   `

#### Class Imbalance Issues

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Enable bias-adjustable softmax  python run_config.py --bias-softmax  # Adjust class weights in loss function (modify code)   `

### Performance Optimization

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Enable optimized attention (modify model creation)  model = create_effnet_backbone(backbone_name, num_classes, use_memory_efficient_attention=True)  # Use DataLoader optimizations  train_loader = DataLoader(      dataset,       batch_size=batch_size,      num_workers=4,          # Adjust based on CPU cores      pin_memory=True,        # Faster GPU transfer      prefetch_factor=2,      # Prefetch batches  )   `

📚 Additional Resources
-----------------------

### Paper References

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   @inproceedings{ahmed2021prnet,    title={PRNet: Progressive Resolution based Network for Radiograph based disease classification},    author={Ahmed, Salman and Atif, Haasha bin and Shabbir, Muhammad Bilal and Naveed, Hammad},    booktitle={2021 Ethics and Explainability for Responsible Data Science (EE-RDS)},    pages={1--6},    year={2021},    organization={IEEE}  }   `

### Related Work

*   [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
    
*   [Albumentations: Fast and Flexible Image Augmentations](https://arxiv.org/abs/1809.06839)
    
*   [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
    

🤝 Contributing
---------------

We welcome contributions! Please follow these steps:

1.  Fork the repository
    
2.  Create a feature branch (git checkout -b feature/amazing-feature)
    
3.  Commit changes (git commit -m 'Add amazing feature')
    
4.  Push to branch (git push origin feature/amazing-feature)
    
5.  Open a Pull Request
    

### Development Setup

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Install development dependencies  pip install -r requirements-dev.txt  # Run tests  python -m pytest tests/  # Code formatting  black . --line-length 88  isort .  # Type checking  mypy src/   `

📄 License
----------

This project is licensed under the MIT License - see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

🙏 Acknowledgments
------------------

*   **National Center in Big Data and Cloud Computing (NCBC)**
    
*   **National Center of Artificial Intelligence**
    
*   **National University of Computer and Emerging Sciences (NUCES-FAST), Lahore, Pakistan**
    
*   **EE-RDS Challenge Organizers**
    

📞 Contact
----------

*   **Salman Ahmed** - s.ahmed@nu.edu.pk
    
*   **Hammad Naveed** - hammad.naveed@nu.edu.pk
    

🔗 Links
--------

*   [Paper (IEEE Xplore)](https://ieeexplore.ieee.org/document/9708553)
    
*   [EE-RDS Challenge](https://eerds2021.github.io/)
    
*   [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
    

**⭐ If you find this work useful, please consider starring the repository and citing our paper!**