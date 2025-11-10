# PatchTST Phase 1 Experiments 🧬⚡

This directory contains the complete experimental framework for **PatchTST (Patch Time-Series Transformer)** applied to CER appliance detection without pretraining.

## 📋 Overview

PatchTST is a state-of-the-art time series transformer that uses a patching mechanism similar to Vision Transformers (ViTs). This implementation focuses on **Phase 1: Direct Classification** without pretraining, providing a comprehensive baseline for time series appliance detection.

### Key Features
- 🔬 **Comprehensive hyperparameter optimization** with grid search
- 📊 **Detailed logging and JSON metrics storage**
- 🎯 **Multiple evaluation metrics** (F1-macro, accuracy, ROC-AUC)
- 📈 **Statistical validation** with multiple random seeds
- 🖥️ **GPU acceleration** support (CUDA)
- 📊 **Advanced result analysis** and visualization

## 🗂️ File Structure

```
experiments_patch/
├── RunPatchTSTClassification.py     # Main experiment runner
├── AnalyzePatchTSTResults.py        # Comprehensive results analysis
├── LaunchPatchTSTExperiments.sh     # Automated experiment launcher
├── requirements_patch.txt           # Additional dependencies
└── README.md                        # This documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# From the TransApp root directory
cd /home/user/vansh/ISP/TransApp

# Create virtual environment (if not exists)
python -m venv venv
source venv/bin/activate

# Install base requirements
pip install -r requirements.txt

# Install PatchTST requirements
pip install -r experiments_patch/requirements_patch.txt
```

### 2. Run Experiments

#### Option A: Using the Launcher Script (Recommended)
```bash
# Make executable (if needed)
chmod +x experiments_patch/LaunchPatchTSTExperiments.sh

# Run interactive launcher
./experiments_patch/LaunchPatchTSTExperiments.sh
```

#### Option B: Direct Python Execution
```bash
cd experiments_patch
python RunPatchTSTClassification.py
```

### 3. Analyze Results
```bash
cd experiments_patch
python AnalyzePatchTSTResults.py

# Or analyze specific case
python AnalyzePatchTSTResults.py --case cooker_case
```

## 🎯 Experimental Focus

### Key Parameters (Similar to TST Approach)
This implementation focuses on the **most impactful hyperparameters** for time series transformers:

1. **Hidden Size (64, 128, 256)**: Inner model dimension affecting capacity
2. **Attention Heads (4, 8, 16)**: Multi-head attention parallelization
3. **Layer Depth (2, 3, 4)**: Transformer network depth

### Fixed Parameters (Optimal Defaults)
- **Patch Length**: 16 (optimal for 15-min intervals)
- **Stride**: 8 (50% overlap for good coverage)
- **Dropout**: 0.1 (standard regularization)
- **Learning Rate**: 1e-4 (proven optimal for transformers)
- **Embeddings**: None (simplified approach)

### Statistical Validation
- **Multiple Seeds**: 3 random seeds (0, 1, 2) for reproducibility
- **Total Experiments**: 81 per case (manageable and focused)
- **Comparison Ready**: Same parameter focus as TST experiments

### Experiment Types

| Option | Description | Experiments | Duration |
|--------|-------------|-------------|----------|
| **Quick Test** | Single case, minimal grid | 1 | 2-5 minutes |
| **Single Case Focused** | One case, key parameters | 81 | 2-4 hours |
| **All Cases Quick** | All cases, minimal grid | 5 | 10-20 minutes |
| **All Cases Focused** | Complete focused evaluation | 405 | 8-16 hours |

### Hyperparameter Grid

#### Focused Grid (Similar to TST Experiments)
```python
{
    # Key architecture parameters (most impactful)
    'hidden_size': [64, 128, 256],         # Inner model dimension
    'num_attention_heads': [4, 8, 16],    # Multi-head attention  
    'num_hidden_layers': [2, 3, 4],       # Transformer depth
    
    # Fixed optimal parameters
    'patch_length': [16],                  # Standard patch size
    'stride': [8],                         # 50% overlap
    'dropout': [0.1],                      # Standard dropout
    'learning_rate': [1e-4],               # Standard learning rate
    'embed_type': [0],                     # No temporal embeddings
    'random_seed': [0, 1, 2]               # Statistical validation
}
# Total: 3 × 3 × 3 × 3 = 81 experiments per case
```

#### Quick Test Grid
```python
{
    'hidden_size': [128],                  # Medium size
    'num_attention_heads': [8],           # Standard heads
    'num_hidden_layers': [3],             # Medium depth
    'patch_length': [16],                 # Standard patch
    'stride': [8],                        # Standard stride
    'dropout': [0.1],                     # Standard dropout
    'learning_rate': [1e-4],              # Standard LR
    'embed_type': [0],                    # No embeddings
    'random_seed': [0]                    # Single seed
}
# Total: 1 experiment per case
```

## 📊 Model Architecture

### PatchTST Configuration
```python
@dataclass
class PatchTSTHyperparameters:
    context_length: int = 96          # 24h @ 15-min intervals
    patch_length: int = 16            # Patch size
    stride: int = 8                   # Overlap between patches
    hidden_size: int = 128            # Embedding dimension
    num_hidden_layers: int = 3        # Transformer layers
    num_attention_heads: int = 8      # Multi-head attention
    dropout: float = 0.1              # Regularization
    learning_rate: float = 1e-4       # Optimizer learning rate
    epochs: int = 15                  # Training epochs
    batch_size: int = 32              # Batch size
```

### Key Model Features
- **Patching Mechanism**: Converts time series into overlapping patches
- **Positional Embeddings**: Maintains temporal order information
- **Classification Head**: Binary appliance detection (ON/OFF)
- **Early Stopping**: Prevents overfitting with validation monitoring
- **Learning Rate Scheduling**: Adaptive learning rate reduction

## 📈 Evaluation Metrics

### Primary Metrics
- **F1-Macro**: Balanced performance across both classes
- **F1-Binary**: Performance on appliance detection (positive class)
- **Accuracy**: Overall correct classifications
- **ROC-AUC**: Discrimination capability

### Secondary Metrics
- **Precision/Recall**: Detailed classification performance
- **Confusion Matrix**: Error analysis
- **Training Metrics**: Loss curves, epochs, convergence

## 📁 Results Structure

```
results/PatchTSTResults/
├── patchtst_cooker_case_phase1_YYYYMMDD_HHMMSS.json
├── patchtst_dishwasher_case_phase1_YYYYMMDD_HHMMSS.json
└── ...

results/PatchTSTAnalysis/
├── patchtst_comprehensive_analysis.png
├── patchtst_case_comparison.png
├── patchtst_detailed_results.csv
└── patchtst_summary_report_YYYYMMDD_HHMMSS.txt

logs/patchtst/
├── patchtst_phase1_cooker_case_YYYYMMDD_HHMMSS.log
└── ...
```

### JSON Result Schema
```json
{
  "experiment_info": {
    "experiment_name": "patchtst_phase1_cooker_case",
    "case_name": "cooker_case",
    "timestamp": "2024-10-XX...",
    "total_experiments": 729,
    "model_type": "PatchTST",
    "phase": "Phase_1_Direct_Classification"
  },
  "hyperparameter_grid": {...},
  "all_results": [
    {
      "hyperparameters": {...},
      "training_info": {
        "training_history": [...],
        "best_val_loss": 0.XX,
        "total_epochs": XX
      },
      "test_metrics": {
        "f1_macro": 0.XXXX,
        "f1_binary": 0.XXXX,
        "accuracy": 0.XXXX,
        "roc_auc": 0.XXXX,
        "confusion_matrix": [[...]]
      },
      "model_info": {
        "total_parameters": XXXXX,
        "trainable_parameters": XXXXX
      }
    }
  ]
}
```

## 🎯 Expected Performance

### Baseline Expectations (Phase 1)
- **F1-Macro**: 0.65 - 0.75 (target range)
- **F1-Binary**: 0.70 - 0.80 (appliance detection)
- **ROC-AUC**: 0.75 - 0.85 (discrimination)
- **Training Time**: 2-5 minutes per experiment (GPU)

### Case-Specific Variations
- **High Performance**: `cooker_case`, `waterheater_case`
- **Medium Performance**: `dishwasher_case`, `tumbledryer_case`
- **Challenging**: `tv_greater21inch_case`, `laptopcomputer_case`

## 🔍 Analysis Features

### Automated Analysis
```bash
python AnalyzePatchTSTResults.py
```

**Generated Outputs:**
- 📊 **Performance visualizations** (box plots, correlations, distributions)
- 📈 **Hyperparameter sensitivity analysis**
- 📄 **Comprehensive text report** with statistical summaries
- 💾 **CSV export** for further analysis
- 🏆 **Best configuration identification**

### Key Analysis Components
1. **Performance Distribution**: Cross-case comparison
2. **Hyperparameter Importance**: Correlation analysis
3. **Model Complexity Analysis**: Parameter vs performance
4. **Training Efficiency**: Epoch requirements and convergence
5. **Temporal Embeddings Impact**: With/without comparison

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Install missing packages
pip install transformers torch scikit-learn numpy pandas matplotlib seaborn
```

#### 2. CUDA Out of Memory
```python
# Reduce batch size in hyperparameters
batch_size: int = 16  # instead of 32
```

#### 3. No Results Found
```bash
# Check results directory
ls -la results/PatchTSTResults/

# Verify experiments completed
tail -f logs/patchtst/*.log
```

#### 4. Slow Training
- **Use GPU**: Ensure CUDA is available
- **Reduce Grid**: Use quick test grid first
- **Adjust Epochs**: Reduce from 15 to 10 for testing

### Performance Optimization
```python
# Quick testing configuration
hyperparameter_grid = {
    'patch_length': [16],        # Single value
    'hidden_size': [128],        # Medium size
    'epochs': [10],              # Fewer epochs
    'random_seed': [0]           # Single seed
}
```

## 🔗 Integration with TransApp

### Data Pipeline Compatibility
- Uses existing `CER_get_data_case()` function
- Compatible with CER dataset structure
- Maintains preprocessing consistency

### Result Format Consistency
- JSON structure matches TST experiments
- Metrics align with existing evaluation framework
- Visualization follows established patterns

## 📚 References

### PatchTST Paper
- **Title**: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
- **Authors**: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam
- **Venue**: ICLR 2023
- **Key Innovation**: Patch-based approach for time series

### Implementation Details
- **Framework**: Hugging Face Transformers
- **Base Model**: `PatchTSTForClassification`
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout + early stopping

## 🚧 Future Phases

### Phase 2: Pretraining (Planned)
- Masked patch modeling
- Self-supervised pretraining
- Transfer learning evaluation

### Phase 3: Advanced Features (Planned)
- Multi-variate input support
- Attention visualization
- Ensemble methods
- Real-time inference

## 📞 Support

For issues or questions:
1. Check the logs in `logs/patchtst/`
2. Verify requirements installation
3. Review hyperparameter configurations
4. Check GPU availability and memory

**Happy experimenting with PatchTST! 🚀**