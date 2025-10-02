# TransApp with TST Architecture - Experimental Suite

This directory contains experimental implementations for testing the Time Series Transformer (TST) architecture integrated with TransApp, without modifying the original codebase.

## 📁 Directory Structure

```
experiments_tst/
├── RunTransAppPretraining_TST.py    # TST pretraining script
├── CompareArchitectures.py          # Architecture comparison tool
├── launcher_tst.py                  # Easy experiment launcher
└── README.md                        # This file
```

## 🏗️ Architecture Overview

### TransApp_TST Features:
- **TST Multi-Head Attention**: Advanced attention mechanism optimized for time series
- **BatchNorm vs LayerNorm**: Configurable normalization for comparison
- **Learnable Positional Encoding**: TST-style position embeddings
- **Residual Attention**: Enhanced attention with residual connections
- **Same Interface**: Drop-in replacement for original TransApp

## 🚀 Quick Start

### 1. Basic Usage

```bash
# Navigate to TST experiments directory
cd experiments_tst/

# Run single TST experiment
python RunTransAppPretraining_TST.py 0 64 BatchNorm

# Compare architectures
python CompareArchitectures.py 0 64 10

# Use interactive launcher
python launcher_tst.py
```

### 2. Command Line Options

#### TST Pretraining:
```bash
python RunTransAppPretraining_TST.py <embed_type> <dim_model> [norm_type]
```
- `embed_type`: 0 (no exogenous vars) or 1 (temporal encoding)
- `dim_model`: Model dimension (64, 96, 128, etc.)
- `norm_type`: "BatchNorm" (default) or "LayerNorm"

#### Architecture Comparison:
```bash
python CompareArchitectures.py <embed_type> <dim_model> [epochs]
```
- Compares Standard TransApp vs TST variants
- Outputs performance metrics and efficiency analysis

#### Launcher (Recommended):
```bash
python launcher_tst.py single --embed-type 0 --dim-model 64 --norm-type BatchNorm
python launcher_tst.py compare --embed-type 0 --dim-model 64 --epochs 10
python launcher_tst.py batch  # Run predefined batch experiments
```

## 📊 Experiment Types

### 1. Single TST Experiment
Tests TST architecture with specific configuration:
- Configurable normalization (BatchNorm/LayerNorm)
- Learnable positional encoding
- Enhanced attention mechanisms

### 2. Architecture Comparison
Systematic comparison between:
- **Standard TransApp**: Original architecture with LayerNorm
- **TST TransApp (BatchNorm)**: TST with BatchNorm (recommended)
- **TST TransApp (LayerNorm)**: TST with LayerNorm (compatibility)

Metrics compared:
- Training time
- Model parameters
- Final loss
- Memory usage

### 3. Batch Experiments
Predefined experiments testing:
- Different embedding types (None vs Temporal)
- Various model dimensions (64, 96, 128)
- Normalization comparisons (BatchNorm vs LayerNorm)

## 📈 Expected Results

### Performance Characteristics:
- **TST with BatchNorm**: Best for time series, faster convergence
- **TST with LayerNorm**: Compatible with original, stable training
- **Parameter Count**: ~10-20% increase due to enhanced attention
- **Training Time**: Similar or slightly faster due to better convergence

### When to Use TST:
- ✅ Time series classification tasks
- ✅ Need better temporal pattern recognition
- ✅ Want improved model performance
- ✅ Have sufficient computational resources

## 📁 Output Structure

Results are saved in separate directories to avoid conflicts:

```
results/
├── TransAppPretrained/           # Original TransApp results
└── TransAppPretrained_TST/       # TST results
    ├── None/                     # No exogenous variables
    │   ├── TransApp_TST_64_BatchNorm.pt
    │   └── TransApp_TST_64_LayerNorm.pt
    └── Embed/                    # With temporal encoding
        ├── TransApp_TST_64_BatchNorm.pt
        └── TransApp_TST_64_LayerNorm.pt

results/Architecture_Comparison/   # Comparison results
├── Standard/                     # Standard TransApp models
├── TST/                         # TST models
└── comparison_*.json            # Comparison reports
```

## 🔧 Advanced Usage

### Custom TST Configuration:
```python
from src.TransAppModel.TransApp_TST import get_transapp_tst_model

model = get_transapp_tst_model(
    m=1, win=1024, dim_model=64,
    mode="pretraining",
    use_tst_pos_encoding=True,
    norm="BatchNorm",
    res_attention=True
)
```

### Integration with Existing Code:
```python
# Replace in your existing scripts:
# from src.TransAppModel.TransApp import TransApp
from src.TransAppModel.TransApp_TST import TransApp_TST as TransApp

# Use normally - same interface!
model = TransApp(max_len=1024, c_in=1, mode="pretraining", ...)
```

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size in scripts
   dict_params = {'batch_size': 8, ...}  # Instead of 16
   ```

2. **Import Errors**:
   ```bash
   # Ensure you're in experiments_tst directory
   cd experiments_tst/
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

3. **Missing Dependencies**:
   ```bash
   # Install missing packages
   pip install torch torchvision sklearn pandas numpy
   ```

## 📋 Experimental Validation

To validate TST improvements:

1. **Run Comparison**: 
   ```bash
   python launcher_tst.py compare --embed-type 0 --dim-model 64 --epochs 20
   ```

2. **Check Results**: 
   ```bash
   # Look at comparison JSON files in results/Architecture_Comparison/
   cat results/Architecture_Comparison/comparison_*.json
   ```

3. **Analyze Performance**:
   - Lower final loss = better pretraining
   - Similar training time = efficient
   - Stable convergence = robust

## 🎯 Next Steps

After successful TST pretraining:

1. **Fine-tuning**: Use TST pretrained models for classification tasks
2. **Evaluation**: Compare classification performance on appliance detection
3. **Optimization**: Tune TST-specific hyperparameters
4. **Integration**: Incorporate successful configurations into main pipeline

## 📞 Support

For issues or questions:
1. Check the comparison results in JSON format
2. Review training logs for error messages  
3. Verify GPU memory usage during training
4. Compare parameter counts between architectures

---

**Note**: This experimental suite is designed to be completely independent of the original TransApp codebase, ensuring no interference with existing functionality.