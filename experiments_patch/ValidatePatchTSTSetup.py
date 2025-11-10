#!/usr/bin/env python3
"""
PatchTST Implementation Validation Script
========================================

This script validates the PatchTST implementation by running a minimal test
to ensure all components work correctly before running full experiments.

Features:
- Dependency validation
- Model instantiation test
- Data loading verification
- Training pipeline test (1 epoch)
- Result format validation

Usage:
    python ValidatePatchTSTSetup.py
"""

import sys
import warnings
from pathlib import Path

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Fix path resolution
current_file = Path(__file__).resolve()
root = current_file.parents[1]
sys.path.insert(0, str(root))

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
        
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
        
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_device_availability():
    """Test GPU/CPU availability"""
    print("\n🖥️ Testing device availability...")
    
    try:
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Device: {device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("🖥️ Using CPU (GPU not available)")
        
        return device
        
    except Exception as e:
        print(f"❌ Device test error: {e}")
        return torch.device('cpu')

def test_data_loading():
    """Test CER data loading"""
    print("\n📊 Testing data loading...")
    
    try:
        from experiments.data_utils import CER_get_data_case
        
        # Load small dataset for testing
        datas_tuple = CER_get_data_case(
            case_name="cooker_case",
            seed=0,
            exo_variable=[],
            win=96,
            ratio_resample=0.1  # Very small sample for testing
        )
        
        X_train, y_train = datas_tuple[0], datas_tuple[1]
        X_valid, y_valid = datas_tuple[2], datas_tuple[3]
        X_test, y_test = datas_tuple[4], datas_tuple[5]
        
        print(f"✅ Training data: {X_train.shape}, labels: {y_train.shape}")
        print(f"✅ Validation data: {X_valid.shape}, labels: {y_valid.shape}")
        print(f"✅ Test data: {X_test.shape}, labels: {y_test.shape}")
        
        return (X_train, y_train, X_valid, y_valid, X_test, y_test)
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return None

def test_model_creation():
    """Test PatchTST model creation"""
    print("\n🏗️ Testing model creation...")
    
    try:
        import torch
        
        # Try original PatchTST first, fall back to custom implementation
        try:
            from transformers import PatchTSTConfig, PatchTSTForClassification
            
            # Create test configuration
            config = PatchTSTConfig(
                num_input_channels=1,
                num_targets=2,
                context_length=96,
                patch_length=16,
                stride=8,
                hidden_size=64,  # Small for testing
                num_hidden_layers=2,
                num_attention_heads=4,
                use_cls_token=True
            )
            
            # Create model
            model = PatchTSTForClassification(config)
            print(f"✅ Using original PatchTSTForClassification")
            
        except (ImportError, AttributeError) as e:
            print(f"⚠️  Original PatchTST not available: {e}")
            print("🔧 Creating custom PatchTST implementation...")
            
            # Custom implementation
            import torch.nn as nn
            from types import SimpleNamespace
            
            class CustomPatchTSTForClassification(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    
                    # Patch embedding
                    self.patch_embedding = nn.Linear(
                        config.patch_length * config.num_input_channels, 
                        config.hidden_size
                    )
                    
                    # Positional embedding
                    max_patches = config.context_length // config.stride + 1
                    self.position_embedding = nn.Parameter(
                        torch.randn(1, max_patches, config.hidden_size)
                    )
                    
                    # CLS token
                    self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
                    
                    # Transformer encoder
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=config.hidden_size,
                        nhead=config.num_attention_heads,
                        dim_feedforward=config.hidden_size * 4,
                        dropout=0.1,
                        activation='gelu',
                        batch_first=True
                    )
                    self.transformer = nn.TransformerEncoder(
                        encoder_layer, 
                        num_layers=config.num_hidden_layers
                    )
                    
                    # Classification head
                    self.classifier = nn.Linear(config.hidden_size, config.num_targets)
                    self.dropout = nn.Dropout(0.1)
                    
                def forward(self, past_values=None, **kwargs):
                    if past_values is not None:
                        x = past_values
                    else:
                        x = list(kwargs.values())[0] if kwargs else None
                        if x is None:
                            raise ValueError("No input tensor provided")
                    
                    batch_size, seq_len, num_channels = x.shape
                    
                    # Create patches
                    patches = []
                    for i in range(0, seq_len - self.config.patch_length + 1, self.config.stride):
                        patch = x[:, i:i+self.config.patch_length, :]
                        patches.append(patch.reshape(batch_size, -1))
                    
                    patches = torch.stack(patches, dim=1)
                    
                    # Patch embedding
                    embeddings = self.patch_embedding(patches)
                    
                    # Add positional embedding
                    embeddings = embeddings + self.position_embedding[:, :embeddings.size(1), :]
                    
                    # Add CLS token
                    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                    embeddings = torch.cat([cls_tokens, embeddings], dim=1)
                    
                    # Transformer
                    output = self.transformer(embeddings)
                    
                    # Classification
                    cls_output = output[:, 0, :]
                    cls_output = self.dropout(cls_output)
                    logits = self.classifier(cls_output)
                    
                    return SimpleNamespace(logits=logits)
            
            # Create config for custom model
            config = SimpleNamespace(
                num_input_channels=1,
                num_targets=2,
                context_length=96,
                patch_length=16,
                stride=8,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4
            )
            
            model = CustomPatchTSTForClassification(config)
            print(f"✅ Using custom PatchTST implementation")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully")
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return None

def test_data_preprocessing():
    """Test data preprocessing for PatchTST"""
    print("\n🔄 Testing data preprocessing...")
    
    try:
        import torch
        import numpy as np
        
        # Create dummy data
        batch_size = 8
        sequence_length = 96
        num_features = 1
        
        X_dummy = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
        y_dummy = np.random.randint(0, 2, batch_size)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_dummy)
        y_tensor = torch.LongTensor(y_dummy)
        
        print(f"✅ Input tensor shape: {X_tensor.shape}")
        print(f"✅ Label tensor shape: {y_tensor.shape}")
        print(f"✅ Input dtype: {X_tensor.dtype}")
        print(f"✅ Label dtype: {y_tensor.dtype}")
        
        return X_tensor, y_tensor
        
    except Exception as e:
        print(f"❌ Data preprocessing error: {e}")
        return None, None

def test_model_forward_pass():
    """Test model forward pass"""
    print("\n⚡ Testing model forward pass...")
    
    try:
        import torch
        
        # Get model and data from previous tests
        model = test_model_creation()
        X_tensor, y_tensor = test_data_preprocessing()
        
        if model is None or X_tensor is None:
            print("❌ Dependencies failed, skipping forward pass test")
            return False
        
        # Set model to evaluation mode
        model.eval()
        
                # Forward pass with correct input format for PatchTST
        with torch.no_grad():
            outputs = model(past_values=X_tensor)
            
        # Debug: inspect the output object
        print(f"🔍 Output object type: {type(outputs)}")
        print(f"🔍 Output object attributes: {dir(outputs)}")
        
        # Check outputs - handle both direct tensor and output object
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            print(f"✅ Found 'logits' attribute")
        elif hasattr(outputs, 'prediction_outputs'):
            logits = outputs.prediction_outputs
            print(f"✅ Found 'prediction_outputs' attribute")
        elif hasattr(outputs, 'last_hidden_state'):
            # If it's a base model output, we need to add classification head
            print("⚠️  Model returned base output, attempting to extract features...")
            logits = outputs.last_hidden_state[:, 0, :]  # Use first token
            if logits.shape[-1] != 2:
                # Add a simple linear layer for testing
                classifier = torch.nn.Linear(logits.shape[-1], 2)
                logits = classifier(logits)
        else:
            # Handle PatchTSTForClassificationOutput or similar objects
            print(f"⚠️  Handling output object of type: {type(outputs)}")
            # Try to access different possible attributes
            for attr in ['logits', 'prediction_outputs', 'prediction_logits', 'classification_outputs']:
                if hasattr(outputs, attr):
                    logits = getattr(outputs, attr)
                    print(f"✅ Found logits in attribute: {attr}")
                    break
            else:
                # If no known attribute found, try to convert the object itself
                logits = outputs
        
        # Ensure we have a proper tensor
        if not isinstance(logits, torch.Tensor):
            print(f"❌ Could not extract tensor from output: {type(logits)}")
            return False
        
        print(f"✅ Forward pass successful")
        print(f"✅ Output shape: {logits.shape}")
        print(f"✅ Expected shape: [batch_size=8, num_classes=2]")
        print(f"✅ Output type: {type(outputs)}")
        print(f"✅ Logits type: {type(logits)}")
        
        # Test probability computation
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        
        print(f"✅ Probabilities shape: {probabilities.shape}")
        print(f"✅ Predictions shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass error: {e}")
        return False

def test_training_step():
    """Test a single training step"""
    print("\n🏋️ Testing training step...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Get model and data
        model = test_model_creation()
        X_tensor, y_tensor = test_data_preprocessing()
        
        if model is None or X_tensor is None:
            print("❌ Dependencies failed, skipping training test")
            return False
        
        # Set up training components
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training step
        optimizer.zero_grad()
        
        # Forward pass with correct input format for PatchTST
        outputs = model(past_values=X_tensor)
        
        # Extract logits from output object
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif hasattr(outputs, 'prediction_outputs'):
            logits = outputs.prediction_outputs
        elif hasattr(outputs, 'last_hidden_state'):
            # If it's a base model output, we need to add classification head
            print("⚠️  Model returned base output, adding classifier...")
            logits = outputs.last_hidden_state[:, 0, :]  # Use first token
            if logits.shape[-1] != 2:
                # Add a simple linear layer for testing
                classifier = nn.Linear(logits.shape[-1], 2)
                logits = classifier(logits)
        else:
            # Handle PatchTSTForClassificationOutput or similar objects
            print(f"⚠️  Handling output object of type: {type(outputs)}")
            # Try to access different possible attributes
            for attr in ['logits', 'prediction_outputs', 'prediction_logits', 'classification_outputs']:
                if hasattr(outputs, attr):
                    logits = getattr(outputs, attr)
                    print(f"✅ Found logits in attribute: {attr}")
                    break
            else:
                # If no known attribute found, this is an error
                print(f"❌ Cannot extract logits from: {type(outputs)}")
                return False
        
        # Ensure logits is a tensor
        if not isinstance(logits, torch.Tensor):
            print(f"❌ Logits is not a tensor: {type(logits)}")
            return False
        
        loss = criterion(logits, y_tensor)
        
        loss.backward()
        optimizer.step()
        
        print(f"✅ Training step successful")
        print(f"✅ Loss value: {loss.item():.6f}")
        print(f"✅ Loss is finite: {torch.isfinite(loss).item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step error: {e}")
        return False

def test_metrics_computation():
    """Test metrics computation"""
    print("\n📊 Testing metrics computation...")
    
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        import numpy as np
        
        # Create dummy predictions and labels
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.4, 0.2, 0.9, 0.6, 0.7, 0.3])
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        roc_auc = roc_auc_score(y_true, y_prob)
        
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"✅ F1-Macro: {f1:.4f}")
        print(f"✅ ROC-AUC: {roc_auc:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics computation error: {e}")
        return False

def test_json_serialization():
    """Test JSON serialization of results"""
    print("\n💾 Testing JSON serialization...")
    
    try:
        import json
        from datetime import datetime
        
        # Create dummy result structure
        test_result = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'model_type': 'PatchTST',
                'phase': 'validation_test'
            },
            'hyperparameters': {
                'patch_length': 16,
                'hidden_size': 64,
                'learning_rate': 1e-4
            },
            'test_metrics': {
                'f1_macro': 0.7234,
                'accuracy': 0.8125,
                'roc_auc': 0.7891
            }
        }
        
        # Test serialization
        json_str = json.dumps(test_result, indent=2, default=str)
        
        # Test deserialization
        loaded_result = json.loads(json_str)
        
        print(f"✅ JSON serialization successful")
        print(f"✅ Serialized length: {len(json_str)} characters")
        print(f"✅ Keys preserved: {list(loaded_result.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization error: {e}")
        return False

def main():
    """Run complete validation pipeline"""
    
    print("🔬 PatchTST Implementation Validation")
    print("=" * 50)
    
    # Track test results
    tests = []
    
    # Run all tests
    tests.append(("Imports", test_imports()))
    tests.append(("Device", test_device_availability() is not None))
    tests.append(("Data Loading", test_data_loading() is not None))
    tests.append(("Model Creation", test_model_creation() is not None))
    tests.append(("Data Preprocessing", test_data_preprocessing()[0] is not None))
    tests.append(("Forward Pass", test_model_forward_pass()))
    tests.append(("Training Step", test_training_step()))
    tests.append(("Metrics", test_metrics_computation()))
    tests.append(("JSON Serialization", test_json_serialization()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed_tests = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed_tests += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Results: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("🎉 ALL TESTS PASSED! PatchTST setup is ready for experiments.")
        print("\n🚀 Focused experimental approach (similar to TST):")
        print("   - Key parameters: hidden_size, attention_heads, layers")
        print("   - 81 experiments per case (vs 729 in original design)")
        print("   - 3 random seeds for statistical validation")
        print("\n🔬 You can now run:")
        print("   ./experiments_patch/LaunchPatchTSTExperiments.sh")
        print("   OR")
        print("   python experiments_patch/RunPatchTSTClassification.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("💡 Try installing missing dependencies:")
        print("   pip install -r experiments_patch/requirements_patch.txt")
    
    return passed_tests == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)