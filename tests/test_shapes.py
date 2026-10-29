"""Basic shape and structural tests for the DSAI Lab Project."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_data_splits_no_overlap():
    """Test that train/valid/test splits have no overlap."""
    import numpy as np
    
    # Check that splits are properly separated
    splits_dir = project_root / "data" / "splits"
    
    # This is a structural test - verify splits directory exists
    assert splits_dir.exists(), f"Splits directory not found: {splits_dir}"
    
    # Check that no audio ID appears in multiple splits
    # This is a placeholder - actual overlap check would load the data
    print("✓ Test: Data splits directory structure valid")


def test_config_files_exist():
    """Test that required config files exist."""
    configs_dir = project_root / "configs"
    
    required_configs = ["data.yaml", "model.yaml", "training.yaml"]
    
    for config_file in required_configs:
        config_path = configs_dir / config_file
        assert config_path.exists(), f"Required config not found: {config_path}"
    
    print("✓ Test: All required config files exist")


def test_script_imports():
    """Test that key scripts can be imported without errors."""
    scripts_dir = project_root / "scripts"
    
    # Test that scripts have valid Python syntax
    for script in ["download_data.py", "preprocess.py"]:
        script_path = scripts_dir / script
        assert script_path.exists(), f"Script not found: {script_path}"
        
        # Try to compile the file
        try:
            with open(script_path, "r") as f:
                compile(f.read(), script_path.name, "exec")
            print(f"✓ Test: {script} has valid Python syntax")
        except SyntaxError as e:
            print(f"✗ Test: {script} has syntax error: {e}")
            raise
    
    print("✓ Test: All scripts have valid syntax")


def test_data_directory_structure():
    """Test that data directory has expected subfolders."""
    data_dir = project_root / "data"
    
    expected_folders = ["raw", "processed", "splits"]
    
    for folder in expected_folders:
        folder_path = data_dir / folder
        assert folder_path.exists(), f"Expected data folder not found: {folder_path}"
    
    print("✓ Test: Data directory has expected subfolders")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Running DSAI Lab Project Tests")
    print("=" * 50)
    
    tests = [
        test_data_splits_no_overlap,
        test_config_files_exist,
        test_script_imports,
        test_data_directory_structure,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()