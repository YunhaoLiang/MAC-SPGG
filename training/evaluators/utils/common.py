"""
Common utility functions for evaluator training and testing.
"""

import os
import gc
import torch


def get_device():
    """Get the available compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def clear_memory():
    """Clear GPU memory to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def setup_directories(base_dir="/content/Evaluator"):
    """
    Set up working directories for training.
    
    Args:
        base_dir: Base directory for evaluator outputs.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    os.chdir(base_dir)
    print(f"Working directory: {os.getcwd()}")


def mount_google_drive():
    """
    Mount Google Drive in Colab environment.
    
    Returns:
        True if mounted successfully, False otherwise.
    """
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
            print("Google Drive mounted successfully")
        return True
    except ImportError:
        print("Not running in Colab environment")
        return False
    except Exception as e:
        print(f"Failed to mount Google Drive: {e}")
        return False
