"""
CUDA & Environment Diagnostic Script for MediFusion.

Reports PyTorch hardware configuration, CUDA capability, GPU memory,
and dynamic dataset/checkpoint directory resolution for local & Kaggle execution.
"""

import sys
from pathlib import Path

# Add project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from configs.config import (
    DATASET_DIR,
    METADATA_CSV_PATH,
    OUTPUT_MODELS_DIR,
    RESULTS_DIR,
    DEVICE,
    USE_AMP,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    BATCH_SIZE,
    LEARNING_RATE
)


def run_cuda_diagnostic() -> dict:
    """
    Run environment diagnostic and print clean markdown formatted report.

    Returns:
        dict: Diagnostic details dictionary.
    """
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else "N/A (CPU Mode)"
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A (CPU Only)"
    
    if cuda_available:
        gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = round(gpu_mem_bytes / (1024 ** 3), 2)
    else:
        gpu_memory_gb = 0.0

    diag_info = {
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "selected_device": DEVICE,
        "amp_available": USE_AMP,
        "dataset_dir": str(DATASET_DIR),
        "metadata_csv_path": str(METADATA_CSV_PATH),
        "output_models_dir": str(OUTPUT_MODELS_DIR),
        "results_dir": str(RESULTS_DIR),
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }

    print("=" * 70)
    print(" MediFusion — Hardware & Environment Diagnostic Report")
    print("=" * 70)
    print(f"  • PyTorch Version:      {diag_info['pytorch_version']}")
    print(f"  • CUDA Available:       {diag_info['cuda_available']}")
    print(f"  • CUDA Version:         {diag_info['cuda_version']}")
    print(f"  • GPU Device Name:      {diag_info['gpu_name']}")
    print(f"  • GPU Memory:           {diag_info['gpu_memory_gb']} GB")
    print(f"  • Selected PyTorch Dev: {diag_info['selected_device']}")
    print(f"  • AMP Enabled:          {diag_info['amp_available']}")
    print("-" * 70)
    print(" Configured Paths:")
    print(f"  • Dataset Directory:    {diag_info['dataset_dir']}")
    print(f"  • Metadata CSV Path:    {diag_info['metadata_csv_path']}")
    print(f"  • Checkpoint Directory: {diag_info['output_models_dir']}")
    print(f"  • Results Directory:    {diag_info['results_dir']}")
    print("-" * 70)
    print(" Configured DataLoader Settings:")
    print(f"  • Num Workers:          {diag_info['num_workers']}")
    print(f"  • Pin Memory:           {diag_info['pin_memory']}")
    print(f"  • Persistent Workers:   {diag_info['persistent_workers']}")
    print("=" * 70)

    return diag_info


if __name__ == "__main__":
    run_cuda_diagnostic()
