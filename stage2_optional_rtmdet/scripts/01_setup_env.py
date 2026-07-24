import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(1)

def main():
    print("Setting up MMDetection Environment...")
    
    # Check PyTorch installation
    try:
        import torch
        print(f"PyTorch found: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not found. Installing PyTorch (CPU version by default, please modify if CUDA is needed)...")
        run_cmd(f"{sys.executable} -m pip install torch torchvision torchaudio")

    # Install OpenMMLab dependencies
    print("Installing OpenMMLab dependencies...")
    run_cmd(f"{sys.executable} -m pip install -U openmim")
    run_cmd("mim install mmengine")
    run_cmd("mim install \"mmcv>=2.0.0\"")
    run_cmd(f"{sys.executable} -m pip install mmdet")

    print("Environment setup complete!")

if __name__ == "__main__":
    main()
