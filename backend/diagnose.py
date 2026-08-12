import sys
import os

print(f"1. Python executable: {sys.executable}")
print(f"2. Python version: {sys.version}")

try:
    import torch
    print(f"5. Torch installed: Yes")
    print(f"   Torch version: {torch.__version__}")
    print(f"6. CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   CUDA device name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print(f"5. Torch installed: No")
    print(f"6. CUDA available: N/A")

print("7. Attempting to import whistress...")
# We need to add WhiStress to sys.path like whistress_service.py does
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "WhiStress"))

try:
    import whistress
    print(f"   WhiStress imported successfully.")
    print(f"8. WhiStress path: {whistress.__file__}")
    
    from whistress import WhiStressInferenceClient
    print(f"9. WhiStressInferenceClient available.")
    
    import inspect
    print("   Code path for WhiStressInferenceClient:")
    print(inspect.getsourcefile(WhiStressInferenceClient))
    
    # Try to initialize the client to see what model weights it loads
    try:
        print("   Attempting to instantiate WhiStressInferenceClient...")
        client = WhiStressInferenceClient(device="cpu")
        print("   Client instantiated successfully.")
    except Exception as e:
        print(f"   Client instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"   WhiStress import failed: {e}")
except Exception as e:
    print(f"   Other error during WhiStress import: {e}")
    
print("10/11/12/13. Searching for model weights...")
whistress_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WhiStress")
if os.path.exists(whistress_dir):
    print(f"   WhiStress source directory exists: {whistress_dir}")
    # Search for .pt, .pth, .bin files in this directory
    for root, dirs, files in os.walk(whistress_dir):
        for f in files:
            if f.endswith(('.pt', '.pth', '.bin', '.ckpt')):
                print(f"   Found weight file: {os.path.join(root, f)}")
else:
    print(f"   WhiStress source directory NOT FOUND: {whistress_dir}")

print("Checking HF cache...")
hf_cache = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(hf_cache):
    print(f"   HF cache exists at: {hf_cache}")
    # List directories under hub
    hub_dir = os.path.join(hf_cache, "hub")
    if os.path.exists(hub_dir):
        print(f"   HF hub directories:")
        for d in os.listdir(hub_dir):
            print(f"     - {d}")
else:
    print("   HF cache does not exist.")
