import sys
import os
import torch
import librosa
import numpy as np
import urllib.request

# Adjust path to import whistress
sys.path.append(os.path.join(os.path.dirname(__file__), "WhiStress"))

from whistress.inference_client.whistress_client import WhiStressInferenceClient

def main():
    print("Loading WhiStress client...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = WhiStressInferenceClient(device=device)
    
    # Download a tiny sample audio
    audio_path = "sample.flac"
    if not os.path.exists(audio_path):
        print("Downloading sample audio...")
        urllib.request.urlretrieve("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac", audio_path)
        
    print("Loading audio...")
    audio_array, sr = librosa.load(audio_path, sr=16000)
    
    print("Preparing audio features...")
    input_features = client.whistress.processor.feature_extractor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    )["input_features"].to(client.device)
    
    print("\n--- Testing WhiStress Inference Pipeline ---")
    pairs = client.predict(audio_array, return_pairs=True)
    for p in pairs:
        print(f"Word: {p[0]!r:<10} | Stress: {p[1]} | Start: {p[2]:.2f} | End: {p[3]:.2f}")

if __name__ == "__main__":
    main()
