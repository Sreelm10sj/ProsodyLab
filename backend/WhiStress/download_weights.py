from huggingface_hub import hf_hub_download
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

repo_id = "slprl/WhiStress"
local_dir = "whistress/weights"

# create local directory if it doesn't exist
# Expected dir structure:
# ```
# whistress/
# ├── weights/
# │   └── additional_decoder_block.pt
# │   └── classifier.pt
# │   └── metadata.json
# ├── ...
# README.md
# download_weights.py
# ...
# ```
local_weights_path = CURRENT_DIR / local_dir
local_weights_path.mkdir(parents=True, exist_ok=True)

files = [
    "classifier.pt",
    "additional_decoder_block.pt",
    "metadata.json"
]

for file in files:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=file,
        local_dir=f'{CURRENT_DIR}/{local_dir}',
        local_dir_use_symlinks=False
    )
    print(f"✅ Downloaded {file} to {local_path}")
