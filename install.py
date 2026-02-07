"""
@dataset{makhresearch_skin_lesion_2025,
  title     = {Skin Lesion Segmentation and Classification Dataset},
  author    = {MakhResearch},
  year      = {2025},
  url       = {https://huggingface.co/datasets/makhresearch/skin-lesion-segmentation-classification}
}

"""

import requests
from zipfile import ZipFile
from io import BytesIO
from pathlib import Path

url = "https://huggingface.co/datasets/makhresearch/skin-lesion-segmentation-classification/resolve/main/skin-lesion-segmentation-classification.zip"
print("Downloading the dataset ZIP file...")
response = requests.get(url)
response.raise_for_status()
print("✅ Download complete.")

print("\nExtracting files...")
with ZipFile(BytesIO(response.content)) as zf:
    zf.extractall(".")
print("✅✅✅ Extraction complete.")