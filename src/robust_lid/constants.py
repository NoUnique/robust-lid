import os
from pathlib import Path

# Package Root
PACKAGE_ROOT = Path(__file__).parent
RESOURCES_DIR = PACKAGE_ROOT / "resources"
CACHE_DIR = Path.home() / ".cache" / "robust_lid"

# Model URLs
FASTTEXT_176_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_218E_URL = "https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin"
GLOTLID_V3_URL = "https://huggingface.co/cis-lmu/glotlid/resolve/main/model_v3.bin"

# Model Filenames
FASTTEXT_176_FILENAME = "lid.176.bin"
FASTTEXT_218E_FILENAME = "lid.218e.bin"
GLOTLID_V3_FILENAME = "glotlid_v3.bin"

# Resource Files
GLOTSCRIPT_TSV = RESOURCES_DIR / "GlotScript.tsv"
