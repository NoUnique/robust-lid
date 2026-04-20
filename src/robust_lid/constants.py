from pathlib import Path
from typing import Final

PACKAGE_ROOT: Final[Path] = Path(__file__).parent
RESOURCES_DIR: Final[Path] = PACKAGE_ROOT / "resources"
CACHE_DIR: Final[Path] = Path.home() / ".cache" / "robust_lid"

FASTTEXT_176_URL: Final[str] = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)
FASTTEXT_218E_URL: Final[str] = (
    "https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin"
)
GLOTLID_V3_URL: Final[str] = "https://huggingface.co/cis-lmu/glotlid/resolve/main/model_v3.bin"

FASTTEXT_176_FILENAME: Final[str] = "lid.176.bin"
FASTTEXT_218E_FILENAME: Final[str] = "lid.218e.bin"
GLOTLID_V3_FILENAME: Final[str] = "glotlid_v3.bin"

GLOTSCRIPT_TSV: Final[Path] = RESOURCES_DIR / "GlotScript.tsv"

UNDEFINED_LANG: Final[str] = "und"
UNDEFINED_SCRIPT: Final[str] = "Zyyy"
DOWNLOAD_TIMEOUT_SEC: Final[int] = 30
