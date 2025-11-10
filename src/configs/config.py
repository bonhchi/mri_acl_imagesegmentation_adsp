# src/configs/config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file located near the project root.
load_dotenv()

ENV_KEYS = [
    "FASTMRI_ROOT",
    "OAI_ZIB_ROOT",
    "KAGGLE_KNEE_PCK_ROOT",
    "KNEE_MRI_ROOT",
    "SKM_TEA_MAIN_ROOT",
]

_env = {key: os.getenv(key) for key in ENV_KEYS}

FASTMRI_ROOT = _env["FASTMRI_ROOT"]
OAI_ZIB_ROOT = _env["OAI_ZIB_ROOT"]
KAGGLE_KNEE_PCK_ROOT = _env["KAGGLE_KNEE_PCK_ROOT"]
KNEE_MRI_ROOT = _env["KNEE_MRI_ROOT"]
SKM_TEA_MAIN_ROOT = _env["SKM_TEA_MAIN_ROOT"]

__all__ = [
    "FASTMRI_ROOT",
    "OAI_ZIB_ROOT",
    "KAGGLE_KNEE_PCK_ROOT",
    "KNEE_MRI_ROOT",
    "SKM_TEA_MAIN_ROOT",
]
