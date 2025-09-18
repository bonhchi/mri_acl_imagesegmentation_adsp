# src/configs/config.py
import os
from dotenv import load_dotenv

# load .env (tìm từ project root)
load_dotenv()

FASTMRI_ROOT = os.getenv("FASTMRI_ROOT")
OAI_ZIB_ROOT = os.getenv("OAI_ZIB_ROOT")
KAGGLE_KNEE_PCK_ROOT = os.getenv("KAGGLE_KNEE_PCK_ROOT")

# check & cảnh báo nếu thiếu
if not FASTMRI_ROOT:
    print("[WARN] FASTMRI_ROOT is not set in .env or environment")
if not OAI_ZIB_FRAMELAST_ROOT:
    print("[WARN] OAI_ZIB_FRAMELAST_ROOT is not set in .env or environment")
if not KNEE_MRI_ROOT:
    print("[WARN] KNEE_MRI_ROOT is not set in .env or environment")
if not SKM_TEA_MAIN_ROOT:
    print("[WARN] SKM_TEA_MAIN_ROOT is not set in .env or environment")