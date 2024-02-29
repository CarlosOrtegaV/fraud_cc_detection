from pathlib import Path
import os

PARENT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PARENT_DIR / 'data'

RAW_DATA_DIR = DATA_DIR / 'raw'
PREPROCESSED_DATA_DIR = DATA_DIR / 'preprocessed'

# sys.path.append('../') 

if not Path(DATA_DIR).exists():
    os.makedirs(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.makedirs(RAW_DATA_DIR)
    
if not Path(PREPROCESSED_DATA_DIR).exists():
    os.makedirs(PREPROCESSED_DATA_DIR)
    
