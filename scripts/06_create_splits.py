from pathlib import Path
import sys

# Add project root to python path to allow importing src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.create_splits import create_cv_splits

if __name__ == "__main__":
    print("Starting cross-validation split generation...")
    create_cv_splits()
    print("Split generation completed successfully.")
