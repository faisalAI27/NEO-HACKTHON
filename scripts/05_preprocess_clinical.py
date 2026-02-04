import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.clinical_preprocessing import ClinicalPreprocessor

def main():
    print("Starting Clinical Preprocessing Pipeline...")
    processor = ClinicalPreprocessor()
    try:
        output = processor.process_and_save()
        print(f"Successfully processed Clinical data: {output}")
    except Exception as e:
        print(f"Clinical Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
