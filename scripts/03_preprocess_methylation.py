import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.methylation_preprocessing import MethylationPreprocessor

def main():
    print("Starting Methylation Preprocessing Pipeline...")
    processor = MethylationPreprocessor()
    try:
        output = processor.process_and_save(n_top_genes=3000)
        print(f"Successfully processed Methylation data: {output}")
    except Exception as e:
        print(f"Methylation Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
