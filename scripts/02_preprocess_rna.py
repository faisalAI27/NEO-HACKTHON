import sys
from pathlib import Path

# Add src to python path so we can import modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.rna_preprocessing import RNAPreprocessor

def main():
    print("Starting RNA Preprocessing Pipeline...")
    processor = RNAPreprocessor()
    try:
        output = processor.process_and_save(n_top_genes=3000)
        print(f"Successfully processed RNA data: {output}")
    except Exception as e:
        print(f"RNA Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
