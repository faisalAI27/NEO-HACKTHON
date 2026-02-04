import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.mutation_preprocessing import MutationPreprocessor

def main():
    print("Starting Mutation Preprocessing Pipeline...")
    processor = MutationPreprocessor()
    try:
        # Lower min_samples because we have small cohort (82 patients)
        output = processor.process_and_save(min_samples=2)
        print(f"Successfully processed Mutation data: {output}")
    except Exception as e:
        print(f"Mutation Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
