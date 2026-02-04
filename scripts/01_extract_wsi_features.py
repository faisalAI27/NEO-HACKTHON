import os
import sys
import yaml
import glob
from pathlib import Path
from tqdm import tqdm
import time
import argparse

# Add src to python path so we can import modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.wsi_preprocessing import WSITiler
from src.data.wsi_feature_extraction import UNIFeatureExtractor

def process_single_slide(slide_path, tiler, extractor, output_dir):
    """
    Process a single slide: Tiling -> Feature Extraction
    """
    slide_name = Path(slide_path).stem
    h5_path = output_dir / f"{slide_name}.h5"
    
    # Step 1: Tiling
    # WSITiler checks internally if output exists, but we can double check here 
    # if we want to force re-run or skip steps.
    # For now relying on WSITiler's check.
    
    try:
        # Extract tiles (saves images to HDF5)
        # Returns the path to the h5 file if successful, or None if failed/empty
        tiling_result = tiler.extract_tiles(slide_path)
        
        if tiling_result is None:
            print(f"Skipping feature extraction for {slide_name} due to tiling failure or no tissue.")
            return False
            
        # Step 2: Feature Extraction
        # Appends features to the same HDF5 file
        shape = extractor.extract_features(tiling_result)
        
        if shape is None:
            print(f"Feature extraction failed for {slide_name}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error processing {slide_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="WSI Feature Extraction Pipeline")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml", help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for dataloader")
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Could not load config from {args.config} ({e}), using defaults.")
        config = {}

    # Setup directories
    raw_dir = Path("data/raw/svs")
    output_dir = Path("data/processed/wsi_features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of slides
    slides = sorted(list(raw_dir.glob("*.svs")))
    print(f"Found {len(slides)} slides to process.")
    
    # Initialize components
    tiler = WSITiler(config)
    extractor = UNIFeatureExtractor() # Loads model
    
    # Processing statistics
    successful = 0
    failed = 0
    start_time = time.time()
    
    # Process loop
    pbar = tqdm(slides, desc="Processing Slides")
    for slide_path in pbar:
        status = process_single_slide(slide_path, tiler, extractor, output_dir)
        if status:
            successful += 1
        else:
            failed += 1
        
        pbar.set_postfix({"Success": successful, "Fail": failed})
        
    duration = time.time() - start_time
    print("\n" + "="*50)
    print(f"Pipeline Completed in {duration/60:.2f} minutes")
    print(f"Total Slides: {len(slides)}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")
    print("="*50)

if __name__ == "__main__":
    main()
