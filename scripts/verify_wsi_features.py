import h5py
import glob
import os
import numpy as np

features_dir = "data/processed/wsi_features"
h5_files = sorted(glob.glob(os.path.join(features_dir, "*.h5")))

if not h5_files:
    print("No .h5 files found to verify.")
    exit(1)

print(f"Found {len(h5_files)} H5 files. Checking a sample...")

# Check first, last, and a few random ones
indices = [0, len(h5_files)-1]
if len(h5_files) > 10:
    indices.extend(list(np.random.choice(range(1, len(h5_files)-1), 3, replace=False)))
indices = sorted(list(set(indices)))

all_good = True

for idx in indices:
    fpath = h5_files[idx]
    filename = os.path.basename(fpath)
    try:
        with h5py.File(fpath, 'r') as f:
            print(f"\nChecking {filename}:")
            
            # Check features
            if 'features' in f:
                feats = f['features']
                print(f"  - Features shape: {feats.shape}")
                if len(feats.shape) == 2 and feats.shape[1] == 1024:
                    print("  - Features shape OK")
                else:
                    print(f"  - Features shape MISMATCH (expected flow (N, 1024))")
                    all_good = False
            else:
                print("  - MISSING 'features' dataset")
                all_good = False
                
            # Check coords
            if 'coords' in f:
                coords = f['coords']
                print(f"  - Coords shape: {coords.shape}")
            else:
                print("  - MISSING 'coords' dataset")
                # Coords might be optional for the feature extraction part if we only care about features, 
                # but WSITiler saves them.
                
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        all_good = False

if all_good:
    print("\nVerification SUCCESS: All checked files have valid structure.")
else:
    print("\nVerification FAILED: Some files are invalid.")
