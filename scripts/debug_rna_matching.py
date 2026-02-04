import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.patient_registry import PatientRegistry

print("Loading registry...")
registry = PatientRegistry()
df_reg = registry.build_registry()
print(f"Registry shape: {df_reg.shape}")
print(f"First 5 registry indices: {df_reg.index[:5].tolist()}")

print("\nLoading RNA...")
try:
    df = pd.read_csv("data/raw/transcriptomics.txt", sep='\t', index_col=0)
    print(f"RNA shape: {df.shape}")
    df_t = df.T
    print(f"Transposed RNA index head: {df_t.index[:5].tolist()}")
    
    first_sample = df_t.index[0]
    derived_pid = "-".join(first_sample.split("-")[:3])
    print(f"\nTest Matching:")
    print(f"Sample: '{first_sample}'")
    print(f"Derived Patient ID: '{derived_pid}'")
    print(f"Is '{derived_pid}' in registry? {derived_pid in df_reg.index}")

    # Check for hidden characters
    print(f"Registry index representation: {[repr(x) for x in df_reg.index[:2]]}")
    print(f"Derived PID representation: {repr(derived_pid)}")

except Exception as e:
    print(f"Error: {e}")
