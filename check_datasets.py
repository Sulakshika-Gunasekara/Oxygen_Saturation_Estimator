import glob
import numpy as np

files_A = glob.glob('*_sequences.npz')
files_B = [f for f in glob.glob('*.npz') if not f.endswith('_sequences.npz')]

print(f'Dataset A: {len(files_A)} files')
print(f'Dataset B: {len(files_B)} files')

# Check first file from each
d_a = np.load(files_A[0])
d_b = np.load(files_B[0])

print(f'\nDataset A sample:')
print(f'  File: {files_A[0]}')
print(f'  Features shape: {d_a["features"].shape}')
print(f'  Labels shape: {d_a["labels"].shape}')

print(f'\nDataset B sample:')
print(f'  File: {files_B[0]}')
print(f'  Features shape: {d_b["features"].shape}')
print(f'  Labels shape: {d_b["labels"].shape}')

# Check if all files in each dataset have consistent shapes
print(f'\nChecking all Dataset A files...')
for f in files_A[:5]:
    d = np.load(f)
    print(f'  {f}: features={d["features"].shape}, labels={d["labels"].shape}')

print(f'\nChecking all Dataset B files...')
for f in files_B[:5]:
    d = np.load(f)
    print(f'  {f}: features={d["features"].shape}, labels={d["labels"].shape}')
