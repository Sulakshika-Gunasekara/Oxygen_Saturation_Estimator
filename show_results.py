import pandas as pd

# Read training history
history = pd.read_csv('final/history_run.csv')
results = pd.read_csv('final/results_15_15_70_dual_dataset.csv')

print("="*60)
print("FINAL TRAINING RESULTS - 15/15/70 Split")
print("="*60)

# Last epoch metrics
last_epoch = history.iloc[-1]
print(f"\nTotal Epochs Trained: {int(last_epoch['epoch'])}")

print("\n📊 VALIDATION SET METRICS (best model):")
print(f"  • MAE:    {results.iloc[0]['val_MAE']:.4f}")
print(f"  • RMSE:   {last_epoch['val_rmse']:.4f}")
print(f"  • R:      {last_epoch['val_r']:.4f}")
print(f"  • Acc±1:  {last_epoch['val_acc1']*100:.2f}%")
print(f"  • Acc±2:  {last_epoch['val_acc2']*100:.2f}%")

print("\n📊 TEST SET METRICS:")
print(f"  • MAE:    {results.iloc[0]['test_MAE']:.4f}")

print("\n📊 TRAINING SET METRICS:")
print(f"  • MAE:    {results.iloc[0]['train_MAE']:.4f}")

print("\n💾 MODEL SPECIFICATIONS:")
print(f"  • Parameters: {int(results.iloc[0]['Params']):,}")
print(f"  • Model Size: {results.iloc[0]['ModelMB']:.2f} MB")
print(f"  • CPU Latency: {results.iloc[0]['CPUms']:.2f} ms/sequence")

print("\n📁 DATASET SPLIT:")
print(f"  Dataset A (N1={int(results.iloc[0]['N1'])}):")
print(f"    - Train: {int(results.iloc[0]['Train_A'])} (15%)")
print(f"    - Val:   {int(results.iloc[0]['Val_A'])} (15%)")
print(f"    - Test:  {int(results.iloc[0]['Test_A'])} (70%)")
print(f"  Dataset B (N2={int(results.iloc[0]['N2'])}):")
print(f"    - Train: {int(results.iloc[0]['Train_B'])} (15%)")
print(f"    - Val:   {int(results.iloc[0]['Val_B'])} (15%)")
print(f"    - Test:  {int(results.iloc[0]['Test_B'])} (70%)")

print("\n" + "="*60)
