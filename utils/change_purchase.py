npz_path = 'data/purchase100.npz'   # <-- change this
csv_out_path = 'data/dataset_purchase.csv'

# ---- Step 1: Inspect the npz so we know what we're working with ----
data = np.load(npz_path, allow_pickle=True)
print("Keys found:", data.files)
for k in data.files:
    arr = data[k]
    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    
features = data['features']   # shape (N, 600), binary
labels = data['labels']       # shape (N,) or (N, 100) one-hot

# If labels are one-hot encoded, convert to class indices
if labels.ndim == 2:
    labels = labels.argmax(axis=1)

# Purchase-100 classes are conventionally stored 1-indexed in the CSV
# (get_purchase_dataset() does `y = purchase_data.iloc[:, 0] - 1` to shift back to 0-99)
labels_1_indexed = labels.astype(int) + 1

# Build the combined DataFrame: label first column, then features
df = pd.DataFrame(features)
df.insert(0, 'label', labels_1_indexed)

df.to_csv(csv_out_path, index=False, header=False)
print(f"Saved {df.shape[0]} rows, {df.shape[1]} columns to {csv_out_path}")
