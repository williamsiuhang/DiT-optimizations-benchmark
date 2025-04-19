from datasets import load_dataset
import os
# adjust the split name & size as needed
ds = load_dataset("ChristophSchuhmann/improved_aesthetics_6plus", split="train[:10000]")
out_csv = os.path.join("data/phase1", "subset.csv")
ds.to_csv(out_csv)
print(f"Wrote metadata to {out_csv}")