import os
from datasets import load_dataset

print("Downloading WikiText-103...")
ds = load_dataset("wikitext", "wikitext-103-raw-v1")

os.makedirs("raw_data", exist_ok=True)

out_full = "raw_data/wikitext103.txt"
out_50m  = "raw_data/wikitext50m.txt"

print(f"Writing {out_full}...")
with open(out_full, "w") as f:
    for split in ["train", "validation", "test"]:
        for row in ds[split]:
            text = row["text"].strip()
            if text:
                f.write(text + "\n")

print(f"Writing {out_50m} (first 50MB)...")
written = 0
limit = 50 * 1024 * 1024
with open(out_50m, "w") as f:
    for split in ["train", "validation", "test"]:
        for row in ds[split]:
            text = row["text"].strip()
            if text:
                line = text + "\n"
                f.write(line)
                written += len(line.encode("utf-8"))
                if written >= limit:
                    break
        if written >= limit:
            break

print(f"Done. Full: {out_full}, 50MB slice: {out_50m}")
