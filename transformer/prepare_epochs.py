import os
import random

WIKI   = "wikitext103.txt"
OWEB   = "openwebtext.txt"
OUTDIR = "epochs"
N      = 4      # number of epoch files
SEED   = 42

os.makedirs(OUTDIR, exist_ok=True)

print("Reading wikitext103.txt...")
with open(WIKI, encoding="utf-8") as f:
    lines = f.readlines()
print(f"  {len(lines):,} lines  ({sum(len(l) for l in lines)/1e9:.2f}GB)")

print("Reading openwebtext.txt...")
with open(OWEB, encoding="utf-8") as f:
    lines += f.readlines()
print(f"  {len(lines):,} lines total  ({sum(len(l) for l in lines)/1e9:.2f}GB)")

print("Shuffling...")
random.seed(SEED)
random.shuffle(lines)

chunk = len(lines) // N
for i in range(N):
    start = i * chunk
    end   = start + chunk if i < N - 1 else len(lines)
    path  = os.path.join(OUTDIR, f"epoch_{3 + i}.txt")
    print(f"Writing {path}  ({end - start:,} lines)...")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines[start:end])

print("Done. Epoch files in", OUTDIR)
