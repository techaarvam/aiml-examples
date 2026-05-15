import os
from datasets import load_dataset

# Target ~300M tokens. wikitext103.txt = 515MB for ~104M tokens (~5 bytes/token).
# 300M tokens * 5 bytes = ~1.5GB
TARGET_BYTES = int(1.5 * 1024 * 1024 * 1024)

out = "raw_data/openwebtext.txt"
os.makedirs("raw_data", exist_ok=True)

print("Streaming OpenWebText (target ~1.5GB / ~300M tokens)...")
ds = load_dataset("openwebtext", split="train", streaming=True)

written = 0
docs = 0
with open(out, "w", encoding="utf-8") as f:
    for item in ds:
        text = item["text"].strip()
        if not text:
            continue
        line = text + "\n"
        f.write(line)
        written += len(line.encode("utf-8"))
        docs += 1
        if docs % 100_000 == 0:
            print(f"  {written / 1e9:.2f}GB  {docs:,} docs")
        if written >= TARGET_BYTES:
            break

print(f"Done. {written / 1e9:.2f}GB written to {out}")
os._exit(0)  # skip cleanup of HuggingFace streaming threads to avoid SIGABRT
