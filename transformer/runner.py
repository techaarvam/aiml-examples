
# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
# Runner — loads a TOML run profile and launches training
# with a reproducible, self-contained output directory.
#
# Usage:
#   python runner.py runs.toml <profile> [--key value ...]
#   python runner.py runs.toml smoke
#   python runner.py runs.toml server --epochs 50
#   python runner.py runs.toml          # list profiles
# --------------------------------------------------

import sys, os, json, datetime, tomllib, shutil

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Args ─────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python runner.py <runs.toml> [profile] [--key value ...]")
    sys.exit(1)

config_file = sys.argv[1]
if not os.path.exists(config_file):
    print(f"Config file not found: {config_file}")
    sys.exit(1)

with open(config_file, "rb") as f:
    config = tomllib.load(f)

profiles = [k for k in config if k != "defaults"]

if len(sys.argv) < 3:
    print(f"Available profiles in {config_file}:")
    for p in profiles:
        desc = config[p].get("_desc", "")
        print(f"  {p:<12} {desc}")
    sys.exit(0)

profile_name = sys.argv[2]
if profile_name not in config:
    print(f"Profile '{profile_name}' not found. Available: {profiles}")
    sys.exit(1)

# Merge defaults → profile → CLI overrides
args = {**config.get("defaults", {}), **config[profile_name]}

# Parse any extra --key value overrides from CLI
extra = sys.argv[3:]
i = 0
while i < len(extra):
    key = extra[i].lstrip("-")
    if i + 1 < len(extra) and not extra[i + 1].startswith("--"):
        val = extra[i + 1]
        # Attempt type coercion to match profile value type
        if key in args:
            t = type(args[key])
            try:
                val = t(val)
            except (ValueError, TypeError):
                pass
        args[key] = val
        i += 2
    else:
        args[key] = True  # boolean flag
        i += 1

# ── Run directory ─────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir   = os.path.abspath(os.path.join("runs", f"{profile_name}_{timestamp}"))
os.makedirs(run_dir, exist_ok=True)

save_model = os.path.join(run_dir, "model.pth")
onnx_model = os.path.join(run_dir, "model.onnx")
log_file   = os.path.join(run_dir, "train.log")

args["save_model"] = save_model


# ── Reproducibility dump ──────────────────────────
meta = dict(args)
meta.update(profile=profile_name, run_dir=run_dir,
            log_file=log_file, launched_at=timestamp)
with open(os.path.join(run_dir, "args.json"), "w") as f:
    json.dump(meta, f, indent=2)

# ── Build trainer command ─────────────────────────
BOOL_FLAGS = {"use_custom_norm", "resume", "quantize", "grad_checkpoint", "validate"}

def build_cmd(overrides=None):
    a = dict(args)
    if overrides:
        a.update(overrides)
    parts = [f"python -u {os.path.join(HERE, 'trainer.py')}"]
    for k, v in a.items():
        if k.startswith("_") or v is None:
            continue
        if k in BOOL_FLAGS:
            if v:
                parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {v}")
    return " \\\n    ".join(parts)

train_cmd  = build_cmd()
resume_cmd = build_cmd({"resume": True, "model_file": save_model})

# ── Write helper scripts ──────────────────────────
def write_sh(path, body):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {HERE}\n")
        f.write(body + "\n")
    os.chmod(path, 0o755)

write_sh(os.path.join(run_dir, "cmd_train.sh"),
    f"{train_cmd} \\\n    2>&1 | tee {log_file}")

write_sh(os.path.join(run_dir, "cmd_resume.sh"),
    f"{resume_cmd} \\\n    2>&1 | tee -a {log_file}")

write_sh(os.path.join(run_dir, "cmd_pth2onnx.sh"), "\n".join([
    f"python {os.path.join(HERE, 'pth2onnx.py')} \\",
    f"    --embedding_type {args['embedding_type']} \\",
    f"    --vecDims {args['vecDims']} \\",
    f"    --output_type {args['output_type']} \\",
    f"    --num_heads {args['num_heads']} \\",
    f"    --num_layers {args['num_layers']} \\",
    f"    --window_size {args['window_size']} \\",
    f"    --model_file {save_model} \\",
    f"    --output {onnx_model}",
]))

write_sh(os.path.join(run_dir, "cmd_netron.sh"),
    f"python -c \"import netron; netron.start('{onnx_model}', "
    f"address=('0.0.0.0', 8081), browse=False); "
    f"input('Serving on 0.0.0.0:8081 — press Enter to stop\\n')\"")

infer_cmd = build_cmd({"model_file": save_model, "resume": False, "input": None,
                       "epochs": None, "save_model": None})
write_sh(os.path.join(run_dir, "cmd_infer.sh"), infer_cmd)

write_sh(os.path.join(run_dir, "cmd_webserver.sh"),
    f"cd {run_dir}\n"
    f"echo 'Serving at http://0.0.0.0:9090/web_infer.html'\n"
    f"python -m http.server 9090")

write_sh(os.path.join(run_dir, "cmd_monitor.sh"),
    f"watch -n 2 cat {os.path.join(run_dir, 'progress.txt')}")

# ── Copy + patch web_infer.html ───────────────────
src_html = os.path.join(HERE, "web_infer.html")
if os.path.exists(src_html):
    seq_len = str(args['window_size'] - 1)
    html = open(src_html).read()
    html = html.replace("'transformer_model.onnx'", "'model.onnx'")
    # Substitute all window-size-dependent hardcoded values
    html = html.replace("const seqLen = 63;",         f"const seqLen = {seq_len};")
    html = html.replace(".slice(0, 63)",               f".slice(0, {seq_len})")
    html = html.replace("|| 63;",                      f"|| {seq_len};")
    html = html.replace("/63 tokens`",                 f"/{seq_len} tokens`")
    html = html.replace(", 63);",                      f", {seq_len});")
    html = html.replace("up to 63 tokens",             f"up to {seq_len} tokens")
    html = html.replace('value="63"',                  f'value="{seq_len}"')
    html = html.replace("[1,63]",                      f"[1,{seq_len}]")
    html = html.replace(
        "Tech Aarvam · 11M params · WikiText-103 · 30K vocab",
        f"Tech Aarvam · {profile_name} · "
        f"L{args['num_layers']} H{args['num_heads']} D{args['vecDims']} "
        f"W{args['window_size']} · {timestamp}"
    )
    open(os.path.join(run_dir, "web_infer.html"), "w").write(html)

# ── Summary ───────────────────────────────────────
print("=" * 60)
print(f"Profile  : {profile_name}")
print(f"Run dir  : {run_dir}")
print(f"Log      : {log_file}")
print(f"Model    : {save_model}")
print(f"Config   : layers={args['num_layers']} heads={args['num_heads']} "
      f"dims={args['vecDims']} window={args['window_size']} "
      f"batch={args['batch_size']} epochs={args['epochs']}")
print(f"Data     : {args.get('input', 'N/A')}")
print(f"Vocab    : {args.get('max_vocab_size', 'unlimited')}")
print("=" * 60)

# ── Launch ────────────────────────────────────────
os.system(f"{train_cmd} 2>&1 | tee {log_file}")
