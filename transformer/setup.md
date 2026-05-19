# Setup — Tech Aarvam Transformer

## 1. Clone the repo

```bash
git clone https://github.com/techaarvam/aiml-examples.git
cd aiml-examples/transformer
```

## 2. Install dependencies

```bash
pip install tiktoken datasets tqdm
```

> `torch`, `numpy` are pre-installed in the Vast.ai PyTorch image.
> `tomllib` is built into Python 3.11+. If on 3.10: `pip install tomli` and patch runner.py import.

## 3. Verify Python + torch + tiktoken

```bash
python -c "
import sys, torch, tiktoken
print('Python :', sys.version)
print('torch  :', torch.__version__)
print('CUDA   :', torch.cuda.is_available(), torch.version.cuda)
enc = tiktoken.get_encoding('p50k_base')
print('tiktoken p50k_base vocab:', enc.n_vocab)
"
```

## 4. Copy training data (from local machine)

Run this **locally** for each machine (substitute port/IP/machine number):

```bash
scp -P <port> -r raw_data/btm_machines/machineN/ root@<ip>:~/transformer/raw_data/btm_machines/
```

## 5. Run training

```bash
bash raw_data/btm_machines/machineN/cmd_run.sh
```

Logs go to `raw_data/btm_machines/machineN/train_mN.log`.
Monitor with: `tail -f raw_data/btm_machines/machineN/train_mN.log`

## 6. After all 4 machines finish — merge weights

Copy all 4 `model.pth` files back locally, then:

```bash
python merge_checkpoints.py btm_merged.pth \
    runs/btm_w64_m1/model.pth \
    runs/btm_w64_m2/model.pth \
    runs/btm_w64_m3/model.pth \
    runs/btm_w64_m4/model.pth
```
