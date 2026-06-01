import torch
import numpy as np
from debug import *
from argParser import *
from torch import nn
from attention import *
import DataInput
import common
from torch.utils.checkpoint import checkpoint as ckpt

class _MLP(nn.Module):
    """FFN block with optional gradient boost on new dims after an inner_dims expansion.

    When mlp_boost_old_d > 0, hooks multiply the gradient on new-dimension rows/cols
    by `boost` before the optimizer sees them, accelerating warm-up of the freshly
    initialised weights without disturbing the old-dim update scale.

    Call remove_boost() after the warm-up epochs to drop the hooks.
    """
    def __init__(self, inner_dims, old_d=0, boost=4.0):
        super().__init__()
        self.fc1   = nn.Linear(inner_dims, inner_dims * 4)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(inner_dims * 4, inner_dims)
        self._hooks = []
        if 0 < old_d < inner_dims:
            self._register_boost_hooks(old_d, inner_dims, boost)

    def _register_boost_hooks(self, old_d, d, boost):
        # W_up  shape [4d, d]: rows 0..4*old_d-1 are old hidden units,
        #                       rows 4*old_d..   are new hidden units.
        def w_up_hook(grad):
            g = grad.clone()
            g[4*old_d:, :]      *= boost   # new hidden units (all input dims)
            g[:4*old_d, old_d:] *= boost   # old hidden units, new input cols
            return g

        # W_down shape [d, 4d]: rows 0..old_d-1 are old output dims,
        #                        rows old_d..    are new output dims.
        def w_down_hook(grad):
            g = grad.clone()
            g[old_d:, :]        *= boost   # new output dims (all hidden cols)
            g[:old_d, 4*old_d:] *= boost   # old output dims, new hidden input cols
            return g

        self._hooks = [
            self.fc1.weight.register_hook(w_up_hook),
            self.fc2.weight.register_hook(w_down_hook),
        ]

    def remove_boost(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHead (nn.Module):
    def __init__(self):
        super().__init__()

        vecDims   = common.vecDims
        innerDims = common.innerDims
        isGlove   = args.embedding_type == "glove-fixed"

        if isGlove:
            usedVocabVecs = []
            for w in common.wordDict.keys():
                if w in common.wordVecs:
                    usedVocabVecs.append( common.wordVecs[w] )
                else: usedVocabVecs.append( np.zeros(vecDims) )
            self.register_buffer('wv', torch.tensor(np.array(usedVocabVecs), dtype=torch.float32))
        else:
            self.embedding = nn.Embedding(common.vocabSize, vecDims)

        # positional embedding used by both modes
        self.posEmbedding = nn.Embedding(args.window_size, vecDims)

        # upscale/downscale adapters: bridge embedding dim (vecDims) ↔ transformer dim (innerDims)
        if innerDims != vecDims:
            self.upscale   = nn.Linear(vecDims, innerDims, bias=False)
            self.downscale = nn.Linear(innerDims, vecDims, bias=False)

        self.attentionHeads = nn.ModuleList([ Attention() for _ in range (0, args.num_layers) ])
        # Shape Notes:
           # Wo works with concatenated attention heads. reduces to VecDims
           # Wo is the learned re-combination weights, instead of a simple mean.

        self.Wo = nn.ParameterList([nn.Parameter(torch.randn(innerDims, innerDims) * (1.0 / innerDims ** 0.5)) for _ in range(args.num_layers)])
        if (args.use_custom_norm):
            self.learnedMeanShift1 = nn.ParameterList([nn.Parameter(torch.zeros(1,1,innerDims)) for _ in range(args.num_layers)])
            self.learnedStdScale1  = nn.ParameterList([nn.Parameter(torch.ones(1,1,innerDims))  for _ in range(args.num_layers)])
            self.learnedMeanShift2 = nn.ParameterList([nn.Parameter(torch.zeros(1,1,innerDims)) for _ in range(args.num_layers)])
            self.learnedStdScale2  = nn.ParameterList([nn.Parameter(torch.ones(1,1,innerDims))  for _ in range(args.num_layers)])
        else:
            self.norm1 = nn.ModuleList([nn.LayerNorm(innerDims) for _ in range(args.num_layers)])
            self.norm2 = nn.ModuleList([nn.LayerNorm(innerDims) for _ in range(args.num_layers)])

        # Standard FFN: d -> 4d -> d with GELU
        self.mlp = nn.ModuleList([
            _MLP(innerDims, old_d=args.mlp_boost_old_d, boost=args.mlp_boost)
            for _ in range(args.num_layers)
        ])
        if (args.output_type == "indices"):
            self.outputLinear = nn.Linear(vecDims, common.vocabSize)
        elif (args.output_type == "vecs"):
            self.outputLinear = nn.Linear(vecDims, vecDims-1)
        
    def _layer_forward(self, X, layer):
        normedX = self.normalize(X, self.norm1[layer],
            (self.learnedMeanShift1[layer], self.learnedStdScale1[layer]) if args.use_custom_norm else None)

        
        attentionOutput = self.attentionHeads[layer].forward(normedX)
        attentionOutput = attentionOutput @ self.Wo[layer]


        X = X + attentionOutput
        normedX = self.normalize(X, self.norm2[layer],
            (self.learnedMeanShift2[layer], self.learnedStdScale2[layer]) if args.use_custom_norm else None)
        X = X + self.mlp[layer](normedX)
        return X

    def dbg_output_health_check(self):
        """Print Wo inactive singular directions and all matrices with cond > 1e5."""
        THRESH = 1e5
        DEAD   = 1e-6
        wo_inactive = []
        ill_cond    = []

        for L in range(args.num_layers):
            # Attention (delegates to Attention.dbg_output_health_check)
            for name, cond, n_inactive, total in self.attentionHeads[L].dbg_output_health_check():
                if cond > THRESH:
                    ill_cond.append(f'attn{L}.{name}:{cond:.1e}')

            # Wo
            S    = torch.linalg.svdvals(self.Wo[L].detach().cpu().float())
            n    = (S <= DEAD).sum().item()
            cond = (S[0] / S[-1]).item() if S[-1] > 0 else float('inf')
            if n > 0:
                wo_inactive.append(f'L{L}:{n}/{len(S)}')
            if cond > THRESH:
                ill_cond.append(f'Wo.{L}:{cond:.1e}')

            # FFN
            for fc, tag in [(self.mlp[L].fc1, f'FFN_up.{L}'), (self.mlp[L].fc2, f'FFN_dn.{L}')]:
                W    = fc.weight.detach().cpu().float()
                S    = torch.linalg.svdvals(W)
                cond = (S[0] / S[-1]).item() if S[-1] > 0 else float('inf')
                if cond > THRESH:
                    ill_cond.append(f'{tag}:{cond:.1e}')

        # upscale / downscale
        for attr, tag in [('upscale', 'upscale'), ('downscale', 'downscale')]:
            if hasattr(self, attr):
                W    = getattr(self, attr).weight.detach().cpu().float()
                S    = torch.linalg.svdvals(W)
                cond = (S[0] / S[-1]).item() if S[-1] > 0 else float('inf')
                if cond > THRESH:
                    ill_cond.append(f'{tag}:{cond:.1e}')

        dbg_output("  Wo inactive: " + (" ".join(wo_inactive) if wo_inactive else "none"))
        dbg_output("  cond>1e5:    " + (", ".join(ill_cond)   if ill_cond    else "none"))

    def normalize (self, X, norm, learnedParams=None):
        # Instead of using nn.LayerNorm, using mean/std operations,
        # since this is a learning project and the goal is to break
        # the transformer down to simplest operations possible.

        if (args.use_custom_norm):
            learnedMeanShift, learnedStdScale = learnedParams
            mean = X.mean(dim=-1, keepdim=True)
            std = X.std(dim=-1, keepdim=True)
            normedX = learnedStdScale * (X - mean) / (std + 1e-6) + learnedMeanShift
        else:
            normedX = norm(X)
        return normedX


    def forward(self, X, targets=None, chunk_size=8192):
        positions = torch.arange(X.shape[1], device=X.device)
        if args.embedding_type == "glove-fixed":
            X = X + self.posEmbedding(positions)
        else:
            X = self.embedding(X) + self.posEmbedding(positions)
        if hasattr(self, 'upscale'):
            X = self.upscale(X)
        for layer in range(0, args.num_layers):
            if args.grad_checkpoint and self.training:
                fn = lambda X, l=layer: self._layer_forward(X, l)
                X = ckpt(fn, X, use_reentrant=False)
            else:
                X = self._layer_forward(X, layer)
        if hasattr(self, 'downscale'):
            X = self.downscale(X)
        if targets is None:
            return self.outputLinear(X)
        flat_h = X.reshape(-1, X.shape[-1])
        flat_t = targets.reshape(-1)
        n      = flat_h.shape[0]
        ce     = nn.CrossEntropyLoss(reduction='sum')
        if not self.outputLinear.weight.requires_grad:
            # outputLinear frozen: surrogate trick — one backward pass through transformer
            W      = self.outputLinear.weight.float()
            with torch.no_grad():
                total  = torch.zeros(1, device=flat_h.device)
                grad_h = torch.zeros_like(flat_h)
                for start in range(0, n, chunk_size):
                    end    = min(start + chunk_size, n)
                    logits = flat_h[start:end].float() @ W.T
                    if self.outputLinear.bias is not None:
                        logits += self.outputLinear.bias.float()
                    total += ce(logits, flat_t[start:end])
                    # in-place stable softmax — avoids a second [chunk, V] allocation
                    logits.sub_(logits.max(dim=-1, keepdim=True).values).exp_()
                    logits.div_(logits.sum(dim=-1, keepdim=True))
                    logits[torch.arange(end - start, device=flat_h.device), flat_t[start:end]] -= 1.0
                    logits.div_(n)
                    grad_h[start:end] = (logits @ W).to(flat_h.dtype)
            # This trick is done to do gradient propogation as one flow
            # instead of looping through each chunk.
            # Chunked cross entropy is done only at the output linear step. Only for loop
            # The cross-entropy on the entire vocabulary * batch-size * seq_len is a huge tensor
            # prevents running out of VRAM in consumer hardware. 
            # May be need even in data-center runs to allow running multiple processes in parallel that use VRAM 
            # effectively

            surrogate = (flat_h * grad_h).sum()
            return surrogate + (total / n - surrogate).detach()
        else:
            # outputLinear trainable: per-chunk backward so weight grads accumulate correctly
            loss = 0.0
            for start in range(0, n, chunk_size):
                end        = min(start + chunk_size, n)
                chunk_loss = ce(self.outputLinear(flat_h[start:end]).float(), flat_t[start:end]) / n
                chunk_loss.backward(retain_graph=(end < n))
                loss += chunk_loss.item()
            return loss

