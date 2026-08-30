"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
import argparse, collections, time, json, os
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate
from _agent_utils import to_native

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- item popularity（官方 baseline） ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, temp_base=1.0, alpha_temp=0.0, margin_weight=0.1):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2, self.temp_base, self.alpha_temp, self.margin_weight = lr, l2, temp_base, alpha_temp, margin_weight
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y, sample_weights):
        B = len(y)
        z, E, S = self.logits(X)
        # Per-sample adaptive temperature: larger error -> higher temp (softer)
        pred = sigmoid(z / self.temp_base)
        error = np.abs(y - pred)
        temp_i = self.temp_base * (1.0 + self.alpha_temp * error)
        temp_i = np.clip(temp_i, 0.5, 2.0)  # stable bounds
        z_scaled = z / temp_i
        z_clipped = np.clip(z_scaled, -3.0, 3.0)
        pred = sigmoid(z_clipped)
        
        eps = 0.05
        y_smoothed = y * (1.0 - eps) + eps
        
        # Margin-aware gradient reweighting
        margin = y * z
        contrastive_weight = 1.0 + self.margin_weight * np.clip(margin, -5.0, 5.0)
        g = ((pred - y_smoothed) * sample_weights * contrastive_weight / B).astype(np.float32)
        
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        
        self.t += 1
        b1, b2, eps_adam = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps_adam)
        self.b -= self.lr * (g * B).sum()
        
        eps_log = 1e-9
        loss = -np.mean(sample_weights * contrastive_weight * (y * np.log(pred + eps_log) + (1 - y) * np.log(1 - pred + eps_log)))
        return float(loss)

    def predict(self, X, bs=200_000):
        preds = []
        for i in range(0, len(X), bs):
            z, _, _ = self.logits(X[i:i+bs])
            pred = sigmoid(z / self.temp_base)
            preds.append(sigmoid(np.clip(z / self.temp_base, -3.0, 3.0)))
        return np.concatenate(preds)

def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True, 
           recency_scale=1.0, margin_weight=0.1, temp_base=0.85, alpha_temp=0.3):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    
    # Precompute recency weights
    N = len(ytr)
    idx_order = np.arange(N)
    recency = (idx_order + 1) / N
    sample_weights = np.ones(N, dtype=np.float32) + recency_scale * (recency - 0.5)
    sample_weights = np.clip(sample_weights, 0.5, 2.0)
    
    m = FM(dim, k=k, lr=lr, seed=seed, temp_base=temp_base, 
           alpha_temp=alpha_temp, margin_weight=margin_weight)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        w_batch = sample_weights[idx]
        losses = []
        for i in range(0, len(idx), bs):
            batch_idx = idx[i:i+bs]
            batch_w = w_batch[i:i+bs]
            losses.append(m.step(Xtr[batch_idx], ytr[batch_idx], batch_w))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}

# --- Appended by agent_loop.py, not part of the original baseline.py ---
if __name__ == "__main__":
    splits = load(os.environ["KUAIRAND_DATA_DIR"])
    res = run_fm(splits, k=16, lr=0.001, epochs=40, seed=0, verbose=False,
                 recency_scale=0.5, margin_weight=0.2, temp_base=0.85, alpha_temp=0.3)
    print("RESULT_JSON:" + json.dumps(to_native(res)))
