"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
import argparse, collections, time, os, json
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate
from _agent_utils import to_native, save_test_scores

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
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, temperature=1.0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.temperature = float(temperature)
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0
        # bucket offset: 6 scalars (one per bucket)
        self.b_bucket = np.zeros(6, dtype=np.float32)
        self.mb_bucket = np.zeros_like(self.b_bucket)
        self.vb_bucket = np.zeros_like(self.b_bucket)
        # per-user temperature (scalar per user, init at 1.0)
        self.user_temp = None
        self.user_temp_map = None
        self.m_user_temp = None
        self.v_user_temp = None

    def logits(self, X, bucket_ids=None, user_ids=None):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        z = self.b + self.W[X].sum(1) + inter
        # Add bucket offset
        if bucket_ids is not None:
            z += self.b_bucket[bucket_ids]
        # Add user bias (scalar per row, before user temp)
        if user_ids is not None and self.user_bias is not None:
            z += self.user_bias[user_ids]
        return z, E, S

    def step(self, X, y, bucket_ids=None, user_ids=None):
        B = len(y)
        z_raw, E, S = self.logits(X, bucket_ids, user_ids)
        # Apply user temperature (if available) BEFORE sigmoid
        if user_ids is not None and self.user_temp is not None:
            T = self.user_temp[user_ids]
            z = z_raw / np.clip(T, 0.2, 5.0)  # clamp for stability
        else:
            z = z_raw / self.temperature
        p = sigmoid(z * self.temperature)  # reapply temperature for grad (see below)
        g = ((p - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E) * self.temperature)
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        # Update bucket offset via Adam
        if bucket_ids is not None:
            gb_bucket = np.zeros(6, dtype=np.float32)
            np.add.at(gb_bucket, bucket_ids, g * self.temperature)
            self.mb_bucket *= b1; self.mb_bucket += (1 - b1) * gb_bucket
            self.vb_bucket *= b2; self.vb_bucket += (1 - b2) * (gb_bucket * gb_bucket)
            self.b_bucket -= self.lr * (self.mb_bucket / (1 - b1 ** self.t)) / (np.sqrt(self.vb_bucket / (1 - b2 ** self.t)) + eps)
        # Update user bias via Adam (train-only)
        if user_ids is not None and self.user_bias is not None:
            g_user = np.zeros(len(self.user_bias), dtype=np.float32)
            np.add.at(g_user, user_ids, g * self.temperature)
            self.m_user_bias *= b1; self.m_user_bias += (1 - b1) * g_user
            self.v_user_bias *= b2; self.v_user_bias += (1 - b2) * (g_user * g_user)
            self.user_bias -= self.lr * (self.m_user_bias / (1 - b1 ** self.t)) / (np.sqrt(self.v_user_bias / (1 - b2 ** self.t)) + eps)
        # Update user temperature via Adam (train-only) — *only* if user_ids provided
        if user_ids is not None and self.user_temp is not None:
            # gradient wrt T_user: dL/dT = dL/dz * dz/dT = g * (-z_raw / T^2)
            # use clamped T to avoid NaN
            T_clamped = np.clip(self.user_temp[user_ids], 0.2, 5.0)
            g_T = g * (-z_raw / (T_clamped ** 2)) * self.temperature  # chain rule
            # sum per-user
            g_user_temp = np.zeros(len(self.user_temp), dtype=np.float32)
            np.add.at(g_user_temp, user_ids, g_T)
            self.m_user_temp *= b1; self.m_user_temp += (1 - b1) * g_user_temp
            self.v_user_temp *= b2; self.v_user_temp += (1 - b2) * (g_user_temp * g_user_temp)
            self.user_temp -= self.lr * (self.m_user_temp / (1 - b1 ** self.t)) / (np.sqrt(self.v_user_temp / (1 - b2 ** self.t)) + eps)
        # Clamp user_temp to positive after update
        if self.user_temp is not None:
            np.clip(self.user_temp, 0.2, 5.0, out=self.user_temp)
        self.b -= self.lr * g.sum() / self.temperature
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def predict(self, X, bucket_ids=None, user_ids=None, bs=200_000):
        logits = []
        for i in range(0, len(X), bs):
            logits.append(self.logits(X[i:i + bs], bucket_ids[i:i + bs] if bucket_ids is not None else None,
                                      user_ids[i:i + bs] if user_ids is not None else None)[0])
        z_raw = np.concatenate(logits)
        # Apply user temperature at inference time
        if user_ids is not None and self.user_temp is not None:
            T = self.user_temp[user_ids]
            z = z_raw / np.clip(T, 0.2, 5.0)
        else:
            z = z_raw / self.temperature
        return sigmoid(z * self.temperature)

def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True, temperature=1.0):
    # Encode data first (train-only vocab, UNK for unseen)
    enc, dim = encode(splits)
    Xtr, ytr, u_tr = enc['train']; Xva, yva, u_va = enc['valid']; Xte, yte, u_te = enc['test']

    # Compute per-user train stats ONLY (no leakage)
    user_pos = collections.defaultdict(int)
    user_imp = collections.defaultdict(int)
    for i in range(len(ytr)):
        uid = u_tr[i]
        user_imp[uid] += 1
        user_pos[uid] += int(ytr[i])

    # Bucket into 5 bins: 0..5 -> bucket id = int(rate*5), mapped to [1..6] (0=UNK)
    bucket_vocab = {i: i + 1 for i in range(6)}
    UNK_BUCKET = 0

    def bucket_rate(uid):
        if user_imp[uid] == 0:
            return UNK_BUCKET
        r = user_pos[uid] / user_imp[uid]
        return bucket_vocab.get(int(np.floor(r * 5)), UNK_BUCKET)

    # Build bucket feature arrays for all splits (train-only vocab, UNK=0)
    bucket_tr = np.array([bucket_rate(uid) for uid in u_tr], dtype=np.int32)
    bucket_va = np.array([bucket_rate(uid) for uid in u_va], dtype=np.int32)
    bucket_te = np.array([bucket_rate(uid) for uid in u_te], dtype=np.int32)

    # Append bucket feature as last column
    Xtr_ext = np.hstack([Xtr, bucket_tr[:, None]])
    Xva_ext = np.hstack([Xva, bucket_va[:, None]])
    Xte_ext = np.hstack([Xte, bucket_te[:, None]])

    # bucket ids for bucket offset: raw bucket id = bucket_ids (0..5), NOT the +1-shifted vocab id
    bucket_ids_tr = bucket_tr - 1
    bucket_ids_tr[bucket_ids_tr < 0] = 0
    bucket_ids_va = bucket_va - 1
    bucket_ids_va[bucket_ids_va < 0] = 0
    bucket_ids_te = bucket_te - 1
    bucket_ids_te[bucket_ids_te < 0] = 0

    dim_ext = dim + 6  # dim from encode() + bucket vocab size (0..5, but 0=UNK)

    # ---------------- PER-USER BIAS + TEMP (train-only, no leakage) ----------------
    # Build dense user index map from train users only
    train_users_set = list(dict.fromkeys(u_tr))  # unique, preserving order
    user_to_idx = {uid: i for i, uid in enumerate(train_users_set)}
    n_users = len(train_users_set)

    # Initialize user bias and temperature, plus Adam buffers (zero init for bias, one init for temp)
    user_bias = np.zeros(n_users, dtype=np.float32)
    m_user_bias = np.zeros_like(user_bias)
    v_user_bias = np.zeros_like(user_bias)
    user_temp = np.ones(n_users, dtype=np.float32)  # start at 1.0
    m_user_temp = np.zeros_like(user_temp)
    v_user_temp = np.zeros_like(user_temp)

    # Map each split's user list to dense index (UNK=0 means bias/temp default)
    # Convert u_tr to numpy first (since it's a plain Python list)
    u_tr_arr = np.asarray(u_tr)
    u_va_arr = np.asarray(u_va)
    u_te_arr = np.asarray(u_te)

    def map_users_to_idx(u_list, map_fn):
        # Return an array of indices (len(u_list)), UNK users get index = 0
        idx = np.zeros(len(u_list), dtype=np.int32)
        for i, uid in enumerate(u_list):
            idx[i] = map_fn.get(uid, 0)
        return idx

    user_idx_tr = map_users_to_idx(u_tr, user_to_idx)
    user_idx_va = map_users_to_idx(u_va, user_to_idx)
    user_idx_te = map_users_to_idx(u_te, user_to_idx)

    m = FM(dim_ext, k=k, lr=lr, seed=seed, temperature=temperature)
    # Inject per-user bias and temp state into FM
    m.user_bias = user_bias
    m.user_temp = user_temp
    m.user_bias_map = user_to_idx
    m.m_user_bias = m_user_bias
    m.v_user_bias = v_user_bias
    m.m_user_temp = m_user_temp
    m.v_user_temp = v_user_temp

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = []
        for i in range(0, len(idx), bs):
            batch_idx = idx[i:i + bs]
            losses.append(m.step(Xtr_ext[batch_idx], ytr[batch_idx],
                                 bucket_ids=bucket_ids_tr[batch_idx],
                                 user_ids=user_idx_tr[batch_idx]))
        scores_valid = m.predict(Xva_ext, bucket_ids=bucket_ids_va, user_ids=user_idx_va)
        va = evaluate(u_va, yva, scores_valid)
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            # Save full state: V, W, b, temp, bucket_offset, user_bias, user_temp
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b), m.temperature,
                          m.b_bucket.copy(), m.user_bias.copy(), m.user_temp.copy())
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    # Restore best state
    m.V, m.W, m.b, _, m.b_bucket, m.user_bias, m.user_temp = best_state
    # Predict
    scores_valid = m.predict(Xva_ext, bucket_ids=bucket_ids_va, user_ids=user_idx_va)
    scores_test = m.predict(Xte_ext, bucket_ids=bucket_ids_te, user_ids=user_idx_te)
    return {'valid': evaluate(u_va, yva, scores_valid),
            'test':  evaluate(u_te, yte, scores_test),
            '_test_scores': scores_test}

if __name__ == "__main__":
    splits = load(os.environ["KUAIRAND_DATA_DIR"])
    res = run_fm(splits, k=16, lr=0.001, epochs=40, seed=0, verbose=False, temperature=1.0)
    save_test_scores(res.pop("_test_scores"))
    print("RESULT_JSON:" + json.dumps(to_native(res)))
