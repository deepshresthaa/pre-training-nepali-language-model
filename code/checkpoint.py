import numpy as np
import os


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flat(d: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict of arrays into 'a/b/c' keys."""
    out = {}
    for k, v in d.items():
        full = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, full))
        else:
            out[full] = np.asarray(v)
    return out


def _nest(flat: dict) -> dict:
    """Inverse of _flat: 'a/b/c' → nested dict."""
    out: dict = {}
    for key, val in flat.items():
        parts = key.split("/")
        d = out
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def collect_weights(embedding, pos_enc, encoder, mlm_head,
                    epoch: int = 0, step: int = 0,
                    loss_history: np.ndarray = None) -> dict:
    """
    Pull every trainable Parameter out of the model objects and return a flat
    dict ready for np.savez.

    Parameters
    ----------
    embedding   : InputEmbedding
    pos_enc     : PositionalEncoding
    encoder     : Encoder
    mlm_head    : MLMHead
    epoch, step : bookkeeping scalars
    loss_history: 1-D float32 array of per-step CE losses

    Returns
    -------
    flat dict  {str → np.ndarray}
    """
    weights: dict = {}

    # ── embedding ────────────────────────────────────────────────────────────
    weights["embed/weight"] = embedding.weight.data

    # ── positional encoding (fixed, but save for reproducibility) ────────────
    weights["pos_enc/pe"] = pos_enc.pe

    # ── encoder layers ───────────────────────────────────────────────────────
    for i, layer in enumerate(encoder.layers):
        p = f"encoder/layer_{i}"
        attn = layer.self_attention
        ff   = layer.feed_forward
        r0   = layer.residual_connections[0]
        r1   = layer.residual_connections[1]

        weights[f"{p}/attn/W_q"] = attn.W_q.data
        weights[f"{p}/attn/W_k"] = attn.W_k.data
        weights[f"{p}/attn/W_v"] = attn.W_v.data
        weights[f"{p}/attn/W_o"] = attn.W_o.data

        weights[f"{p}/ff/W1"] = ff.W1.data
        weights[f"{p}/ff/b1"] = ff.b1.data
        weights[f"{p}/ff/W2"] = ff.W2.data
        weights[f"{p}/ff/b2"] = ff.b2.data

        weights[f"{p}/res0/norm/alpha"] = r0.norm.alpha.data
        weights[f"{p}/res0/norm/bias"]  = r0.norm.bias.data
        weights[f"{p}/res1/norm/alpha"] = r1.norm.alpha.data
        weights[f"{p}/res1/norm/bias"]  = r1.norm.bias.data

    # ── encoder final norm ───────────────────────────────────────────────────
    weights["encoder/final_norm/alpha"] = encoder.norm.alpha.data
    weights["encoder/final_norm/bias"]  = encoder.norm.bias.data

    # ── MLM head ─────────────────────────────────────────────────────────────
    weights["mlm_head/W"] = mlm_head.W.data
    weights["mlm_head/b"] = mlm_head.b.data

    # ── metadata ─────────────────────────────────────────────────────────────
    weights["meta/epoch"] = np.array(epoch, dtype=np.int32)
    weights["meta/step"]  = np.array(step,  dtype=np.int32)
    weights["meta/train_loss_history"] = (
        np.array(loss_history, dtype=np.float32)
        if loss_history is not None
        else np.array([], dtype=np.float32)
    )

    return weights


def save_checkpoint(path: str, embedding, pos_enc, encoder, mlm_head,
                    epoch: int = 0, step: int = 0,
                    loss_history: np.ndarray = None) -> None:
    """
    Save all model weights to *path* (will add .npz if missing).

    Example
    -------
    >>> save_checkpoint("checkpoints/epoch_1.npz", embedding, pos_enc,
    ...                 encoder, mlm_head, epoch=1, step=400,
    ...                 loss_history=np.array([3.1, 2.8, 2.5]))
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    weights = collect_weights(embedding, pos_enc, encoder, mlm_head,
                              epoch, step, loss_history)
    np.savez(path, **weights)
    size_mb = os.path.getsize(path + ".npz" if not path.endswith(".npz") else path) / 1e6
    print(f"[checkpoint] saved → {path}  ({size_mb:.1f} MB,  epoch={epoch}, step={step})")


def load_checkpoint(path: str, embedding, pos_enc, encoder, mlm_head):
    """
    Load weights from *path* back into the model objects **in-place**.

    Returns
    -------
    epoch : int
    step  : int
    loss_history : np.ndarray  (float32, 1-D)

    Example
    -------
    >>> epoch, step, history = load_checkpoint("checkpoints/epoch_1.npz",
    ...                                         embedding, pos_enc,
    ...                                         encoder, mlm_head)
    """
    if not path.endswith(".npz"):
        path = path + ".npz"
    data = np.load(path)

    # ── embedding ────────────────────────────────────────────────────────────
    embedding.weight.data = data["embed/weight"].copy()

    # ── encoder layers ───────────────────────────────────────────────────────
    for i, layer in enumerate(encoder.layers):
        p   = f"encoder/layer_{i}"
        attn = layer.self_attention
        ff   = layer.feed_forward
        r0   = layer.residual_connections[0]
        r1   = layer.residual_connections[1]

        attn.W_q.data = data[f"{p}/attn/W_q"].copy()
        attn.W_k.data = data[f"{p}/attn/W_k"].copy()
        attn.W_v.data = data[f"{p}/attn/W_v"].copy()
        attn.W_o.data = data[f"{p}/attn/W_o"].copy()

        ff.W1.data = data[f"{p}/ff/W1"].copy()
        ff.b1.data = data[f"{p}/ff/b1"].copy()
        ff.W2.data = data[f"{p}/ff/W2"].copy()
        ff.b2.data = data[f"{p}/ff/b2"].copy()

        r0.norm.alpha.data = data[f"{p}/res0/norm/alpha"].copy()
        r0.norm.bias.data  = data[f"{p}/res0/norm/bias"].copy()
        r1.norm.alpha.data = data[f"{p}/res1/norm/alpha"].copy()
        r1.norm.bias.data  = data[f"{p}/res1/norm/bias"].copy()

    # ── encoder final norm ───────────────────────────────────────────────────
    encoder.norm.alpha.data = data["encoder/final_norm/alpha"].copy()
    encoder.norm.bias.data  = data["encoder/final_norm/bias"].copy()

    # ── MLM head ─────────────────────────────────────────────────────────────
    mlm_head.W.data = data["mlm_head/W"].copy()
    mlm_head.b.data = data["mlm_head/b"].copy()

    epoch        = int(data["meta/epoch"])
    step         = int(data["meta/step"])
    loss_history = data["meta/train_loss_history"].copy()

    print(f"[checkpoint] loaded ← {path}  (epoch={epoch}, step={step}, "
          f"loss_history length={len(loss_history)})")
    return epoch, step, loss_history


def print_weight_summary(embedding, encoder, mlm_head) -> None:
    """Print a human-readable table of every parameter tensor and its stats."""
    print("\n" + "═" * 72)
    print(f"{'LAYER':<45} {'SHAPE':<20} {'MEAN':>7}  {'STD':>7}")
    print("═" * 72)

    def row(name, arr):
        print(f"  {name:<43} {str(arr.shape):<20} {arr.mean():+7.4f}  {arr.std():7.4f}")

    row("embed/weight", embedding.weight.data)

    for i, layer in enumerate(encoder.layers):
        p    = f"layer_{i}"
        attn = layer.self_attention
        ff   = layer.feed_forward
        row(f"{p}/attn/W_q", attn.W_q.data)
        row(f"{p}/attn/W_k", attn.W_k.data)
        row(f"{p}/attn/W_v", attn.W_v.data)
        row(f"{p}/attn/W_o", attn.W_o.data)
        row(f"{p}/ff/W1",    ff.W1.data)
        row(f"{p}/ff/W2",    ff.W2.data)

    row("encoder/final_norm/alpha", encoder.norm.alpha.data)
    row("mlm_head/W", mlm_head.W.data)
    row("mlm_head/b", mlm_head.b.data)
    print("═" * 72 + "\n")
