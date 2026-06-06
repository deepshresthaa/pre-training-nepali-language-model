import numpy as np
import math
import argparse
import os
import time
import psutil

from encoder import (
    Parameter, xavier_uniform, softmax, relu, dropout,
    InputEmbedding, PositionalEncoding, LayerNormalization,
    FeedForward, MultiHeadAttention, ResidualConnection,
    EncoderBlock, Encoder, build_encoder,
)
from checkpoint import save_checkpoint, load_checkpoint, print_weight_summary


# ═══════════════════════════════════════════════════════════════════════════════
# Vocab-file tokenizer  (no SentencePiece binary required)
# ═══════════════════════════════════════════════════════════════════════════════

class VocabTokenizer:
    """
    Minimal tokenizer built directly from nepali_bpe.vocab.

    File format: one piece per line  (optionally "piece<TAB>score").
    Special tokens are injected at fixed ids:
        0  <pad>   1  <mask>   2  <cls>   3  <sep>   4  <unk>
    All vocab pieces start at id 5.
    """
    PAD_ID  = 0
    MASK_ID = 1
    CLS_ID  = 2
    SEP_ID  = 3
    UNK_ID  = 4

    def __init__(self, vocab_path: str):
        self.piece2id = {
            "<pad>":  self.PAD_ID,
            "<mask>": self.MASK_ID,
            "<cls>":  self.CLS_ID,
            "<sep>":  self.SEP_ID,
            "<unk>":  self.UNK_ID,
        }
        idx = 5
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                piece = line.strip().split("\t")[0]
                if piece and piece not in self.piece2id:
                    self.piece2id[piece] = idx
                    idx += 1
        self.id2piece   = {v: k for k, v in self.piece2id.items()}
        self.vocab_size = idx
        print(f"  [Tokenizer] {self.vocab_size:,} pieces loaded from {vocab_path}")

    def encode(self, text: str) -> list:
        """Space-split word lookup; unknown words → <unk>."""
        return [self.piece2id.get(w, self.UNK_ID) for w in text.strip().split()]

    def special_ids(self) -> set:
        return {self.PAD_ID, self.MASK_ID, self.CLS_ID,
                self.SEP_ID, self.UNK_ID}


# ═══════════════════════════════════════════════════════════════════════════════
# Safe batch-size heuristic
# ═══════════════════════════════════════════════════════════════════════════════

def get_safe_batch_size(d_model: int = 256) -> int:
    ram_gb = psutil.virtual_memory().available / 1e9
    if   ram_gb < 4:  return 1
    elif ram_gb < 8:  return 2
    elif ram_gb < 16: return 4
    else:             return 8


# ═══════════════════════════════════════════════════════════════════════════════
# Data pipeline
# ═══════════════════════════════════════════════════════════════════════════════

SEQ_LEN = 512


def encode_file(file_path: str, tokenizer: VocabTokenizer,
                max_lines: int = None) -> list:
    """Stream-tokenise ne.txt; append <sep> after every line."""
    buf = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            buf.extend(tokenizer.encode(line))
            buf.append(tokenizer.SEP_ID)
    return buf


def create_sequences(token_ids: list, seq_len: int, pad_id: int) -> np.ndarray:
    """Chunk flat token list into (N, seq_len) int32 array."""
    samples = []
    for i in range(0, len(token_ids), seq_len):
        chunk = token_ids[i : i + seq_len]
        if len(chunk) < seq_len:
            chunk = chunk + [pad_id] * (seq_len - len(chunk))
        samples.append(chunk)
    return np.array(samples, dtype=np.int32)


def batch_generator(data: np.ndarray, batch_size: int):
    """Yield randomly shuffled (batch_size, seq_len) batches."""
    idx  = np.random.permutation(len(data))
    data = data[idx]
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        if len(batch) == batch_size:
            yield batch


def load_dataset(file_path: str, tokenizer: VocabTokenizer,
                 seq_len: int, max_lines: int = None):
    if file_path and os.path.exists(file_path):
        print(f"  Loading: {file_path}")
        tids = encode_file(file_path, tokenizer, max_lines)
        seqs = create_sequences(tids, seq_len, tokenizer.PAD_ID)
        print(f"  {len(tids):,} tokens → {len(seqs):,} sequences × {seq_len}")
        return seqs
    print("  [Warning] data file not found — using synthetic data.")
    return None


def generate_batch_synthetic(batch_size, seq_len, tokenizer):
    lo  = max(tokenizer.special_ids()) + 1
    hi  = tokenizer.vocab_size
    tok = np.random.randint(lo, hi, (batch_size, seq_len - 2), dtype=np.int32)
    cls = np.full((batch_size, 1), tokenizer.CLS_ID, dtype=np.int32)
    sep = np.full((batch_size, 1), tokenizer.SEP_ID, dtype=np.int32)
    return np.concatenate([cls, tok, sep], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# MLM Head
# ═══════════════════════════════════════════════════════════════════════════════

class MLMHead:
    def __init__(self, d_model, vocab_size, embedding, tie_weights=True):
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.tie        = tie_weights
        self.embedding  = embedding
        self.W    = embedding.weight if tie_weights else \
                    Parameter(xavier_uniform((d_model, vocab_size)))
        self.b    = Parameter(np.zeros(vocab_size, dtype=np.float32))
        self.norm = LayerNormalization(d_model)

    def forward(self, x):
        x = self.norm.forward(x)
        self._x_normed = x
        W = self.W.data.T if self.tie else self.W.data
        return x @ W + self.b.data

    def backward(self, d_logits):
        self.b.grad += d_logits.sum(axis=(0, 1))
        B_S = d_logits.shape[0] * d_logits.shape[1]
        if self.tie:
            d_xn = d_logits @ self.W.data
            self.W.grad += (d_logits.reshape(B_S, self.vocab_size).T
                            @ self._x_normed.reshape(B_S, self.d_model))
        else:
            d_xn = d_logits @ self.W.data.T
            self.W.grad += (self._x_normed.reshape(B_S, self.d_model).T
                            @ d_logits.reshape(B_S, self.vocab_size))
        alpha = self.norm.alpha.data
        std   = self._x_normed.std(axis=-1, keepdims=True) + 1e-6
        x_hat = self._x_normed / std
        self.norm.alpha.grad += (d_xn * x_hat).sum(axis=(0, 1))
        self.norm.bias.grad  += d_xn.sum(axis=(0, 1))
        return d_xn * alpha / std


# ═══════════════════════════════════════════════════════════════════════════════
# Adam
# ═══════════════════════════════════════════════════════════════════════════════

class Adam:
    def __init__(self, params, lr=3e-4, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01):
        self.params = params; self.lr = lr
        self.beta1 = beta1;   self.beta2 = beta2
        self.eps = eps;       self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        lr_t = self.lr * math.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            p.data   -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# MLM masking
# ═══════════════════════════════════════════════════════════════════════════════

def apply_mlm_mask(tokens, mask_prob, mask_id, vocab_size, special_ids):
    B, S = tokens.shape
    masked = tokens.copy()
    labels = np.full((B, S), -1, dtype=np.int32)
    cand   = np.ones((B, S), dtype=bool)
    for sid in special_ids:
        cand &= (tokens != sid)
    rand = np.random.rand(B, S)
    sel  = cand & (rand < mask_prob)
    labels[sel] = tokens[sel]
    sub   = np.random.rand(B, S)
    masked[sel & (sub < 0.80)] = mask_id
    rnd = sel & (sub >= 0.80) & (sub < 0.90)
    if rnd.sum():
        lo = max(special_ids) + 1
        masked[rnd] = np.random.randint(lo, vocab_size, size=rnd.sum())
    return masked, labels, sel


# ═══════════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════════

def mlm_cross_entropy(logits, labels, mask):
    probs = softmax(logits, axis=-1)
    pb, ps = np.where(mask)
    if len(pb) == 0:
        return 0.0, np.zeros_like(logits)
    ps_ = probs[pb, ps, :]; ls_ = labels[pb, ps]; N = len(pb)
    loss = -np.log(ps_[np.arange(N), ls_] + 1e-12).mean()
    d = np.zeros_like(logits)
    dp = ps_.copy(); dp[np.arange(N), ls_] -= 1.0
    d[pb, ps, :] = dp / N
    return loss, d


# ═══════════════════════════════════════════════════════════════════════════════
# Backward helpers
# ═══════════════════════════════════════════════════════════════════════════════

def layer_norm_backward(d_out, x_in, ln):
    eps, alpha = ln.epsilon, ln.alpha.data
    mean = x_in.mean(axis=-1, keepdims=True)
    var  = x_in.var(axis=-1,  keepdims=True)
    std  = np.sqrt(var + eps)
    x_hat = (x_in - mean) / std
    N = x_in.shape[-1]
    ln.alpha.grad += (d_out * x_hat).reshape(-1, N).sum(axis=0)
    ln.bias.grad  += d_out.reshape(-1, N).sum(axis=0)
    d_xh  = d_out * alpha
    d_var = (-0.5 * d_xh * (x_in - mean) / (var + eps)**1.5).sum(axis=-1, keepdims=True)
    d_mu  = (-d_xh / std).sum(axis=-1, keepdims=True) \
            + d_var * (-2.0 * (x_in - mean)).mean(axis=-1, keepdims=True)
    return d_xh / std + d_var * 2.0 * (x_in - mean) / N + d_mu / N


def feed_forward_backward(d_out, x_in, ff):
    z1 = x_in @ ff.W1.data + ff.b1.data
    a1 = relu(z1)
    B, S, D = d_out.shape; fd = ff.W2.data.shape[0]
    ff.b2.grad += d_out.reshape(-1, D).sum(axis=0)
    ff.W2.grad += a1.reshape(-1, fd).T @ d_out.reshape(-1, D)
    dz1 = (d_out @ ff.W2.data.T) * (z1 > 0).astype(np.float32)
    ff.b1.grad += dz1.reshape(-1, fd).sum(axis=0)
    ff.W1.grad += x_in.reshape(-1, x_in.shape[-1]).T @ dz1.reshape(-1, fd)
    return dz1 @ ff.W1.data.T


def attention_backward(d_out, q_in, attn):
    B, S, D = d_out.shape; h, dk = attn.h, attn.d_k
    def proj(x, W): return (x @ W.data).reshape(B, S, h, dk).swapaxes(1, 2)
    Q = proj(q_in, attn.W_q); K = proj(q_in, attn.W_k); V = proj(q_in, attn.W_v)
    w = softmax(Q @ K.swapaxes(-2, -1) / math.sqrt(dk), axis=-1)
    cc = (w @ V).swapaxes(1, 2).reshape(B, S, D)
    attn.W_o.grad += cc.reshape(-1, D).T @ d_out.reshape(-1, D)
    dh = (d_out @ attn.W_o.data.T).reshape(B, S, h, dk).swapaxes(1, 2)
    dV = w.swapaxes(-2, -1) @ dh
    dw = dh @ V.swapaxes(-2, -1)
    ds = w * (dw - (dw * w).sum(axis=-1, keepdims=True)) / math.sqrt(dk)
    dQ = ds @ K; dK = ds.swapaxes(-2, -1) @ Q
    dQf = dQ.swapaxes(1, 2).reshape(B, S, D)
    dKf = dK.swapaxes(1, 2).reshape(B, S, D)
    dVf = dV.swapaxes(1, 2).reshape(B, S, D)
    attn.W_q.grad += q_in.reshape(-1, D).T @ dQf.reshape(-1, D)
    attn.W_k.grad += q_in.reshape(-1, D).T @ dKf.reshape(-1, D)
    attn.W_v.grad += q_in.reshape(-1, D).T @ dVf.reshape(-1, D)
    return (dQf + dKf + dVf) @ attn.W_q.data.T


def encoder_block_backward(d_out, x_in, block):
    attn, ff = block.self_attention, block.feed_forward
    r0, r1 = block.residual_connections[0], block.residual_connections[1]
    xn  = r0.norm.forward(x_in)
    x1  = x_in + dropout(attn.forward(xn, xn, xn, training=False), 0.0)
    x1l = r1.norm.forward(x1.copy())
    dff = feed_forward_backward(d_out.copy(), x1l, ff)
    dx1 = d_out.copy() + layer_norm_backward(dff, x1.copy(), r1.norm)
    da  = attention_backward(dx1.copy(), r0.norm.forward(x_in), attn)
    return dx1.copy() + layer_norm_backward(da, x_in.copy(), r0.norm)


def encoder_backward(d_out, x_pe, encoder):
    xs, x = [x_pe], x_pe
    for blk in encoder.layers:
        xn  = blk.residual_connections[0].norm.forward(x)
        xa  = x + blk.self_attention.forward(xn, xn, xn, training=False)
        xn2 = blk.residual_connections[1].norm.forward(xa)
        x   = xa + blk.feed_forward.forward(xn2, training=False)
        xs.append(x)
    dx = layer_norm_backward(d_out, xs[-1], encoder.norm)
    for i in range(len(encoder.layers) - 1, -1, -1):
        dx = encoder_block_backward(dx, xs[i], encoder.layers[i])
    return dx


def embedding_backward(d_x, tokens, embedding):
    np.add.at(embedding.weight.grad, tokens,
              d_x / math.sqrt(embedding.d_model))


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter collection
# ═══════════════════════════════════════════════════════════════════════════════

def collect_parameters(embedding, encoder, mlm_head):
    p = [embedding.weight]
    for blk in encoder.layers:
        a, ff = blk.self_attention, blk.feed_forward
        p += [a.W_q, a.W_k, a.W_v, a.W_o, ff.W1, ff.b1, ff.W2, ff.b2]
        for rc in blk.residual_connections:
            p += [rc.norm.alpha, rc.norm.bias]
    p += [encoder.norm.alpha, encoder.norm.bias]
    if not mlm_head.tie:
        p.append(mlm_head.W)
    p += [mlm_head.b, mlm_head.norm.alpha, mlm_head.norm.bias]
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    n_epochs        = 3,
    steps_per_epoch = 1000,
    batch_size      = None,
    seq_len         = SEQ_LEN,
    d_model         = 256,
    n_layers        = 4,
    n_heads         = 8,
    d_ff            = 1024,
    dropout_rate    = 0.1,
    lr              = 3e-4,
    mask_prob       = 0.15,
    checkpoint_dir  = "checkpoints",
    resume_path     = None,
    log_every       = 50,
    data_file       = "../dataset/ne.txt",
    vocab_file      = "../dataset/nepali_bpe.vocab",
    max_lines       = None,
):
    np.random.seed(42)

    tokenizer  = VocabTokenizer(vocab_file)
    VOCAB_SIZE = tokenizer.vocab_size
    SPECIAL    = tokenizer.special_ids()
    MASK_ID    = tokenizer.MASK_ID

    if batch_size is None:
        batch_size = get_safe_batch_size(d_model)

    print("\n" + "═" * 66)
    print("  Nepali Text LM — MLM Training")
    print("═" * 66)
    print(f"  vocab={VOCAB_SIZE:,}  d_model={d_model}  layers={n_layers}"
          f"  heads={n_heads}  d_ff={d_ff}")
    print(f"  batch={batch_size}  seq_len={seq_len}  mask={mask_prob:.0%}  lr={lr}")
    print("═" * 66 + "\n")

    dataset = load_dataset(data_file, tokenizer, seq_len, max_lines)

    embedding, pos_enc, encoder = build_encoder(
        src_vocab_size = VOCAB_SIZE,
        src_seq_len    = seq_len,
        d_model        = d_model,
        N              = n_layers,
        h              = n_heads,
        dropout_rate   = dropout_rate,
        d_ff           = d_ff,
    )
    mlm_head  = MLMHead(d_model, VOCAB_SIZE, embedding, tie_weights=True)
    params    = collect_parameters(embedding, encoder, mlm_head)
    optimiser = Adam(params, lr=lr, weight_decay=0.01)

    start_epoch = 0; global_step = 0; loss_history = []
    if resume_path and os.path.exists(resume_path):
        start_epoch, global_step, hist = load_checkpoint(
            resume_path, embedding, pos_enc, encoder, mlm_head)
        loss_history = list(hist); start_epoch += 1

    os.makedirs(checkpoint_dir, exist_ok=True)
    print_weight_summary(embedding, encoder, mlm_head)

    for epoch in range(start_epoch, start_epoch + n_epochs):
        epoch_losses = []
        t0 = time.time()
        biter = batch_generator(dataset, batch_size) if dataset is not None else None

        for step in range(steps_per_epoch):
            if biter is not None:
                try:
                    tokens = next(biter)
                except StopIteration:
                    biter  = batch_generator(dataset, batch_size)
                    tokens = next(biter)
            else:
                tokens = generate_batch_synthetic(batch_size, seq_len, tokenizer)

            masked, labels, sel = apply_mlm_mask(
                tokens, mask_prob, MASK_ID, VOCAB_SIZE, SPECIAL)

            optimiser.zero_grad()

            x   = embedding.forward(masked)
            x   = pos_enc.forward(x, training=True)
            xp  = x.copy()
            mem = encoder.forward(x, training=True)
            lg  = mlm_head.forward(mem)

            loss, dlg = mlm_cross_entropy(lg, labels, sel)

            dm  = mlm_head.backward(dlg)
            dx  = encoder_backward(dm, xp, encoder)
            embedding_backward(dx, masked, embedding)

            gn = math.sqrt(sum(float(np.sum(p.grad**2)) for p in params))
            if gn > 1.0:
                sf = 1.0 / (gn + 1e-6)
                for p in params: p.grad *= sf

            optimiser.step()
            epoch_losses.append(loss); loss_history.append(loss); global_step += 1

            if (step + 1) % log_every == 0 or step == 0:
                nm = sel.sum()
                print(f"  epoch {epoch+1:>3} | step {step+1:>5}/{steps_per_epoch}"
                      f" | loss {loss:.4f}"
                      f" | masked {nm} ({nm/(batch_size*seq_len)*100:.1f}%)"
                      f" | gnorm {gn:.3f}")

        ml = float(np.mean(epoch_losses))
        print(f"\n  ▶ Epoch {epoch+1} done | mean_loss={ml:.4f}"
              f" | {time.time()-t0:.1f}s\n")
        ckpt = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.npz")
        save_checkpoint(ckpt, embedding, pos_enc, encoder, mlm_head,
                        epoch=epoch+1, step=global_step,
                        loss_history=np.array(loss_history, dtype=np.float32))

    print_weight_summary(embedding, encoder, mlm_head)
    print("Training complete.")
    return embedding, pos_enc, encoder, mlm_head, loss_history


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_file",      default="../dataset/ne.txt")
    p.add_argument("--vocab_file",     default="../dataset/nepali_bpe.vocab")
    p.add_argument("--resume",         default=None)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--steps",          type=int,   default=1000)
    p.add_argument("--batch",          type=int,   default=None)
    p.add_argument("--seq_len",        type=int,   default=512)
    p.add_argument("--d_model",        type=int,   default=256)
    p.add_argument("--layers",         type=int,   default=4)
    p.add_argument("--heads",          type=int,   default=8)
    p.add_argument("--d_ff",           type=int,   default=1024)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--mask_prob",      type=float, default=0.15)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_every",      type=int,   default=50)
    p.add_argument("--max_lines",      type=int,   default=None)
    a = p.parse_args()

    train(
        n_epochs        = a.epochs,
        steps_per_epoch = a.steps,
        batch_size      = a.batch,
        seq_len         = a.seq_len,
        d_model         = a.d_model,
        n_layers        = a.layers,
        n_heads         = a.heads,
        d_ff            = a.d_ff,
        lr              = a.lr,
        mask_prob       = a.mask_prob,
        checkpoint_dir  = a.checkpoint_dir,
        resume_path     = a.resume,
        log_every       = a.log_every,
        data_file       = a.data_file,
        vocab_file      = a.vocab_file,
        max_lines       = a.max_lines,
    )