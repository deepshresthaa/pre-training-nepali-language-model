import numpy as np
import math


# ── helpers ──────────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def dropout(x, rate, training=True):
    if not training or rate == 0.0:
        return x
    mask = (np.random.rand(*x.shape) > rate).astype(x.dtype)
    return x * mask / (1.0 - rate)


# ── parameter container (replaces nn.Parameter / nn.Module) ──────────────────

class Parameter:
    """Wraps a numpy array so it's clearly a trainable weight."""
    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(self.data)


def xavier_uniform(shape):
    fan_in, fan_out = shape[-2], shape[-1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape).astype(np.float32)


# ── InputEmbedding ────────────────────────────────────────────────────────────

class InputEmbedding:
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.weight     = Parameter(xavier_uniform((vocab_size, d_model)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (batch, seq_len)  →  (batch, seq_len, d_model)
        return self.weight.data[x] * math.sqrt(self.d_model)


# ── PositionalEncoding ────────────────────────────────────────────────────────

class PositionalEncoding:
    def __init__(self, d_model: int, seq_len: int, dropout_rate: float):
        self.dropout_rate = dropout_rate
        pe       = np.zeros((seq_len, d_model), dtype=np.float32)
        position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)   # (seq_len, 1)
        j        = np.arange(0, d_model // 2, dtype=np.float32)
        div_term = 10000 ** (2 * j / d_model)                                # (d_model//2,)
        pe[:, 0::2] = np.sin(position / div_term)
        pe[:, 1::2] = np.cos(position / div_term)
        self.pe = pe[np.newaxis, :, :]                                        # (1, seq_len, d_model)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.shape[1], :]
        return dropout(x, self.dropout_rate, training)


# ── LayerNormalization ────────────────────────────────────────────────────────

class LayerNormalization:
    def __init__(self, features: int, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.alpha   = Parameter(np.ones(features,  dtype=np.float32))
        self.bias    = Parameter(np.zeros(features, dtype=np.float32))

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std  = x.std(axis=-1,  keepdims=True)
        return self.alpha.data * (x - mean) / (std + self.epsilon) + self.bias.data


# ── FeedForward ───────────────────────────────────────────────────────────────

class FeedForward:
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        self.dropout_rate = dropout_rate
        self.W1 = Parameter(xavier_uniform((d_model, d_ff)))
        self.b1 = Parameter(np.zeros(d_ff,    dtype=np.float32))
        self.W2 = Parameter(xavier_uniform((d_ff, d_model)))
        self.b2 = Parameter(np.zeros(d_model, dtype=np.float32))

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        # x: (batch, seq_len, d_model)
        x = relu(x @ self.W1.data + self.b1.data)          # (batch, seq_len, d_ff)
        x = dropout(x, self.dropout_rate, training)
        return x @ self.W2.data + self.b2.data              # (batch, seq_len, d_model)


# ── MultiHeadAttention ────────────────────────────────────────────────────────

class MultiHeadAttention:
    def __init__(self, d_model: int, h: int, dropout_rate: float):
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model      = d_model
        self.h            = h
        self.d_k          = d_model // h
        self.dropout_rate = dropout_rate
        self.W_q = Parameter(xavier_uniform((d_model, d_model)))
        self.W_k = Parameter(xavier_uniform((d_model, d_model)))
        self.W_v = Parameter(xavier_uniform((d_model, d_model)))
        self.W_o = Parameter(xavier_uniform((d_model, d_model)))

    @staticmethod
    def attention(query, key, value, mask, dropout_rate, training):
        d_k = query.shape[-1]
        # (batch, h, seq_len, seq_len)
        scores = query @ key.swapaxes(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        weights = softmax(scores, axis=-1)
        weights = dropout(weights, dropout_rate, training)
        return weights @ value, weights                     # (batch, h, seq_len, d_k)

    def forward(self, q, k, v, mask=None, training: bool = True):
        B, S, _ = q.shape

        def project(x, W):
            # (batch, seq_len, d_model) → (batch, h, seq_len, d_k)
            return (x @ W.data).reshape(B, -1, self.h, self.d_k).swapaxes(1, 2)

        Q = project(q, self.W_q)
        K = project(k, self.W_k)
        V = project(v, self.W_v)

        x, self.attention_weights = MultiHeadAttention.attention(
            Q, K, V, mask, self.dropout_rate, training
        )
        # (batch, h, seq_len, d_k) → (batch, seq_len, d_model)
        x = x.swapaxes(1, 2).reshape(B, S, self.d_model)
        return x @ self.W_o.data                            # (batch, seq_len, d_model)


# ── ResidualConnection ────────────────────────────────────────────────────────

class ResidualConnection:
    def __init__(self, features: int, dropout_rate: float):
        self.dropout_rate = dropout_rate
        self.norm         = LayerNormalization(features)

    def forward(self, x: np.ndarray, sublayer, training: bool = True) -> np.ndarray:
        return x + dropout(sublayer(self.norm.forward(x)), self.dropout_rate, training)


# ── EncoderBlock ──────────────────────────────────────────────────────────────

class EncoderBlock:
    def __init__(self, features: int, self_attention: MultiHeadAttention,
                 feed_forward: FeedForward, dropout_rate: float):
        self.self_attention       = self_attention
        self.feed_forward         = feed_forward
        self.residual_connections = [
            ResidualConnection(features, dropout_rate),
            ResidualConnection(features, dropout_rate),
        ]

    def forward(self, x: np.ndarray, src_mask=None, training: bool = True) -> np.ndarray:
        x = self.residual_connections[0].forward(
            x,
            lambda z: self.self_attention.forward(z, z, z, src_mask, training),
            training,
        )
        x = self.residual_connections[1].forward(
            x,
            lambda z: self.feed_forward.forward(z, training),
            training,
        )
        return x


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder:
    def __init__(self, features: int, layers: list):
        self.layers = layers
        self.norm   = LayerNormalization(features)

    def forward(self, x: np.ndarray, mask=None, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, mask, training)
        return self.norm.forward(x)


# ── build_encoder factory ─────────────────────────────────────────────────────

def build_encoder(
    src_vocab_size: int,
    src_seq_len:    int,
    d_model:        int   = 4096,
    N:              int   = 6,
    h:              int   = 32,
    dropout_rate:   float = 0.1,
    d_ff:           int   = 4*4096,
):
    embedding = InputEmbedding(src_vocab_size, d_model)
    pos_enc   = PositionalEncoding(d_model, src_seq_len, dropout_rate)

    layers = []
    for _ in range(N):
        layers.append(EncoderBlock(
            features       = d_model,
            self_attention = MultiHeadAttention(d_model, h, dropout_rate),
            feed_forward   = FeedForward(d_model, d_ff, dropout_rate),
            dropout_rate   = dropout_rate,
        ))

    encoder = Encoder(d_model, layers)
    return embedding, pos_enc, encoder



