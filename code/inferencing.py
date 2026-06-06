"""
inferencing.py  —  Inference / testing for the trained Nepali Text LM
─────────────────────────────────────────────────────────────────────────────
Folder layout expected
  NEPAL.../
  ├── code/
  │   ├── encoder.py
  │   ├── checkpoint.py
  │   ├── protein_lm.py       ← VocabTokenizer lives here
  │   └── inferencing.py      ← this file
  ├── dataset/
  │   └── nepali_bpe.vocab
  └── checkpoints/
      └── epoch_3.npz

Four inference modes
─────────────────────
  1. fill_mask      — give a sentence with [MASK], model predicts the token
  2. score          — log-probability score of a complete sentence
  3. embed          — get the [CLS] embedding vector for a sentence
  4. top_k          — show top-K predictions for every [MASK] in a sentence

Usage
─────
  # fill a mask
  python inferencing.py --mode fill_mask \
      --text "नेपाल एक [MASK] देश हो"

  # score a sentence
  python inferencing.py --mode score \
      --text "नेपाल एक सुन्दर देश हो"

  # get embedding vector (prints shape + first 8 dims)
  python inferencing.py --mode embed \
      --text "नेपाल एक सुन्दर देश हो"

  # top-5 predictions for each mask
  python inferencing.py --mode top_k --top_k 5 \
      --text "नेपाल एक [MASK] देश हो र यहाँ [MASK] छ"
"""

import numpy as np
import math
import argparse
import os

from encoder import (
    softmax, build_encoder,
)
from checkpoint import load_checkpoint
from training import VocabTokenizer          # reuse the tokenizer we built


# ═══════════════════════════════════════════════════════════════════════════════
# Model loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str,
               vocab_file: str,
               seq_len: int   = 512,
               d_model: int   = 256,
               n_layers: int  = 4,
               n_heads: int   = 8,
               d_ff: int      = 1024,
               dropout_rate: float = 0.1):
    tokenizer = VocabTokenizer(vocab_file)

    embedding, pos_enc, encoder = build_encoder(
        src_vocab_size = tokenizer.vocab_size,
        src_seq_len    = seq_len,
        d_model        = d_model,
        N              = n_layers,
        h              = n_heads,
        dropout_rate   = dropout_rate,
        d_ff           = d_ff,
    )

    # MLMHead is needed only to satisfy load_checkpoint signature
    from training import MLMHead
    mlm_head = MLMHead(d_model, tokenizer.vocab_size, embedding, tie_weights=True)

    epoch, step, loss_history = load_checkpoint(
        checkpoint_path, embedding, pos_enc, encoder, mlm_head)

    print(f"  Loaded checkpoint: {checkpoint_path}")
    print(f"  Trained for {epoch} epoch(s), {step} steps")
    if len(loss_history):
        print(f"  Last recorded loss: {loss_history[-1]:.4f}")

    return tokenizer, embedding, pos_enc, encoder, mlm_head


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenise a single sentence → padded (1, seq_len) array
# ═══════════════════════════════════════════════════════════════════════════════

MASK_PLACEHOLDER = "[MASK]"   # what the user types in their input string

def tokenise_input(text: str, tokenizer: VocabTokenizer,
                   seq_len: int) -> tuple:
    """
    Convert a raw string into a (1, seq_len) token array.
    '[MASK]' in the text is converted to MASK_ID automatically.

    Returns
    -------
    token_ids   : (1, seq_len) int32
    mask_positions : list of int   — indices where [MASK] appears
    """
    words    = text.strip().split()
    ids      = [tokenizer.CLS_ID]

    for w in words:
        if w == MASK_PLACEHOLDER:
            ids.append(tokenizer.MASK_ID)
        else:
            ids.append(tokenizer.piece2id.get(w, tokenizer.UNK_ID))

    ids.append(tokenizer.SEP_ID)

    # record mask positions (before padding)
    mask_positions = [i for i, tok in enumerate(ids) if tok == tokenizer.MASK_ID]

    # pad / truncate to seq_len
    if len(ids) < seq_len:
        ids = ids + [tokenizer.PAD_ID] * (seq_len - len(ids))
    else:
        ids = ids[:seq_len]

    return np.array([ids], dtype=np.int32), mask_positions


# ═══════════════════════════════════════════════════════════════════════════════
# Forward pass (no masking, no dropout)
# ═══════════════════════════════════════════════════════════════════════════════

def forward(token_ids, embedding, pos_enc, encoder, mlm_head):
    """
    Full forward pass → logits (1, seq_len, vocab_size).
    Dropout is disabled (training=False).
    """
    x      = embedding.forward(token_ids)
    x      = pos_enc.forward(x, training=False)
    memory = encoder.forward(x, training=False)
    logits = mlm_head.forward(memory)                  # (1, seq_len, vocab_size)
    return logits, memory


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 1 — fill_mask
# ═══════════════════════════════════════════════════════════════════════════════

def fill_mask(text: str, tokenizer, embedding, pos_enc, encoder, mlm_head,
              seq_len: int, top_k: int = 1):
    """
    Predict the most likely token(s) at each [MASK] position.
    """
    token_ids, mask_positions = tokenise_input(text, tokenizer, seq_len)

    if not mask_positions:
        print("  [fill_mask] No [MASK] token found in input. "
              "Add '[MASK]' where you want a prediction.")
        return

    logits, _ = forward(token_ids, embedding, pos_enc, encoder, mlm_head)
    probs     = softmax(logits[0], axis=-1)            # (seq_len, vocab_size)

    words = text.strip().split()
    print(f"\n  Input : {text}")

    for pos in mask_positions:
        top_ids    = np.argsort(probs[pos])[::-1][:top_k]
        top_probs  = probs[pos][top_ids]
        top_pieces = [tokenizer.id2piece.get(i, f"<id:{i}>") for i in top_ids]

        print(f"\n  [MASK] at position {pos}:")
        for piece, prob in zip(top_pieces, top_probs):
            filled = text.replace(MASK_PLACEHOLDER, f"[{piece}]", 1)
            print(f"    {piece:<20}  prob={prob:.4f}   → \"{filled}\"")


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 2 — score
# ═══════════════════════════════════════════════════════════════════════════════

def score_sentence(text: str, tokenizer, embedding, pos_enc, encoder, mlm_head,
                   seq_len: int):
    """
    Compute the mean log-probability of every non-special token in the sentence
    (pseudo-log-likelihood, a standard MLM scoring approach).

    Higher (less negative) = model thinks this sentence is more natural.
    """
    words    = text.strip().split()
    base_ids = [tokenizer.CLS_ID]
    for w in words:
        base_ids.append(tokenizer.piece2id.get(w, tokenizer.UNK_ID))
    base_ids.append(tokenizer.SEP_ID)

    special = tokenizer.special_ids()
    scorable_positions = [i for i, t in enumerate(base_ids)
                          if t not in special]

    if not scorable_positions:
        print("  [score] No scorable tokens found.")
        return

    total_log_prob = 0.0

    for pos in scorable_positions:
        # mask the token at this position, run forward, read back log p(true)
        masked = base_ids.copy()
        true_id        = masked[pos]
        masked[pos]    = tokenizer.MASK_ID

        # pad to seq_len
        if len(masked) < seq_len:
            masked = masked + [tokenizer.PAD_ID] * (seq_len - len(masked))
        else:
            masked = masked[:seq_len]

        token_ids = np.array([masked], dtype=np.int32)
        logits, _ = forward(token_ids, embedding, pos_enc, encoder, mlm_head)
        probs     = softmax(logits[0], axis=-1)        # (seq_len, vocab)
        log_p     = math.log(float(probs[pos, true_id]) + 1e-12)
        total_log_prob += log_p

    mean_log_prob = total_log_prob / len(scorable_positions)
    perplexity    = math.exp(-mean_log_prob)

    print(f"\n  Input          : {text}")
    print(f"  Mean log-prob  : {mean_log_prob:.4f}")
    print(f"  Perplexity     : {perplexity:.2f}   (lower = model finds it more natural)")


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 3 — embed
# ═══════════════════════════════════════════════════════════════════════════════

def get_embedding(text: str, tokenizer, embedding, pos_enc, encoder, mlm_head,
                  seq_len: int):
    """
    Return the [CLS] token's encoder output as a sentence embedding.
    This is a (d_model,) vector representing the whole sentence.
    """
    token_ids, _ = tokenise_input(text, tokenizer, seq_len)
    _, memory    = forward(token_ids, embedding, pos_enc, encoder, mlm_head)

    cls_vec = memory[0, 0, :]                          # (d_model,)

    print(f"\n  Input     : {text}")
    print(f"  Embedding shape : {cls_vec.shape}")
    print(f"  First 8 dims    : {cls_vec[:8].round(4)}")
    print(f"  Norm            : {np.linalg.norm(cls_vec):.4f}")
    return cls_vec


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 4 — top_k  (alias — same as fill_mask but always shows top-k)
# ═══════════════════════════════════════════════════════════════════════════════

def top_k_predictions(text: str, tokenizer, embedding, pos_enc, encoder,
                      mlm_head, seq_len: int, top_k: int = 5):
    """Show top-K candidates for every [MASK] in the input."""
    fill_mask(text, tokenizer, embedding, pos_enc, encoder, mlm_head,
              seq_len, top_k=top_k)


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive REPL  (bonus — run without --text for a prompt loop)
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_loop(tokenizer, embedding, pos_enc, encoder, mlm_head,
                     seq_len: int, top_k: int = 5):
    print("\n" + "═" * 60)
    print("  Interactive mode — type a sentence with [MASK]")
    print("  Commands:  :score <text>  |  :embed <text>  |  :quit")
    print("═" * 60)

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue
        if raw == ":quit":
            break
        elif raw.startswith(":score "):
            score_sentence(raw[7:], tokenizer, embedding, pos_enc,
                           encoder, mlm_head, seq_len)
        elif raw.startswith(":embed "):
            get_embedding(raw[7:], tokenizer, embedding, pos_enc,
                          encoder, mlm_head, seq_len)
        else:
            # default: fill_mask
            if MASK_PLACEHOLDER not in raw:
                print(f"  Tip: include [MASK] in your sentence, "
                      f"or prefix with :score / :embed")
            top_k_predictions(raw, tokenizer, embedding, pos_enc,
                               encoder, mlm_head, seq_len, top_k)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nepali LM — Inference")

    # paths
    parser.add_argument("--checkpoint",  default="checkpoints/epoch_3.npz",
                        help="Path to trained checkpoint .npz file")
    parser.add_argument("--vocab_file",  default="../dataset/nepali_bpe.vocab")

    # must match training config exactly
    parser.add_argument("--seq_len",  type=int,   default=512)
    parser.add_argument("--d_model",  type=int,   default=256)
    parser.add_argument("--layers",   type=int,   default=4)
    parser.add_argument("--heads",    type=int,   default=8)
    parser.add_argument("--d_ff",     type=int,   default=1024)

    # inference options
    parser.add_argument("--mode",   default="fill_mask",
                        choices=["fill_mask", "score", "embed", "top_k",
                                 "interactive"],
                        help="Inference mode")
    parser.add_argument("--text",   default=None,
                        help="Input sentence (use [MASK] for fill_mask/top_k)")
    parser.add_argument("--top_k",  type=int, default=5,
                        help="Number of top predictions to show")

    args = parser.parse_args()

    # ── load model ────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Loading model...")
    print("═" * 60)

    tokenizer, embedding, pos_enc, encoder, mlm_head = load_model(
        checkpoint_path = args.checkpoint,
        vocab_file      = args.vocab_file,
        seq_len         = args.seq_len,
        d_model         = args.d_model,
        n_layers        = args.layers,
        n_heads         = args.heads,
        d_ff            = args.d_ff,
    )

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.mode == "interactive" or args.text is None:
        interactive_loop(tokenizer, embedding, pos_enc, encoder, mlm_head,
                         args.seq_len, args.top_k)

    elif args.mode == "fill_mask":
        fill_mask(args.text, tokenizer, embedding, pos_enc, encoder, mlm_head,
                  args.seq_len, top_k=1)

    elif args.mode == "top_k":
        top_k_predictions(args.text, tokenizer, embedding, pos_enc, encoder,
                          mlm_head, args.seq_len, top_k=args.top_k)

    elif args.mode == "score":
        score_sentence(args.text, tokenizer, embedding, pos_enc, encoder,
                       mlm_head, args.seq_len)

    elif args.mode == "embed":
        get_embedding(args.text, tokenizer, embedding, pos_enc, encoder,
                      mlm_head, args.seq_len)