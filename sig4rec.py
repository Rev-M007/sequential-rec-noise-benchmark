"""
sig4rec.py
==========

Sig4Rec ("Disentangling Temporal Evolution in Frequency Domain for Sequential
Recommendation", He et al., IEEE Trans. Computational Social Systems 13(2),
2026) as a custom RecBole model. Equation numbers below refer to the paper.

Components
----------
1. Temporal evolution signal (Sec. IV-A). Each relation r has a learnable
   complex spectrum F_r with Ns frequency components. The influence of a
   historical item at log-scaled interval Δt (Eq. 1) is obtained by phase-
   modulating the spectrum (Fourier shift theorem, Eq. 4) and taking the n = 0
   sample of its inverse DFT (Eqs. 5-6):
       S_r(Δt) = Re{ IDFT( F_r(k) · e^{j2πkΔt/Ns} )[0] }
2. User preference (Sec. IV-B). For each relation, history items are weighted
   by a softmax similarity to the last item C (Eq. 8) and by S_r (Eq. 7); the
   relation representations are average-pooled into p_u (Eq. 9).
3. Prediction (Sec. IV-C): y_ui = (e_u + p_u) · e_i + b_i (Eq. 10), trained with
   BPR (Eq. 11).
4. Joint learning (Sec. IV-D): a TransE-style margin loss on the relation
   network (Eqs. 12-13), added with weight alpha (Eq. 14).

Relations (Sec. V-A1 and Table I)
---------------------------------
* "shared attribute": two items share at least one attribute value. For ml-1m
  this is the paper's Shared Genre relation, read from ``movies.dat``.
* "sequential": the default relation; every earlier item precedes the last one.

Implementation choices where the paper is not explicit
------------------------------------------------------
* Which history items enter Eq. 7 for relation r (``relation_mode``):
    - "related" (default): every earlier item that has relation r with the
      last item i_n. Summing over *all* items for every relation would make
      the |R| relations collapse into one averaged spectrum (Eq. 9 is linear),
      so relation existence must select items.
    - "neighbor": only the r-relation neighbor of Definition 1 (the most
      recent related item).
  In both modes i_n itself is included ("S_u includes the last observed item
  i_n", Sec. IV-B) with Δt = 0, and an explicit relation with no related item
  falls back to the immediately preceding item (Definition 1).
* Δt' in Eq. 4 is Δt from Eq. 1 without further scaling. Eq. 1 yields
  log2(minutes), at most ~21 for ml-1m, so Ns >= 24 avoids wrap-around.
* Eq. 13 sums over all relation triples; here each training step samples
  ``rel_batch_size`` positive triples and averages the hinge loss.
* Ns, margin γ and rel_batch_size are not reported; defaults are below. The
  paper reports alpha = 0.6, d = 128 and lr = 5e-3 as best for ml-1m.
* RecBole stores timestamps as float32 (~64 s resolution at Unix-time scale),
  which only affects intervals of a few minutes after the log transform.

Usage
-----
    python train_model.py --model Sig4Rec --dataset ml-1m
"""

import io
import os
import urllib.request
import zipfile

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

SIG4REC_DEFAULTS = {
    "embedding_size": 64,
    "n_freq": 64,              # Ns, number of frequency components
    "time_scalar": 60.0,       # Eq. 1: seconds -> minutes
    "time_bias": 1e-6,         # Eq. 1
    "relation_mode": "related",
    "alpha": 0.6,              # Eq. 14, paper's best value for ml-1m
    "margin": 1.0,             # γ in Eq. 13
    "rel_batch_size": 1024,    # relation triples sampled per training step
    "init_std": 0.01,          # "Gaussian with mean 0 and std 0.01" (Sec. V-A4)
    "loss_type": "BPR",
    "relation_file": None,     # item -> attribute file, e.g. dataset/ml-1m/ml-1m.genre
}

RELATIONS = ["shared_attribute", "sequential"]
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


# --------------------------------------------------------------------------- #
# Relation data
# --------------------------------------------------------------------------- #
def prepare_ml1m_genres(data_path: str) -> str:
    """Write ``<data_path>/ml-1m/ml-1m.genre`` (item id -> genres) from GroupLens' movies.dat."""
    out_file = os.path.join(data_path, "ml-1m", "ml-1m.genre")
    if os.path.exists(out_file):
        return out_file

    print(f"[data] Downloading ml-1m movie genres from {ML1M_URL} ...")
    with urllib.request.urlopen(ML1M_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    # movies.dat lines look like "1::Toy Story (1995)::Animation|Children's|Comedy".
    movies = archive.read("ml-1m/movies.dat").decode("latin-1").splitlines()

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write("item_id:token\tgenre:token_seq\n")
        for line in movies:
            if line.strip():
                movie_id, _, genres = line.split("::")
                f.write(f"{movie_id}\t{genres.replace('|', ' ')}\n")
    print(f"[data] Wrote genres for {len(movies):,} movies to {out_file}")
    return out_file


def load_item_attributes(path: str, item_tokens) -> torch.Tensor:
    """Multi-hot attribute matrix [n_items, n_attributes] aligned with RecBole's item ids.

    Row 0 (padding) and items missing from the file are all-zero, i.e. they have
    no explicit relation to any item.
    """
    token_to_id = {str(token): idx for idx, token in enumerate(item_tokens)}
    rows, vocab = [], {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            item, _, attrs = line.rstrip("\n").partition("\t")
            if item in token_to_id:
                ids = [vocab.setdefault(a, len(vocab)) for a in attrs.split()]
                rows.append((token_to_id[item], ids))

    matrix = torch.zeros(len(item_tokens), max(len(vocab), 1))
    for item_id, attr_ids in rows:
        matrix[item_id, attr_ids] = 1.0
    return matrix


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Sig4Rec(SequentialRecommender):
    """Sig4Rec for RecBole (SequentialRecommender is RecBole's AbstractRecommender
    subclass for sequence models; it supplies the field names, n_items and
    max_seq_length that RecBole's sequential data loaders and trainer rely on)."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        def param(key):
            value = config[key]
            return SIG4REC_DEFAULTS[key] if value is None else value

        self.embedding_size = param("embedding_size")
        self.n_freq = param("n_freq")
        self.time_scalar = param("time_scalar")
        self.time_bias = param("time_bias")
        self.relation_mode = param("relation_mode")
        self.alpha = param("alpha")
        self.margin = param("margin")
        self.rel_batch_size = param("rel_batch_size")
        self.init_std = param("init_std")
        self.loss_type = param("loss_type")
        if self.relation_mode not in ("related", "neighbor"):
            raise ValueError("relation_mode must be 'related' or 'neighbor'.")

        self.TIME_SEQ = config["TIME_FIELD"] + config["LIST_SUFFIX"]
        self.n_users = dataset.num(self.USER_ID)
        self.n_relations = len(RELATIONS)

        relation_file = param("relation_file")
        if relation_file is None or not os.path.exists(relation_file):
            raise FileNotFoundError(
                f"Sig4Rec needs an item attribute file (config 'relation_file'), got {relation_file}.")
        item_attrs = load_item_attributes(relation_file, dataset.field2id_token[self.ITEM_ID])
        self.register_buffer("item_attrs", item_attrs)
        self._build_relation_sampler(item_attrs)

        # ---- Embeddings (Eq. 10) and relation embedding (Eq. 12) ------------ #
        d = self.embedding_size
        self.item_embedding = nn.Embedding(self.n_items, d, padding_idx=0)
        self.user_embedding = nn.Embedding(self.n_users, d)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.relation_embedding = nn.Embedding(1, d)  # explicit (attribute) relation only

        # ---- Learnable spectra F_r: real and imaginary parts (Sec. IV-A) ------ #
        self.freq_real = nn.Parameter(torch.empty(self.n_relations, self.n_freq))
        self.freq_imag = nn.Parameter(torch.empty(self.n_relations, self.n_freq))
        self.register_buffer("freq_index", torch.arange(self.n_freq, dtype=torch.float32))

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' is 'BPR' or 'CE'.")

        self._init_weights()

    def _init_weights(self) -> None:
        for param in self.parameters():
            nn.init.normal_(param, mean=0.0, std=self.init_std)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def _build_relation_sampler(self, item_attrs: torch.Tensor) -> None:
        """Lookup tables for sampling positive (head, tail) pairs that share an attribute."""
        has_attr = item_attrs.sum(1) > 0
        has_attr[0] = False
        self.register_buffer("relation_heads", has_attr.nonzero().squeeze(1))

        members = [item_attrs[:, a].nonzero().squeeze(1) for a in range(item_attrs.size(1))]
        counts = torch.tensor([len(m) for m in members])
        table = torch.zeros(len(members), max(int(counts.max()), 1), dtype=torch.long)
        for a, m in enumerate(members):
            table[a, :len(m)] = m
        self.register_buffer("attr_members", table)
        self.register_buffer("attr_counts", counts)

    # ---- Sec. IV-A: temporal evolution signal ------------------------------- #
    def relation_interval(self, gap_seconds: torch.Tensor) -> torch.Tensor:
        """Eq. 1: Δt = max(log2(gap / scalar + bias), 0)."""
        return torch.log2(gap_seconds / self.time_scalar + self.time_bias).clamp(min=0)

    def temporal_signal(self, delta_t: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
        """S_r(Δt) for every relation (Eqs. 4-6). Input [...], output [..., R].

        The modulated spectrum (Eq. 4) is inverse-transformed with torch.fft.ifft
        (which includes the 1/Ns factor of Eq. 5) and the n = 0 sample is kept (Eq. 6).
        Processed in chunks to bound memory during full-ranking evaluation.
        """
        spectrum = torch.complex(self.freq_real, self.freq_imag)          # [R, Ns]
        flat = delta_t.reshape(-1)
        out = []
        for part in flat.split(chunk):
            phase = 2 * torch.pi * part[:, None] * self.freq_index / self.n_freq   # [n, Ns]
            shift = torch.polar(torch.ones_like(phase), phase)                     # e^{jφ}
            modulated = spectrum[None] * shift[:, None, :]                         # [n, R, Ns]
            out.append(torch.fft.ifft(modulated, dim=-1)[..., 0].real)             # [n, R]
        return torch.cat(out).view(*delta_t.shape, self.n_relations)

    # ---- Sec. IV-B: user preference --------------------------------------- #
    def relation_masks(self, item_seq, valid, last_pos, last_item) -> torch.Tensor:
        """Which history positions contribute to each relation's I_r. Output [B, L, R]."""
        pos = torch.arange(item_seq.size(1), device=item_seq.device)[None]       # [1, L]
        earlier = valid & (pos < last_pos[:, None])
        is_last = pos == last_pos[:, None]
        previous = pos == (last_pos - 1)[:, None]

        shared = (self.item_attrs[item_seq] * self.item_attrs[last_item][:, None]).sum(-1) > 0
        shared = shared & earlier

        if self.relation_mode == "related":
            explicit, sequential = shared, earlier
        else:  # Definition 1: most recent related item only
            neighbor = torch.where(shared, pos, torch.full_like(pos, -1)).max(dim=1).values
            explicit = (pos == neighbor[:, None]) & (neighbor[:, None] >= 0)
            sequential = previous & earlier

        no_relation = ~explicit.any(dim=1, keepdim=True)
        explicit = explicit | (no_relation & previous & earlier)   # fallback to preceding item
        return torch.stack([explicit | is_last, sequential | is_last], dim=-1).float()

    def forward(self, item_seq, time_seq, user) -> torch.Tensor:
        """Query vector e_u + p_u for each sequence ([B, d])."""
        valid = item_seq != 0
        last_pos = valid.sum(dim=1) - 1
        last_item = item_seq.gather(1, last_pos[:, None]).squeeze(1)
        last_time = time_seq.gather(1, last_pos[:, None])

        seq_emb = self.item_embedding(item_seq)                            # [B, L, d]
        last_emb = self.item_embedding(last_item)                          # [B, d]

        # Eq. 8: softmax over the history of -||e_in - e_ik||^2
        distance = ((seq_emb - last_emb[:, None]) ** 2).sum(-1)
        correlation = torch.softmax(distance.masked_fill(~valid, float("inf")).neg(), dim=1)

        # Eq. 1 and Eqs. 4-6
        delta_t = self.relation_interval((last_time - time_seq).clamp(min=0))
        signal = self.temporal_signal(delta_t)                             # [B, L, R]

        # Eq. 7 per relation, then Eq. 9 average pooling over relations
        weights = self.relation_masks(item_seq, valid, last_pos, last_item) * signal
        weights = weights * correlation[..., None]
        relation_repr = torch.einsum("blr,bld->brd", weights, seq_emb)      # [B, R, d]
        preference = relation_repr.mean(dim=1)

        return self.user_embedding(user) + preference

    def _query(self, interaction) -> torch.Tensor:
        return self.forward(interaction[self.ITEM_SEQ], interaction[self.TIME_SEQ],
                            interaction[self.USER_ID])

    # ---- Sec. IV-D: relation network ------------------------------------------ #
    def _sample_unrelated(self, anchors: torch.Tensor, tries: int = 10) -> torch.Tensor:
        """Random items sharing no attribute with each anchor (corrupted triples)."""
        sample = torch.randint(1, self.n_items, anchors.shape, device=anchors.device)
        for _ in range(tries):
            related = (self.item_attrs[sample] * self.item_attrs[anchors]).sum(-1) > 0
            if not related.any():
                break
            sample[related] = torch.randint(1, self.n_items, (int(related.sum()),),
                                            device=anchors.device)
        return sample

    def relation_loss(self) -> torch.Tensor:
        """Eq. 13 on a batch of sampled positive triples (h, r, t) with head/tail corruption."""
        n = self.rel_batch_size
        device = self.relation_heads.device
        heads = self.relation_heads[torch.randint(len(self.relation_heads), (n,), device=device)]
        attrs = torch.multinomial(self.item_attrs[heads], 1).squeeze(1)
        slots = (torch.rand(n, device=device) * self.attr_counts[attrs]).long()
        tails = self.attr_members[attrs, slots]

        emb, rel = self.item_embedding, self.relation_embedding.weight[0]

        def score(h, t):  # Eq. 12
            return ((emb(h) + rel - emb(t)) ** 2).sum(-1)

        positive = score(heads, tails)
        loss = (torch.relu(self.margin + positive - score(heads, self._sample_unrelated(heads)))
                + torch.relu(self.margin + positive - score(self._sample_unrelated(tails), tails)))
        return loss.mean()

    # ---- RecBole interface -------------------------------------------------- #
    def calculate_loss(self, interaction):
        """L = L_rec + alpha * L_rel (Eq. 14). Returned as a tuple so RecBole logs both parts."""
        query = self._query(interaction)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":  # Eq. 11
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_score = (query * self.item_embedding(pos_items)).sum(-1) + self.item_bias(pos_items).squeeze(-1)
            neg_score = (query * self.item_embedding(neg_items)).sum(-1) + self.item_bias(neg_items).squeeze(-1)
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:
            logits = query @ self.item_embedding.weight.T + self.item_bias.weight.T
            rec_loss = self.loss_fct(logits, pos_items)

        if self.alpha > 0:
            return rec_loss, self.alpha * self.relation_loss()
        return rec_loss

    def predict(self, interaction) -> torch.Tensor:
        """Eq. 10 for one candidate item per sequence (sampled evaluation)."""
        query = self._query(interaction)
        items = interaction[self.ITEM_ID]
        return (query * self.item_embedding(items)).sum(-1) + self.item_bias(items).squeeze(-1)

    def full_sort_predict(self, interaction) -> torch.Tensor:
        """Eq. 10 for every item: (e_u + p_u) E^T + b  ->  [B, n_items]."""
        query = self._query(interaction)
        return query @ self.item_embedding.weight.T + self.item_bias.weight.T
