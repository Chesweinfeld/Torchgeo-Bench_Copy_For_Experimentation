"""Faithful port of the Tessera v1.1 encoder architecture.

Ported from ``ucam-eo/tessera``'s ``tessera_infer_QAT/src/models/{modules,ssl_model_v1_1}.py``
(``v1.1`` branch) so real checkpoint weights load with matching module
names/shapes. Kept as close to the original as possible — including the
GRU-based temporal pooling, which is a no-op for the single-timestep inputs
GeoBench provides (see :class:`CustomTemporalAwarePooling`) but must still
exist for :meth:`torch.nn.Module.load_state_dict` to line up.
"""

import math

import torch
import torch.nn as nn


class CustomGRUCell(nn.Module):
    """Single-timestep GRU cell, implemented from primitive ops (matches upstream)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ir = nn.Linear(input_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        r_t = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_prev) + self.b_r)
        z_t = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_prev) + self.b_z)
        h_tilde = torch.tanh(self.W_ih(x_t) + self.W_hh(r_t * h_prev) + self.b_h)
        return (1 - z_t) * h_prev + z_t * h_tilde


class CustomGRU(nn.Module):
    """Sequence-level wrapper around :class:`CustomGRUCell` (matches upstream)."""

    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gru_cell = CustomGRUCell(input_size, hidden_size)

    def forward(
        self, x: torch.Tensor, h_0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape
            x = x.transpose(0, 1)

        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        outputs = []
        h_t = h_0
        for t in range(seq_len):
            h_t = self.gru_cell(x[:, t, :], h_t)
            outputs.append(h_t)
        stacked = torch.stack(outputs, dim=1)
        if not self.batch_first:
            stacked = stacked.transpose(0, 1)
        return stacked, h_t


class CustomTemporalAwarePooling(nn.Module):
    """GRU + attention pooling over the time axis (matches upstream).

    ``T == 1`` (GeoBench's case) short-circuits to a plain squeeze — the
    GRU/attention parameters still exist (for state-dict compatibility) but
    are not exercised.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.temporal_context = CustomGRU(input_dim, input_dim, batch_first=True)
        self.query = nn.Linear(input_dim, 1)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if t == 0:
            return torch.zeros(b, d, device=x.device, dtype=x.dtype)
        if t == 1:
            return x.squeeze(1)

        x_context, _ = self.temporal_context(x)
        x_context = self.layer_norm(x_context)
        attn_weights = torch.softmax(self.query(x_context), dim=1)
        return (attn_weights * x).sum(dim=1)


class TemporalPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding keyed by day-of-year, not sequence index."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        position = doy.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=doy.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class TransformerEncoder(nn.Module):
    """Per-sensor encoder: embed bands, add DOY position encoding, transform, pool.

    Args:
        band_num: Number of spectral bands (10 for S2, 2 for S1).
        latent_dim: Tessera's ``latent_dim`` config value; internal width is
            ``latent_dim * 4``.
        nhead: Attention heads.
        num_encoder_layers: Transformer encoder layers.
        dim_feedforward: Transformer feedforward width.
        dropout: Transformer dropout.
    """

    def __init__(
        self,
        band_num: int,
        latent_dim: int,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        width = latent_dim * 4

        self.embedding = nn.Sequential(
            nn.Linear(band_num, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.temporal_encoder = TemporalPositionalEncoder(d_model=width)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.attn_pool = CustomTemporalAwarePooling(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, seq_len, band_num + 1)`` — last column is day-of-year."""
        bands = x[:, :, :-1]
        doy = x[:, :, -1]
        bands_embedded = self.embedding(bands)
        x = bands_embedded + self.temporal_encoder(doy)
        x = self.transformer_encoder(x)
        return self.attn_pool(x)


def build_dim_reducer(in_dim: int, out_dim: int) -> nn.Sequential:
    """``Linear(in, in*2) -> LayerNorm(in*2) -> ReLU -> Dropout(0.2) -> Linear(in*2, out)``."""
    return nn.Sequential(
        nn.Linear(in_dim, in_dim * 2),
        nn.LayerNorm(in_dim * 2),
        nn.ReLU(inplace=False),
        nn.Dropout(0.2),
        nn.Linear(in_dim * 2, out_dim),
    )
