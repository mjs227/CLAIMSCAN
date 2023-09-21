
import torch
import torch.nn as nn


def get_windows(in_ten, l_pad_size, r_pad_size, pad_tok=(0, 1)):
    _, in_len, pt_dim = tuple(in_ten.shape)

    if pad_tok is None:
        l_pad = in_ten[:, 0, :].repeat(1, l_pad_size, 1)
        r_pad = in_ten[:, -1, :].repeat(1, r_pad_size, 1)
        in_ten = in_ten[:, 1:-1, :]
        in_len -= 2
    else:
        pad_tok_ = pad_tok if isinstance(pad_tok, tuple) else (pad_tok, pad_tok)
        l_pad = torch.full(
            (1, l_pad_size, pt_dim),
            pad_tok_[0],
            device=DEVICE,
            dtype=in_ten.dtype
        )
        r_pad = torch.full(
            (1, r_pad_size, pt_dim),
            pad_tok_[1],
            device=DEVICE,
            dtype=in_ten.dtype
        )

    pad_len = in_len + l_pad_size + r_pad_size
    indices = torch.arange(in_len, device=DEVICE).view((in_len, 1)).repeat((1, pad_len))
    rng = torch.arange(pad_len, device=DEVICE).view((1, pad_len)).repeat((in_len, 1))
    rng = ((rng + indices) % pad_len).unsqueeze(-1).expand(-1, -1, pt_dim)

    in_ten = torch.cat((l_pad, in_ten, r_pad), dim=1).repeat(in_len, 1, 1)
    in_ten = torch.gather(in_ten, 1, rng)

    r_windows = in_ten[:, l_pad_size + 1:l_pad_size + r_pad_size + 1, :]
    centers = in_ten[:, l_pad_size, :].unsqueeze(1)
    l_windows = in_ten[:, :l_pad_size, :]

    return torch.cat((l_windows, r_windows, centers), dim=1)


class PTLayer(nn.Module):
    def __init__(self, in_dim, window, p):
        super(PTLayer, self).__init__()
        self.IN_DIM = in_dim
        self.WINDOW = window
        self.q = nn.Linear(in_dim, in_dim)
        self.k = nn.Conv1d(window + 1, (window + 1) * in_dim, in_dim, stride=in_dim, groups=(window + 1))
        self.ff_window = nn.Conv1d(window, window * in_dim, in_dim, stride=in_dim, groups=window)
        self.ff_cent = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        q = self.q(x[:, -1, :])
        k = self.k(x).reshape(x.size(0), self.WINDOW + 1, self.IN_DIM)
        attns = torch.matmul(q.unsqueeze(1), torch.transpose(k, -2, -1))
        attns = torch.softmax(attns, dim=2)

        x_cent = torch.matmul(attns, x)
        x_cent = torch.nn.functional.normalize(x_cent, p=2, dim=2)
        x_cent = self.dropout(self.ff_cent(x_cent))
        x_cent = torch.nn.functional.normalize(x_cent + x[:, -1, :].unsqueeze(1), p=2, dim=2)

        x_window = self.ff_window(x[:, :-1, :]).reshape(x.size(0), self.WINDOW, self.IN_DIM)
        x_window = self.dropout(x_window + x[:, :-1, :])
        x_window = torch.nn.functional.normalize(x_window, p=2, dim=2)

        return torch.cat((x_window, x_cent), dim=1)


class PTFinalLayer(nn.Module):
    def __init__(self, in_dim, window):
        super(PTFinalLayer, self).__init__()
        self.IN_DIM = in_dim
        self.WINDOW = window
        self.q = nn.Linear(in_dim, in_dim)
        self.k = nn.Conv1d(window + 1, (window + 1) * in_dim, in_dim, stride=in_dim, groups=(window + 1))

    def forward(self, x):
        q = self.q(x[:, -1, :])
        k = self.k(x).reshape(x.size(0), self.WINDOW + 1, self.IN_DIM)
        attns = torch.matmul(q.unsqueeze(1), torch.transpose(k, -2, -1))
        attns = torch.softmax(attns, dim=2)
        x_cent = torch.matmul(attns, x) + x[:, -1, :].unsqueeze(1)

        return torch.nn.functional.normalize(x_cent, p=2, dim=2)


class PositionalTransformer(nn.Module):
    def __init__(self, d_model=128, n_layers=3, window=1, p=0.1):
        super(PositionalTransformer, self).__init__()
        self.layers = nn.ModuleList([PTLayer(d_model, window, p) for _ in range(n_layers)])
        self.final_layer = PTFinalLayer(d_model, window)
        self.d_model = d_model

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.final_layer(x)
