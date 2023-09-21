
import torch
import torch.nn as nn
import torch.optim as op
from copy import deepcopy
from positional_transformer import PositionalTransformer, get_windows


class LinearAsConv(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(LinearAsConv, self).__init__()
        self.conv = nn.Conv1d(1, out_features, in_features, stride=in_features, **kwargs)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1).flatten(start_dim=-2))

        return torch.transpose(x, -1, -2)


class HeadModel(nn.Module):
    def __init__(self, lm_dim, l_window, r_window, pt_kwargs, p=0.1):
        super(HeadModel, self).__init__()
        self.l_window = l_window
        self.r_window = r_window
        self.pt = PositionalTransformer(
            **{**pt_kwargs, **{'window': l_window + r_window}}
        )
        self.pt_dim = self.pt.d_model
        self.linear1 = LinearAsConv(lm_dim, self.pt_dim)
        self.linear2 = LinearAsConv(self.pt_dim, 1)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = self.linear1(x)
        x = get_windows(x, self.l_window, self.r_window, pad_tok=None)
        x = self.pt(x).squeeze().unsqueeze(0)

        return self.linear2(x).flatten(start_dim=1)


class ClaimscanModel:
    def __init__(
        self,
        lm_,
        optimizer=op.SGD,
        optimizer_kwargs=None,
        optimizer_lm=None,
        optimizer_lm_kwargs=None,
        lm_dim=768,
        head_p=0.1,
        l_window=3,
        r_window=3,
        pt_kwargs=None
    ):
        self.lm = lm_
        self.lm.eval()
        self.device = self.lm.device
        self.head = HeadModel(
            lm_dim, 
            l_window,
            r_window,
            pt_kwargs=({} if pt_kwargs is None else pt_kwargs),
            p=head_p
        )
        self.head.to(self.device)
        self.head.eval()
        self.forward_fn = self.forward_eval

        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        optimizer_lm = optimizer if optimizer_lm is None else optimizer_lm

        self.optimizer_head = optimizer(self.head.parameters(), **optimizer_kwargs)
        self.optimizer_lm = optimizer_lm(
            self.lm.parameters(),
            **(optimizer_lm_kwargs if optimizer_lm_kwargs else optimizer_kwargs)
        )

    def to(self, device):
        self.device = device
        self.lm.to(device)
        self.head.to(device)

    def zero_grad(self):
        self.optimizer_lm.zero_grad()
        self.optimizer_head.zero_grad()

    def step(self):
        self.optimizer_head.step()
        self.optimizer_lm.step()

    def eval(self):
        self.lm.eval()
        self.head.eval()
        self.forward_fn = self.forward_eval

    def train(self):
        self.lm.train()
        self.head.train()
        self.forward_fn = self.forward_train

    def state_dict(self):
        device = self.device
        self.to('cpu')
        state_dict = (self.lm.state_dict(), self.head.state_dict())
        self.to(device)

        return deepcopy(state_dict)

    def load_state_dict(self, state_dict):
        device = self.device
        self.to('cpu')
        self.lm.load_state_dict(state_dict[0])
        self.head.load_state_dict(state_dict[1])
        self.to(device)

    def forward_train(self, x):
        x = self.lm(x).last_hidden_state

        return self.head(x)

    def forward_eval(self, x):
        x = self.forward_train(x)

        return torch.sigmoid(x.flatten())

    def __call__(self, x):
        return self.forward_fn(x)
