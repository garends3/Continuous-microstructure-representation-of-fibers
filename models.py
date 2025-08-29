import torch.nn as nn
import torch
from functools import partial


def positional_encoding(
    x: torch.Tensor, lpos: int, sigma: float = None
) -> torch.Tensor:
    if not sigma:
        js = 2 ** torch.arange(lpos).to(x.device) * torch.pi
    else:
        js = 2 ** torch.linspace(0, sigma, lpos).to(x.device) * torch.pi

    jx = torch.einsum("ix, j -> ijx", x, js)
    sin_out = torch.sin(jx).reshape(x.shape[0], -1)
    cos_out = torch.cos(jx).reshape(x.shape[0], -1)
    return torch.cat([x, sin_out, cos_out], dim=-1)


def input_mapping(x, B=None):
    if B is None:
        return x
    else:
        x_proj = (2.0 * torch.pi * x) @ B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def progressive_emb(x_emb: torch.Tensor, t_frac: float) -> torch.Tensor:
    a = torch.ones(x_emb.shape[1]).to(x_emb.device)
    start = int(t_frac * x_emb.shape[1] + 3)
    end = int(t_frac * x_emb.shape[1] + 4)
    a[start:end] = (t_frac * x_emb.shape[1]) - int(t_frac * x_emb.shape[1])
    a[int(end) :] = 0

    return x_emb * a.unsqueeze(dim=0)


class Zeppelin_NeSH(nn.Module):
    def __init__(
        self,
        lpos=10,
        hidden_dim=256,
        n_layers=11,
        sigma=None,
        gaussian=True,
    ) -> None:
        super().__init__()

        output_size = 3
        input_size = lpos * 2 if gaussian else lpos * 6 + 3

        self.mlp = nn.Sequential(
            *(
                [nn.Linear(input_size, hidden_dim), nn.ReLU()]
                + [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_layers - 2)
                + [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            )
        )

        self.weight = nn.Sequential(*[nn.Linear(hidden_dim, 1), nn.Sigmoid()])
        self.Dpar = nn.Sequential(*[nn.Linear(hidden_dim, 1), nn.Sigmoid()])
        self.Dperp = nn.Sequential(*[nn.Linear(hidden_dim, 1), nn.Sigmoid()])


        self.Lpos = lpos
        self.sigma = sigma
        self.B = nn.Parameter(torch.randn([lpos, 3]) * sigma, requires_grad=False) if gaussian else None

    def forward(self, x, t_frac=None) -> torch.Tensor:
        if self.B is not None:
            x_emb = input_mapping(x, self.B.to(x.device))
        else:
            x_emb = positional_encoding(x, self.Lpos, self.sigma)

        if t_frac is not None:
            x_emb = progressive_emb(x_emb, t_frac)

        x_mlp = self.mlp(x_emb)

        return torch.cat(
            [
                self.weight(x_mlp),
                self.Dpar(x_mlp) * 4 + 0,
                self.Dperp(x_mlp) * 4 + 0
            ],
            dim=-1,
        )


def create_zeppelin_model(cfg: dict, model: nn.Module) -> nn.Module:
    train_cfg = cfg["train_cfg"]
    sigma = train_cfg["sigma"]
    lpos = train_cfg["lpos"]
    hidden_dim = train_cfg["hidden_dim"]
    n_layers = train_cfg["n_layers"]

    return model(
        lpos=lpos,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        sigma=sigma,
        gaussian=cfg["gaussian_encoding"],
    )


MODELS = {
    "Zeppelin_model": partial(create_zeppelin_model, model=Zeppelin_NeSH),
}


def get_model(cfg: dict) -> nn.Module:
    constructor = MODELS.get(cfg["model_name"], None)
    if constructor is None:
        raise Exception("Model name not recognized")
    return constructor(cfg)
