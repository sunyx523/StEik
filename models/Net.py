# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS

import numpy as np
import torch
import torch.nn as nn
from models.SIREN import FCBlock

class Decoder(nn.Module):
    def forward(self, *args, **kwargs):
        return self.fc_block(*args, **kwargs)

class Network(nn.Module):
    def __init__(self, latent_size, in_dim=3, decoder_hidden_dim=256, nl='sine', encoder_type=None,
                 decoder_n_hidden_layers=8, init_type='siren', neuron_type='quadratic', sphere_init_params=[1.6, 1.0]):
        super().__init__()
        self.encoder_type = encoder_type
        self.init_type = init_type
        if encoder_type == 'autodecoder':
            # latent_size will stay as input latent size
            pass
        elif encoder_type == 'none':
            latent_size = 0
        else:
            raise ValueError("unsupported encoder type")
        self.decoder = Decoder()
        if neuron_type == 'linear':
            self.decoder.fc_block = FCBlock(in_dim + latent_size, 1, num_hidden_layers=decoder_n_hidden_layers, hidden_features=decoder_hidden_dim,
                                    outermost_linear=True, nonlinearity=nl, init_type=init_type,
                                    sphere_init_params=sphere_init_params)  # SIREN decoder
        elif neuron_type == 'quadratic':
            self.decoder.fc_block = QuaNet(d_in=in_dim + latent_size, nl=nl, n_layers=decoder_n_hidden_layers, d_hidden=decoder_hidden_dim,
                                                    init_type=init_type, sphere_init_params=sphere_init_params)
        else:
            raise ValueError("unsupported neuron type")
                                                    

    def forward(self, non_mnfld_pnts, mnfld_pnts=None):
        # shape is (bs, npoints, in_dim+latent_size) for both inputs, npoints could be different sizes
        batch_size = non_mnfld_pnts.shape[0]
        if not mnfld_pnts is None and self.encoder_type == 'autodecoder':
            # Assume inputs have latent vector concatted with [xyz, latent]
            latent = non_mnfld_pnts[:,:,3:]
            latent_reg = latent.norm(dim=-1).mean()
            manifold_pnts_pred = self.decoder(mnfld_pnts.view(-1, mnfld_pnts.shape[-1])).reshape(batch_size, -1)
        elif mnfld_pnts is not None:
            manifold_pnts_pred = self.decoder(mnfld_pnts.view(-1, mnfld_pnts.shape[-1])).reshape(batch_size, -1)
            latent = None
            latent_reg = None
        else:
            manifold_pnts_pred = None
            latent = None
            latent_reg = None

        # Off manifold points
        nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts.view(-1, non_mnfld_pnts.shape[-1])).reshape(batch_size, -1)

        return {"manifold_pnts_pred": manifold_pnts_pred,
                "nonmanifold_pnts_pred": nonmanifold_pnts_pred,
                "latent_reg": latent_reg,
                "latent": latent}


class QuaNet(nn.Module):
    def __init__(
            self,
            d_in=3,
            d_out=1,
            d_hidden=256,
            n_layers=8,
            nl='softplus',
            sphere_init_params=[1.6, 1.0],
            init_type='siren',
    ):
        super().__init__()
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)
        self.init_type = init_type
        self.sphere_init_params = sphere_init_params

        for l in range(self.num_layers - 1):
            if l == 0:
                qua = QuadraticLayer(dims[l], dims[l + 1])
            else:
                qua = QuadraticLayer(dims[l], dims[l + 1])
            setattr(self, "qua" + str(l), qua)
        if self.init_type == "mfgi":
            self.apply(geom_sine_init)
            self.qua0.apply(first_layer_mfgi_init)
            self.qua1.apply(second_layer_mfgi_init)
            getattr(self, "qua" + str(self.num_layers - 3)).apply(second_last_layer_geom_sine_init)
            getattr(self, "qua" + str(self.num_layers - 2)).apply(last_layer_geom_sine_init)
        elif self.init_type == "siren":
            self.apply(sine_init)
            self.qua0.apply(first_layer_sine_init)
        elif init_type == 'geometric_sine':
            self.apply(geom_sine_init)
            self.qua0.apply(first_layer_geom_sine_init)
            getattr(self, "qua" + str(self.num_layers - 3)).apply(second_last_layer_geom_sine_init)
            getattr(self, "qua" + str(self.num_layers - 2)).apply(last_layer_geom_sine_init)

        if nl == 'softplus':
            self.activation = nn.Softplus(beta=100)
        elif nl == 'tanh':
            self.activation = nn.Tanh()
        elif nl == "sine":
            self.activation = Sine()
        elif nl == "relu":
            self.activation = nn.ReLU()

    def forward(self, inputs):
        x = inputs
        for l in range(self.num_layers - 1):
            qua = getattr(self, "qua" + str(l))
            if l == self.num_layers - 2:
                x = qua(x)
            else:
                x = qua(x)
                x = self.activation(x)

        if self.init_type == "mfgi" or self.init_type == 'geometric_sine':
            radius, scaling = self.sphere_init_params
            x = torch.sign(x) * torch.sqrt(x.abs() + 1e-8)
            x -= radius  # 1.6
            x *= scaling  # 1.0
        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.unsqueeze(1)


class QuadraticLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_out)
        self.lin2 = nn.Linear(d_in, d_out)
        self.lin3 = nn.Linear(d_in, d_out)

    def forward(self, x):
        return torch.mul(self.lin1(x), self.lin2(x)) + self.lin3(torch.square(x))


def init_lin2_lin3(m):
    nn.init.normal_(m.lin2.weight, mean=0.0, std=1e-5)
    nn.init.ones_(m.lin2.bias)
    nn.init.normal_(m.lin3.weight, mean=0.0, std=1e-5)
    nn.init.zeros_(m.lin3.bias)


################################# SIREN's initialization ###################################
# This file is borrowed from DiGS: https://github.com/Chumbyte/DiGS
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, "lin1") and hasattr(m, "lin2") and hasattr(m, "lin3"):
            num_input = m.lin1.weight.size(-1)
            # See SIREN paper supplement Sec. 1.5 for discussion of factor 30
            m.lin1.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
            init_lin2_lin3(m)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "lin1") and hasattr(m, "lin2") and hasattr(m, "lin3"):
            num_input = m.lin1.weight.size(-1)
            # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.lin1.weight.uniform_(-1 / num_input, 1 / num_input)
            init_lin2_lin3(m)


################################# sine geometric initialization ###################################
# This file is borrowed from DiGS: https://github.com/Chumbyte/DiGS

def geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "lin1") and hasattr(m, "lin2") and hasattr(m, "lin3"):
            num_output = m.lin1.weight.size(0)
            m.lin1.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.lin1.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.lin1.weight.data /= 30
            m.lin1.bias.data /= 30
            init_lin2_lin3(m)


def first_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "lin1") and hasattr(m, "lin2") and hasattr(m, "lin3"):
            num_output = m.lin1.weight.size(0)
            m.lin1.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.lin1.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.lin1.weight.data /= 30
            m.lin1.bias.data /= 30


def second_last_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "lin1") and hasattr(m, "lin2") and hasattr(m, "lin3"):
            num_output = m.lin1.weight.size(0)
            assert m.lin1.weight.shape == (num_output, num_output)
            m.lin1.weight.data = 0.5 * np.pi * torch.eye(num_output) + 0.001 * torch.randn(num_output, num_output)
            m.lin1.bias.data = 0.5 * np.pi * torch.ones(num_output, ) + 0.001 * torch.randn(num_output)
            m.lin1.weight.data /= 30
            m.lin1.bias.data /= 30
            init_lin2_lin3(m)


def last_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "lin1") and hasattr(m, "lin2") and hasattr(m, "lin3"):
            num_input = m.lin1.weight.size(-1)
            assert m.lin1.weight.shape == (1, num_input)
            assert m.lin1.bias.shape == (1,)
            # m.lin1.weight.data = -1 * torch.ones(1, num_input) + 0.001 * torch.randn(num_input)
            m.lin1.weight.data = -1 * torch.ones(1, num_input) + 0.00001 * torch.randn(num_input)
            m.lin1.bias.data = torch.zeros(1) + num_input
            init_lin2_lin3(m)


################################# multi frequency geometric initialization ###################################
# This file is borrowed from DiGS: https://github.com/Chumbyte/DiGS
periods = [1, 30]  # Number of periods of sine the values of each section of the output vector should hit
# periods = [1, 60] # Number of periods of sine the values of each section of the output vector should hit
portion_per_period = np.array([0.25, 0.75])  # Portion of values per section/period


def first_layer_mfgi_init(m):
    global periods
    global portion_per_period
    with torch.no_grad():
        if hasattr(m, "lin1") and hasattr(m, "lin2") and hasattr(m, "lin3"):
            num_input = m.lin1.weight.size(-1)
            num_output = m.lin1.weight.size(0)
            num_per_period = (portion_per_period * num_output).astype(int)  # Number of values per section/period
            assert len(periods) == len(num_per_period)
            assert sum(num_per_period) == num_output
            weights = []
            for i in range(0, len(periods)):
                period = periods[i]
                num = num_per_period[i]
                scale = 30 / period
                weights.append(torch.zeros(num, num_input).uniform_(-np.sqrt(3 / num_input) / scale,
                                                                    np.sqrt(3 / num_input) / scale))
            W0_new = torch.cat(weights, axis=0)
            m.lin1.weight.data = W0_new
            init_lin2_lin3(m)


def second_layer_mfgi_init(m):
    global portion_per_period
    with torch.no_grad():
        if hasattr(m, "lin1") and hasattr(m, "lin2") and hasattr(m, "lin3"):
            num_input = m.lin1.weight.size(-1)
            assert m.lin1.weight.shape == (num_input, num_input)
            num_per_period = (portion_per_period * num_input).astype(int)  # Number of values per section/period
            k = num_per_period[0]  # the portion that only hits the first period
            # W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input), np.sqrt(3 / num_input) / 30) * 0.00001
            W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input),
                                                                np.sqrt(3 / num_input) / 30) * 0.0005
            W1_new_1 = torch.zeros(k, k).uniform_(-np.sqrt(3 / num_input) / 30, np.sqrt(3 / num_input) / 30)
            W1_new[:k, :k] = W1_new_1
            m.lin1.weight.data = W1_new
            init_lin2_lin3(m)


class Sine(nn.Module):
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


