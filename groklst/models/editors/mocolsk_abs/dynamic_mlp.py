import torch
import torch.nn as nn


class FCResLayer(nn.Module):
    def __init__(self, linear_size=256):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=False)
        self.nonlin2 = nn.ReLU(inplace=False)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class FCNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_filts=256):
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.feats = nn.Sequential(
            nn.Linear(num_inputs, num_filts),
            nn.ReLU(inplace=False),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
        )
        self.class_emb = nn.Linear(num_filts, num_outputs, bias=self.inc_bias)

    def forward(self, x, return_feats=False, class_of_interest=None):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb  # [b, num_filts]
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        return class_pred  # [b, num_class]

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])


class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(
            conv,
        )
        if not bias:
            self.conv.add_module("ln", nn.LayerNorm(out_channels))
        self.conv.add_module("relu", nn.ReLU(inplace=False))

    def forward(self, x):
        out = self.conv(x)
        return out


class Dynamic_MLP_A(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.get_weight = nn.Linear(loc_planes, inplanes * planes)
        self.norm = nn.LayerNorm(planes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, img_fea, loc_fea):
        if loc_fea.ndim == 3:
            B, N, C = loc_fea.shape
            weight = self.get_weight(loc_fea)
            weight = weight.view(B, N, self.inplanes, self.planes)
        elif loc_fea.ndim == 2:
            B, C = loc_fea.shape
            weight = self.get_weight(loc_fea)
            weight = weight.view(B, self.inplanes, self.planes)

        # img_fea = torch.bmm(img_fea.unsqueeze(-2), weight).squeeze(1)
        img_fea = img_fea.unsqueeze(-2) @ weight
        img_fea = img_fea.squeeze(-2)
        img_fea = self.norm(img_fea)
        img_fea = self.relu(img_fea)

        return img_fea


class Dynamic_MLP_B(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.conv11 = Basic1d(inplanes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=False),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        _ndim = 3
        if loc_fea.ndim == 2:
            _ndim = 2
            loc_fea = loc_fea.unsqueeze(-2)
            img_fea = img_fea.unsqueeze(-2)
        B, N, C = loc_fea.shape  #

        weight11 = self.conv11(img_fea)
        weight12 = self.conv12(weight11)

        weight21 = self.conv21(loc_fea)
        # weight22 = self.conv22(weight21).view(-1, self.inplanes, self.planes)
        weight22 = self.conv22(weight21).view(B, N, self.inplanes, self.planes)  #

        img_fea = (weight12.unsqueeze(-2) @ weight22).squeeze(-2)
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        if _ndim == 2:
            img_fea = img_fea.squeeze(-2)

        return img_fea


class Dynamic_MLP_C(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.conv11 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=False),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        _ndim = 3
        if loc_fea.ndim == 2:
            _ndim = 2
            loc_fea = loc_fea.unsqueeze(-2)
            img_fea = img_fea.unsqueeze(-2)

        B, N, C = loc_fea.shape  #
        cat_fea = torch.cat([img_fea, loc_fea], -1)

        weight11 = self.conv11(cat_fea)
        weight12 = self.conv12(weight11)

        weight21 = self.conv21(cat_fea)
        # weight22 = self.conv22(weight21).view(-1, self.inplanes, self.planes)
        weight22 = self.conv22(weight21).view(B, N, self.inplanes, self.planes)  #

        img_fea = (weight12.unsqueeze(-2) @ weight22).squeeze(-2)
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)
        
        if _ndim == 2:
            img_fea = img_fea.squeeze(-2)

        return img_fea


class RecursiveBlock(nn.Module):
    def __init__(self, inplanes, planes, loc_planes, mlp_type="c"):
        super().__init__()
        if mlp_type.lower() == "a":
            MLP = Dynamic_MLP_A
        elif mlp_type.lower() == "b":
            MLP = Dynamic_MLP_B
        elif mlp_type.lower() == "c":
            MLP = Dynamic_MLP_C

        self.dynamic_conv = MLP(inplanes, planes, loc_planes)

    def forward(self, img_fea, loc_fea):
        img_fea = self.dynamic_conv(img_fea, loc_fea)
        return img_fea, loc_fea


class FusionModule(nn.Module):
    """FusionModule for MoCoLSKConv.
    """
    def __init__(self, inplanes=2048, planes=256, hidden=64, num_layers=2, mlp_type="a", kernel_size=3):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.hidden = hidden

        self.conv1 = nn.Linear(inplanes, planes)

        conv2 = []
        if num_layers == 1:
            conv2.append(RecursiveBlock(planes, planes, loc_planes=planes, mlp_type=mlp_type))
        else:
            conv2.append(RecursiveBlock(planes, hidden, loc_planes=planes, mlp_type=mlp_type))
            for _ in range(1, num_layers - 1):
                conv2.append(RecursiveBlock(hidden, hidden, loc_planes=planes, mlp_type=mlp_type))
            conv2.append(RecursiveBlock(hidden, planes, loc_planes=planes, mlp_type=mlp_type))
        self.conv2 = nn.ModuleList(conv2)
        weights_dim = 2 * 2 * kernel_size**2
        self.conv3 = nn.Linear(planes, weights_dim)
        self.norm3 = nn.LayerNorm(weights_dim)
        self.conv4 = nn.Linear(inplanes, weights_dim)

    def forward(self, img_fea, loc_fea):
        """
        img_fea: (N, channel), backbone输出经过全局池化的feature
        loc_fea: (N, fea_dim)
        """
        identity = img_fea
        img_fea = self.conv1(img_fea)
        for m in self.conv2:
            img_fea, loc_fea = m(img_fea, loc_fea)

        img_fea = self.conv3(img_fea)
        img_fea = self.norm3(img_fea)

        img_fea = img_fea + self.conv4(identity)
        img_fea = torch.mean(img_fea, dim=0, keepdim=True)

        return img_fea


class FusionModule12(nn.Module):
    """FusionModule for MoCoOneLKConv.
    """
    def __init__(self, inplanes=2048, planes=256, hidden=64, num_layers=2, mlp_type="a", kernel_size=3):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.hidden = hidden

        self.conv1 = nn.Linear(inplanes, planes)

        conv2 = []
        if num_layers == 1:
            conv2.append(RecursiveBlock(planes, planes, loc_planes=planes, mlp_type=mlp_type))
        else:
            conv2.append(RecursiveBlock(planes, hidden, loc_planes=planes, mlp_type=mlp_type))
            for _ in range(1, num_layers - 1):
                conv2.append(RecursiveBlock(hidden, hidden, loc_planes=planes, mlp_type=mlp_type))
            conv2.append(RecursiveBlock(hidden, planes, loc_planes=planes, mlp_type=mlp_type))
        self.conv2 = nn.ModuleList(conv2)
        weights_dim = 1 * 2 * kernel_size**2
        self.conv3 = nn.Linear(planes, weights_dim)
        self.norm3 = nn.LayerNorm(weights_dim)
        self.conv4 = nn.Linear(inplanes, weights_dim)

    def forward(self, img_fea, loc_fea):
        """
        img_fea: (N, channel), backbone输出经过全局池化的feature
        loc_fea: (N, fea_dim)
        """
        identity = img_fea
        img_fea = self.conv1(img_fea)
        for m in self.conv2:
            img_fea, loc_fea = m(img_fea, loc_fea)

        img_fea = self.conv3(img_fea)
        img_fea = self.norm3(img_fea)

        img_fea = img_fea + self.conv4(identity)
        img_fea = torch.mean(img_fea, dim=0, keepdim=True)

        return img_fea



class CSFusionModule(nn.Module):
    """CSFusionModule for MoCoLSKCSConv.
    """
    def __init__(self, inplanes=2048, planes=256, hidden=64, num_layers=2, mlp_type="a", weights_dim=32):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.hidden = hidden

        self.conv1 = nn.Linear(inplanes, planes)

        conv2 = []
        if num_layers == 1:
            conv2.append(RecursiveBlock(planes, planes, loc_planes=planes, mlp_type=mlp_type))
        else:
            conv2.append(RecursiveBlock(planes, hidden, loc_planes=planes, mlp_type=mlp_type))
            for _ in range(1, num_layers - 1):
                conv2.append(RecursiveBlock(hidden, hidden, loc_planes=planes, mlp_type=mlp_type))
            conv2.append(RecursiveBlock(hidden, planes, loc_planes=planes, mlp_type=mlp_type))
        self.conv2 = nn.ModuleList(conv2)
        self.conv3 = nn.Linear(planes, weights_dim)
        self.norm3 = nn.LayerNorm(weights_dim)
        self.conv4 = nn.Linear(inplanes, weights_dim)

    def forward(self, img_fea, loc_fea):
        """
        img_fea: (N, channel), backbone输出经过全局池化的feature
        loc_fea: (N, fea_dim)
        """
        identity = img_fea
        img_fea = self.conv1(img_fea)
        for m in self.conv2:
            img_fea, loc_fea = m(img_fea, loc_fea)

        img_fea = self.conv3(img_fea)
        img_fea = self.norm3(img_fea)

        img_fea = img_fea + self.conv4(identity) # B N C
        # img_fea = torch.mean(img_fea, dim=0, keepdim=True)

        return img_fea 