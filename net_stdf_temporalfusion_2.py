import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.dcn.deform_conv import ModulatedDeformConv

import torch
import torch.nn as nn
import torch.nn.functional as F

# Supondo que ModulatedDeformConv já esteja implementado ou importado
# from deform_conv import ModulatedDeformConv

class SpatialTemporalAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SpatialTemporalAttention, self).__init__()
        self.temporal_attn = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        b, t, c, h, w = x.size()
        x_reshape = x.view(b, t * c, h, w)
        spatial_weight = self.spatial_attn(x_reshape).view(b, 1, c, h, w)
        temporal_weight = self.temporal_attn(x).view(b, t, c, 1, 1)
        return x * spatial_weight * temporal_weight

class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: número de canais de entrada (número de quadros, ex.: 7 quadros de luminância).
            out_nc: número de canais de saída.
            nf: número de filtros (channels) para as camadas convolucionais.
            nb: número de camadas na arquitetura U-shape.
            deform_ks: tamanho do kernel deformável.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # Backbone U-shape para extração espacial
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, f'dn_conv{i}',
                nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, f'up_conv{i}',
                nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )

        # Ramo para extração de informações temporais
        self.temporal_fusion = nn.Sequential(*[SpatialTemporalAttention(in_nc, nf) for _ in range(1)])

        # Cabeça de regressão para offset e mask
        # Cada mapa utiliza offset e mask individuais para cada ponto.
        self.offset_mask = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )

        # Convolução deformável com grupos individuais por mapa (in_nc)
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc
        )

    def forward(self, inputs):
        """
        inputs: tensor de forma (B, in_nc, H, W), onde cada canal corresponde a um quadro.
        """
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # Extração de features espaciais (backbone U-shape)
        out_lst = [self.in_conv(inputs)]
        for i in range(1, nb):
            dn_conv = getattr(self, f'dn_conv{i}')
            out_lst.append(dn_conv(out_lst[i - 1]))
        out = self.tr_conv(out_lst[-1])
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, f'up_conv{i}')
            out = up_conv(torch.cat([out, out_lst[i]], dim=1))

        # Extração de informações temporais usando convoluções 3D
        temporal_feat = self.temporal_fusion(inputs)

        # Fusão dos recursos espacial e temporal
        fused_feat = out + temporal_feat  # Fusão simples via soma
        fused_feat = F.relu(fused_feat, inplace=True)

        # Cálculo do offset e mask para a convolução deformável
        off_msk = self.offset_mask(self.out_conv(fused_feat))
        off = off_msk[:, :in_nc * 2 * n_off_msk, ...]
        msk = torch.sigmoid(off_msk[:, in_nc * 2 * n_off_msk:, ...])

        # Aplicação da convolução deformável para fusionar os quadros
        output = F.relu(self.deform_conv(inputs, off, msk), inplace=True)
        return output




class ResBlock(nn.Module):
    """Bloco residual básico para preservação de detalhes."""
    def __init__(self, nf, base_ks=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, base_ks, padding=base_ks//2)
        self.conv2 = nn.Conv2d(nf, nf, base_ks, padding=base_ks//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res  # Skip connection

class EnhancedQE(nn.Module):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3):
        super(EnhancedQE, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=1),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(*[ResBlock(nf, base_ks) for _ in range(nb)])

        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.res_blocks(out)
        out = self.out_conv(out)
        return out


# ==========
# MFVQE network
# ==========

class MFVQE(nn.Module):
    """STDF -> EnhancedQE -> Dynamic Kernel -> residual."""

    def __init__(self, opts_dict):
        """
        Args:
            opts_dict: network parameters defined in YAML.
        """
        super(MFVQE, self).__init__()

        self.radius = opts_dict['radius']
        self.input_len = 2 * self.radius + 1
        self.in_nc = opts_dict['stdf']['in_nc']

        # STDF: fusão espaço-temporal
        self.ffnet = STDF(
            in_nc=self.in_nc * self.input_len,
            out_nc=opts_dict['stdf']['out_nc'],
            nf=opts_dict['stdf']['nf'],
            nb=opts_dict['stdf']['nb'],
            deform_ks=opts_dict['stdf']['deform_ks']
        )

        # EnhancedQE: rede de melhoria global
        self.qenet = EnhancedQE(
            in_nc=opts_dict['qenet']['in_nc'],
            nf=opts_dict['qenet']['nf'],
            nb=opts_dict['qenet']['nb'],
            out_nc=opts_dict['qenet']['out_nc']
        )

     

    def forward(self, x):
        out = self.ffnet(x)        # Passa pelo STDF
        out = self.qenet(out)      # Passa pelo EnhancedQE

        # Residual: adicionar quadro central
        frm_lst = [self.radius + idx_c * self.input_len for idx_c in range(self.in_nc)]
        out += x[:, frm_lst, ...]  # Adiciona quadro central como resíduo

        return out

