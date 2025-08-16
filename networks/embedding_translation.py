# Embedding translation
import logging, torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def GN(c, g=8): 
    return nn.GroupNorm(min(g, c), c)
def drop_path(x, p=0., training=False):
    if p == 0. or not training: 
        return x
    keep = torch.rand(x.size(0), 1, 1, 1, device=x.device) > p
    return x.div(1 - p) * keep

class DropPath(nn.Module):
    def __init__(self, p): 
        super().__init__(); 
        self.p = p
    def forward(self, x): 
        return drop_path(x, self.p, self.training)

class DepthWiseGatedConv(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.dw  = nn.Conv2d(cin, cin, k, s, p, groups=cin, bias=False)
        self.pw  = nn.Conv2d(cin, cout * 2, 1, bias=False)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.act(self.dw(x))
        a, b = self.pw(x).chunk(2, 1)
        return a * torch.sigmoid(b)

class MBConv(nn.Module):
    def __init__(self, cin, cout, expansion=2, drop_rate=0.1):
        super().__init__()
        mid = cin * expansion
        self.use_res = cin == cout
        self.block = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False), GN(mid), nn.SiLU(),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            GN(mid), nn.SiLU(),
            nn.Conv2d(mid, cout, 1, bias=False), GN(cout)
        )
        self.drop = DropPath(drop_rate)
    def forward(self, x):
        y = self.block(x)
        if self.use_res:
            y = self.drop(y)
            return x + y
        return y

class SSA(nn.Module):
    def __init__(self, c, r=2):
        super().__init__()
        for h in (8, 6, 4, 3, 2, 1):
            if c % h == 0:
                self.h = h
                break
        self.q = nn.Conv2d(c, c, 1, bias=False)
        self.k = nn.Conv2d(c, c, 1, bias=False)
        self.v = nn.Conv2d(c, c, 1, bias=False)
        self.nq, self.nk = GN(c), GN(c)
        self.r, self.scale = r, (c // self.h) ** -0.5
        self.proj = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.q(self.nq(x)).reshape(B, self.h, C // self.h, H * W)
        kv = F.avg_pool2d(x, self.r)
        hk, wk = kv.shape[-2:]
        k = self.k(self.nk(kv)).reshape(B, self.h, C // self.h, hk * wk)
        v = self.v(self.nk(kv)).reshape(B, self.h, C // self.h, hk * wk)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(-1)
        y = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        y = y.reshape(B, C, H, W)
        return x + self.proj(y)

class FEM(nn.Module):
    def __init__(self, c, rates=(1, 6, 12)):
        super().__init__()
        self.br = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=r, dilation=r, groups=c, bias=False),
                nn.Conv2d(c, c, 1, bias=False), nn.SiLU()
            ) for r in rates
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(len(rates) * c, c, 3, padding=1, bias=False),
            GN(c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False), GN(c)
        )
    def forward(self, x):
        y = self.fuse(torch.cat([b(x) for b in self.br], 1))
        return x + y

class MSFA(nn.Module):
    def __init__(self, chs, out):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, out, 1, bias=False) for c in chs])
        self.ref  = nn.Sequential(
            nn.Conv2d(len(chs) * out, out, 3, padding=1, bias=False),
            GN(out), nn.SiLU(),
            nn.Conv2d(out, out, 3, padding=1, bias=False), GN(out), nn.SiLU()
        )
    def forward(self, feats):
        H, W = feats[0].shape[-2:]
        up = [self.proj[i](F.interpolate(f, (H, W), mode='nearest'))
              for i, f in enumerate(feats)]
        return self.ref(torch.cat(up, 1))

class MainPipeline(nn.Module):
    def __init__(self, in_channel=4, out_channels=4,
                 ch=32, ch_mult=(1, 2, 3, 3), drop_rate=0.1):
        super().__init__()
        self.stem = nn.Conv2d(in_channel, ch, 3, padding=1, bias=True)

        c1, c2, c3, c4 = [ch * m for m in ch_mult]
        mb = partial(MBConv, drop_rate=drop_rate)

        self.enc1 = mb(ch,  c1)
        self.enc2 = mb(c1,  c2)
        self.enc3 = mb(c2,  c3)
        self.enc4 = mb(c3,  c4)
        self.bottleneck = SSA(c4)

        self.s1 = nn.Conv2d(c1, c1 // 2, 1, bias=False)
        self.s2 = nn.Conv2d(c2, c2 // 2, 1, bias=False)
        self.s3 = nn.Conv2d(c3, c3 // 2, 1, bias=False)

        self.dec3 = nn.Sequential(mb(c3 // 2 + c4, c3), FEM(c3))
        self.dec2 = nn.Sequential(mb(c2 // 2 + c3, c2), FEM(c2))
        self.dec1 = nn.Sequential(mb(c1 // 2 + c2, c1), FEM(c1))
        self.dec0 = nn.Sequential(mb(ch + c1,  ch), FEM(ch))

        def head(cin):
            return nn.Sequential(GN(cin), nn.SiLU(),
                                 nn.Conv2d(cin, out_channels, 3, padding=1, bias=False))
        self.h3, self.h2, self.h1, self.h0 = head(c3), head(c2), head(c1), head(ch)

        self.msfa = MSFA([c1, ch, ch], ch)
        self._init(); 

    @staticmethod
    def _up(x, ref): return F.interpolate(x, size=ref.shape[-2:], mode='nearest')

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        x3 = self.enc3(F.avg_pool2d(x2, 2))
        x4 = self.enc4(F.avg_pool2d(x3, 2))
        x4 = self.bottleneck(x4)

        d3 = self.dec3(torch.cat([self.s3(x3), self._up(x4, x3)], 1))
        d2 = self.dec2(torch.cat([self.s2(x2), self._up(d3, x2)], 1))
        d1 = self.dec1(torch.cat([self.s1(x1), self._up(d2, x1)], 1))
        d0 = self.dec0(torch.cat([x0,          self._up(d1, x0)], 1))

        fused = self.msfa([d1, d0, x0])

        return {
            'level3': self.h3(d3),
            'level2': self.h2(d2),
            'level1': self.h1(d1),
            'out'   : self.h0(fused)
        }

    # utils
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)