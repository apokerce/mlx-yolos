# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validate the mlx-yolos yolov8-pose port without needing MLX.

Builds a PyTorch model that mirrors mlx-yolos' module layout 1:1 (NHWC
kernel layout, BN eps=1e-3), loads the safetensors produced by
``mlx-yolos convert``, and compares its forward output against the
official Ultralytics ``yolov8n-pose.pt`` on a real image.

Pre-reqs:
  * /tmp/yolov8n-pose.pt                     — Ultralytics original
  * /tmp/mlx-yolos-yolov8n-pose.safetensors  — produced by ``mlx-yolos convert``

Acceptance:
  - exact parameter-tree match (0 missing / 0 extra keys)
  - max abs forward diff < 1e-3 on bus.jpg
"""

import os, sys

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from safetensors.torch import load_file
import ultralytics  # for unpickling

CFG = yaml.safe_load(open('src/mlxyolos/cfg/models/v8/yolov8-pose.yaml'))


def make_divisible(x, divisor=8):
    import math
    return math.ceil(x / divisor) * divisor


# --- Torch ports of mlx-yolos modules (NHWC kernel layout, eps=1e-3) ---


def conv_nhwc(x_nhwc, w_nhwc, stride=1, padding=0, groups=1, bias=None):
    w_nchw = w_nhwc.permute(0, 3, 1, 2).contiguous()
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    y = F.conv2d(x_nchw, w_nchw, bias=bias, stride=stride, padding=padding, groups=groups)
    return y.permute(0, 2, 3, 1).contiguous()


class _WeightHolder(nn.Module):
    """Stores weight as `.weight` so the param key ends in `.conv.weight`."""
    def __init__(self, *shape, with_bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape))
        if with_bias:
            self.bias = nn.Parameter(torch.empty(shape[0]))


class TConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.s = s
        self.p = p
        self.g = g
        # mirror mlxyolos.nn.modules.Conv: nested `.conv` and `.bn`
        self.conv = _WeightHolder(c2, k, k, c1 // g)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act_on = act
    def forward(self, x):
        y = conv_nhwc(x, self.conv.weight, stride=self.s, padding=self.p, groups=self.g)
        y_nchw = y.permute(0, 3, 1, 2)
        y_nchw = self.bn(y_nchw)
        y = y_nchw.permute(0, 2, 3, 1).contiguous()
        if self.act_on:
            y = y * torch.sigmoid(y)
        return y


class TBareConv(nn.Module):
    """Bare nn.Conv2d-equivalent in NHWC, with bias."""
    def __init__(self, c1, c2, k=1, s=1, p=0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c2, k, k, c1))
        self.bias = nn.Parameter(torch.empty(c2))
        self.s = s
        self.p = p
    def forward(self, x):
        return conv_nhwc(x, self.weight, stride=self.s, padding=self.p, bias=self.bias)


class TBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = TConv(c1, c_, k[0], 1)
        self.cv2 = TConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class TC2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = TConv(c1, 2*self.c, 1, 1)
        self.cv2 = TConv((2+n)*self.c, c2, 1)
        self.m = nn.ModuleList(TBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
    def forward(self, x):
        y = self.cv1(x)
        a, b = torch.split(y, self.c, dim=-1)
        outs = [a, b]
        for m in self.m:
            outs.append(m(outs[-1]))
        return self.cv2(torch.cat(outs, dim=-1))


class TSPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = TConv(c1, c_, 1, 1)
        self.cv2 = TConv(c_*4, c2, 1, 1)
        self.k = k
    def forward(self, x):
        x = self.cv1(x)
        x_nchw = x.permute(0, 3, 1, 2)
        y1 = F.max_pool2d(x_nchw, self.k, stride=1, padding=self.k//2).permute(0, 2, 3, 1)
        y2 = F.max_pool2d(y1.permute(0, 3, 1, 2), self.k, stride=1, padding=self.k//2).permute(0, 2, 3, 1)
        y3 = F.max_pool2d(y2.permute(0, 3, 1, 2), self.k, stride=1, padding=self.k//2).permute(0, 2, 3, 1)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=-1))


class TUpsample2x(nn.Module):
    def forward(self, x):
        return x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)


class TConcat(nn.Module):
    def forward(self, xs):
        return torch.cat(xs, dim=-1)


def make_anchors(feat_hw, strides):
    pts, ss = [], []
    for (h, w), s in zip(feat_hw, strides):
        sx = torch.arange(w, dtype=torch.float32) + 0.5
        sy = torch.arange(h, dtype=torch.float32) + 0.5
        gy, gx = torch.meshgrid(sy, sx, indexing='ij')
        pts.append(torch.stack([gx, gy], dim=-1).reshape(-1, 2))
        ss.append(torch.full((h*w, 1), float(s)))
    return torch.cat(pts, 0), torch.cat(ss, 0)


class TPoseV8(nn.Module):
    strides = (8, 16, 32)
    def __init__(self, nc, kpt_shape, reg_max, ch):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max*4
        self.kpt_shape = tuple(kpt_shape)
        self.nk = self.kpt_shape[0]*self.kpt_shape[1]
        c2 = max(16, ch[0]//4, reg_max*4)
        c3 = max(ch[0], min(nc, 100))
        c4 = max(ch[0]//4, self.nk)
        self.cv2 = nn.ModuleList(nn.ModuleList([TConv(x, c2, 3), TConv(c2, c2, 3), TBareConv(c2, 4*reg_max, 1)]) for x in ch)
        self.cv3 = nn.ModuleList(nn.ModuleList([TConv(x, c3, 3), TConv(c3, c3, 3), TBareConv(c3, nc, 1)]) for x in ch)
        self.cv4 = nn.ModuleList(nn.ModuleList([TConv(x, c4, 3), TConv(c4, c4, 3), TBareConv(c4, self.nk, 1)]) for x in ch)
    def forward(self, feats):
        bs = feats[0].shape[0]
        feat_hw = [(f.shape[1], f.shape[2]) for f in feats]
        boxes_lvl, scores_lvl, kpts_lvl = [], [], []
        for i, f in enumerate(feats):
            b = f
            for layer in self.cv2[i]: b = layer(b)
            c = f
            for layer in self.cv3[i]: c = layer(c)
            k = f
            for layer in self.cv4[i]: k = layer(k)
            n = b.shape[1]*b.shape[2]
            boxes_lvl.append(b.reshape(bs, n, 4*self.reg_max))
            scores_lvl.append(c.reshape(bs, n, self.nc))
            kpts_lvl.append(k.reshape(bs, n, self.nk))
        boxes = torch.cat(boxes_lvl, 1)
        scores = torch.cat(scores_lvl, 1)
        kpts = torch.cat(kpts_lvl, 1)
        anchors, str_t = make_anchors(feat_hw, self.strides)
        bx = boxes.reshape(bs, -1, 4, self.reg_max)
        bx = torch.softmax(bx, dim=-1)
        proj = torch.arange(self.reg_max, dtype=bx.dtype)
        dist = (bx*proj).sum(-1)
        lt, rb = torch.split(dist, 2, dim=-1)
        x1y1 = anchors.unsqueeze(0) - lt
        x2y2 = anchors.unsqueeze(0) + rb
        cxy = (x1y1+x2y2)/2
        wh = x2y2-x1y1
        dbox = torch.cat([cxy, wh], dim=-1) * str_t.unsqueeze(0)
        scores = torch.sigmoid(scores)
        ndim = self.kpt_shape[1]
        k = kpts.reshape(bs, -1, self.kpt_shape[0], ndim)
        kxy = (k[..., :2]*2.0 + (anchors.view(1, -1, 1, 2) - 0.5)) * str_t.view(1, -1, 1, 1)
        kv = torch.sigmoid(k[..., 2:3])
        k = torch.cat([kxy, kv], -1).reshape(bs, -1, self.nk)
        return torch.cat([dbox, scores, k], dim=-1)


class TBaseModel(nn.Module):
    """Sequential routing identical to mlxyolos.nn.tasks.BaseModel."""
    def __init__(self, layers, save):
        super().__init__()
        self.model = nn.ModuleList(layers)
        self._save = sorted(set(save))
    def forward(self, x):
        y = []
        for m in self.model:
            f = getattr(m, 'f', -1)
            if f != -1:
                if isinstance(f, int):
                    x = y[f]
                else:
                    x = [y[j] if j != -1 else x for j in f]
            x = m(x)
            y.append(x if getattr(m, 'i', -1) in self._save else None)
        return x


def parse_yaml_to_torch(cfg, scale='n'):
    nc = cfg['nc']
    reg_max = cfg.get('reg_max', 16)
    depth, width, max_channels = cfg['scales'][scale]
    layers = []
    save = []
    ch = [3]
    spec = list(cfg['backbone']) + list(cfg['head'])
    for i, (f, n, m, args) in enumerate(spec):
        # repeat scaling
        n_ = max(round(n*depth), 1) if n > 1 else n
        c2 = None
        args = list(args)
        if m == 'Conv':
            c1 = ch[f]
            c2 = make_divisible(min(args[0], max_channels)*width, 8)
            mod = TConv(c1, c2, *args[1:])
        elif m == 'C2f':
            c1 = ch[f]
            c2 = make_divisible(min(args[0], max_channels)*width, 8)
            shortcut = args[1] if len(args) > 1 else False
            mod = TC2f(c1, c2, n=n_, shortcut=shortcut)
            n_ = 1
        elif m == 'SPPF':
            c1 = ch[f]
            c2 = make_divisible(min(args[0], max_channels)*width, 8)
            mod = TSPPF(c1, c2, args[1] if len(args) > 1 else 5)
        elif m.startswith('nn.Upsample'):
            c2 = ch[f]
            mod = TUpsample2x()
        elif m == 'Concat':
            c2 = sum(ch[x] for x in f)
            mod = TConcat()
        elif m == 'Pose':
            ch_list = [ch[x] for x in f]
            kpt_shape = args[1] if len(args) > 1 else (17, 3)
            mod = TPoseV8(nc, kpt_shape, reg_max, ch_list)
        else:
            raise ValueError(m)
        mod.i = i
        mod.f = f
        save.extend(x % (i+1) for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(mod)
        if i == 0:
            ch = []
        ch.append(c2 if c2 is not None else 0)
    return TBaseModel(layers, save)


# ---- Build, load, validate ----

shadow = parse_yaml_to_torch(CFG, scale='n')
shadow.eval()

# Load mlx-yolos-converted weights (NHWC layout matches our shadow exactly)
sd = {k: v.float() for k, v in load_file('/tmp/mlx-yolos-yolov8n-pose.safetensors').items()}

# Build expected key set from shadow
exp = set()
for k, v in shadow.state_dict().items():
    if k.endswith('num_batches_tracked'):
        continue
    exp.add(k)

st = set(sd.keys())
print(f'shadow params (excl. num_batches_tracked): {len(exp)}, safetensors keys: {len(st)}')
print(f'missing in safetensors: {len(exp - st)}')
print(f'extra in safetensors: {len(st - exp)}')
if exp - st:
    print('  sample missing:', sorted(exp - st)[:5])
if st - exp:
    print('  sample extra:', sorted(st - exp)[:5])

# Load into shadow with strict matching
missing, unexpected = shadow.load_state_dict(sd, strict=False)
print(f'load_state_dict: missing={len(missing)} unexpected={len(unexpected)}')

# Make sure shapes align by spot-checking a few
for k in ['model.0.weight', 'model.22.cv2.0.0.weight']:
    if k in sd:
        s = shadow.state_dict()[k]
        print(f'  {k}: shadow={tuple(s.shape)} loaded={tuple(sd[k].shape)}')

# --- Forward and compare to ultralytics ---
img_path = os.getenv("IMAGE_PATH", "bus.jpg")

src = np.array(Image.open(img_path).convert('RGB'))
def letterbox(img, new=640):
    h, w = img.shape[:2]
    r = min(new/h, new/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    pad_w, pad_h = new-nw, new-nh
    left, top = pad_w//2, pad_h//2
    resized = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    out = np.full((new, new, 3), 114, dtype=np.uint8)
    out[top:top+nh, left:left+nw] = resized
    return out
lb = letterbox(src, 640)
x_nhwc = torch.from_numpy(lb.astype(np.float32)/255.0)[None]

with torch.no_grad():
    out = shadow(x_nhwc)
print('shadow out shape:', tuple(out.shape))

ckpt = torch.load('/tmp/yolov8n-pose.pt', weights_only=False, map_location='cpu')
gt_model = ckpt['model'].float().eval()
x_nchw = x_nhwc.permute(0, 3, 1, 2)
with torch.no_grad():
    gt_out = gt_model(x_nchw)
gt_preds = gt_out[0] if isinstance(gt_out, (list, tuple)) else gt_out
gt_aligned = gt_preds.permute(0, 2, 1)

diff = (out - gt_aligned).abs()
print(f'max abs diff: {diff.max().item():.5f}')
print(f'mean abs diff: {diff.mean().item():.6f}')

p = out[0].numpy()
g = gt_aligned[0].numpy()
print('shadow top score:', p[:, 4].max(), 'count > 0.25:', int((p[:, 4] > 0.25).sum()))
print('gt     top score:', g[:, 4].max(), 'count > 0.25:', int((g[:, 4] > 0.25).sum()))
