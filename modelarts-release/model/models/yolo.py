import argparse
import logging
import sys
from copy import deepcopy

sys.path.append("/home/mind/model/")
sys.path.append('./')
logger = logging.getLogger(__name__)
from models.common import *
from models.experimental import *
from utils1.autoanchor import check_anchor_order
from utils1.general import make_divisible, check_file, set_logging
from utils1.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from utils1.loss import SigmoidBin

try:
    import thop
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None
    export = False
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super(Detect, self).__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)


class IDetect(nn.Module):
    stride = None
    export = False
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super(IDetect, self).__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def fuseforward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs * self.na, self.no, ny, nx).permute(0, 2, 3, 1).contiguous()

            if not self.training:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].reshape(-1, 1, 1, 2)
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 3)
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data.reshape(-1, 1, 1, 2))
                    y = torch.cat((xy, wh, conf), 3)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        print("IDetect.fuse")
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)
            
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32, device=z.device)
        box @= convert_matrix                          
        return box, score


class IKeypoint(nn.Module):
    stride = None
    export = False

    def __init__(self, nc=80, anchors=(), nkpt=17, ch=(), inplace=True, dw_conv_kpt=False):
        super(IKeypoint, self).__init__()
        self.nc = nc
        self.nkpt = nkpt
        self.dw_conv_kpt = dw_conv_kpt
        self.no_det = (nc + 5)
        self.no_kpt = 3*self.nkpt
        self.no = self.no_det + self.no_kpt
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.flip_test = False
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no_det * self.na, 1) for x in ch)
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)
        
        if self.nkpt is not None:
            if self.dw_conv_kpt:
                self.m_kpt = nn.ModuleList(
                            nn.Sequential(DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1)) for x in ch)
            else:
                self.m_kpt = nn.ModuleList(nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch)

        self.inplace = inplace

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            if self.nkpt is None or self.nkpt == 0:
                x[i] = self.im[i](self.m[i](self.ia[i](x[i])))
            else:
                x[i] = torch.cat([self.im[i](self.m[i](self.ia[i](x[i]))), self.m_kpt[i](x[i])], axis=1)

            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_det = x[i][..., :6]
            x_kpt = x[i][..., 6:]

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.nkpt == 0:
                    y = x[i].sigmoid()
                else:
                    y = x_det.sigmoid()

                if self.inplace:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)
                    if self.nkpt != 0:
                        x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 +
                                            kpt_grid_x.repeat(1, 1, 1, 1, 17)) * self.stride[i]
                        x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 +
                                            kpt_grid_y.repeat(1, 1, 1, 1, 17)) * self.stride[i]
                        x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                    y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)

                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    if self.nkpt != 0:
                        y[..., 6:] = (y[..., 6:] * 2. - 0.5 +
                                      self.grid[i].repeat((1, 1, 1, 1, self.nkpt))) * self.stride[i]
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class IAuxDetect(nn.Module):
    stride = None
    export = False
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super(IAuxDetect, self).__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl:])
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[:self.nl])

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            x[i+self.nl] = self.m2[i](x[i+self.nl])
            x[i+self.nl] = x[i+self.nl].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x[:self.nl])

    def fuseforward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)            
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    def fuse(self):
        print("IAuxDetect.fuse")
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32, device=z.device)
        box @= convert_matrix                          
        return box, score


class IBin(nn.Module):
    stride = None
    export = False

    def __init__(self, nc=80, anchors=(), ch=(), bin_count=21):
        super(IBin, self).__init__()
        self.nc = nc
        self.bin_count = bin_count

        self.w_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        self.h_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        self.no = nc + 3 + \
            self.w_bin_sigmoid.get_length() + self.h_bin_sigmoid.get_length()
        
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):

        self.w_bin_sigmoid.use_fw_regression = True
        self.h_bin_sigmoid.use_fw_regression = True
        
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                
                pw = self.w_bin_sigmoid.forward(y[..., 2:24]) * self.anchor_grid[i][..., 0]
                ph = self.h_bin_sigmoid.forward(y[..., 24:46]) * self.anchor_grid[i][..., 1]

                y[..., 2] = pw
                y[..., 3] = ph
                
                y = torch.cat((y[..., 0:4], y[..., 46:]), dim=-1)
                
                z.append(y.view(bs, -1, y.shape[-1]))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None):
        super(Model, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()
        if isinstance(m, IDetect):
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()
        if isinstance(m, IAuxDetect):
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:4]])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_aux_biases()
        if isinstance(m, IBin):
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases_bin()
        if isinstance(m, IKeypoint):
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases_kpt()

        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]
            s = [1, 0.83, 0.67]
            f = [None, 3, None]
            y = []
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]
                yi[..., :4] /= si
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]
                y.append(yi)
            return torch.cat(y, 1), None
        else:
            return self.forward_once(x, profile)

    def forward_once(self, x, profile=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            if not hasattr(self, 'traced'):
                self.traced == False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect) \
                        or isinstance(m, IKeypoint):
                    break

            if profile:
                c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)
            
            y.append(x if m.i in self.save else None)

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):
        m = self.model[-1]
        for mi, mi2, s in zip(m.m, m.m2, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _initialize_biases_bin(self, cf=None):
        m = self.model[-1]
        bc = m.bin_count
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            old = b[:, (0,1,2,bc+3)].data
            obj_idx = 2*bc+4
            b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)
            b[:, (obj_idx+1):].data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            b[:, (0, 1, 2, bc+3)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_kpt(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]
        for mi in m.m:
            b = mi.bias.detach().view(m.na, -1).T
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif isinstance(m, RepConv_OREPA):
                m.switch_to_deploy()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
            elif isinstance(m, (IDetect, IAuxDetect)):
                m.fuse()
                m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):
        present = type(self.model[-1]) is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()
            m.f = -1
            m.i = self.model[-1].i + 1
            self.model.add_module(name='%s' % m.i, module=m)
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]
        return self

    def autoshape(self):
        print('Adding autoShape... ')
        m = autoShape(self)
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
        return m

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)


def parse_model(d, ch):
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n
        if m in [nn.Conv2d, Conv, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC,
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv, 
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,  
                 Res, ResCSPA, ResCSPB, ResCSPC, 
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC, 
                 ResXCSPA, ResXCSPB, ResXCSPC,
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC, conv_bn_relu_maxpool, Shuffle_Block, DWConvblock,
                 C3, RepVGGBlock, SEBlock, CBH, LC_Block, Dense]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC, 
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC, 
                     ResCSPA, ResCSPB, ResCSPC, 
                     RepResCSPA, RepResCSPB, RepResCSPC, 
                     ResXCSPA, ResXCSPB, ResXCSPC, 
                     GhostCSPA, GhostCSPB, GhostCSPC, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is ADD:
            c2 = sum([ch[x] for x in f]) // 2
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m in [Detect, IDetect, IAuxDetect, IBin, IKeypoint]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolor-csp-c.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)
    set_logging()
    device = select_device(opt.device)

    model = Model(opt.cfg).to(device)
    model.train()
    
    if opt.profile:
        img = torch.rand(1, 3, 640, 640).to(device)
        y = model(img, profile=True)
