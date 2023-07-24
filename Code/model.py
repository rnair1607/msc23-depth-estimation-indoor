import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as torch_nn_func
import math

def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
        m.eval()                      # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.cuda())
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.cuda())

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params  
        if params.encoder == 'densenet161':
            self.selected_model = models.densenet161(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [96, 96, 192, 384, 2208]

        else:
            self.selected_model = models.densenet121(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [64, 64, 128, 256, 1024]

    def forward(self, x):
        feature = x
        skip_feat = []
        i = 1
        for k, v in self.selected_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            print("Feature before:::",feature)
            print("Feature type:::",type(feature))

            feature = v(feature)
            print("Feature after:::",feature)
            print("Feature type:::",type(feature))
            if any(x in k for x in self.feat_names):
                    skip_feat.append(feature)
            i += 1
        return skip_feat
    
class Atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(Atrous_conv, self).__init__()
        self.Atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.Atrous_conv.add_module('bn1', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True, track_running_stats=True, eps=1.1e-5))
        
        self.Atrous_conv.add_module('at_conv_seq', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, bias=False, kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels*2, momentum=0.01, affine=True, track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation), dilation=dilation)))

    def forward(self, x):
        return self.Atrous_conv.forward(x)
    

class Upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(Upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio
        
    def forward(self, x):
        up_x = torch_nn_func.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out


class Reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(Reduction_1x1, self).__init__()        
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()
        
        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                                                                                 kernel_size=1, stride=1, padding=0),
                                                                       nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2
    
    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)
        
        return net

class Local_planar_guidance(nn.Module):
    def __init__(self, upratio):
        super(Local_planar_guidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq, focal):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]
        
        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio
        
        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3)
    
class Decoder(nn.Module):
    def __init__(self, params, feat_out_channels, num_features=512):
        super(Decoder, self).__init__()
        self.params = params

        self.Upconv5    = Upconv(feat_out_channels[4], num_features)
        self.bn5        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        
        self.conv5      = torch.nn.Sequential(nn.Conv2d(num_features + feat_out_channels[3], num_features, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.Upconv4    = Upconv(num_features, num_features // 2)
        self.bn4        = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4      = torch.nn.Sequential(nn.Conv2d(num_features // 2 + feat_out_channels[2], num_features // 2, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.bn4_2      = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        
        self.daspp_3    = Atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6    = Atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[2], num_features // 4, 6)
        self.daspp_12   = Atrous_conv(num_features + feat_out_channels[2], num_features // 4, 12)
        self.daspp_18   = Atrous_conv(num_features + num_features // 4 + feat_out_channels[2], num_features // 4, 18)
        self.daspp_24   = Atrous_conv(num_features + num_features // 2 + feat_out_channels[2], num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.reduc8x8   = Reduction_1x1(num_features // 4, num_features // 4, self.params.max_depth)
        self.lpg8x8     = Local_planar_guidance(8)
        
        self.Upconv3    = Upconv(num_features // 4, num_features // 4)
        self.bn3        = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3      = torch.nn.Sequential(nn.Conv2d(num_features // 4 + feat_out_channels[1] + 1, num_features // 4, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.reduc4x4   = Reduction_1x1(num_features // 4, num_features // 8, self.params.max_depth)
        self.lpg4x4     = Local_planar_guidance(4)
        
        self.Upconv2    = Upconv(num_features // 4, num_features // 8)
        self.bn2        = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2      = torch.nn.Sequential(nn.Conv2d(num_features // 8 + feat_out_channels[0] + 1, num_features // 8, 3, 1, 1, bias=False),
                                              nn.ELU())
        
        self.reduc2x2   = Reduction_1x1(num_features // 8, num_features // 16, self.params.max_depth)
        self.lpg2x2     = Local_planar_guidance(2)
        
        self.Upconv1    = Upconv(num_features // 8, num_features // 16)
        self.reduc1x1   = Reduction_1x1(num_features // 16, num_features // 32, self.params.max_depth, is_final=True)
        self.conv1      = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.get_depth  = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                              nn.Sigmoid())

    def forward(self, features, focal):
        skip0, skip1, skip2, skip3 = features[0], features[1], features[2], features[3]
        dense_features = torch.nn.ReLU()(features[4])
        Upconv5 = self.Upconv5(dense_features) # H/16
        Upconv5 = self.bn5(Upconv5)
        concat5 = torch.cat([Upconv5, skip3], dim=1)
        iconv5 = self.conv5(concat5)
        
        Upconv4 = self.Upconv4(iconv5) # H/8
        Upconv4 = self.bn4(Upconv4)
        concat4 = torch.cat([Upconv4, skip2], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)
        
        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)
        
        reduc8x8 = self.reduc8x8(daspp_feat)
        plane_normal_8x8 = reduc8x8[:, :3, :, :]
        plane_normal_8x8 = torch_nn_func.normalize(plane_normal_8x8, 2, 1)
        plane_dist_8x8 = reduc8x8[:, 3, :, :]
        plane_eq_8x8 = torch.cat([plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)
        depth_8x8 = self.lpg8x8(plane_eq_8x8, focal)
        depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.params.max_depth
        depth_8x8_scaled_ds = torch_nn_func.interpolate(depth_8x8_scaled, scale_factor=0.25, mode='nearest')
        
        Upconv3 = self.Upconv3(daspp_feat) # H/4
        Upconv3 = self.bn3(Upconv3)
        concat3 = torch.cat([Upconv3, skip1, depth_8x8_scaled_ds], dim=1)
        iconv3 = self.conv3(concat3)
        
        reduc4x4 = self.reduc4x4(iconv3)
        plane_normal_4x4 = reduc4x4[:, :3, :, :]
        plane_normal_4x4 = torch_nn_func.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]
        plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        depth_4x4 = self.lpg4x4(plane_eq_4x4, focal)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.params.max_depth
        depth_4x4_scaled_ds = torch_nn_func.interpolate(depth_4x4_scaled, scale_factor=0.5, mode='nearest')
        
        Upconv2 = self.Upconv2(iconv3) # H/2
        Upconv2 = self.bn2(Upconv2)
        concat2 = torch.cat([Upconv2, skip0, depth_4x4_scaled_ds], dim=1)
        iconv2 = self.conv2(concat2)
        
        reduc2x2 = self.reduc2x2(iconv2)
        plane_normal_2x2 = reduc2x2[:, :3, :, :]
        plane_normal_2x2 = torch_nn_func.normalize(plane_normal_2x2, 2, 1)
        plane_dist_2x2 = reduc2x2[:, 3, :, :]
        plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        depth_2x2 = self.lpg2x2(plane_eq_2x2, focal)
        depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.params.max_depth
        
        Upconv1 = self.Upconv1(iconv2)
        reduc1x1 = self.reduc1x1(Upconv1)
        concat1 = torch.cat([Upconv1, reduc1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], dim=1)
        iconv1 = self.conv1(concat1)
        final_depth = self.params.max_depth * self.get_depth(iconv1)
                
        return depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth
    
class AcaModel(nn.Module):
    def __init__(self, params):
        super(AcaModel, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params, self.encoder.feat_out_channels, params.bts_size)
        for name, para in self.encoder.named_parameters():
          if para.requires_grad and 'denselayer6' or 'denselayer5' or 'denselayer3' or 'denselayer2' or 'denselayer4' or 'denseblock3' or 'denseblock4' or 'denseblock2' in name:
             para.requires_grad = False

    def forward(self, x, focal):
        # print("From Model file:::")
        # print("X is ::",x)
        # print("Focal is ::",focal)
        return self.decoder(self.encoder(x), focal)