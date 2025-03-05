import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class Final_Model(nn.Module):

    def __init__(self, backbone_net, semantic_head):
        super(Final_Model, self).__init__()
        self.backend = backbone_net
        self.semantic_head = semantic_head

    def forward(self, x):
        middle_feature_maps = self.backend(x)

        semantic_output = self.semantic_head(middle_feature_maps)

        return semantic_output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None):
        super(BasicBlock, self).__init__()
        self.if_BN = if_BN
        if self.if_BN:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.if_BN:
            self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = conv3x3(planes, planes)
        if self.if_BN:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class DilationConvBlock(nn.Module):
    
    def __init__(self, dilation=None, downsample=False):
        super(DilationConvBlock, self).__init__()
        padding=dilation
            
        if downsample:
            self.conv1 = nn.Conv2d(128, 128, 3, stride=(1, 2), padding=padding, dilation=dilation)
            self.ds_block = nn.Conv2d(128, 128, 1, stride=(1, 2))
        else:
            self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=padding, dilation=dilation)
            self.ds_block = None
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.Hardswish()
        
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.ds_block is not None:
            identity = self.ds_block(identity)

        out += identity
        out = self.relu(out)
        return out


class HybridDilatedConvModule(nn.Module):
    
    def __init__(self, dilations=[(1, 1), (1, 2), (1, 3), (1, 5)], in_dim=128):
        super(HybridDilatedConvModule, self).__init__()
        
        self.dilation_convs = nn.ModuleList()
        for d in dilations:
            self.dilation_convs.append(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1,
                                    padding=d,
                                    dilation=d))
        n_dilations = len(dilations) 
        self.fuse_conv = nn.Conv2d(n_dilations*in_dim, in_dim, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        feats = []
        for d_conv in self.dilation_convs:
            feats.append(d_conv(x))
        
        # (b, ch, h, w)
        fuse_feats = torch.cat(feats, dim=1)
        out = self.fuse_conv(fuse_feats)
        out = out + x
        return out
    

class SemanticHead(nn.Module):

    def __init__(self,num_class=14,input_channel=1024):
        super(SemanticHead,self).__init__()
  
        self.conv_1=nn.Conv2d(input_channel, 512, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu_1 = nn.LeakyReLU()

        self.conv_2=nn.Conv2d(512, 128, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu_2 = nn.LeakyReLU()

        self.semantic_output=nn.Conv2d(128, num_class, 1)

    def forward(self, input_tensor):
        res=self.conv_1(input_tensor)
        res=self.bn1(res)
        res=self.relu_1(res)
        
        res=self.conv_2(res)
        res=self.bn2(res)
        res=self.relu_2(res)
        
        res=self.semantic_output(res)
        return res
    

class ResNet_34(nn.Module):

    def __init__(self, nclasses, aux, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True, zero_init_residual=False,
                 norm_layer=None, groups=1, width_per_group=64):
        super(ResNet_34, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.aux = aux

        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group


        self.conv1 = BasicConv2d(5, 64, kernel_size=1)
        self.conv2 = BasicConv2d(64, 128, kernel_size=1)
        self.conv3 = BasicConv2d(128, 256, kernel_size=1)
        self.conv4 = BasicConv2d(256, 512, kernel_size=1)
        self.inplanes = 512

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.d_layer3 = self._make_variable_scale_downsample_layer([1, 2, 3, 1, 2, 3], blocks=layers[2])
        self.d_layer4 = self._make_variable_scale_downsample_layer([1, 2, 3], blocks=layers[3])

        self.d_fuse1 = HybridDilatedConvModule()
        self.d_fuse2 = HybridDilatedConvModule()
        self.d_fuse3 = HybridDilatedConvModule()
        self.d_fuse4 = HybridDilatedConvModule()

        self.head = SemanticHead(num_class=14)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))
        return nn.Sequential(*layers)
    
    def _make_variable_scale_downsample_layer(self, dilations:list, blocks):
        assert len(dilations) == blocks
        assert dilations[0] == 1
        layers = []

        layers.append(DilationConvBlock(dilation=1, downsample=True))
        print(dilations[1::])
        for dilation in dilations[1::]:
            layers.append(DilationConvBlock(dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x_1 = self.layer1(x)  # (batch, 128, 64, 2048)
        x_1 = self.d_fuse1(x_1)
        x_2 = self.layer2(x_1)  # 1/2 (batch, 128, 32, 1024)
        x_2 = self.d_fuse2(x_2)
        x_3 = self.d_layer3(x_2)
        # x_3 = self.layer3(x_2)  # 1/4 (batch, 128, 16, 512)
        x_3 = self.d_fuse3(x_3)
        x_4 = self.d_layer4(x_3)
        # x_4 = self.layer4(x_3)  # 1/8 (batch, 128, 8, 256)
        x_4 = self.d_fuse4(x_4)

        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        res = [x, x_1, res_2, res_3, res_4]
        out = torch.cat(res, dim=1)
        out = self.head(out)
        out = F.softmax(out, dim=1)

        return out

if __name__ == "__main__":

    import time
    model = ResNet_34(20).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    time_train = []
    for i in range(20):
        inputs = torch.randn(1, 5, 64, 2048).cuda()
        model.eval()
        with torch.no_grad():
          start_time = time.time()
          outputs = model(inputs)
        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        fwt = time.time() - start_time
        time_train.append(fwt)
        print ("Forward time per img: %.3f (Mean: %.3f)" % (
          fwt / 1, sum(time_train) / len(time_train) / 1))
        time.sleep(0.15)

