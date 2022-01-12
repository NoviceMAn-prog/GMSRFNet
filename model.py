import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b', 'res2net50_v1b_26w_4s']

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b lib.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model

def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        #model_state = torch.load('/media/nercms/NERCMS/GepengJi/Medical_Seqmentation/CRANet/models/res2net50_v1b_26w_4s-3cf99910.pth')
        #model.load_state_dict(model_state)
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'],map_location=torch.device('cpu')))
        return model
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class downsample_2x(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, dilation=1):
        super(downsample_2x, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        #self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        #x = self.bn(x)
        return x
class downsample_4x(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, dilation=1):
        super(downsample_4x, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x
class downsample_8x(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, dilation=1):
        super(downsample_8x, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x
class upsample_2x(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, dilation=1):
        super(upsample_2x, self).__init__()
        self.conv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        #self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        #x = self.bn(x)
        return x
class upsample_4x(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, dilation=1):
        super(upsample_4x, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x
class upsample_8x(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, dilation=1):
        super(upsample_8x, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.ConvTranspose2d(out_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x
class form_attention_map(nn.Module):
    def __init__(self, in_planes,kernel_size=3, stride=1, padding=1, dilation=1):
        super(form_attention_map, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 32,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.conv2 = nn.Conv2d(32, 1,
                              kernel_size=1, stride=1,
                              padding=0, dilation=dilation, bias=False)
        self.sig = nn.Sigmoid()
        
        
    def forward(self,one,two,three):
        x = torch.cat([one,two,three],axis=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sig(x)
        x = torch.add(x,1)
        return x
    


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        p = self.avg_pool(x).view(b, c)
        y = self.fc(p).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class exchange(nn.Module):
    def __init__(self,scale1,scale2,scale3,scale4,k_1,k_2):
        super().__init__()
        self.k_1 = 32
        self.k_2 = 64
        self.layers1 = [scale1,scale1+self.k_1+3*k_1,scale1+2*self.k_1+3*k_1,scale1+3*self.k_1+3*k_1]
        self.layers2 = [scale2,scale2+self.k_1+3*k_1,scale2+2*self.k_1+3*k_1,scale2+3*self.k_1+3*k_1]
        self.layers3 = [scale3,scale3+self.k_2+3*k_1,scale3+2*self.k_2+3*k_1,scale3+3*self.k_2+3*k_1]
        self.layers4 = [scale4,scale4+self.k_2+3*k_1,scale4+2*self.k_2+3*k_1,scale4+3*self.k_2+3*k_1]
        
        
        self.x1 = BasicConv2d(self.layers1[0],self.k_1,kernel_size=3,stride=1,padding=1)
        self.y1 = BasicConv2d(self.layers2[0],self.k_1,kernel_size=3,stride=1,padding=1)
        self.v1 = BasicConv2d(self.layers3[0],self.k_2,kernel_size=3,stride=1,padding=1)
        self.w1 = BasicConv2d(self.layers4[0],self.k_2,kernel_size=3,stride=1,padding=1)
        
        self.x1_y1 = upsample_2x(self.k_1,self.k_1)
        self.x1_v1 = upsample_4x(self.k_2,self.k_1)
        self.x1_w1 = upsample_8x(self.k_2,self.k_1)
        self.att_x1 = form_attention_map(3*self.k_1)
        
        self.x2_input = BasicConv2d(self.layers1[1],32,kernel_size=1,stride=1,padding=0)
        
        self.y1_x1 = downsample_2x(self.k_1,self.k_1)
        self.y1_v1 = upsample_2x(self.k_2,self.k_1)
        self.y1_w1 = upsample_4x(self.k_2,self.k_1)
        self.att_y1 = form_attention_map(3*self.k_1)
        
        self.y2_input = BasicConv2d(self.layers2[1],64,kernel_size=1,stride=1,padding=0)
        
        self.v1_x1 = downsample_4x(self.k_1,self.k_1)
        self.v1_y1 = downsample_2x(self.k_1,self.k_1)
        self.v1_w1 = upsample_2x(self.k_2,self.k_1)
        self.att_v1 = form_attention_map(3*self.k_1)
        
        self.v2_input = BasicConv2d(self.layers3[1],128,kernel_size=1,stride=1,padding=0)
        
        self.w1_x1 = downsample_8x(self.k_1,self.k_1)
        self.w1_y1 = downsample_4x(self.k_1,self.k_1)
        self.w1_v1 = downsample_2x(self.k_2,self.k_1)
        self.att_w1 = form_attention_map(3*self.k_1)
        
        self.w2_input = BasicConv2d(self.layers4[1],256,kernel_size=1,stride=1,padding=0)
        
        self.x2 = BasicConv2d(32,self.k_1,kernel_size=3,stride=1,padding=1)
        self.x2_se = SELayer(32)
        
        self.y2 = BasicConv2d(64,self.k_1,kernel_size=3,stride=1,padding=1)
        self.y2_se = SELayer(64)
        
        self.v2 = BasicConv2d(128,self.k_2,kernel_size=3,stride=1,padding=1)
        self.v2_se = SELayer(128)
        
        self.w2 = BasicConv2d(256,self.k_2,kernel_size=3,stride=1,padding=1)
        self.w2_se = SELayer(256)
        ######### 3 calculation
        
        self.x2_y2 = upsample_2x(self.k_1,self.k_1)
        self.x2_v2 = upsample_4x(self.k_2,self.k_1)
        self.x2_w2 = upsample_8x(self.k_2,self.k_1)
        self.att_x2 = form_attention_map(3*self.k_1)
        
        self.x3_input = BasicConv2d(self.layers1[2],32,kernel_size=1,stride=1,padding=0)
        
        self.y2_x2 = downsample_2x(self.k_1,self.k_1)
        self.y2_v2 = upsample_2x(self.k_2,self.k_1)
        self.y2_w2 = upsample_4x(self.k_2,self.k_1)
        self.att_y2 = form_attention_map(3*self.k_1)
        
        self.y3_input = BasicConv2d(self.layers2[2],64,kernel_size=1,stride=1,padding=0)
        
        self.v2_x2 = downsample_4x(self.k_1,self.k_1)
        self.v2_y2 = downsample_2x(self.k_1,self.k_1)
        self.v2_w2 = upsample_2x(self.k_2,self.k_1)
        self.att_v2 = form_attention_map(3*self.k_1)
        
        self.v3_input = BasicConv2d(self.layers3[2],128,kernel_size=1,stride=1,padding=0)
        
        self.w2_x2 = downsample_8x(self.k_1,self.k_1)
        self.w2_y2 = downsample_4x(self.k_1,self.k_1)
        self.w2_v2 = downsample_2x(self.k_2,self.k_1)
        self.att_w2 = form_attention_map(3*self.k_1)
        
        self.w3_input = BasicConv2d(self.layers4[2],256,kernel_size=1,stride=1,padding=0)
        
        self.x3 = BasicConv2d(32,self.k_1,kernel_size=3,stride=1,padding=1)
        self.x3_se = SELayer(32)
        
        self.y3 = BasicConv2d(64,self.k_1,kernel_size=3,stride=1,padding=1)
        self.y3_se = SELayer(64)
        
        self.v3 = BasicConv2d(128,self.k_2,kernel_size=3,stride=1,padding=1)
        self.v3_se = SELayer(128)
        
        self.w3 = BasicConv2d(256,self.k_2,kernel_size=3,stride=1,padding=1)
        self.w3_se = SELayer(256)
        
        ################# 4 calculations
        
        self.x3_y3 = upsample_2x(self.k_1,self.k_1)
        self.x3_v3 = upsample_4x(self.k_2,self.k_1)
        self.x3_w3 = upsample_8x(self.k_2,self.k_1)
        self.att_x3 = form_attention_map(3*self.k_1)
        
        self.x4_input = BasicConv2d(self.layers1[3],self.layers1[3],kernel_size=1,stride=1,padding=0)
        
        self.y3_x3 = downsample_2x(self.k_1,self.k_1)
        self.y3_v3 = upsample_2x(self.k_2,self.k_1)
        self.y3_w3 = upsample_4x(self.k_2,self.k_1)
        self.att_y3 = form_attention_map(3*self.k_1)
        
        self.y4_input = BasicConv2d(self.layers2[3],self.layers2[3],kernel_size=1,stride=1,padding=0)
        
        self.v3_x3 = downsample_4x(self.k_1,self.k_1)
        self.v3_y3 = downsample_2x(self.k_1,self.k_1)
        self.v3_w3 = upsample_2x(self.k_2,self.k_1)
        self.att_v3 = form_attention_map(3*self.k_1)
        
        self.v4_input = BasicConv2d(self.layers3[3],self.layers3[3],kernel_size=1,stride=1,padding=0)
        
        self.w3_x3 = downsample_8x(self.k_1,self.k_1)
        self.w3_y3 = downsample_4x(self.k_1,self.k_1)
        self.w3_v3 = downsample_2x(self.k_2,self.k_1)
        self.att_w3 = form_attention_map(3*self.k_1)
        
        self.w4_input = BasicConv2d(self.layers4[3],self.layers4[3],kernel_size=1,stride=1,padding=0)
        
        self.x4 = BasicConv2d(self.layers1[3],scale1,kernel_size=3,stride=1,padding=1)
        self.x4_se = SELayer(self.layers1[3])
        
        self.y4 = BasicConv2d(self.layers2[3],scale2,kernel_size=3,stride=1,padding=1)
        self.y4_se = SELayer(self.layers2[3])
        
        self.v4 = BasicConv2d(self.layers3[3],scale3,kernel_size=3,stride=1,padding=1)
        self.v4_se = SELayer(self.layers3[3])
        
        self.w4 = BasicConv2d(self.layers4[3],scale4,kernel_size=3,stride=1,padding=1)
        self.w4_se = SELayer(self.layers4[3])
        
        

    def forward(self,x,y,v,w):
        x1 = self.x1(x)
        y1 = self.y1(y)
        v1 = self.v1(v)
        w1 = self.w1(w)
        
        x1_y1 = self.x1_y1(y1)
        x1_v1 = self.x1_v1(v1)
        x1_w1 = self.x1_w1(w1)
        att_x1 = self.att_x1(x1_y1,x1_v1,x1_w1)
        
        x2_input = torch.cat([x,x1,x1_y1,x1_v1,x1_w1],axis=1)
        x2_input = self.x2_input(x2_input)
        x2_input = x2_input*att_x1
        
        y1_x1 = self.y1_x1(x1)
        y1_v1 = self.y1_v1(v1)
        y1_w1 = self.y1_w1(w1)
        att_y1 = self.att_y1(y1_x1,y1_v1,y1_w1)
        
        y2_input = torch.cat([y,y1,y1_x1,y1_v1,y1_w1],axis=1)
        y2_input = self.y2_input(y2_input)
        y2_input = y2_input*att_y1
        
        v1_x1 = self.v1_x1(x1)
        v1_y1 = self.v1_y1(y1)
        v1_w1 = self.v1_w1(w1)
        att_v1 = self.att_v1(v1_x1,v1_y1,v1_w1)
        
        v2_input = torch.cat([v,v1,v1_x1,v1_y1,v1_w1],axis=1)
        v2_input = self.v2_input(v2_input)
        v2_input = v2_input*att_v1
        
        w1_x1 = self.w1_x1(x1)
        w1_y1 = self.w1_y1(y1)
        w1_v1 = self.w1_v1(v1)
        att_w1 = self.att_w1(w1_x1,w1_y1,w1_v1)
    
        w2_input = torch.cat([w,w1,w1_x1,w1_y1,w1_v1],axis=1)
        w2_input = self.w2_input(w2_input)
        w2_input = w2_input*att_w1
        
        x2 = self.x2_se(x2_input)
        x2 = self.x2(x2)
        
        y2 = self.y2_se(y2_input)
        y2 = self.y2(y2)
        
        v2 = self.v2_se(v2_input)
        v2 = self.v2(v2)
        
        w2 = self.w2_se(w2_input)
        w2 = self.w2(w2)
        
        
        #####################
        x2_y2 = self.x2_y2(y2)
        x2_v2 = self.x2_v2(v2)
        x2_w2 = self.x2_w2(w2)
        att_x2 = self.att_x2(x2_y2,x2_v2,x2_w2)
        
        x3_input = torch.cat([x,x1,x2,x2_y2,x2_v2,x2_w2],axis=1)
        x3_input = self.x3_input(x3_input)
        x3_input = x3_input*att_x2
        
        y2_x2 = self.y2_x2(x2)
        y2_v2 = self.y2_v2(v2)
        y2_w2 = self.y2_w2(w2)
        att_y2 = self.att_y2(y2_x2,y2_v2,y2_w2)
        
        y3_input = torch.cat([y,y1,y2,y2_x2,y2_v2,y2_w2],axis=1)
        y3_input = self.y3_input(y3_input)
        y3_input = y3_input*att_y2
        
        v2_x2 = self.v2_x2(x2)
        v2_y2 = self.v2_y2(y2)
        v2_w2 = self.v2_w2(w2)
        att_v2 = self.att_v2(v2_x2,v2_y2,v2_w2)
        
        v3_input = torch.cat([v,v1,v2,v2_x2,v2_y2,v2_w2],axis=1)
        v3_input = self.v3_input(v3_input)
        v3_input = v3_input*att_v2
        
        w2_x2 = self.w2_x2(x2)
        w2_y2 = self.w2_y2(y2)
        w2_v2 = self.w2_v2(v2)
        att_w2 = self.att_w2(w2_x2,w2_y2,w2_v2)
    
        w3_input = torch.cat([w,w1,w2,w2_x2,w2_y2,w2_v2],axis=1)
        w3_input = self.w3_input(w3_input)
        w3_input = w3_input*att_w2        
        
        x3 = self.x3_se(x3_input)
        x3 = self.x3(x3)
        
        y3 = self.y3_se(y3_input)
        y3 = self.y3(y3)
        
        v3 = self.v3_se(v3_input)
        v3 = self.v3(v3)
        
        w3 = self.w3_se(w3_input)
        w3 = self.w3(w3)
        
        ################ 4 calculations
        x3_y3 = self.x3_y3(y3)
        x3_v3 = self.x3_v3(v3)
        x3_w3 = self.x3_w3(w3)
        att_x3 = self.att_x3(x3_y3,x3_v3,x3_w3)
        
        x4_input = torch.cat([x,x1,x2,x3,x3_y3,x3_v3,x3_w3],axis=1)
        x4_input = self.x4_input(x4_input)
        x4_input = x4_input*att_x3
        
        y3_x3 = self.y3_x3(x3)
        y3_v3 = self.y3_v3(v3)
        y3_w3 = self.y3_w3(w3)
        att_y3 = self.att_y3(y3_x3,y3_v3,y3_w3)
        
        y4_input = torch.cat([y,y1,y2,y3,y3_x3,y3_v3,y3_w3],axis=1)
        y4_input = self.y4_input(y4_input)
        y4_input = y4_input*att_y3
        
        v3_x3 = self.v3_x3(x3)
        v3_y3 = self.v3_y3(y3)
        v3_w3 = self.v3_w3(w3)
        att_v3 = self.att_v3(v3_x3,v3_y3,v3_w3)
        
        v4_input = torch.cat([v,v1,v2,v3,v3_x3,v3_y3,v3_w3],axis=1)
        v4_input = self.v4_input(v4_input)
        v4_input = v4_input*att_v3
        
        w3_x3 = self.w3_x3(x3)
        w3_y3 = self.w3_y3(y3)
        w3_v3 = self.w3_v3(v3)
        att_w3 = self.att_w3(w3_x3,w3_y3,w3_v3)
    
        w4_input = torch.cat([w,w1,w2,w3,w3_x3,w3_y3,w3_v3],axis=1)
        w4_input = self.w4_input(w4_input)
        w4_input = w4_input*att_w3
        
        x4 = self.x4_se(x4_input)
        x4 = self.x4(x4)
        y4 = self.y4_se(y4_input)
        y4 = self.y4(y4)
        v4 = self.v4_se(v4_input)
        v4 = self.v4(v4)
        w4 = self.w4_se(w4_input)
        w4 = self.w4(w4)
        
        
        
        
        return x+x4,y+y4,v+v4,w+w4

class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()

        self.en1 = BasicConv2d(3,32,kernel_size=3,stride=1,padding=1)
        self.se1 = SELayer(32)
        
        self.en2 = BasicConv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.se2 = SELayer(64)
        
        self.en3 = BasicConv2d(64, 128,kernel_size=3,stride=1,padding=1)
        self.se3 = SELayer(128)
        
        self.en4 = BasicConv2d(128,256,kernel_size=3,stride=1,padding=1)
        self.se4 = SELayer(256)
        
        self.pooling = nn.MaxPool2d((2, 2))
        
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        channel = 32
        self.conv1 = BasicConv2d(3,64, kernel_size=(3, 3), padding=(1, 1))
        self.conv_red = BasicConv2d(256,32, kernel_size=(1, 1), padding=(0, 0))
        self.rfb2_1 = RFB_modified(512,64)
        self.rfb3_1 = RFB_modified(1024,128)
        self.rfb4_1 = RFB_modified(2048,256)

        self.msrfv21 = exchange(32,64,128,256,32,64)
        self.msrfv22 = exchange(32,64,128,256,32,64)
        self.msrfv23 = exchange(32,64,128,256,32,64)
        
        self.u4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.level3_selection = nn.Conv2d(256, 128,kernel_size=3, stride=1,padding=1, bias=False)
        self.u3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.level2_selection = nn.Conv2d(128,64,kernel_size=3, stride=1,padding=1, bias=False)
        self.u1 = nn.ConvTranspose2d(64,32, kernel_size=4, stride=2, padding=1)
        
        self.final_1 = BasicConv2d(64,32,kernel_size=3,stride=1,padding=1)
        self.final_2 = BasicConv2d(32,1,kernel_size=3,stride=1,padding=1)
        self.sig = nn.Sigmoid()
        
        self.dsv8 = nn.Conv2d(256,1,kernel_size=1,stride=1,padding=0)
        self.dsv4 = nn.Conv2d(128,1,kernel_size=1,stride=1,padding=0)
        self.dsv2 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)
        self.upsamp8 = torch.nn.Upsample(scale_factor=8)
        self.upsamp4 = torch.nn.Upsample(scale_factor=4)
        self.upsamp2 = torch.nn.Upsample(scale_factor=2)
        

    def forward(self, x):
        x = self.conv1(x)
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)
        x1 = self.conv_red(x1)
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)
        n11,n21,n31,n41 = x1,x2_rfb,x3_rfb,x4_rfb

        n13,n23,n33,n43 = self.msrfv21(n11,n21,n31,n41)
        #n13,n23,n33,n43 = self.msrfv22(n12,n22,n32,n42)
        n14,n24,n34,n44 = self.msrfv23(n13,n23,n33,n43)
        
        n14,n24,n34,n44 = n14+n11,n24+n21,n34+n31,n44+n41
        
        
        dsv_up8 = self.dsv8(n44)
        dsv_up8 = self.upsamp8(dsv_up8)
        u3 = self.u4(n44)
        level3 = torch.cat([n34,u3],axis=1)
        level3 = self.level3_selection(level3)
        dsv_up4 = self.dsv4(level3)
        dsv_up4 = self.upsamp4(dsv_up4)
        u2 = self.u3(level3)
        level2 = torch.cat([n24,u2],axis=1)
        level2 = self.level2_selection(level2)
        dsv_up2 = self.dsv2(level2)
        dsv_up2 = self.upsamp2(dsv_up2)
        
        level1 = self.u1(level2)
        level1 = torch.cat([n14,level1],axis=1)
        level1 = self.final_1(level1)
        level1 = self.final_2(level1)
        seg_map = self.sig(level1)
        
        return seg_map,dsv_up2,dsv_up4,dsv_up8

model =EncoderBlock()
img = torch.randn(1, 3, 256,256)
preds = model(img) # (1, 1000)
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
print(get_n_params(model))
print(preds[0].shape,preds[1].shape,preds[2].shape,preds[3].shape)
