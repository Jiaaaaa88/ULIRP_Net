import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from egasr import EGASR


class SAOM(nn.Module):
    def __init__(self, oup_channels, group_num=16, gate_treshold=0.5):
        super().__init__()
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num)
        self.gate_treshold = nn.Parameter(torch.tensor(gate_treshold))
        self.sigmoid = nn.Sigmoid()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(oup_channels, oup_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(oup_channels // 4, oup_channels, 1),
            nn.Sigmoid()
        )

        self.reconstruct_weights = nn.Parameter(torch.ones(2, 2))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        gn_x = self.gn(x)

        channel_weights = self.channel_attention(x)
        w_gamma = (self.gn.weight / sum(self.gn.weight)).view(1, -1, 1, 1)
        reweights = self.sigmoid(gn_x * w_gamma * channel_weights)

        w1 = torch.where(reweights > self.gate_treshold,
                         torch.ones_like(reweights),
                         reweights)
        w2 = torch.where(reweights > self.gate_treshold,
                         torch.zeros_like(reweights),
                         reweights)

        x_1 = w1 * x
        x_2 = w2 * x
        return self.adaptive_reconstruct(x_1, x_2)

    def adaptive_reconstruct(self, x_1, x_2):
        weights = self.softmax(self.reconstruct_weights)
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)

        out1 = weights[0][0] * x_11 + weights[0][1] * x_22
        out2 = weights[1][0] * x_12 + weights[1][1] * x_21
        return torch.cat([out1, out2], dim=1)


class CAOM(nn.Module):
    def __init__(self, op_channel, out_channel=None, alpha=1 / 2, squeeze_radio=2, group_size=2):
        super().__init__()
        self.op_channel = op_channel
        self.out_channel = out_channel if out_channel is not None else op_channel
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel

        self.squeeze1 = nn.Sequential(
            nn.Conv2d(up_channel, up_channel // squeeze_radio, 1, bias=False),
            nn.BatchNorm2d(up_channel // squeeze_radio),
            nn.ReLU()
        )

        self.squeeze2 = nn.Sequential(
            nn.Conv2d(low_channel, low_channel // squeeze_radio, 1, bias=False),
            nn.BatchNorm2d(low_channel // squeeze_radio),
            nn.ReLU()
        )

        self.GWC_3x3 = nn.Conv2d(up_channel // squeeze_radio,
                                 self.out_channel // 2,
                                 3,
                                 padding=1,
                                 groups=group_size)
        self.GWC_5x5 = nn.Conv2d(up_channel // squeeze_radio,
                                 self.out_channel // 2,
                                 5,
                                 padding=2,
                                 groups=group_size)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channel, self.out_channel // 8, 1),
            nn.ReLU(),
            nn.Conv2d(self.out_channel // 8, self.out_channel, 1),
            nn.Sigmoid()
        )

        self.channel_adj = nn.Conv2d(self.out_channel + low_channel // squeeze_radio, self.out_channel, 1)

        # 残差连接
        self.shortcut = nn.Conv2d(op_channel, self.out_channel, 1)
        self.norm = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)

        # 分支处理
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 多尺度特征提取
        Y1_3x3 = self.GWC_3x3(up)
        Y1_5x5 = self.GWC_5x5(up)
        Y1 = torch.cat([Y1_3x3, Y1_5x5], dim=1)

        # 特征融合
        out = torch.cat([Y1, low], dim=1)

        # 调整通道数
        out = self.channel_adj(out)

        # 注意力引导
        attention_weights = self.attention(out)
        out = out * attention_weights

        # 残差连接
        out = self.relu(self.norm(out + identity))
        return out


class CSRD(nn.Module):
    def __init__(self, op_channel, out_channel=None, group_num=4, gate_treshold=0.5, alpha=1/2, squeeze_radio=2, group_size=2):
        super().__init__()
        self.out_channel = out_channel if out_channel is not None else op_channel
        self.SAOM = SAOM(op_channel, group_num, gate_treshold)
        self.CAOM = CAOM(op_channel, self.out_channel, alpha, squeeze_radio, group_size)

        self.feature_adaptation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channel, self.out_channel // 8, 1),
            nn.ReLU(),
            nn.Conv2d(self.out_channel // 8, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.feature_adaptation(self.CAOM(x))
        SAOM_out = self.SAOM(x)
        CAOM_out = self.CAOM(x)
        out = weights[:, 0:1] * SAOM_out + weights[:, 1:2] * CAOM_out
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Swin Transformer
        self.swin_transformer_input = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            img_size=384,
            features_only=True,
            drop_path_rate=0.2
        )


        self.conv_layer0 = nn.Conv2d(in_channels=1536, out_channels=1024, kernel_size=1)
        self.add_module("conv_layer0", self.conv_layer0)
        self.conv_layer1 = nn.Conv2d(in_channels=1792, out_channels=1024, kernel_size=1)
        self.add_module("conv_layer1", self.conv_layer1)
        self.conv_layer2 = nn.Conv2d(in_channels=1408, out_channels=768, kernel_size=1)
        self.add_module("conv_layer2", self.conv_layer2)
        self.conv_layer3 = nn.Conv2d(in_channels=960, out_channels=3, kernel_size=1)
        self.add_module("conv_layer3", self.conv_layer3)

        self.CSRD_layer1 = CSRD(op_channel=1024, out_channel=1024)
        self.CSRD_layer2 = CSRD(op_channel=1024, out_channel=1024)
        self.CSRD_layer3 = CSRD(op_channel=768, out_channel=768)

        self.EGASR = EGASR()


        checkpoint_path = r'C:/Users/Lee/.cache/huggingface/hub/models--timm--swin_base_patch4_window7_224.ms_in22k_ft_in1k/pytorch_model.bin'
        self.load_pretrained_weights(self.swin_transformer_input, checkpoint_path)

    def load_pretrained_weights(self, model, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = model.state_dict()

        # 过滤掉不匹配的键
        filtered_state_dict = {k: v for k, v in state_dict.items() if
                               k in model_state_dict and v.size() == model_state_dict[k].size()}

        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)

        #self.add_module("sc_conv_layer1", self.sc_conv_layer1)
    def forward(self, input1, input2):
        # Process input1 through Swin Transformer and store each layer's output
        '''''''''''''''''''''''''''encoder'''''''''''''''''''''''''''
        # Process input1 through Swin Transformer and store each layer's output
        #print("in1&2",input1.shape,input2.shape)
        layer_outputs_input1 = self.swin_transformer_input(input1)

        en_rgb_feature1 = layer_outputs_input1[0]
        en_rgb_feature1 = en_rgb_feature1.permute(0, 3, 1, 2)
        en_rgb_feature2 = layer_outputs_input1[1]
        en_rgb_feature2 = en_rgb_feature2.permute(0, 3, 1, 2)
        en_rgb_feature3 = layer_outputs_input1[2]
        en_rgb_feature3 = en_rgb_feature3.permute(0, 3, 1, 2)
        en_rgb_feature4 = layer_outputs_input1[3]
        en_rgb_feature4 = en_rgb_feature4.permute(0, 3, 1, 2)

        layer_outputs_input2 = self.swin_transformer_input(input2)
        en_dolp_feature1 = layer_outputs_input2[0]
        en_dolp_feature1 = en_dolp_feature1.permute(0, 3, 1, 2)
        en_dolp_feature2 = layer_outputs_input2[1]
        en_dolp_feature2 = en_dolp_feature2.permute(0, 3, 1, 2)
        en_dolp_feature3 = layer_outputs_input2[2]
        en_dolp_feature3 = en_dolp_feature3.permute(0, 3, 1, 2)
        en_dolp_feature4 = layer_outputs_input2[3]
        en_dolp_feature4 = en_dolp_feature4.permute(0, 3, 1, 2)
        rgb_dolp_feature1 = torch.cat((en_rgb_feature4, en_dolp_feature4), dim=1)
        # print('rgb_dolp_feature1', rgb_dolp_feature1.shape)

        '''''''''''''''''''''''''''decoder'''''''''''''''''''''''''''
        rgb_dolp_feature1 = F.interpolate(rgb_dolp_feature1, scale_factor=2, mode='bilinear')
        de_feature0 = self.conv_layer0(rgb_dolp_feature1)
        de_feature0 = self.CSRD_layer1(de_feature0)
        de_orignfeature0 = torch.cat((en_rgb_feature3, en_dolp_feature3), dim=1)
        de_feature00 = torch.cat((de_feature0, de_orignfeature0), dim=1)
        de_feature00 = F.interpolate(de_feature00, scale_factor=2, mode='bilinear')
        de_feature1 = self.conv_layer1(de_feature00)
        de_feature1 = self.CSRD_layer2(de_feature1)
        # print('sc2', de_feature1.shape)
        de_orignfeature1 = torch.cat((en_rgb_feature2, en_dolp_feature2), dim=1)
        de_feature11 = torch.cat((de_feature1, de_orignfeature1), dim=1)
        de_feature11 = F.interpolate(de_feature11, scale_factor=2, mode='bilinear')
        de_feature2 = self.conv_layer2(de_feature11)
        de_feature2 = self.CSRD_layer3(de_feature2)
        de_orignfeature2 = torch.cat((en_rgb_feature1, en_dolp_feature1), dim=1)
        de_feature22 = torch.cat((de_feature2, de_orignfeature2), dim=1)

        output = self.conv_layer3(de_feature22)
        output = self.EGASR(output)
        output = output.clamp(0, 1)
        return output


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input1 = torch.randn(3, 3, 384, 384)
    input2 = torch.randn(3, 3, 384, 384)
    print("input11", input1.shape)
    model = Model()
    K = model(input1, input2)
    print("output", K.shape)
