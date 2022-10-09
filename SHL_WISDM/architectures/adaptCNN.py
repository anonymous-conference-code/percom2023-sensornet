import torch.utils.data
import torchaudio
import torchvision
from torchvision.transforms import InterpolationMode
import torch
import torch.nn as nn

"""
SENSORNET:
(1) Spectrogram Conversion
(2) Adaptive Convolution
(3) Head-importance Learning

"""

if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")


"""
position encoding

Pan, X., Ge, C., Lu, R., Song, S., Chen, G., Huang, Z. and Huang, G., 2022. 
On the integration of self-attention and convolution. 
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 815-825).
"""


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


"""
(2) Adaptive Convolution
*  dense layer
** attention layer
"""


class AdaptCNN(nn.Module):
    def __init__(self, in_planes, out_planes, dim, kernel_att=2, kernel_conv=3, dilation=1):
        super(AdaptCNN, self).__init__()
        self.out_planes = out_planes
        self.h = int(out_planes**0.5)
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = kernel_att
        self.dilation = dilation
        self.dim = dim

        """
        * dense layer, as shown in Fig.2
        """

        """
        (a)
        """
        if in_planes == 1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False)
            )
        else:
            """
            (b)
            """
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
            self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

        self.conv_p = nn.Conv2d(2, 1, kernel_size=1, bias=False)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.kernel_att)
        self.fold = torch.nn.Fold(output_size=(dim, dim), kernel_size=(self.kernel_att, self.kernel_att), stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=2)

        self.ap = nn.AdaptiveAvgPool3d((1, dim, dim))

    def forward(self, x):
        res = self.ap(x)
        b, c, m, h, w = x.size()
        x = x.view(b*c, m, h, w)

        """
        Q, K, V in Fig.1
        """
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        q = q.view(b, c, self.out_planes, h, w)
        k = k.view(b, c, self.out_planes, h, w)
        v = v.view(b, c, self.out_planes, h, w)

        scaling = float(h) ** -0.5
        q = q.permute(0, 2, 1, 3, 4).contiguous()
        b_, m_, c_, h_, w_ = q.size()

        pe = self.conv_p(position(h_, w_, x.is_cuda))
        k = k + pe

        q_att = q.view(b_ * m_, c_, h_, w_) * scaling
        k_att = k.view(b_ * m_, c_, h_, w_)
        v_att = v.view(b_ * m_, c_, h_, w_)

        """
        ** attention layer, as shown in Fig.2
        """

        """
        T(.) in Eq.(3) 
        """

        "Q^p"
        unfold_q = self.unfold(q_att).view(b * m_, c_, self.kernel_att * self.kernel_att, -1)

        "K^p"
        unfold_k = self.unfold(k_att).view(b * m_, c_, self.kernel_att * self.kernel_att, -1)

        """
        Q*K in Eq.4
        """
        unfold_q = unfold_q.unsqueeze(2)
        unfold_q = torch.repeat_interleave(unfold_q, repeats=c_, dim=2)
        unfold_rpe = self.unfold(pe).view(1, 1, self.kernel_att * self.kernel_att, -1)
        unfold_k = unfold_k + unfold_rpe
        unfold_k = unfold_k.unsqueeze(1)
        unfold_k = unfold_k.repeat(1, c_, 1, 1, 1)
        att = (unfold_q * unfold_k).sum(3)
        att = self.softmax(att)

        """
        A(q, k) in Eq.5
        """
        att = att.unsqueeze(3)
        att = att.repeat(1, 1, 1, self.kernel_att * self.kernel_att, 1)

        "V^p"
        out_att = self.unfold(v_att).view(b * m_, c_, self.kernel_att * self.kernel_att, -1)
        out_att = out_att.unsqueeze(1)
        out_att = (att * out_att).sum(2)

        """
        T(.)^-1 in Eq.(6) 
        """
        out_att = out_att.view(b * m_, c_ * self.kernel_att * self.kernel_att, -1)
        out_att = self.fold(out_att)
        out_att = out_att.view(b_, m_, c_, h_, w_)
        out_att = out_att.permute(0, 2, 1, 3, 4).contiguous()
        out_att += res
        return out_att


class SensorNet(nn.Module):

    def __init__(self, duration_window, duration_overlap, fs):

        super(SensorNet, self).__init__()

        """
        (1) Spectrogram Conversion
        size: 48 * 48 
        """
        img = 48

        self.nperseg = int(duration_window * fs)
        self.noverlap = int(duration_overlap * fs)

        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=self.nperseg, hop_length=self.nperseg - self.noverlap, center=True)

        self.resize = torchvision.transforms.Resize((img, img), interpolation=InterpolationMode.BILINEAR)


        """
        The setting of attention layer in TABLE II
        """

        self.attncnn1 = AdaptCNN(1, 64, img)
        self.attncnn2 = AdaptCNN(64, 128, img//2)
        self.attncnn3 = AdaptCNN(128, 256, img//4)

        self.ap1 = nn.AdaptiveAvgPool3d((64, img // 2, img // 2))
        self.ap2 = nn.AdaptiveAvgPool3d((128, img // 4, img // 4))
        self.ap3 = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.norm1 = nn.LayerNorm([64, img, img])
        self.norm2 = nn.LayerNorm([128, img//2, img//2])
        self.norm3 = nn.LayerNorm([256, img//4, img//4])

        self.FC0 = nn.Linear(256, 512)
        """
            SHL 2018: 8 classes
            WISDM: 18 classes

            self.FC1 = nn.Linear(512, x)
        """
        self.FC1 = nn.Linear(512, 18)

        self.Attention = nn.Linear(256, 256)

        self.relu_FC0 = nn.ReLU()
        self.dropout0 = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x):
        """
        Spectrogram Conversion in Fig.1
        """
        x = self.spectrogram(x).log2()
        x = self.resize(x)
        x = x.unsqueeze(2)

        x = self.ap1(self.relu_FC0(self.norm1(self.attncnn1(x))))
        x = self.ap2(self.relu_FC0(self.norm2(self.attncnn2(x))))

        """
        last layer: global feature extraction in Fig.2 *
        """
        x = self.relu_FC0(self.norm3(self.attncnn3(x)))
        x = x.permute(0, 2, 1, 3, 4)
        x = self.ap3(x)

        """
        (3) Head-importance Learning, as shown in Fig. 1       
        """
        x = x.view(x.shape[0], -1)
        attention_weight = self.Attention(x)
        attention_weight = torch.sigmoid(attention_weight)
        x = x * attention_weight

        if self.training:
            x = self.dropout0(x)
        x = x .view(x .size()[0], -1)
        x = self.relu_FC0(self.FC0(x))
        if self.training:
            x = self.dropout1(x)
        x = self.FC1(x)

        return x
