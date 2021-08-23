import torch
import torch.nn as nn
import os
import numpy as np
from torch.nn import functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



class MIPNet(nn.Module):
    def __init__(self):
        super(MIPNet, self).__init__()
        # Main Encoder Part

        self.en = Encoder()
        self.de = Decoder()

        self.nonlocal_list = nn.ModuleList([NonLocalBlockND(64,2**i) for i in range(6)])

    def forward(self, mixture):
        """
        :param mixture: [B, T, F] B: Batch; T: Timestep
                            F: Feature;
        :return:
        """
        time_shape = mixture.shape[-2]
        mixture = mixture.unsqueeze(dim=1)   # [B,1,T,F]
        x = mixture
        batch_size, feat_dim = mixture.size(0), mixture.size(2)

        x_list = []
        x,s = self.en(x)
        for id in range(6):
            x = self.nonlocal_list[id](x)
        x, GA = self.de(x, s)
       
        return x.squeeze(), GA


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.en1 = block(1,8)
        self.en2 = block(8,16)
        self.en3 = block(16,32)
        self.en4 = block(32,64)
      

    def forward(self, x):
        
        en_list = []

        x = self.en1(x)
        en_list.append(x)
   
        x = self.en2(x)
        en_list.append(x)
 
        x = self.en3(x)
        en_list.append(x)
       
        x = self.en4(x)
        en_list.append(x)

        return x, en_list

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.de4 = block(128,32)
        self.de3 = block(64,16)
        self.de2 = block(32,8)
        self.de1 = block(16,1)
        self.skip4 = skip_att(64)
        self.skip3 = skip_att(32)
        self.skip2 = skip_att(16)
        self.skip1 = skip_att(8)

    def forward(self, x, x_list):

     
        GA = []
        GA.append(self.skip4(x, x_list[-1]))
        x = self.de4(torch.cat((x, self.skip4(x, x_list[-1])),dim=1))
        GA.append(self.skip3(x, x_list[-2]))
        x = self.de3(torch.cat((x, self.skip3(x, x_list[-2])),dim=1))
        GA.append(self.skip2(x, x_list[-3]))
        x = self.de2(torch.cat((x, self.skip2(x, x_list[-3])),dim=1))
        GA.append(self.skip1(x, x_list[-4]))
        x = self.de1(torch.cat((x, self.skip1(x, x_list[-4])),dim=1))
       

        return x, GA

class block(nn.Module):
    def __init__(self,inchan,oucha):
        super(block, self).__init__()

        self.b3lock1 = nn.Sequential(
            nn.Conv2d(in_channels=inchan, out_channels=oucha, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(oucha))
        self.b3lock2= nn.Sequential(
            nn.Conv2d(in_channels=oucha, out_channels=oucha, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(oucha))
        self.b3lock3 = nn.Sequential(
            nn.Conv2d(in_channels=oucha, out_channels=oucha, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(oucha))
        self.dilation = nn.Sequential(
            nn.Conv2d(in_channels=inchan, out_channels=oucha,  kernel_size=5, stride=(1, 1), padding=8, dilation=4),
            nn.BatchNorm2d(4),
            # nn.Sigmoid(),
        )
       

    def forward(self,x):

        t = self.b3lock1(x)
        t = self.b3lock2(t)
        t = self.b3lock3(t)
        d = self.dilation(x)

        return t+d



class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, dila_rate,inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d
        self.in_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1)
        self.g_fre = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                         kernel_size=3, stride=1, padding= np.int((dila_rate * 2) / 2), dilation= dila_rate)
        self.g_cha = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                         kernel_size=3, stride=1, padding=np.int((dila_rate * 2) / 2), dilation=dila_rate)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta_fre = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                             kernel_size=3, stride=1, padding= np.int((dila_rate * 2) / 2), dilation= dila_rate)
        # self.theta_time = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
        #                          kernel_size=1, stride=1, padding=0)
        self.theta_channels = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                                  kernel_size=3, stride=1, padding= np.int((dila_rate * 2) / 2), dilation= dila_rate)

        self.phi_fre = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                           kernel_size=3, stride=1, padding= np.int((dila_rate * 2) / 2), dilation= dila_rate)
        # self.phi_time = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
        #                        kernel_size=1, stride=1, padding=0)
        self.phi_channels = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                                kernel_size=3, stride=1, padding= np.int((dila_rate * 2) / 2), dilation= dila_rate)
        self.att_mix = nn.Conv2d(in_channels=self.inter_channels*2, out_channels=self.inter_channels, kernel_size=1, stride=1)

        # if sub_sample:
        #     self.g = nn.Sequential(self.g, max_pool_layer)
        #     self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)
        frequence_shape = x.shape[3]
        channel_num = self.inter_channels
        time_step = x.shape[2]
        x_in = self.in_conv(x)
        g_x_fre = self.g_fre(x_in)   ###[B,C,T,F]
        g_x_cha = self.g_cha(x_in)
        ######frequency_nonlocal_value计算############
        theta_x_fre = self.theta_fre(x_in) ##[B,C,T,F]
        theta_x_fre = theta_x_fre.permute(0, 1, 3,2)                        #[B,C,F,T]
        theta_x_fre = torch.reshape(theta_x_fre,[batch_size,frequence_shape,-1])      ####[B,F,CT]
        phi_x_fre = self.phi_fre(x_in)  #####[B,C,T,F]
        phi_x_fre = phi_x_fre.permute(0,2,1,3)     #[B,T,C,F]
        phi_x_fre = torch.reshape(phi_x_fre,[batch_size,-1,frequence_shape])    ###[B,CT,F]
        f_fre = torch.matmul(theta_x_fre, phi_x_fre)
        f_div_C_fre = F.softmax(f_fre, dim=-1)         #####[B,F,F]
        #####frequency_nonlocal_value计算############
        # #####time_nonlocal_value计算############
        # theta_x_time = self.theta_time(x)  #[B,C,T,F]
        # theta_x_time = torch.reshape(theta_x_time,[batch_size,time_step,-1])     ##[B,T,CF]
        # phi_x_time = self.phi_time(x)  #[B,C,T,F]
        # phi_x_time = phi_x_time.permute(0,3,1,2)            #[B,F,C,T]
        # phi_x_time = torch.reshape(phi_x_time,[batch_size,-1,time_step])    #[B,CF,T]
        # f_time = torch.matmul(theta_x_time, phi_x_time)          ###[B,T,T]
        # f_div_C_time = F.softmax(f_time, dim=-1)
        # #####time_nonlocal_value计算############
        #####channel_nonlocal_value计算############
        theta_x_channels = self.theta_channels(x_in)  # [B,C,T,F]
        theta_x_channels = torch.reshape(theta_x_channels, [batch_size, channel_num, -1])  ##[B,C,FT]
        phi_x_channels = self.phi_channels(x_in)  # [B,C,T,F]
        phi_x_channels = phi_x_channels.permute(0, 3, 2, 1)  # [B,F,T,C]
        phi_x_channels = torch.reshape(phi_x_channels, [batch_size, -1, channel_num])  # [B,FT,C]
        f_channels = torch.matmul(theta_x_channels, phi_x_channels)  ###[B,C,C]
        f_div_C_channels= F.softmax(f_channels, dim=-1)
        #####channel_nonlocal_value计算###########
        y_cha = torch.reshape(g_x_cha,[batch_size,channel_num,-1])    #[B,C,TF]
        y_cha = torch.matmul(f_div_C_channels,y_cha)                ######[B,C,TF]
        y_cha = torch.reshape(y_cha,[batch_size,channel_num,time_step,frequence_shape])   ####[B,C,T,F]
        # y = y.permute(0,2,1,3)                 ####[B,T,C,F]
        # y = torch.reshape(y,[batch_size,time_step,-1])   #####[B,T,CF]
        # y = torch.matmul(f_div_C_time,y)       #####[B,T,CF]
        # y = torch.reshape(y,[batch_size,time_step,channel_num,frequence_shape])   ####[B,T,C,F]
        # y = y.permute(0,2,1,3)           #####[B,C,T,F]
        y_fre = g_x_fre.permute(0,3,2,1)          ######[B,F,T,C]
        y_fre = torch.reshape(y_fre,[batch_size,frequence_shape,-1])  #####[B,F,TC]
        y_fre = torch.matmul(f_div_C_fre,y_fre)     ######[B,F,TC]
        y_fre = torch.reshape(y_fre,[batch_size,frequence_shape,time_step,channel_num])    #[B,F,T,C]
        y_fre = y_fre.permute(0,3,2,1)    #[B,C,T,F]
        y = torch.cat((y_cha,y_fre),dim=1)
        y = self.att_mix(y)
        W_y = self.W(y)
        z = W_y + x
        return z


class skip_att(nn.Module):
    def __init__(self,in_c):
        super(skip_att, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 161))
        self.lin1 = nn.Sequential(nn.Conv1d(kernel_size=3,in_channels=in_c,out_channels=16,dilation=2,padding=2),
                                  nn.ELU()
                                  )
        self.lin2 = nn.Sequential(nn.Conv1d(kernel_size=3,in_channels=16,out_channels=16,dilation=4,padding=4),
                                  nn.ELU()
                                  )
        # self.lin4 = nn.Sequential(nn.Conv1d(kernel_size=3, in_channels=16, out_channels=16, dilation=8, padding=8),
        #                           nn.ELU()
        #                           )
        self.lin3 = nn.Sequential(nn.Conv1d(kernel_size=3, in_channels=16, out_channels=in_c, dilation=8, padding=8),
                                  nn.ELU()
                                  )
        self.Sig = nn.Sigmoid()

    def forward(self, en, de):

        # en_cha = self.avg(en)#B,C,1,161
        de_cha = self.avg(de).squeeze(dim=2)
        de = self.lin1(de_cha)
        de = self.lin2(de)
        de = self.lin3(de)
        # de = self.Sig(de)
        #
        # en_fre = self.avg(en.permute(0, 3, 2, 1))#B,F,1,1
        # de_fre = self.avg(de.permute(0, 3, 2, 1)).permute(0,)

        en_ = de.unsqueeze(dim=2) * en
        # de_ = en_cha * de

        return en_


class DilatedConv1(nn.Module):
    def __init__(self):
        super(DilatedConv1, self).__init__()
        self.conv_d1 = nn.Sequential(nn.Conv1d(kernel_size=3,dilation=2,out_channels=16,in_channels=256,padding=2),
                                      #nn.BatchNorm2d(256),
                                      nn.ELU())
        self.conv_d2 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=4, out_channels=16, in_channels=16,padding=4),
                                     # nn.BatchNorm2d(256),
                                     nn.ELU())
        self.conv_d3 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=8, out_channels=16, in_channels=16,padding=8),
                                     # nn.BatchNorm2d(256),
                                     nn.ELU())
        self.conv_d4 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=16, out_channels=16, in_channels=16,padding=16),
                                     # nn.BatchNorm2d(256),
                                     nn.ELU())
        self.conv_d5 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=32, out_channels=16, in_channels=16,padding=32),
                                     # nn.BatchNorm2d(256),
                                     nn.ELU())
        self.conv_d6 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=64, out_channels=16, in_channels=16,padding=64),
                                     # nn.BatchNorm2d(256),
                                     nn.ELU())
        self.conv_d7 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=128, out_channels=16, in_channels=16,padding=128),
                                     # nn.BatchNorm2d(256),
                                     nn.ELU())
    def forward(self,x):
        x = self.conv_d1(x)
        # print(x.shape)
        x = self.conv_d2(x)
        # print(x.shape)

        x = self.conv_d3(x)
        x = self.conv_d4(x)
        x = self.conv_d5(x)
        x = self.conv_d6(x)
        x = self.conv_d7(x)
        return x
