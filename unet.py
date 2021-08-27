import torch
import torch.nn as nn
import torch.nn.functional as functional


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels,dropout = 0, pool = True) -> None:
        super(EncoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=(3,3), padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding = 1)
        self.relu = nn.ReLU()
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=2)
        else:
            self.pool = None

        if dropout>0:
            self.drop = nn.Dropout()
        else:
            self.drop = None
    
    def forward(self, x):

        skip = self.relu(self.conv2(self.relu(self.conv1(x))))
        out = skip
        if self.pool:

            out = self.pool(out)

        if self.drop:

            out = self.drop(out)

        return [out, skip]   


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(DecoderBlock, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,padding=1,output_padding=1)#, padding='same')
        self.conv1 = nn.Conv2d(out_channels*2, out_channels, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, padding=1, kernel_size=3)
        self.relu = nn.ReLU()
        #self.skip =skip

    def forward(self, x, skip):

        x = self.trans_conv(x)
        x = torch.cat((skip,x), dim=1)
        x = self.relu(self.conv2(self.relu(self.conv1(x))))
        return x

    
class Unet(nn.Module):

    def __init__(self, in_channels, classes, n_filters=32) -> None:
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.n_filters = n_filters
        self.classes = classes
        self.e1 = EncoderBlock(in_channels=self.in_channels, out_channels=self.n_filters, dropout=0.2)
        self.e2 = EncoderBlock(in_channels=self.n_filters, out_channels=self.n_filters*2, dropout=0.2)
        self.e3 = EncoderBlock(in_channels=self.n_filters*2, out_channels=self.n_filters*4, dropout=0.2)
        self.e4 = EncoderBlock(in_channels=self.n_filters*4, out_channels=self.n_filters*8, dropout=0.2)
        self.e5 = EncoderBlock(in_channels=self.n_filters*8, out_channels=self.n_filters*16, dropout=0.2, pool=False)
        self.d6 = DecoderBlock(in_channels=self.n_filters*16, out_channels=self.n_filters*8)
        self.d7 = DecoderBlock(in_channels=self.n_filters*8, out_channels=self.n_filters*4)
        self.d8 = DecoderBlock(in_channels=self.n_filters*4, out_channels=self.n_filters*2)
        self.d9 = DecoderBlock(in_channels=self.n_filters*2, out_channels=self.n_filters)
        self.conv1 = nn.Conv2d(in_channels=self.n_filters, out_channels=self.n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.n_filters, out_channels=classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, x):

        x1 = self.e1(x)
        #print(x1[1].shape)
        x2 = self.e2(x1[0])
        #print(x2[1].shape)
        x3 = self.e3(x2[0])
        #print(x3[1].shape)
        x4 = self.e4(x3[0])
        #print(x4[1].shape)
        x5 = self.e5(x4[0])
        #print(x5[0].shape)
        x6 = self.d6(x5[0], x4[1])
        #print(x6.shape)
        x7 = self.d7(x6, x3[1])
        x8 = self.d8(x7, x2[1])
        x9 = self.d9(x8, x1[1])
        x10 = self.relu(self.conv1(x9))
        x11 = self.conv2(x10)

        return x11
