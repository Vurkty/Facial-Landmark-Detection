import torch.nn as nn
from torch.nn import Upsample

class HourGlass(nn.Module):
    def __init__(self,n=4,f=128):
        super(HourGlass,self).__init__()
        self._n = n
        self._f = f
        self._init_layers(self._n,self._f)

    def _init_layers(self,n,f):
        setattr(self,'res'+str(n)+'_1',Residual(f,f))
        setattr(self,'pool'+str(n)+'_1',nn.MaxPool2d(2,2))
        setattr(self,'res'+str(n)+'_2',Residual(f,f))
        if n > 1:
            self._init_layers(n-1,f)
        else:
            self.res_center = Residual(f,f)
        setattr(self,'res'+str(n)+'_3',Residual(f,f))
        setattr(self,'unsample'+str(n),Upsample(scale_factor=2))


    def _forward(self,x,n,f):
        up1 = x
        up1 = eval('self.res'+str(n)+'_1')(up1)
        low1 = eval('self.pool'+str(n)+'_1')(x)
        low1 = eval('self.res'+str(n)+'_2')(low1)
        if n > 1:
            low2 = self._forward(low1,n-1,f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.'+'res'+str(n)+'_3')(low3)
        up2 = eval('self.'+'unsample'+str(n)).forward(low3)

        return up1+up2

    def forward(self,x):
        return self._forward(x,self._n,self._f)

class Residual(nn.Module):
    def __init__(self,ins,outs):
        super(Residual,self).__init__()
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins,outs/2,1),
            nn.BatchNorm2d(outs/2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs/2,outs/2,3,1,1),
            nn.BatchNorm2d(outs/2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs/2,outs,1)
        )
        if ins != outs:
            self.skipConv = nn.Conv2d(ins,outs,1)
        self.ins = ins
        self.outs = outs
    def forward(self,x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x

class Lin(nn.Module):
    def __init__(self,numIn, numout):
        super(Lin,self).__init__()
        self.conv = nn.Conv2d(numIn,numout,1)
        self.bn = nn.BatchNorm2d(numout)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class VuHGs(nn.Module):
    def __init__(self):
        super(VuHGs,self).__init__()
        self.__conv1 = nn.Conv2d(1,64,1)
        self.__relu1 = nn.ReLU(inplace=True)
        self.__conv2 = nn.Conv2d(64,128,1)
        self.__relu2 = nn.ReLU(inplace=True)
        self.__hg = HourGlass()
        self.__lin = Lin(128, 68)

    def forward(self,x):
        x = self.__relu1(self.__conv1(x))
        x = self.__relu2(self.__conv2(x))
        x = self.__hg(x)
        x = self.__lin(x)
        return x