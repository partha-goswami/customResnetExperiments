import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom_ResNet(nn.Module):
    def __init__(self):
        '''
        Init Method.
        Here we initialize and define the layers
        '''
        super(Custom_ResNet, self).__init__()

        # PrepLayer - (Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Layer1 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        # Add(X, R1)
        self.layer1_x = self.x_block(64, 128, 3)
        self.layer1_r1 = self.res_block(128, 128, 3)

        # Layer2 -
        # Conv 3x3 [256k]
        # MaxPooling2D
        # BN
        # ReLU
        self.layer2 = self.x_block(128, 256, 3, 1)

        # Layer3 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        # Add(X, R2)
        self.layer3_x = self.X_block(256, 512, 3)
        self.layer3_r2 = self.res_block(512, 512, 3)

        # MaxPooling with Kernel Size 4
        self.pool = nn.MaxPool2d(4, 4)

        # FC Layer
        self.FC = nn.Linear(512, 10, bias=False)


    def res_block(self, in_channels, out_channels, kernel_size, padding=1, bias=False):
        '''
        Responsible for creating convolution block consisting of Conv-BN-ReLU-Conv-BN-ReLU

        :param in_channels: Input channels
        :param out_channels: Output Channels
        :param kernel_size: Kernel Size
        :return: Convolution block
        '''
        conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return conv

    def x_block(self, in_channels, out_channels, kernel_size, padding=1, bias=False):
        '''
        Responsible for creating convolution block consisting of Conv-MaxPool-BN-ReLU

        :param in_channels: Input Channels
        :param out_channels: Output Channels
        :param kernel_size: Kernel Size
        :param padding: Padding
        :param bias: Bias
        :return: Convolution Block
        '''
        conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      bias=bias),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return conv

    def forward(self, x):
        '''
        Responsible for one forward pass in the neural network

        :param x: Input
        :return: SoftMax output
        '''

        # Initial Assumptions -
        # Initial jIn (jump-in) = 1
        # Initial RF (Receptive Field) = 1*1


        # PrepLayer - (Conv 3x3 s1, p1) >> BN >> RELU [64k]
        # Input Size = 32*32
        # Output Size = (32+(2*1)-3)/1 + 1 = 32*32
        # jOut = 1*1 = 1
        # Receptive Field (rOut) = 1+(3-1)*1 = 3*3
        x = self.prep_layer(x)

        # Layer1 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        # Add(X, R1)

        # Input Size = 32*32
        # Output Size = 16*16
        x = self.layer1_x(x)

        # Input Size = 16*16
        # Output Size = 16*16
        r1 = self.layer1_r1(x)
        x = x + r1

        # Layer2 -
        # Conv 3x3 [256k]
        # MaxPooling2D
        # BN
        # ReLU

        # Input Size = 16*16
        # Output Size = 8*8
        x = self.layer2(x)

        # Layer3 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        # Add(X, R2)

        # Input Size = 8*8
        # Output Size = 4*4
        x = self.layer3_x(x)

        # Input Size = 4*4
        # Output Size = 4*4
        r2 = self.layer3_r2(x)
        x = x + r2

        # MaxPooling with Kernel Size 4

        # Input Size = 4*4
        # Output Size = 1*1
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        # Fully Connected Layer
        x = self.FC(x)

        x = x.view(-1, 10)
        return F.softmax(x, dim=-1)

