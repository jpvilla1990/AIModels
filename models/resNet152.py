from torch import flatten
from torch.nn import Module, ModuleList, ReLU, Conv2d, MaxPool2d, Linear

class ResNet152(Module):
    """
    Class to create ResNet152 model
    https://cv-tricks.com/keras/understand-implement-resnets/
    """
    def __init__(self, sizeInput=[3, 32, 32], sizeOutput=4):
        """
        Constructor

        sizeInput : Array 3 elements
        sizeOutput : size output to define last layer
        """
        super(ResNet152, self).__init__()
        inputLastLayer = int(((sizeInput[1] / 4) - 2) * ((sizeInput[2] / 4) - 2) * 2048)
        outputLastLayer = int(sizeOutput)

        self.layersInput = ModuleList()
        self.layersStage1 = ModuleList()
        self.layersStage2 = ModuleList()
        self.layersStage3 = ModuleList()
        self.layersStage4 = ModuleList()
        self.lastLayer = ModuleList()
        self.projectionsIndentiyConnection = ModuleList()

        self.reLu = ReLU()

        self.layersInput.append(Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2))
        self.layersInput.append(MaxPool2d(kernel_size=3, stride=2))

        self.projectionsIndentiyConnection.append(Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1))
        self.projectionsIndentiyConnection.append(Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1))
        self.projectionsIndentiyConnection.append(Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1))
        self.projectionsIndentiyConnection.append(Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1))


        for _ in range(3):
            self.layersStage1.append(Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1))
            self.layersStage1.append(Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            self.layersStage1.append(Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1))

        for _ in range(8):
            self.layersStage2.append(Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1))
            self.layersStage2.append(Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
            self.layersStage2.append(Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1))

        for _ in range(36):
            self.layersStage3.append(Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1))
            self.layersStage3.append(Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
            self.layersStage3.append(Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1))

        for _ in range(3):
            self.layersStage4.append(Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1))
            self.layersStage4.append(Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
            self.layersStage4.append(Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1))

        self.lastLayer.append(Linear(inputLastLayer, outputLastLayer))

    def initialStage(self, x):
        """
        Method to forward the initial stage
        """
        for layer in self.layersInput:
            x = layer(x)
            x = self.reLu(x)

        return x

    def stage1(self, x):
        """
        Method to forward the stage 1
        """
        x = self.projectionsIndentiyConnection[0](x)
        for index in range(0, len(self.layersStage1), 3):
            res = x
            x = self.layersStage1[index](x)
            x = self.layersStage1[index + 1](x)
            x = self.layersStage1[index + 2](x)
            x = self.reLu(x) + res

        return x

    def stage2(self, x):
        """
        Method to forward the stage 2
        """
        x = self.projectionsIndentiyConnection[1](x)
        for index in range(0, len(self.layersStage2), 3):
            res = x
            x = self.layersStage2[index](x)
            x = self.layersStage2[index + 1](x)
            x = self.layersStage2[index + 2](x)
            x = self.reLu(x) + res

        return x

    def stage3(self, x):
        """
        Method to forward the stage 3
        """
        x = self.projectionsIndentiyConnection[2](x)
        for index in range(0, len(self.layersStage3), 3):
            res = x
            x = self.layersStage3[index](x)
            x = self.layersStage3[index + 1](x)
            x = self.layersStage3[index + 2](x)
            x = self.reLu(x) + res

        return x

    def stage4(self, x):
        """
        Method to forward the stage 4
        """
        x = self.projectionsIndentiyConnection[3](x)
        for index in range(0, len(self.layersStage4), 3):
            res = x
            x = self.layersStage4[index](x)
            x = self.layersStage4[index + 1](x)
            x = self.layersStage4[index + 2](x)
            x = self.reLu(x) + res

        return x

    def stageLast(self, x):
        """
        Method to forward the last layer
        """
        x = flatten(x, start_dim=1)
        x = self.lastLayer[0](x)
        return x

    def forward(self, x):
        """
        Method to execute forward execution
        """
        x = self.initialStage(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stageLast(x)
        return x

    def getNumberParams(self):
        """
        Tool to obtain number of parameters, must be called right after calling
        generateWeights method
        """
        return sum(p.numel() for p in self.parameters())

    def getParameters(self):
        """
        Tool to obtain parameters
        """
        parameters = []
        for p in self.parameters():
            parameters.append(p)

        return parameters