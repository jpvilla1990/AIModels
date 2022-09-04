from torch import flatten
from torch.nn import Module, ModuleList, ReLU, Conv2d, MaxPool2d, Linear, Sigmoid

class CNNsiameseNN(Module):
    """
    Class to create CNN Siamese Neural Network model
    https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

    image 105x105
    """
    def __init__(self, sizeInput=[3, 32, 32], sizeOutput=10):
        """
        Constructor

        sizeInput : Array 3 elements
        sizeOutput : size output to define last layer
        """
        super(CNNsiameseNN, self).__init__()
        inputLastLayer = int((((((((sizeInput[1] - 9) / 2) - 6) / 2) - 3) / 2 ) - 3) *  (((((((sizeInput[2] - 9) / 2) - 6) / 2) - 3) / 2 ) - 3) * 256)
        outputLastLayer = int(sizeOutput)

        self.layersInput = ModuleList()
        self.layersFeaturesMaps = ModuleList()
        self.layersOutput = ModuleList()

        self.reLu = ReLU()

        self.layersInput.append(Conv2d(in_channels=sizeInput[0], out_channels=64, kernel_size=10))
        self.layersInput.append(ReLU())

        self.layersFeaturesMaps.append(MaxPool2d(kernel_size=2))
        self.layersFeaturesMaps.append(Conv2d(in_channels=64, out_channels=128, kernel_size=7))
        self.layersFeaturesMaps.append(ReLU())
        self.layersFeaturesMaps.append(MaxPool2d(kernel_size=2))
        self.layersFeaturesMaps.append(Conv2d(in_channels=128, out_channels=128, kernel_size=4))
        self.layersFeaturesMaps.append(ReLU())
        self.layersFeaturesMaps.append(MaxPool2d(kernel_size=2))
        self.layersFeaturesMaps.append(Conv2d(in_channels=128, out_channels=256, kernel_size=4))
        self.layersFeaturesMaps.append(ReLU())

        self.layersOutput.append(Linear(inputLastLayer, outputLastLayer))
        self.layersOutput.append(Sigmoid())

    def initialStage(self, x):
        """
        Method to forward the initial stage
        """
        for layer in self.layersInput:
            x = layer(x)
        return x

    def featureStage(self, x):
        """
        Method to forward the feature stage
        """
        for layer in self.layersFeaturesMaps:
            x = layer(x)
        return x

    def stageLast(self, x):
        """
        Method to forward the last layer
        """
        x = flatten(x, start_dim=1)
        for layer in self.layersOutput:
            x = layer(x)
        return x

    def forward(self, x):
        """
        Method to execute forward execution
        """
        x = self.initialStage(x)
        x = self.featureStage(x)
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