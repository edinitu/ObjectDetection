import torch
import torch.nn as nn


class NetworkModel(torch.nn.Module):

    def __init__(self):
        super(NetworkModel, self).__init__()

        # define network's convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, bias=False)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, bias=False)
        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, bias=False)
        self.conv19 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False)
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, bias=False)
        self.conv21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, bias=False)
        self.conv22 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, bias=False)
        self.conv24 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, bias=False)

        # define fully connected layers
        self.fc1 = torch.nn.Linear(7 * 7 * 1024, 4096)
        self.fc2 = torch.nn.Linear(4096, 7 * 7 * 42)

    def forward(self, x):
        #print('Initial image shape: ', str(x.shape))
        x = nn.functional.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, (2, 2), 2)
        x = nn.functional.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, (2, 2), 2)
        x = nn.functional.leaky_relu(self.conv3(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv4(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv5(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv6(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, (2, 2), 2)
        x = nn.functional.leaky_relu(self.conv7(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv8(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv9(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv10(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv11(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv12(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv13(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv14(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv15(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv16(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, (2, 2), 2)
        x = nn.functional.leaky_relu(self.conv17(x), negative_slope=0.1)
        x = self.conv18(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv19(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv20(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv21(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv22(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv23(x), negative_slope=0.1)
        x = self.conv24(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)

        x = x.view(-1, self.num_flat_features(x))
        if torch.min(x) < -1 or torch.max(x) > 1:
            raise RuntimeError("Network activations out of range after convolutional layers")
        x = self.fc1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = nn.functional.dropout(x, p=0.5)
        x = nn.functional.relu(self.fc2(x))
        if torch.max(x) > 1:
            print(f'Max value: {torch.max(x)}')
            #raise RuntimeError("Network activations out of range after linear layers")
        #print('Final shape of tensor:', str(x.shape))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def print_network(self):
        print('Network layers: ')
        for elem in self.__dict__.get('_modules').values():
            print(elem)


def example():
    shape = (3, 448, 448)
    rand_tensor = torch.rand(shape)
    model = NetworkModel()
    model.print_network()
    model.forward(rand_tensor)


#example()
