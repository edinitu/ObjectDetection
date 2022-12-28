import torch
import torch.nn as nn


class NetworkModel(torch.nn.Module):

    def __init__(self):
        super(NetworkModel, self).__init__()

        # define network's convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv19 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.conv23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv24 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        # define fully connected layers
        self.fc1 = torch.nn.Linear(7 * 7 * 1024, 4096)  # 6*6 from image dimension
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
        x = nn.functional.leaky_relu(self.conv18(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv19(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv20(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv21(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv22(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv23(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.conv24(x), negative_slope=0.1)

        x = x.view(-1, self.num_flat_features(x))
        x = nn.functional.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.fc2(x)

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
