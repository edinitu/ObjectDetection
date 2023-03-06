import torch
import torch.nn as nn


class NetworkModel(torch.nn.Module):

    def __init__(self, testing=False):
        super(NetworkModel, self).__init__()

        self.testing = testing
        # define network's convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False,
                               dtype=torch.float16)
        self.bn1 = nn.BatchNorm2d(64, dtype=torch.float16)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, bias=False,
                               dtype=torch.float16)
        self.bn2 = nn.BatchNorm2d(192, dtype=torch.float16)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, bias=False,
                               dtype=torch.float16)
        self.bn3 = nn.BatchNorm2d(128, dtype=torch.float16)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False,
                               dtype=torch.float16)
        self.bn4 = nn.BatchNorm2d(256, dtype=torch.float16)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, bias=False,
                               dtype=torch.float16)
        self.bn5 = nn.BatchNorm2d(256, dtype=torch.float16)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False,
                               dtype=torch.float16)
        self.bn6 = nn.BatchNorm2d(512, dtype=torch.float16)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False, dtype=torch.float16)
        self.bn7 = nn.BatchNorm2d(256, dtype=torch.float16)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False,
                               dtype=torch.float16)
        self.bn8 = nn.BatchNorm2d(512, dtype=torch.float16)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False, dtype=torch.float16)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False,
                                dtype=torch.float16)
        self.bn10 = nn.BatchNorm2d(512, dtype=torch.float16)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False,
                                dtype=torch.float16)
        self.bn11 = nn.BatchNorm2d(256, dtype=torch.float16)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False,
                                dtype=torch.float16)
        self.bn12 = nn.BatchNorm2d(512, dtype=torch.float16)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False, dtype=torch.float16)
        self.bn13 = nn.BatchNorm2d(256, dtype=torch.float16)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False,
                                dtype=torch.float16)
        self.bn14 = nn.BatchNorm2d(512, dtype=torch.float16)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, bias=False, dtype=torch.float16)
        self.bn15 = nn.BatchNorm2d(512, dtype=torch.float16)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, bias=False,
                                dtype=torch.float16)
        self.bn16 = nn.BatchNorm2d(1024, dtype=torch.float16)
        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False,
                                dtype=torch.float16)
        self.bn17 = nn.BatchNorm2d(512, dtype=torch.float16)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, bias=False,
                                dtype=torch.float16)
        self.bn18 = nn.BatchNorm2d(1024, dtype=torch.float16)
        self.conv19 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False, dtype=torch.float16)
        self.bn19 = nn.BatchNorm2d(512, dtype=torch.float16)
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, bias=False,
                                dtype=torch.float16)
        self.bn20 = nn.BatchNorm2d(1024, dtype=torch.float16)
        self.conv21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, bias=False,
                                dtype=torch.float16)
        self.bn21 = nn.BatchNorm2d(1024, dtype=torch.float16)
        self.conv22 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False,
                                dtype=torch.float16)
        self.bn22 = nn.BatchNorm2d(1024, dtype=torch.float16)
        self.conv23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, bias=False,
                                dtype=torch.float16)
        self.bn23 = nn.BatchNorm2d(1024, dtype=torch.float16)
        self.conv24 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, bias=False,
                                dtype=torch.float16)
        self.bn24 = nn.BatchNorm2d(1024, dtype=torch.float16)

        # define linear layers
        self.fc1 = nn.Linear(7 * 7 * 1024, 4096, dtype=torch.float16)
        # TODO make this layer configurable based on the number of classes we want to detect
        self.fc2 = nn.Linear(4096, 7 * 7 * 6, dtype=torch.float16)

    def forward(self, x):
        # print('Initial image shape: ', str(x.shape))
        x = nn.functional.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, (2, 2), 2)
        x = nn.functional.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, (2, 2), 2)
        x = nn.functional.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, (2, 2), 2)
        x = nn.functional.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn8(self.conv8(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn9(self.conv9(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn10(self.conv10(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn11(self.conv11(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn12(self.conv12(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn13(self.conv13(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn14(self.conv14(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn15(self.conv15(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn16(self.conv16(x)), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, (2, 2), 2)
        x = nn.functional.leaky_relu(self.bn17(self.conv17(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn18(self.conv18(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn19(self.conv19(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn20(self.conv20(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn21(self.conv21(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn22(self.conv22(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn23(self.conv23(x)), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.bn24(self.conv24(x)), negative_slope=0.1)

        x = x.view(-1, self.num_flat_features(x))
       # if torch.min(x) < -1 or torch.max(x) > 1:
        #    raise RuntimeError("Network activations out of range after convolutional layers")
        x = self.fc1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        if not self.testing:
            x = nn.functional.dropout(x, p=0.5)
        x = nn.functional.relu(self.fc2(x))
        if torch.max(x) > 1.5:
            print(f'Max value: {torch.max(x)}')
            # raise RuntimeError("Network activations out of range after linear layers")
        # print('Final shape of tensor:', str(x.shape))
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

    def set_testing(self):
        self.testing = True

    def set_training(self):
        self.testing = False


def example():
    shape = (3, 448, 448)
    rand_tensor = torch.rand(shape)
    model = NetworkModel()
    model.print_network()
    model.forward(rand_tensor)


#example()
