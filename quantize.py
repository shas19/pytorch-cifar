'''Post training quantization of VGG16.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from models import *
from utils import progress_bar


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=500, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

quantize = False
# net = VGGQ('VGG16')
# net = AlexNet2()
# net = ResNet18Q()
net = ResNet18O()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.

# print('==> Resuming from checkpoint..')
# assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
# checkpoint = torch.load('./checkpoint/ckpt.pth')
# net.load_state_dict(checkpoint['net'])

# Use HPVM checkpoints
assert os.path.isdir('model_params/pytorch'), 'Error: no checkpoint directory found!'
# checkpoint = torch.load('./model_params/pytorch/alexnet2_cifar10.pth.tar')
checkpoint = torch.load('./model_params/pytorch/resnet18_cifar10.pth.tar')

print(checkpoint.keys())
net.load_state_dict(checkpoint)

net.eval()

if quantize:
    # net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    net.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric), 
        weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

    # sanity check if the bias is zero
    # Why cannot the dtype=qint8 work for activation?

    net_fp32_prepared = torch.quantization.prepare(net)
    input_fp32 = torch.randn(500, 3, 32, 32)
    net_fp32_prepared(input_fp32)

    net_int8 = torch.quantization.convert(net_fp32_prepared)


def test(net):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # print(inputs.shape)
            # outputs = net(inputs)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                         % (100.*correct/total, correct, total))


# the main code
if quantize:
    test(net_int8)
else:
    test(net)

