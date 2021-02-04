#!/usr/bin/env python3.6
#
# nemo2rbe_example.py
#
# Renzo Andri <andrire@iis.ee.ethz.ch>
#
# Copyright (C) 2019-2021 ETH Zurich, University of Bologna
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.sw.txt for details.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import nemo
from tqdm import tqdm
from rbe import RBE
import numpy as np
from helper_functions import compare_models_numpy
from nemo.quant.pact import pact_integer_requantize
from matplotlib import pyplot as plt
from math import log2

class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU() # <== Module, not Function!
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU() # <== Module, not Function!
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 256)
        self.fcrelu1 = nn.ReLU() # <== Module, not Function!
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x) # <== Module, not Function!
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x) # <== Module, not Function!
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fcrelu1(x) # <== Module, not Function!
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1) # <== the softmax operation does not need to be quantized, we can keep it as it is
        return output


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
    def update(self, val):
        self.sum += val.cpu()
        self.n += 1
    @property
    def avg(self):
        return self.sum / self.n

def train(model, device, train_loader, optimizer, epoch, verbose=True):
    model.train()
    train_loss = Metric('train_loss')
    with tqdm(total=len(train_loader),
          desc='Train Epoch     #{}'.format(epoch + 1),
          disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss.update(loss)
            t.set_postfix({'loss': train_loss.avg.item()})
            t.update(1)
    return train_loss.avg.item()

def test(model, device, test_loader, integer=False, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    test_acc = Metric('test_acc')
    with tqdm(total=len(test_loader),
          desc='Test',
          disable=not verbose) as t:
        with torch.no_grad():
            for data, target in test_loader:
                if integer:      # <== this will be useful when we get to the
                    data *= 255  #     IntegerDeployable stage
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                test_acc.update((pred == target.view_as(pred)).float().mean())
                t.set_postfix({'acc' : test_acc.avg.item() * 100. })
                t.update(1)
    test_loss /= len(test_loader.dataset)
    return test_acc.avg.item() * 100.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=128, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=128, shuffle=False, **kwargs
)

#wget https://raw.githubusercontent.com/FrancescoConti/nemo_examples_helper/master/mnist_cnn_fp.pt
model = ExampleNet().to(device)
try:
   state_dict = torch.load("mnist_cnn_fp.pt", map_location='cpu')
except FileNotFoundError:
   import os
   os.system("wget https://raw.githubusercontent.com/FrancescoConti/nemo_examples_helper/master/mnist_cnn_fp.pt")
   state_dict = torch.load("mnist_cnn_fp.pt", map_location='cpu')

model.load_state_dict(state_dict, strict=True)
acc = test(model, device, test_loader)
print("\nFullPrecision accuracy: %.02f%%" % acc)

# switch to FakeQuantized and retrain
model = nemo.transform.quantize_pact(model, dummy_input=torch.randn((1,1,28,28)).to(device))
precision = {
    'conv1': {
        'W_bits' : 15
    },
    'conv2': {
        'W_bits' : 15
    },
    'fc1': {
        'W_bits' : 15
    },
    'fc2': {
        'W_bits' : 15
    },
    'relu1': {
        'x_bits' : 16
    },
    'relu2': {
        'x_bits' : 16
    },
    'fcrelu1': {
        'x_bits' : 16
    },
}
model.change_precision(bits=1, min_prec_dict=precision) # what is bits=1
acc = test(model, device, test_loader)
print("\nFakeQuantized @ 16b accuracy (without activation statistics): %.02f%%" % acc)
model.set_statistics_act() # adapt statistics from input data
acc = test(model, device, test_loader)
print("\nFakeQuantized @ 16b accuracy (with activation statistics): %.02f%%" % acc)

# example mixed precission

precision = {
    'conv1': {
        'W_bits' : 7
    },
    'conv2': {
        'W_bits' : 7
    },
    'fc1': {
        'W_bits' : 7
    },
    'fc2': {
        'W_bits' : 3
    },
    'relu1': {
        'x_bits' : 8
    },
    'relu2': {
        'x_bits' : 8
    },
    'fcrelu1': {
        'x_bits' : 8
    },
}
model.change_precision(bits=1, min_prec_dict=precision)
acc = test(model, device, test_loader)
print("\nFakeQuantized @ mixed-precision accuracy: %.02f%%" % acc)
nemo.utils.save_checkpoint(model, None, 0, checkpoint_name='mnist_fq_mixed')

# Folding of batch normalization layers
model.fold_bn()
model.reset_alpha_weights()
acc = test(model, device, test_loader)
print("\nFakeQuantized @ mixed-precision (folded) accuracy: %.02f%%" % acc)

# Deployable setup
state_dict = torch.load('checkpoint/mnist_fq_mixed.pth')['state_dict']
model.load_state_dict(state_dict, strict=True)


state_dict = torch.load('checkpoint/mnist_fq_mixed.pth')['state_dict']
model.load_state_dict(state_dict, strict=True)
model = nemo.transform.bn_quantizer(model)
model.harden_weights()
model.set_deployment(eps_in=1./255)
print(model)
acc = test(model, device, test_loader)
print("\nQuantizedDeployable @ mixed-precision accuracy: %.02f%%" % acc)


model = nemo.transform.integerize_pact(model, eps_in=1.0/255)
print(model)
acc = test(model, device, test_loader, integer=True)
print("\nIntegerDeployable @ mixed-precision accuracy: %.02f%%" % acc)

nemo.utils.export_onnx('mnist_id_mixed.onnx', model, model, (1,28,28))

##### export to RBE

# Generate sample input data
correct = 0
total = 0
#for sample in range(10):
sample = 2
sample_image = test_loader.dataset.data[sample, :, :]
sample_image = np.reshape(sample_image, (1,1,28,28)).float()


sample_expclass = test_loader.dataset.targets[sample]
plt.imshow(sample_image[0,0,:,:])
plt.show()

# run network
output=model(sample_image)
sample_actclass = output.argmax(dim=1);
if sample_actclass == sample_expclass:
    correct += 1
total += 1

print(str(correct)+"/"+str(total)+" act:"+str(int(sample_actclass))+"/exp:"+str(int(sample_expclass)))

# run entire network  layer by layer
input_fm = sample_image
layers = model._modules.items()
for i in range(len(layers)):
    layer = list(layers)[i][1]
    print(type(layer))
    if isinstance(layer, nemo.quant.pact.PACT_Linear):
        input_fm=input_fm.flatten()
    input_fm=layer(input_fm)
output_fm=F.log_softmax(input_fm)



sample_image_small = sample_image[0:1, 0:1, 0:5, 0:5]
# Init RBE with nemo and create
rbe = RBE.init_from_nemo(list(layers)[0][1], list(layers)[1][1], list(layers)[2][1], sample_image_small)
rbe.print_overview()
rbe.run_bittrue_shifted_network(verify_model=True)
#rbe.run_simple_golden_network()

compare_models_numpy(rbe.y_golden, rbe.y_nemo, True)





