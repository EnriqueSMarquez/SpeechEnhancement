import os
import os.path
import time
from wavenet_modules import *
from torch.autograd import Variable
from tqdm import tqdm
import cPickle
from torch import nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DeBottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None,kernel_size=3):
        super(DeBottleneck1D, self).__init__()
        self.conv1 = nn.ConvTranspose1d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv2 = nn.ConvTranspose1d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size-1)/2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.ConvTranspose1d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # we do pre-activation
        out = self.relu(self.bn1(x))
        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        return out


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,kernel_size=3):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size-1)/2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # we do pre-activation
        out = self.relu(self.bn1(x))
        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class DeResNet1D(nn.Module):

    def __init__(self, block, layers, kernel_size=3):
        self.inplanes = 1024
        self.kernel_size = kernel_size
        super(DeResNet1D, self).__init__()
        self.conv1 = nn.ConvTranspose1d(1024, 256*block.expansion, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size-1)/2)
        self.bn1 = nn.BatchNorm1d(256*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 256, layers[0]) #32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=self.kernel_size)#16
        self.layer3 = self._make_layer(block, 64, layers[2], stride=self.kernel_size)#8
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#4
        self.bn2 = nn.BatchNorm1d(64*block.expansion)
        self.last_conv1 = nn.ConvTranspose1d(64*block.expansion,16, kernel_size=self.kernel_size+13, stride=1)
        self.last_conv2 = nn.ConvTranspose1d(16,1, kernel_size=4, stride=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )

            upsample = nn.ConvTranspose1d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample,kernel_size=self.kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.last_conv1(x)
        x = self.relu(x)
        x = self.last_conv2(x)
        return x

class ResNet1D(nn.Module):

    def __init__(self, block, layers, kernel_size=3):
        self.inplanes = 16
        self.kernel_size = kernel_size
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size-1)/2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0]) #32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=self.kernel_size)#16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=self.kernel_size)#8
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#4
        self.bn2 = nn.BatchNorm1d(256*block.expansion)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )

            downsample = nn.Conv1d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,kernel_size=self.kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

def resnet_encoder(depth,block=Bottleneck1D,kernel_size=3):
    n=(depth-2)/9
    model = ResNet1D(block, [n, n, n, n],kernel_size=kernel_size)
    return model

def dresnet_decoder(layers_per_block,block=DeBottleneck1D,kernel_size=3):
    model = DeResNet1D(block, 4*[layers_per_block],kernel_size=kernel_size)
    return model

class Trainer():
    def __init__(self,model,training_loader,optimizer,criterion,test_loader=None,verbose=True,saving_folder=None,resynthesizer=None,device_ids=[0],checkpoint=True):
        self.model = model
        self.verbose = verbose
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.saving_folder = saving_folder
        self.checkpoint = checkpoint
        if not checkpoint or not os.path.isfile(self.saving_folder + 'history.txt'):
            self.history = {'test_loss' : [],
                            'train_loss' : []}
        else:
            self.history = cPickle.load(open(self.saving_folder + 'history.txt','r'))
            self.model = torch.load(self.saving_folder+'model.pth.tar').cuda()
        self.resynthesizer = resynthesizer
        self.devices = len(device_ids)
        self.device_ids = device_ids

    def train(self,nb_epochs,drop_learning_rate=[]):
        best_loss = 100.
        if not hasattr(self.model, 'device_ids'):
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        starting_epoch = len(self.history['test_loss'])
        for epoch in range(starting_epoch,nb_epochs):
            if epoch in drop_learning_rate:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr']*0.1
            if self.verbose:
                print('EPOCH : %d')%(epoch)
            train_loss = self.train_epoch()
            if self.test_loader:
                test_loss = self.test_epoch()
            self.history['train_loss'] += [train_loss]
            self.history['test_loss'] += [test_loss]
            if self.verbose:
                print('TRAINING LOSS : %.4f')%(self.history['train_loss'][-1])
                print('TESTING LOSS : %.4f')%(self.history['test_loss'][-1])
            self.save_history()
            self.save_model()
            if best_loss > self.history['test_loss'][-1]:
                self.save_model('best_')
            if self.resynthesizer != None:
                self.save_sample_audio(epoch)

    def train_epoch(self):
        running_loss = 0.
        training_loader = tqdm(self.training_loader)
        for i,(x,y) in enumerate(training_loader,0):
            batch_loss = self.train_batch(x,y)
            running_loss = 0.99 * running_loss + 0.01 * batch_loss.data[0]
            training_loader.set_postfix(loss=running_loss)
        return running_loss
    def train_batch(self,x,y):
        self.optimizer.zero_grad()
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        out = self.model(x)
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def test_epoch(self):
        running_loss = 0.
        for i,(x,y)  in enumerate(self.test_loader):
            batch_loss = self.test_batch(x,y)
            running_loss = 0.99 * running_loss + 0.01 * batch_loss.data[0]
        return running_loss
    def test_batch(self,x,y):
        x = Variable(x).cuda()
        y = Variable(y).cuda().view(-1)
        out = self.model(x)
        loss = self.criterion(out, y)
        _, predicted = torch.max(out.data, 1)
        # _,truth = torch.max(y.data,1)
        return loss
    def save_history(self):
        with open(self.saving_folder + 'history.txt','w') as fp:
            cPickle.dump(self.history,fp)
    def save_model(self,name=''):
        torch.save(self.model,self.saving_folder+name+'model.pth.tar')
    def save_sample_audio(self,epoch):
        if not os.path.isdir(self.saving_folder +  str(epoch)):
            os.mkdir(self.saving_folder +  str(epoch))
        self.resynthesizer.reconstruct_file(saving_folder=self.saving_folder +  str(epoch)+'/')

class FullyConnectedConv1d_SpeechEnhancement(nn.Module):
    def __init__(self,encoder,decoder):
        super(FullyConnectedConv1d_SpeechEnhancement,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out_act = nn.Hardtanh()
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out_act(x)
        return x

class WaveNet(nn.Module):
    """
    A Complete Wavenet Model
    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model
    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=16,
                 in_channels=1,
                 output_length=32,
                 kernel_size=3,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(WaveNet, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dtype = dtype

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        # self.main_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.in_channels,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                # self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                # self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                #                                         num_channels=residual_channels,
                #                                         dilation=new_dilation,
                #                                         dtype=dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   padding=(kernel_size-1)/2,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 padding=(kernel_size-1)/2,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=in_channels,
                                    kernel_size=1,
                                    # padding=1,
                                    bias=True)

        # self.output_length = 2 ** (layers - 1)
        self.output_length = output_length
        self.receptive_field = receptive_field

    def wavenet(self, input):

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = F.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = F.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            # if x.size(2) != 1:
            #      s = dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def forward(self, input):
        x = self.wavenet(input)
                         # dilation_func=self.wavenet_dilate)

        # reshape output
        # [n, c, l] = x.size()
        # l = self.output_length
        # x = x[:, :, -l:]
        # x = x.transpose(1, 2).contiguous()
        # x = x.view(n * l, c)
        x = x.view(x.size(0),-1)
        return x
