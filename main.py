import data
from model import WaveNet, Trainer,resnet_encoder,dresnet_decoder,FullyConnectedConv1d_SpeechEnhancement
import torch
from torch.autograd import Variable
import os
import os.path
import time
from torch import nn

def quick_transforms(x):
    return torch.from_numpy(x).type(torch.FloatTensor)

serialized_training_data_folder = '/scratch/esm1g14/segan/ser_data/'
serialized_testing_data_folder = '/scratch/esm1g14/segan/ser_test_data/'
saving_folder = './Run9/'
print(saving_folder)
if not os.path.isdir(saving_folder):
    os.mkdir(saving_folder)

train_generator = data.AudioSampleGenerator(serialized_training_data_folder,transforms=quick_transforms,target_transforms=quick_transforms,train=True,train_split=1.)
# train_generator.chop_data(0.02)
test_generator = data.AudioSampleGenerator(serialized_testing_data_folder,transforms=quick_transforms,target_transforms=quick_transforms,train=False,train_split=0.)
training_loader = torch.utils.data.DataLoader(  dataset=train_generator,
                                            batch_size=8*4,  # specified batch size here
                                            shuffle=True,
                                            num_workers=2,
                                            drop_last=True,  # drop the last batch that cannot be divided by batch_size
                                            pin_memory=True)
testing_loader = torch.utils.data.DataLoader(  dataset=test_generator,
                                            batch_size=4,  # specified batch size here
                                            shuffle=True,
                                            num_workers=2,
                                            drop_last=True,  # drop the last batch that cannot be divided by batch_size
                                            pin_memory=True)
print('BUILDING MODEL')
criterion = nn.MSELoss()
encoder = resnet_encoder(7,kernel_size=31).cuda()
decoder = dresnet_decoder(7,kernel_size=31).cuda()
model = FullyConnectedConv1d_SpeechEnhancement(encoder=encoder,decoder=decoder).cuda()
resynthesizer = data.AudioResynthesizer(model=model,data_folder_path=serialized_testing_data_folder,saving_folder=saving_folder,transform=quick_transforms)
# model = WaveNet(layers=3,in_channels=1,output_length=32,kernel_size=3,bias=False,residual_channels=16).cuda()
# optimizer = torch.optim.Adam(model.parameters(),weight_decay=10.e-5)
optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=10.e-5)
trainer = Trainer(model,training_loader,optimizer,criterion,test_loader=testing_loader,verbose=True,saving_folder=saving_folder,resynthesizer=resynthesizer,device_ids=[0,1],checkpoint=True)
trainer.train(70,drop_learning_rate=[10,40,50])
