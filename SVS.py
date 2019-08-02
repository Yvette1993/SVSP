
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
import distiller
from torchsummary import summary
import NET
import pascal 
import argparse
import time

start = time.process_time()

def get_parser():
    parser = argparse.ArgumentParser(description='Depth Esitimation by Single View Stereo Matching')
    parser.add_argument('-root', metavar='DIR', help='path to dataset (default :/home/lisa/SVSP/')
    parser.add_argument('-j', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of total epochs to run (default: 2000')
    parser.add_argument('--batch_size', default=2, type=int,metavar='N',
                        help='mini-batch size (default: 2)')
    parser.add_argument('-LR', default=0.0002, type=float,metavar='N',
                        help='Learning rate(default: 0.0002)')
    parser.add_argument('--disp_range', default=65, type=int,metavar='N',
                        help='how much channels by softmax(default:65)')
    parser.add_argument('--crop_size', default=(192,640), metavar='(w,h)',
                        help='crop size from image (default: (192,640))')
    parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU only. \n'
                        'Flag not set => uses GPUs according to the --gpus flag value.'
                        'Flag set => overrides the --gpus flag')
    return parser

args = get_parser().parse_args()
if args.root is None:
    args.root = '/home/lisa/SVSP/'

if args.j is None:
    args.j = 2

if args.batch_size is None:
    args.batch_size = 2

if args.LR is None:
    args.LR = 0.0002

if args.disp_range is None:
    args.disp_range = 65
    
if args.epochs is None:
    args.epochs = 2000

if args.crop_size is None:
    args.crop_size = (192,640)

if args.cpu or not torch.cuda.is_available():
    # Set GPU index to -1 if using CPU
    args.device = 'cpu'
    args.gpus = -1
else:
    args.device = 'cuda'
    if args.gpus is not None:
        try:
            args.gpus = [int(s) for s in args.gpus.split(',')]
        except ValueError:
            raise ValueError('ERROR: Argument --gpus must be a comma-separated list of integers only')
        available_gpus = torch.cuda.device_count()
        for dev_id in args.gpus:
            if dev_id >= available_gpus:
                raise ValueError('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                 .format(dev_id, available_gpus))
        # Set default device in case the first one on the list != 0
        torch.cuda.set_device(args.gpus[0])



#transforms
#normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
#transform=transforms.Compose([
#    transforms.RandomCrop((192,640)),
#    transforms.RandomHorizontalFlip(p=0.5),
#    transforms.ColorJitter(0.2),
#    transforms.ToTensor(), 
#    #normalize
#])
#target_transform = transforms.Compose([
#    transforms.RandomCrop((192,640)),
#    transforms.RandomHorizontalFlip(p=0.5),
#    transforms.ToTensor(),
#])


####loader data
trainset= pascal.DepthEstimation(args.root,train = True,crop_size=args.crop_size)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.j, drop_last=False)
#print('trainloader=',len(trainloader))
testset=pascal.DepthEstimation(args.root,train =False,crop_size=args.crop_size)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.j, drop_last=False)

#test_x = Variable(torch.unsqueeze(testloader.img,dim=3)).type(torch.FloatTensor).cuda()/255.
#test_y = testloader.target.cuda()

"""




##init weights

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier(m.weight.data)
        xavier(m.bias.data)
"""


#build cnn
SVS = NET.SVS()
SVS.cuda()
#print(SVS)
#SVS.apply(weights_init)
#torch.save(SVS,'/home/lisa/SVSP/SVS.pkl')

optimizer = torch.optim.Adam(SVS.parameters(),lr =args.LR) # optimize all paramenters
loss_func = nn.L1Loss().cuda() 
lossM = nn.MSELoss().cuda()

if __name__ == '__main__':

    for epoch in range(args.epochs): 
         ##training------------------------------------------
        for step,(batch_x,batch_y) in enumerate(trainloader):

            
            #print(batch_y.size(),batch_y) 
            #print(batch_x.size(),batch_x)
            
            b_x = Variable(batch_x).cuda()  #->(2,3,92,640)
            b_y = Variable(batch_y).view(2,1,192,640).cuda() #[2, 1, 192, 640]
            channels=args.disp_range 
            
            right_img,depth1,depth2 = SVS(b_x,args.disp_range) #->(2,3,192,640),(2,65,192,640)
            
            train_loss_right = loss_func(right_img, b_x)
            #loss = loss_func(depth2, b_y)
            loss = lossM(depth2,b_y)
            #print('loss=',loss)
            optimizer.zero_grad()         #clear gradients for this training step
            loss.backward()               #back propagation, comput gradients
            optimizer.step()

            if step %50 ==0:
                print('Epoch:', epoch, '|train_right loss:%.16f'%train_loss_right.item(),'|loss:%.4f'%loss.item()) #'|accuracy:%.4f'%accuracy
        print('Epoch:', epoch, '|loss:%.4f'%loss.item()) 
           
    print('===> Saving models...')
    torch.save({
        'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, '/home/lisa/SVSP/checkpoint/autoencoder.pth')
    
       
    for epoch in range(args.epochs): # total img = 790 training 3 times.   

        print('===> Try resume from checkpoint')
        if os.path.isdir('checkpoint'):
            try:
                checkpoint = torch.load('./checkpoint/autoencoder.pth')
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                print('===> Load last checkpoint data')
            except FileNotFoundError:
                print('Can\'t found autoencoder.pth')
        else:
            start_epoch = 0
            print('===> Start from scratch')

        ##evaluation------------------------
        for step,(batch_x,batch_y) in enumerate(testloader):
            b_x = Variable(batch_x).cuda()  #->(2,3,92,640)
            b_y = Variable(batch_y).view(2,1,192,640).cuda() #[2, 1, 192, 640]
            channels=disp_range 
            
            right_img,depth1,depth2 = SVS(b_x,args.disp_range) #->(2,3,192,640),(2,65,192,640)

            test_loss_right = loss_func(right_img, b_x)
        
            test_loss = loss_func(depth2, b_y)
            pred_y = torch.max(depth2,1)[1].cuda().data.squeeze()
            accuracy = ((depth2 == test_x).sum() / (len(testset)*2*2))
            
        
        print('Epoch:', epoch, '|test_right loss:%.16f'%test_loss_right.item(),'|test_stereo loss:%.4f'%test_loss.item(), '|accuracy:%.4f'%accuracy)


end = time.process_time()
print('Runing Time:%s'%(end-start))


            
"""
#print 10 prediction from test data
test_output = NET.VSN(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
"""
