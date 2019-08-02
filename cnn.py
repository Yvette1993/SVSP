
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

#Hyper Parameters
EPOCH = 1               #train the training data n times,to save time,
BATCH_SIZE = 50
LR = 0.001              #learning rate
#DOWNLOAD_MNIST = True   #using 'True'download the data
DOWNLOAD_MNIST = False   #then False


train_data = torchvision.datasets.MNIST(            #download data form website
    root = './mnist',                               #put data into the folder
    train = True,                                   #using for training
    transform = torchvision.transforms.ToTensor(),  #transform the img(0-255) to tensor(0,1)
    download = DOWNLOAD_MNIST 
)

#plot one example
#print(train_data.train_data.size())  #(60000,28,28)
#print(train_data.train_labels.size()) #(60000)
#plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
#plt.title('%i'%train_data.train_labels[0])
#plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = torchvision.datasets.MNIST(root='./mnist/',train =False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255  #raw img(0-255), /255  manually compact the database to (0,1).

test_y = test_data.test_labels[:2000]              



#build cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(               #(1,28,28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2, #if stride=1, padding=(kernel_size-1)/2=(5-1)/2 ,keep img same size.
            ),    #->(16,28,28)
            nn.ReLU(), #->(16,28,28)
            nn.MaxPool2d(kernel_size=2), #->(16,14,14)
        )
        self.conv2 = nn.Sequential(       #->(16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  #->(32,14,14)
            nn.ReLU(),                   #->(32,14,14)
            nn.MaxPool2d(2),    #->(32,7,7)
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   #(batch, 32, 7, 7)
        x = x.view(x.size(0), -1) # (batch, 32*7*7) "-1":flattening data
        output = self.out(x)
        return output

cnn = CNN()
#print(cnn) #net architecture

optimizer = torch.optim.Adam(cnn.parameters(),lr =LR) # optimize all cnn paramenters
loss_func = nn.CrossEntropyLoss()  #the target label is not one=hotted

#training as testiing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):  #gives batch data, normalize x when iter
        b_x = Variable(x)  #batch x
        b_y = Variable(y)  #batch y

        output = cnn(b_x)   #cnn outputs
        loss = loss_func(output, b_y) #cross entropy loss
        optimizer.zero_grad()         #clear gradients for this training step
        loss.backward()               #back propagation, comput gradients
        optimizer.step()


        if step %50 ==0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = ((pred_y == test_y).sum() / test_y.size(0))
            print('Epoch:', epoch, '|train loss:%.4f'%loss.item(), '|test accuracy:%.2f'%accuracy.item())

#print 10 prediction from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
