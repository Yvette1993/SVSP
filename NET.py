import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary



class SVS(nn.Module):
    def __init__(self):
        super(SVS, self).__init__()
        #####VSN
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(               
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1, #padding=(kernel_size-stride)/2,keep img same size.
            ),    #->(,64,192,640)
            nn.ReLU(), #->(,64,192,640) 
        )
      
        self.conv2_1 = nn.Sequential(       #->(,64,96,320)
            nn.Conv2d(64, 128, 3, 1, 1),  #->(,128,96,320)
            nn.ReLU(),                   #->(,128,96,320)
            #nn.MaxPool2d(2,2),    #->(,128,48,160)
        )
        self.conv3_1 = nn.Sequential(       #->(,128,48,160)
            nn.Conv2d(128, 256, 3, 1, 1),  #->(,256,48,160)
            nn.ReLU(),                   #->(,256,48,160)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), #->(,256,48,160)
            nn.ReLU(),
            #nn.MaxPool2d(2,2),    #->(,256,24,80)
        )
        self.conv4_1 = nn.Sequential(       #->(,256,24,80)
            nn.Conv2d(256, 512, 3, 1, 1),  #->(,512,24,80)
            nn.ReLU(),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), #->(,512,24,80)
            nn.ReLU(),
            #nn.MaxPool2d(2,2),    #->(,512,12,40)
        )
        self.conv5_1 = nn.Sequential(       #->(,512,12,40)
            nn.Conv2d(512, 512, 3, 1, 1),  #->(,512,12,40)
            nn.ReLU(),
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), #->(,512,12,40)
            nn.ReLU(),
            #nn.MaxPool2d(2,2),    #->(,512,6,20)
        )
        
        self.drop =  nn.Dropout(p=0.5)
        self.conv6_1 = nn.Sequential(
            nn.Linear(512*6*20, 4096),
            nn.ReLU(),
           # nn.Dropout(p=0.self)
        )
        self.conv7_1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            #nn.Dropout(p=0.5)
        )
        self.conv8_1 = nn.Sequential(
            nn.Linear(4096, 15360),   #->(128*6*20)
            nn.ReLU(),

        )
        
        self.pred5 = nn.Sequential(
            nn.ConvTranspose2d(128,128,32,16,8), #->(,128,96,320)
        )

        self.pred4 = nn.Sequential(
            nn.BatchNorm2d(512), #->(,512,12,40)
            nn.Conv2d(512, 128, 3, 1, 1), #->(,128,12,40)
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,16,8,4) #->(,128,96,320)
        )
        self.pred3 = nn.Sequential(
            nn.BatchNorm2d(256), #->(,256,24,80)
            nn.Conv2d(256, 128, 3, 1, 1), #->(,128,24,80)
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,8,4,2) #->(,128,96,320)
        )
        self.pred2 = nn.Sequential(
            nn.BatchNorm2d(128), #->(,128,48,160)
            nn.Conv2d(128, 128, 3, 1, 1), #->(128,48,160)
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,4,2,1) #->(,128,96,320)
            
        )
        self.pred1 = nn.Sequential(
            nn.BatchNorm2d(64), #->(,64,96,320)
            nn.Conv2d(64, 128, 3, 1, 1), #->(,128,96,320)
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,1,1,0) #->(,128,96,320)
        )
        self.up = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,4,2,1), #->(,128,192,640)
            nn.ReLU(),
            nn.Conv2d(128, 65, 3, 1, 1) #->(,65,192,640)
        )
         ######Stereo

        self.conv1 = nn.Sequential(
            nn.Conv2d(               
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3, 
            ),    
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(       
            nn.Conv2d(64, 128, 5, 2, 2),  
            nn.ReLU()                      
        )
        
        self.convredir =nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.ReLU() 
        )
        self.conv3a = nn.Sequential(       
            nn.Conv2d(65, 256, 5, 2, 2),  
            nn.ReLU()
        )
        self.conv3b = nn.Sequential(   
            nn.Conv2d(256, 256, 3, 1, 1),  
            nn.ReLU()
        )
        self.conv4a = nn.Sequential(       
            nn.Conv2d(256, 512, 3, 2, 1),  
            nn.ReLU()
        )
        self.conv4b = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),  
            nn.ReLU()
        )
        self.conv5a = nn.Sequential(       
            nn.Conv2d(512, 512, 3, 2, 1),  
            nn.ReLU()
        )
        self.conv5b = nn.Sequential( 
            nn.Conv2d(512, 512, 3, 1, 1),  
            nn.ReLU()
        )
        self.conv6a = nn.Sequential(       
            nn.Conv2d(512, 1024, 3, 2, 1),  
            nn.ReLU()
        )
        self.conv6b = nn.Sequential( 
            nn.Conv2d(1024, 1024, 3, 1, 1),  
            nn.ReLU()
        )
        self.pr6 = nn.Sequential(
            nn.Conv2d(1024, 1, 3, 1, 1)
        )
          
        self.upconv5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  
            nn.ReLU()
        )
        self.iconv5 = nn.Sequential(
            nn.Conv2d(1025, 512, 3, 1, 1),  
            nn.ReLU()
        )
        self.pr5 = nn.Sequential(
            nn.Conv2d(512, 1, 3, 1, 1), 
        )
      
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  
            nn.ReLU()
        )
        self.iconv4 = nn.Sequential(
            nn.Conv2d(769, 256, 3, 1, 1),  
            nn.ReLU()
        )
        self.pr4 = nn.Sequential(
            nn.Conv2d(256, 1, 3, 1, 1), 
        )
          
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  
            nn.ReLU()
        )
        self.iconv3 = nn.Sequential(
            nn.Conv2d(385, 128, 3, 1, 1),  
            nn.ReLU()
        )
        self.pr3 = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1, 1), 
        )
        
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  
            nn.ReLU()
        )
        self.iconv2 = nn.Sequential(
            nn.Conv2d(193, 64, 3, 1, 1),  
            nn.ReLU()
        )
        self.pr2 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1), 
        )
        
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  
            nn.ReLU()
        )
        self.iconv1 = nn.Sequential(
            nn.Conv2d(97, 32, 3, 1, 1),  
            nn.ReLU()
        )
        self.pr1 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1), 
        )

        self.upconv0 = nn.Sequential(
            nn.ConvTranspose2d(32,16, 4, 2, 1),  
            nn.ReLU()
        )
        self.iconv0 = nn.Sequential(
            nn.Conv2d(20, 32, 3, 1, 1),  
            nn.ReLU()
        )
        self.pr0 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1), 
        )


     ###VSN
    def right_img(batch_size,imgs,softmax,channels):
        data = []
        for soft,img in zip(softmax,imgs): # for batch =2 ->(65,192,640)
            i = img.size()
            SUM=torch.zeros(i[0],i[1],i[2]).cuda()
            for j in range(channels):#D1-D65
                D = torch.index_select(soft,0,torch.LongTensor([j]).cuda())  #->(65),(1,192,640)
    
                ##I1-3   #->(3)(1,192,640)
                I1 = torch.index_select(img,0,torch.LongTensor([0]).cuda() )
                I2 = torch.index_select(img,0,torch.LongTensor([1]).cuda())
                I3 = torch.index_select(img,0,torch.LongTensor([2]).cuda())
                    
                #shifted
                s1 = torch.mul(I1,D)    #->(1,192,640)
                s2 = torch.mul(I2,D)
                s3 = torch.mul(I3,D)
                cat= torch.cat((s1,s2,s3),0)  #->(3,192,640)

                SUM =torch.add(SUM, cat)               #65->1 (3,192,640) batch1 data
                #print('SUM=',SUM.size())
            data.append(SUM)  #len(data)=2 SUM=(3,192,640)

        a = data[0].size()
        CAT = torch.zeros(1,a[0],a[1],a[2]).cuda()
        for i in data:
            b=i.size()
            i= i.view(1,b[0],b[1],b[2])
            CAT = torch.cat((CAT,i),0)
        #print('CAT1=',CAT[1:].size())
        return CAT[1:]       #->(2,3,192,640)
    ###Stereo
    def CAT3(batch_size,pr,deconv,conv):
        data = []
        for k,l,r in zip(pr,deconv,conv):
            cat = torch.cat((l,r),0)
            cat = torch.cat((cat,k),0)
            data.append(cat)
        a = data[0].size()
        CAT = torch.zeros(1,a[0],a[1],a[2]).cuda()
        for i in data:
            b=i.size()
            i= i.view(1,b[0],b[1],b[2])
            CAT = torch.cat((CAT,i),0)
        #print('CAT1=',CAT[1:].size())
        return CAT[1:]
    
    def CAT2(batch_size,corr,conv):
        data = []
        for l,r in zip(corr,conv):
            cat = torch.cat((l,r),0)
            data.append(cat)
        a = data[0].size()
        CAT = torch.zeros(1,a[0],a[1],a[2]).cuda()
        for i in data:
            b=i.size()
            i= i.view(1,b[0],b[1],b[2])
            CAT = torch.cat((CAT,i),0)
        #print('CAT2=',CAT[1:].size())
        return CAT[1:]

    """
    def get_loss(batch_size,pr,groundtruth):
        
        S= pr.size()
        #print('S=',S)
        gt = nn.functional.interpolate(groundtruth,size=(S[0],S[1],S[2],S[3]),mode='bilinear')
        loss = nn.L1Loss(pr,gt)
        return loss
    """
      

    def forward(self, left_img,disp_range):

        
        conv1_1 = self.conv1_1(left_img)      #w0   #->(,64,192,640)
        pool1 = self.pool(conv1_1)              #->(,64,96,320)
        conv2_1 = self.conv2_1(pool1)     #w1   #->(,128,96,320)
        pool2 = self.pool(conv2_1)              #->(,128,48,160)
        conv3_1 = self.conv3_1(pool2)     #w2   #->(,256,48,160)
        conv3_2 = self.conv3_2(conv3_1)   #w3   #->(,256,48,160)
        pool3 = self.pool(conv3_2)              #->(,256,24,80)
        conv4_1 = self.conv4_1(pool3)     #w4   #->(,512,24,80)
        conv4_2 = self.conv4_2(conv4_1)   #w5   #->(,512,24,80)
        pool4 = self.pool(conv4_2)              #->(,512,12,40)
        conv5_1 = self.conv5_1(pool4)     #w6   #->(,512,12,40)
        conv5_2 = self.conv5_2(conv5_1)   #w7   #->(,512,12,40)
        pool5 = self.pool(conv5_2)              #->(,512,6,20)
        flatten = pool5.view(pool5.size(0), -1) # (batch, 512*6*20) "-1":flattening data
        conv6_1 = self.conv6_1(flatten)   #w8   4096
        drop6 = self.drop(conv6_1)
        conv7_1 = self.conv7_1(drop6)     #w9   4096
        drop7 = self.drop(conv7_1)
        conv8_1 = self.conv8_1(drop7)     #w10  128*6*20
      
        reshape = conv8_1.view(2,128, 6, 20)  #->(2,128,6,20)
        
        pred5 = self.pred5(reshape)   #->(2,128,96,320)   w11
        pred4 = self.pred4(pool4)  #->(2,128,96,320)      w12
        pred3 = self.pred3(pool3)  #->(2,128,96,320)      w13
        pred2 = self.pred2(pool2)  #->(2,128,96,320)      w14
        pred1 = self.pred1(pool1)  #->(2,128,96,320)      w15

        SUM = torch.add(pred1,pred2)
        SUM = torch.add(SUM,pred3)
        SUM = torch.add(SUM,pred4)
        SUM = torch.add(SUM,pred5)

        depth = self.up(SUM)      #->(2,65,192,640) (0-255)  w16,17
        soft = nn.Softmax2d()(depth)    #->(2,65,192,640) (0-1)  classifer0??
       
        right_img = self.right_img(imgs=left_img,softmax=soft,channels=disp_range)  #->(2,3,192,640)
       
        #####Stereo
        img0_aug=nn.BatchNorm2d(3).cuda()(left_img)       #(2,3,192,640)  
        img1_aug=nn.BatchNorm2d(3).cuda()(right_img)      #(2,3,192,640)
        #disp = groundtruth.float().cuda()    #(2,1,192,640)
        #print('disp=',disp.size())
        
        conv1l = self.conv1(img0_aug)            #->64 (2,64,96,320)     w18
        conv1r = self.conv1(img1_aug)            #->64  (2,64,96,320)
       
        conv2l = self.conv2(conv1l)       #->128 (2,128,48,160)          w19
        conv2r = self.conv2(conv1r)       #->128 (2,128,48,160)
       
        corr1 = nn.PairwiseDistance(2)(conv2l,conv2r).view(2,1,48,160)  #->256 (2,1,48,160)
        conv_redir = self.convredir(conv2l)   #->(2,64,48,160)      w20
        blob20 = self.CAT2(corr=corr1,conv=conv_redir)   #->256 (2,65,48,160)
        
        conv3a = self.conv3a(blob20)    #-> 256  [2, 256, 24, 80]   w21
        conv3b = self.conv3b(conv3a)    #->256  [2, 256, 24, 80]    w22
     
        conv4a = self.conv4a(conv3b)    #->512 ([2, 512, 12, 40])   w23
        conv4b = self.conv4b(conv4a)    #->512  ([2, 512, 12, 40])  w24
       
        conv5a = self.conv5a(conv4b)    #->512  ([2, 512, 6, 20])   w25
        conv5b = self.conv5b(conv5a)    #->512  ([2, 512, 6, 20])   w26
        #print('5b=',conv5b.size())
       
        conv6a = self.conv6a(conv5b)    #->1024 ([2, 1024, 3, 10])  w27
        conv6b = self.conv6b(conv6a)    #->1024  ([2, 1024, 3, 10]) w28
        
        pr6 = self.pr6(conv6b)    #->1  ([2, 1, 3, 10])             w29
        pr6_5= nn.functional.interpolate(pr6,scale_factor=2)   #->1  ([2, 1, 6, 20])  w30
      
        
        deconv5 = self.upconv5(conv6b)  #->512  ([2, 512, 6, 20])    w31
        cat5 = self.CAT3(pr=pr6_5,deconv=deconv5,conv=conv5b)
        
        iconv5 = self.iconv5(cat5)   #->512  (2,512,6,20)         w32
        pr5 = self.pr5(iconv5)  #->1    (2,1,6,20)                w33
     
        pr5_4= nn.functional.interpolate(pr5,scale_factor=2)   #->1  ([2, 1, 12, 40])    w34
        deconv4 = self.upconv4(iconv5) #->256     (2,256,12,40)                          w35
        cat4 = self.CAT3(pr=pr5_4,deconv=deconv4,conv=conv4b) #-> (2,769,12,40)
        
        iconv4 = self.iconv4(cat4)  #->256                 w36
        pr4 = self.pr4(iconv4)   #->1                      w37
       
        pr4_3= nn.functional.interpolate(pr4,scale_factor=2)   #->1  ([2, 1, 24, 80])   w38
        deconv3 = self.upconv3(iconv4) # ->128   (2,128,24,80)                          w39
        cat3 = self.CAT3(pr=pr4_3,deconv=deconv3,conv=conv3b)  #-> (2,385,24,80)
    
        iconv3 = self.iconv3(cat3) #->128  (2,128,24,80)                           w40
        pr3 = self.pr3(iconv3)   #->1                                              w41
     
        pr3_2= nn.functional.interpolate(pr3,scale_factor=2)  #(2,1,48,160)        w42
        deconv2 = self.upconv2(iconv3) # ->64   (2,64,48,160)                      w43
        cat2 = self.CAT3(pr=pr3_2,deconv=deconv2,conv=conv2l) #-> (2,193,48,160)
       
        iconv2 = self.iconv2(cat2) #->                                            w44
        pr2 = self.pr2(iconv2) #->1                                               w45
    
        pr2_1= nn.functional.interpolate(pr2,scale_factor=2)  #(2,1,96,320)       w46
        deconv1 = self.upconv1(iconv2) # ->   (2,32,96,320)                       w47
        cat1 = self.CAT3(pr=pr2_1,deconv=deconv1,conv=conv1l)  #-> (2,97,96,320)
        
        iconv1 = self.iconv1(cat1) #->(2,32,96,320)                               w48
        pr1 = self.pr1(iconv1) #->1                                               w49
 
        pr1_0= nn.functional.interpolate(pr1,scale_factor=2)  #(2,16,192,640)     w50
        deconv0 = self.upconv0(iconv1) # ->   (2,16,192,640)                      w51
        cat0 = self.CAT3(pr=pr1_0,deconv=deconv0,conv=left_img)  #-> (2,20,192,640)
  
        iconv0 = self.iconv0(cat0) #->(2,32,193,640)                              w52
        pr0 = self.pr0(iconv0) #->1 (2,1,192,640)                                 w53
        depth2=torch.mul(pr0,256.)
        
     
        return right_img,depth,depth2
        

         

         

         
         
         
         
        
