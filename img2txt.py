
# -*- coding: utf-8 -*-

 
import os
 
 
def traverse(data,label,root):
    pn = os.path.join(root,data)
    #print('pn=',pn)
    ps = os.listdir(pn)
    #print('ps=',ps)
    ln = os.path.join(root,label)
    ls = os.listdir(ln)
    Pame = []
    Lame = []
    for p1,l1 in zip(ps,ls):
        p_path = os.path.join(pn,p1)
        l_path = os.path.join(ln,l1)
        Pame.append(p_path)
        Lame.append(l_path)
        #print Name
    image2txt = open('/home/lisa/svs-pytorch/data/train.txt',mode='w')
    #image2txt = open('/home/lisa/svs-pytorch/data/val.txt',mode='w')
    for i,j in zip(Pame,Lame):
        image2txt.write(i+" "+j+"\n")
            
data = 'train_22image_02/data/'
label = 'train_22groundtruth_02/'
#data = 'val_13image_02/data/'
#label = 'val_13groundtruth_02/'
root = '/home/lisa/svs-pytorch/data/'
traverse(data,label,root)



"""
def gettxt(path):
    drive_txt = str(path2txt(path))
    print(drive_txt)
  
    drive_txt.split('=',1)
    #print(drive_txt[1])
    drive_txt[1].split(' ' ,1)
    #print(drive_txt[0])
    #drive_file = open(drive_txt[0],mode='rb')
    with open(drive_txt[0],mode='rb') as f:
        for line in f.readlines():
            folder_txt= path2txt(line)#->into _drive_
            print(folder_txt)
            with open(folder_txt,mode='rb') as f:
                for line in f.readlines():
                    folder_txt= path2txt(line)#->into _drive_
            
            folder_name =  open(str(folder_txt),mode='rb')
            for i in folder_name:
                if i == image_02:
                    img_txt = path2txt(i)



if __name__ == '__main__':
    kit_dir = "/home/lisa/Pictures/depth/2011_09_26"
    img_txt = gettxt(kit_dir)

"""
