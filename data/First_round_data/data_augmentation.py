import glob
import random
import os
import numpy as np
import json
import copy

from PIL import Image

import sys

import time

class function(object):

    def __init__(self,x1,y1,x2,y2):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
        self.flag=True
        if self.x2-self.x1!=0:
            self.k=(self.y2-self.y1)/(self.x2-self.x1)
        else:
            self.flag=False
            self.k=0
        self.b=self.y1-self.k*self.x1
    
    def compute(self,x):
        y=self.k*x+self.b
        
        return y

def data_augmentation(train_file, train_json):
    start=time.time()
    with open(train_json, 'r') as f:
        information = json.load(f)
    new_dict=copy.deepcopy(information)
    normal=os.listdir('normal_aug')
    
    img_id=0
    id=0
    print(len(information["images"]))
    for a in information["images"]:
        if img_id<=a["id"]:
            img_id=a["id"]
    for a in information["annotations"]:
        if id<=a["id"]:
            id=a["id"]
    print(img_id,id)
    img_id+=1
    id+=1
    
    normal_file=[]
    for normal_img in normal:
        normal_img=os.path.abspath('./normal_aug')+'/'+normal_img
        normal_file.append(normal_img)
    random.shuffle(normal_file)
    restrict_img=information["images"]
    bbox=None
    random.shuffle(information["annotations"])
    jishu=0
    
    
    for ann in information["annotations"]:
        flname=None
        bbox=ann["bbox"] 
        cat_id=ann["category_id"]
        area=ann["minAreaRect"]
        for fil in restrict_img:
            
            if ann["image_id"]==fil["id"]:
                #print(fil)
                #print(len(restrict_img))
                flname=fil["file_name"]
                #img_id=fil["id"]
                #print("this cycle")
                #print("id founded")
                #print(img_id)
                #print(flname)
                flname=train_file+'/'+flname
            #flname=fil["file_name"]
            #img_id=fil["id"]
            #flname=train_file+'/'+flname
        #if flname=="/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/zhanghaoxu/tianchi/image_augmentation/coco/images/train2014/190105_103936_00151002.jpg":
        #    continue
        res_img=np.array(Image.open(flname))
        #bbox=None
        #for ann in information["annotations"]:
        #    if img_id==ann["image_id"]:
        #        bbox=ann["bbox"]
        #        #bbox=bbox[0]
        #print(bbox)
        if bbox is None:
            continue
        #print(flname)
        
        ########comput lines
        area_xmin=np.argmin([area[0][0],area[1][0],area[2][0],area[3][0]])
        area_ymin=np.argmin([area[0][1],area[1][1],area[2][1],area[3][1]])
        area_xmax=np.argmax([area[0][0],area[1][0],area[2][0],area[3][0]])
        area_ymax=np.argmax([area[0][1],area[1][1],area[2][1],area[3][1]])
        line1=function(area[area_xmin][0],area[area_xmin][1],area[area_ymax][0],area[area_ymax][1])
        line2=function(area[area_xmin][0],area[area_xmin][1],area[area_ymin][0],area[area_ymin][1])
        line3=function(area[area_ymin][0],area[area_ymin][1],area[area_xmax][0],area[area_xmax][1])
        line4=function(area[area_ymax][0],area[area_ymax][1],area[area_xmax][0],area[area_xmax][1])
        flag_cut=True
        if line1.flag==False or line2.flag==False or line3.flag==False or line4.flag==False:
            flag_cut=False
        if flag_cut==True:
            empty_arg=[]
            for i in range(bbox[2]):
                for j in range(bbox[3]):
                    #print(bbox[1])
                    if bbox[1]+j>line1.compute(bbox[0]+i):
                        if [j,i] not in empty_arg:
                            empty_arg.append([j,i])
                        res_img[bbox[1]+j][bbox[0]+i]=0
                    if bbox[1]+j<line2.compute(bbox[0]+i):
                        if [j,i] not in empty_arg:
                            empty_arg.append([j,i])
                        res_img[bbox[1]+j][bbox[0]+i]=0
                    if bbox[1]+j<line3.compute(bbox[0]+i):
                        if [j,i] not in empty_arg:
                            empty_arg.append([j,i])
                        res_img[bbox[1]+j][bbox[0]+i]=0
                    if bbox[1]+j>line4.compute(bbox[0]+i):
                        if [j,i] not in empty_arg:
                            empty_arg.append([j,i])
                        res_img[bbox[1]+j][bbox[0]+i]=0
        
        add_img=res_img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        add_w, add_h=add_img.shape[1], add_img.shape[0]
        
        num_normal_file=len(normal_file)
        i=np.random.randint(0,num_normal_file)
        nor_img=np.array(Image.open(normal_file[i]))
        img_name=normal_file[i]
        print(img_name)
        
        nor_w, nor_h=nor_img.shape[1], nor_img.shape[0]
        flag=False
        if nor_w-add_w-1<0 or nor_h-add_h-1<0:
            continue
            
        random_time=0
        while(flag==False):
            random_time+=1
            if random_time>20:
                break
            
            xmin=np.random.randint(0,nor_w-add_w-1)
            ymin=np.random.randint(0,nor_h-add_h-1)
            #print(nor_img[ymin][xmin])
            #print(id)
            if any(nor_img[ymin][xmin]!=[255,255,255]) and any(nor_img[ymin+add_h][xmin+add_w]!=[255,255,255]) \
                and any(nor_img[ymin+add_h][xmin]!=[255,255,255]) and any(nor_img[ymin][xmin+add_w]!=[255,255,255]):
                flag=True
        nor_img2=copy.deepcopy(nor_img)
        nor_img[ymin:ymin+add_h,xmin:xmin+add_w]=add_img
        if flag_cut==True:
            #for i in range(bbox[2]):
            #    for j in range(bbox[3]):
            #        for k in empty_arg:
            #empty_arg2=[]
            #for k in empty_arg:
            #    k[0]+=ymin
            #    k[1]+=xmin
            #    empty_arg2.append(k)
            #print(empty_arg[1])
            #print(empty_arg2[1])
            for k in empty_arg:
                nor_img[ymin+k[0]][xmin+k[1]]=nor_img2[ymin+k[0]][xmin+k[1]]
        
        finished_img=Image.fromarray(nor_img)
        #finished_img.save('added/'+str(num)+'.jpg')
        finished_img.save(img_name)
        #os.system('sleep 2s')
        img_dict=dict()
        ann_dict=dict()
        #print(img_name.split('/')[-1])
        #print(type(img_name.split('/')))
        #aaa=img_name.split('/')[-1]
        flag=False
        current_img_id=0
        for a in new_dict["images"]:
            if a["file_name"]==img_name.split('/')[-1]:
                #print(a["file_name"])
                current_img_id=a["id"]
                #print(current_img_id)
                flag=True
        if flag==False:
            img_dict["file_name"]=img_name.split('/')[-1]
            print(img_id)
            img_dict["id"]=img_id
            img_dict["license"]=1
            img_dict["height"]=nor_h
            img_dict["width"]=nor_w
            img_dict["data_captured"]=""
            img_dict["coco_url"]=""
            img_dict["flickr_url"]=""
            new_dict["images"].append(img_dict)
            ann_dict["bbox"]=[xmin,ymin,add_w,add_h]
            #print([xmin,ymin,add_w,add_h])
            #print([xmin/nor_w, ymin/nor_h, add_w/nor_w, add_h/nor_h])
            xmax=xmin+add_w-1
            ymax=ymin+add_h-1
            #if xmin/nor_w <0 or xmin/nor_w >1 or ymin/nor_h<0 or ymin/nor_h>1 or xmax/nor_w<0 or xmax/nor_w>1 or ymax/nor_h<0 or ymax/nor_h>1:
            #    err+=1
            #    print([xmin, ymin, add_w, add_h])
            #    print(nor_w, nor_h)
            #    print([xmin/nor_w,ymin/nor_h,xmax/nor_w,ymax/nor_h])
            ann_dict["minAreaRect"]=[]
            ann_dict["area"]=[]
            ann_dict["category_id"]=cat_id
            ann_dict["iscrowd"]=0
            ann_dict["id"]=id
            ann_dict["segmentation"]=[]
            ann_dict["image_id"]=img_id
            new_dict["annotations"].append(ann_dict)
            img_id+=1
        else:
            ann_dict["bbox"]=[xmin,ymin,add_w,add_h]
            xmax=xmin+add_w-1
            ymax=ymin+add_h-1
            #if xmin/nor_w <0 or xmin/nor_w >1 or ymin/nor_h<0 or ymin/nor_h>1 or xmax/nor_w<0 or xmax/nor_w>1 or ymax/nor_h<0 or ymax/nor_h>1:
            #    err+=1
            #    print([xmin, ymin, add_w, add_h])
            #    print(nor_w,nor_h)
            #    print([xmin/nor_w,ymin/nor_h,xmax/nor_w,ymax/nor_h])
            ann_dict["minAreaRect"]=[]
            ann_dict["area"]=[]
            ann_dict["category_id"]=cat_id
            ann_dict["iscrowd"]=0
            ann_dict["id"]=id
            ann_dict["segmentation"]=[]
            ann_dict["image_id"]=current_img_id
            new_dict["annotations"].append(ann_dict)
        
            
        if jishu==1200:
            break
        jishu+=1
        print(jishu,"finished")
        id+=1
        #img_id+=1
    with open("./annotations/train_aug.json","w") as f:
        json.dump(new_dict,f)
    end=time.time()
    print(end-start, 's used')
    


if __name__ == '__main__':
    train_json = 'annotations/train.json'
    train_file = 'jinnan2_round1_train_20190305/restricted'
    data_augmentation(train_file, train_json)
    
