#!/usr/bin/env python
# coding=utf-8
import numpy as np
from os import walk
import os
import os.path
import openslide
from PIL.Image import Image
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt


def train_gen(path, widths=400, heights=400):
    cor_x = []
    cor_y = []
    #***************************************读出所有file存入picture**********************************#
    picture = []
    for (_, _, filenames) in walk(path):
        for filename in filenames:
            if 'Mask' not in filename:
                picture.append(filename)
    print (picture)
    while True:
        del cor_x[:],cor_y[:]
        #************************************随机选择一个 file**************************************#
        file_pic = np.random.choice(picture)
        print ('file_pic:',file_pic)
        #************************************mask level7上判断tumor——>>形成tumor矩形坐标**************#
        file_prefix = os.path.splitext(file_pic)[0]
        file_mask = file_prefix + '_Mask.tif'
        slide_mask = openslide.open_slide(path + file_mask)
        level_dimensions = slide_mask.level_dimensions[slide_mask.level_count-1]
        max_downsamples = int(slide_mask.level_downsamples[-1])
        slide_mask_img = slide_mask.read_region((0,0), slide_mask.level_count-1, level_dimensions)
        slide_mask_gray = Image.convert(slide_mask_img,mode='L')
        slide_mask_arr = np.array(slide_mask_gray)
        mask_y,mask_x = np.nonzero(slide_mask_arr)
        #****************************tumor矩形坐标——>>mask level0上的矩形坐标**************************#
        cor_x0 = (min(mask_x)-1)*max_downsamples
        cor_y0 = (min(mask_y)-1)*max_downsamples
        cor_x1 = (max(mask_x)+1)*max_downsamples
        cor_y1 = (max(mask_y)+1)*max_downsamples
        tumor_weight = cor_x1 - cor_x0
        tumor_height = cor_y1 - cor_y0
        #****************************读入pic,产生0~1之间的随机数****************************************#
        slide_pic = openslide.open_slide(path + file_pic)
        random_choice = np.random.random()
        #****************************随机产生normal/tumor slide***************************************#
            #*****************************产生tumor slide***************************************#
        if random_choice > 0.5:
            # ***********************在mask level0上的矩形区域随机读取一块tumor有效区域********************#
            random_tumor_x = np.random.randint(cor_x0, cor_x0 + tumor_weight - widths)
            random_tumor_y = np.random.randint(cor_y0, cor_y0 + tumor_height - heights)
            random_tumor_cor = (random_tumor_x,random_tumor_y)
            random_tumor_mask_img = slide_mask.read_region(random_tumor_cor,0,(widths,heights))
            random_tumor_arr = np.array(Image.convert(random_tumor_mask_img,mode = "L"))
            tumor_a,tumor_b = np.nonzero(random_tumor_arr)
            while len(tumor_a) is 0:
                random_tumor_x = np.random.randint(cor_x0, cor_x0 + tumor_weight - widths)
                random_tumor_y = np.random.randint(cor_y0, cor_y0 + tumor_height - heights)
                random_tumor_cor = (random_tumor_x, random_tumor_y)
                random_tumor_mask_img = slide_mask.read_region(random_tumor_cor, 0, (widths, heights))
                random_tumor_arr = np.array(Image.convert(random_tumor_mask_img, mode="L"))
                tumor_a,tumor_b = np.nonzero(random_tumor_arr)
            #***********************为这个有效区域打上tumor的标签****************************************#
            random_tumor_img =slide_pic.read_region(random_tumor_cor, 0, (widths, heights))
            random_tumor_con = Image.convert(random_tumor_img, mode="RGB")
            random_tumor_resize = Image.resize(random_tumor_con, (150, 150))
            random_tumor_arr = image.img_to_array(random_tumor_resize)
            x = np.expand_dims(random_tumor_arr, axis=0) / 255.
            y = to_categorical(0, 2)
            #*****************************产生normal slide*******************************************#
        elif random_choice < 0.5:
            #******************************确定effective zone****************************************#
            slide_weight,slide_height = slide_pic.dimensions
            scaling = 1000
            Multiple = slide_weight / scaling
            slide_pic_thumbnail = slide_pic.get_thumbnail((scaling,slide_height*Multiple))
            slide_pic_thumbnail.save('/home/yyydido/project/sg/thumbnail/'+file_pic)
            slide_pic1 = openslide.open_slide('/home/yyydido/project/sg/thumbnail/'+file_pic)
            volumeNum = 100
            RGBListPortrait = list()
            RGBListTransverse = list()
            varListPortrait = list()
            varListTransverse = list()
            slide_pic1_weight, slide_pic1_height = slide_pic1.dimensions
            for i in range(0, slide_pic1_height, volumeNum):
                arr = np.array(slide_pic1.read_region((0, i), 0, (slide_pic1_weight, volumeNum)))
                arrR = np.mean(arr[:, :, :1])
                arrG = np.mean(arr[:, :, 1:2])
                arrB = np.mean(arr[:, :, 2:3])
                RGBListPortrait.append((arrR, arrG, arrB))
            for i in range(0, slide_pic1_weight, volumeNum):
                arr = np.array(slide_pic1.read_region((i, 0), 0, (volumeNum, slide_pic1_height)))
                arrR = np.mean(arr[:, :, :1])
                arrG = np.mean(arr[:, :, 1:2])
                arrB = np.mean(arr[:, :, 2:3])
                RGBListTransverse.append((arrR, arrG, arrB))
            for i, rgbVar in enumerate(RGBListPortrait):
                RGBSpot = np.var(rgbVar)
                if RGBSpot >= 1:
                    varListPortrait.append(i)
            for i, rgbVar in enumerate(RGBListTransverse):
                RGBSpot = np.var(rgbVar)
                if RGBSpot >= 1:
                    varListTransverse.append(i)

            effective_x0 = int(min(varListTransverse) * volumeNum * Multiple)
            effective_y0 = int(min(varListPortrait) * volumeNum * Multiple)
            effective_width = int(((max(varListTransverse) + 1) - (min(varListTransverse))) *volumeNum * Multiple)
            effective_height = int(((max(varListPortrait) + 1) - (min(varListPortrait))) * volumeNum * Multiple)
            #**************************在mask level0上的矩形区域随机读取一块normal有效区域****************************#
            random_normal_x = np.random.randint(effective_x0, effective_x0 + effective_width - widths)
            random_normal_y = np.random.randint(effective_y0, effective_y0 + effective_height - heights)
            random_normal_cor = (random_normal_x, random_normal_y)
            random_normal_mask_img = slide_mask.read_region(random_normal_cor,0,(widths,heights))
            random_normal_arr = np.array(Image.convert(random_normal_mask_img,mode='L'))
            normal_a,normal_b = np.nonzero(random_normal_arr)
            while len(normal_a) is not 0:
                random_normal_x = np.random.randint(effective_x0, effective_x0 + effective_width - widths)
                random_normal_y = np.random.randint(effective_y0, effective_y0 + effective_height - heights)
                random_normal_cor = (random_normal_x, random_normal_y)
                random_normal_mask_img = slide_mask.read_region(random_normal_cor, 0, (widths, heights))
                random_normal_arr = np.array(Image.convert(random_normal_mask_img, mode='L'))
                normal_a, normal_b = np.nonzero(random_normal_arr)
            # ***********************为这个有效区域打上normal的标签****************************************#
            random_normal_img = slide_pic.read_region(random_normal_cor,0,(widths,heights))
            random_normal_con = Image.convert(random_normal_img,mode='RGB')
            random_normal_resize = Image.resize(random_normal_con,(150,150))
            random_normal_arr = image.img_to_array(random_normal_resize)
            x = np.expand_dims(random_normal_arr,axis=0) /255.
            y = to_categorical(1, 2)
        yield (x, y)
train_gen('camelyon/')
