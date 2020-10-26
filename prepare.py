#coding:utf-8

#  Save specified area as an image file from spectrogram and its annotation label file (yolo format)
#  
#  スペクトルグラムとyolo形式のラベルを読んで
#  抽出された図形を表示する。または　書き出す。
#  


import sys
import os
import argparse
import pathlib
import glob
import csv

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg



# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.18.4
#  matplotlib  3.3.1



class Class_Show(object):
    def __init__(self, file_path, save_dir_path='./result_out/', rt_without_plotshow=True):
        #
        self.file_path= file_path
        print(self.file_path)  # print file name
        self.label_txt_file= os.path.splitext(self.file_path)[0] + '.txt'
        self.save_dir_path= save_dir_path
        self.savefile= self.save_dir_path + pathlib.PurePath(file_path).stem
        self.rt_without_plotshow= rt_without_plotshow
        self.category=[]
        self.boxes=[]
        # get base image
        self.base_img= mpimg.imread( self.file_path)
        print ('base_img.shape', self.base_img.shape)
        
        # get label
        self.get_label(self.label_txt_file, img_shape=self.base_img.shape)
        
        # start to draw whole image
        self.plot_image()
        
        #
        self.plot_boxes_image()
        


    def plot_image(self, ):
        
        self.fig_image= self.base_img.copy()
        
    	#
        fig, ax = plt.subplots()
        self.fig= fig
        self.ax= ax
        
        ax.set_axis_off()
        
        self.img0= ax.imshow( self.fig_image, aspect='auto', origin='upper')
        
        self.add_patch1()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.tight_layout()
        
        if not self.rt_without_plotshow:
            plt.show()
    
    
    def get_label(self,fpath, img_shape=None):
        #
        self.category=[]
        self.boxes=[]
        print ('label file', fpath)
        if img_shape is not None:
            width=img_shape[1]
            hight=img_shape[0]
            
        with open(fpath, mode='r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                # [category number] [object center in X] [object center in Y] [object width in X] [object width in Y]
                if len(row) == 5:
                    self.category.append(int(row[0]))
                     # [object start in X] [object start in Y] [object width in X] [object width in Y]
                    if img_shape is not None:
                        self.boxes.append([ \
                           int(width * ( float(row[1]) - float(row[3])/2.0) ), \
                           int(hight * ( float(row[2]) - float(row[4])/2.0) ), \
                           int(width * float(row[3])), \
                           int(hight * float(row[4])) \
                           ])
                    else:
                        self.boxes.append([ \
                           float(row[1]) - float(row[3])/2.0, \
                           float(row[2]) - float(row[4])/2.0, \
                           float(row[3]), \
                           float(row[4]) \
                           ])
        if not self.rt_without_plotshow:
            print ( self.category )
            print ( self.boxes )


    def plot_boxes_image(self,):
        
    	if self.boxes is not None:
            #print (' try to draw self.boxes_previous')
            
            num0=0
            for i, box in enumerate (self.boxes):
                x, y = box[:2]
                w, h = box[2:]
                self.box_img= self.base_img[y:y+h, x:x+w, :].copy()
                
                print (self.savefile+'__' + str(num0) + '_' + str( self.category[i]) + '.jpg') 
                mpimg.imsave( self.savefile+'__' + str(num0) + '_' + str( self.category[i]) + '.jpg', self.box_img)
                
                
                num0+=1
                
                if not self.rt_without_plotshow:
                    fig, ax = plt.subplots()
                    ax.set_axis_off()
                    self.box_img= ax.imshow( self.box_img, aspect='equal', origin='upper')
                    plt.tight_layout()
                    plt.show()
    
    def add_patch1(self,):
        if self.boxes is not None:
            #print (' try to draw self.boxes_previous')
            for box in self.boxes:
                x, y = box[:2]
                w, h = box[2:]
                self.ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='green', fill=None))


def get_jpg_files( dir_path ):
    # get jpg file list which has label txt.
    jpg_list=glob.glob( dir_path + "*.jpg")
    jpg_list_with_labeltxt=[]
    for l in jpg_list:
        if os.path.exists(os.path.splitext(l)[0] + '.txt'):
            jpg_list_with_labeltxt.append(l)
    #
    return jpg_list_with_labeltxt

if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='save labeled portion image of spectrogram ')
    parser.add_argument('--jpg_dir', '-w', default='./result', help='spectrogram jpg and label txt (yolo format) directory')
    parser.add_argument('--out_dir', '-o', default='./result_out', help='output directory')
    args = parser.parse_args()
    
    path0= args.jpg_dir + '/'
    path1= args.out_dir + '/'
    
    #
    jpg_files_list= get_jpg_files(path0)
    print ('number of jpg files', len(jpg_files_list))
    #print (jpg_files_list)
    
    
    for i,f in enumerate(jpg_files_list): # ([jpg_files_list[0]]):
        #
        img1= Class_Show(f, save_dir_path=path1, rt_without_plotshow=True)
        
