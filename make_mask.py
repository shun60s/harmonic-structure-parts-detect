#coding:utf-8

#  Make mask data (npy file) from handwritten mask define image (BMP file)
#  Warning: Due to use only information along x-axis, something like inverted V shape can be defined. 
#
#  抽出された図形を読み込むみ表示する。
#  マスクを作る
#
#  マスクは閉曲面が1個しかない前提とする。　つまり、「Y」のような二股を許さない。　飛び地も許さない。
#  X軸　検索だけのため　「ヘ」の字のようなものしか定義できない。「く」の字のようなものは2つに分かれるため定義できない。
#
#
#  お絵描きソフトのＰＡＩＮＴを使って赤線で囲った画像をマスク定義ファイルとしてＢＭＰで保存すること。
#  その理由はＪＰＧで保存するとＧＲＡＹスケールにならないで、カラーになるため、後段でマスク部分を抽出できなくなるため。

import sys
import os
import argparse
import pathlib
import glob


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg


# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.18.4
#  matplotlib  3.3.1


class Class_Show2(object):
    def __init__(self, file_path):
        #
        self.file_path= file_path
        print(self.file_path)  # print file name
        # get base image
        self.base_img= mpimg.imread( self.file_path)
        print ('base_img.shape', self.base_img.shape)
        
        #
        self.make_mask()
        
        # start to draw whole image
        self.plot_image()
        
    
    
    def plot_image(self, rt_without_plotshow=False):
        
        self.fig_image= self.base_img.copy()
        
    	#
        fig,  [ax1, ax2] = plt.subplots(1, 2)
        self.fig= fig
        self.ax1=ax1
        self.ax2=ax2
        
        #ax1.set_axis_off()
        self.img0= ax1.imshow( self.fig_image, aspect='equal', origin='upper')
        
        #ax2.set_axis_off()
        self.img2= ax2.imshow( self.fig_image2, aspect='equal', origin='upper')
        
        plt.tight_layout()
        
        self.add_patch1(self.xys)
        
        if not rt_without_plotshow:
            plt.show()
        
    """
    閉曲面が1個しかない前提で検索する。、　つまり、「Y」のような二股を許さない。　飛び地も許さない。
    X軸　検索だと　「ヘ」の字のようなものしか定義できない。（デフォルト）
    Y軸　検索だと、「く」の字のようなものしか定義できない。

    """
    def make_mask(self,  show_progress=False): # =True):
        # make available area  per x-aix
        # pick up color portion which  R,G, and, B value is not same 
        self.fig_image2= self.base_img.copy()
        iy, ix, _ = self.fig_image2.shape
        ixs= np.array(np.ones([ix,2]) * -1, dtype=int)  # available area per x-aix, initial value -1
        
        all_line_pos=[]  # every line_pos per x-axis
        for i in range(ix):
            line_pos=[]  # stack per a x-axis
            flag_new= False
            j_start = -1
            j_end = -1
            for j in range(iy):
                if not (self.base_img[j,i,0] == self.base_img[j,i,1] == self.base_img[j,i,2]):
                    if not flag_new:  # check when line starts
                        j_start = j
                        j_end = j
                        flag_new=True
                    else:
                        j_end=j  # update line end
                else:  # append position  when line ends
                    if flag_new:
                        line_pos.append([j_start, j_end])
                        flag_new= False
            if flag_new:  # append position when edge is line
                line_pos.append([j_start, j_end])
                flag_new= False
               
            all_line_pos.append(line_pos)

        # decide available area via line
        # try 1, forward direction
        found_available=False
        for i,line_pos in enumerate (all_line_pos):
            if len(line_pos) == 2:   # x between two lines  is available
                ixs[i,0]= line_pos[0][1]+1
                ixs[i,1]= line_pos[1][0]-1
                found_available=True
            elif len(line_pos) == 0:  # no line, whole x is  not available
                ixs[i,0]= 0
                ixs[i,1]= 0
            elif len(line_pos) > 2:
                """
            　    課題：
            　    間隔が細かく線が2本より多く見えるときは、検出に失敗することがある。
            　    その場合は、細かい間隔の囲い線の部分を　１本の（太い）線にするように　BMP画像を書き直すこと。
                
                """
                if i > 0 and ixs[i-1,0] >= 0:
                    # When there are many available areas, choice only longest ovelap area to previous.
                    overlap_width= np.zeros(len(line_pos)-1)
                    for m in range( len(line_pos)-1 ):
                        overlap_width[m] = min(ixs[i-1,1], line_pos[m+1][0]-1) - max(ixs[i-1,0], line_pos[m][1]+1)
                    
                    if np.amax(overlap_width) >= 3:  # when there is  3 and over overlap
                        ixs[i,0]= line_pos[ np.argmax(overlap_width)][1]+1
                        ixs[i,1]= line_pos[ np.argmax(overlap_width)+1][0]-1
                        found_available=True
                    else: # when there is no overlpa
                        ixs[i,0]= 0
                        ixs[i,1]= 0
            
            elif len(line_pos) == 1:
                if i > 0 and ixs[i-1,0] >= 0:  # check if previous was checked
                    if ixs[i-1,0]==0 and ixs[i-1,1]==0:
                        # there is no available area at previous after once available was found
                        # then set to be no available at current
                        if found_available:
                            ixs[i,0]= 0
                            ixs[i,1]= 0
                    else:
                        if (min(ixs[i-1,1], line_pos[0][0]-1) - max(ixs[i-1,0], 0)) > 0: 
                            # when there is overlap left side
                            ixs[i,0]= 0
                            ixs[i,1]= line_pos[0][0]-1
                        elif (min(ixs[i-1,1], iy-1) - max(ixs[i-1,0],line_pos[0][1]+1 )) > 0: 
                            # when there is overlap right side
                            ixs[i,0]= line_pos[0][1]+1
                            ixs[i,1]= iy -1
                        else:
                            # there is no overlap
                            # then set to be no available at present i
                            ixs[i,0]= 0
                            ixs[i,1]= 0
                            
            if show_progress:
                print ('foward', i, line_pos, ixs[i])
           
        # try 2, backward direction
        for i in range ( len(all_line_pos)-2, -1, -1):
            if ixs[i,0] < 0:
                # only check about forward direction was fault
                line_pos= all_line_pos[i]
                if len(line_pos) > 2:
                    if ixs[i+1,0] >= 0:
                        # When there are many available areas, choice only longest ovelap area to previous.
                        overlap_width= np.zeros(len(line_pos)-1)
                        for m in range( len(line_pos)-1 ):
                            overlap_width[m] = min(ixs[i+1,1], line_pos[m+1][0]-1) - max(ixs[i+1,0], line_pos[m][1]+1)
                        
                        if np.amax(overlap_width)  >= 3:  # when there is  3 and over overlap
                            ixs[i,0]= line_pos[ np.argmax(overlap_width)][1]+1
                            ixs[i,1]= line_pos[ np.argmax(overlap_width)+1][0]-1
                        else: # when there is no overlap
                            ixs[i,0]= 0
                            ixs[i,1]= 0
                
                elif len(line_pos) == 1:

                    if ixs[i+1,0] >= 0:  # check if previous was checked
                        if ixs[i+1,0]==0 and ixs[i+1,1]==0:
                            # there is no available area at previous
                            # then set to be no available at current
                            ixs[i,0]= 0
                            ixs[i,1]= 0
                        else:
                            if (min(ixs[i+1,1], line_pos[0][0]-1) - max(ixs[i+1,0], 0)) > 0: 
                                # when there is overlap left side
                                ixs[i,0]= 0
                                ixs[i,1]= line_pos[0][0]-1
                            elif (min(ixs[i+1,1], iy-1) - max(ixs[i+1,0],line_pos[0][1]+1 )) > 0: 
                                # when there is overlap right side
                                ixs[i,0]= line_pos[0][1]+1
                                ixs[i,1]= iy -1
                            else:
                                # there is no overlap
                                # then set to be no available at present i
                                ixs[i,0]= 0
                                ixs[i,1]= 0
                                
                if show_progress:
                    print ('backward', i, line_pos, ixs[i])
        
        self.unmask = ixs.copy() # available area per x-aix,
        
        #---------------------------------------------------------------
        # black out without available area
        count0=0
        for i,line_pos in enumerate (all_line_pos):
            if show_progress:
                print (i, line_pos, ixs[i])
            
            if ixs[i][0] == ixs[i][1]:  # there is no available
                self.fig_image2[:,i,:]=0
            else:
                self.fig_image2[ 0:ixs[i][0], i  ,:]=0
                self.fig_image2[ ixs[i][1]+1:ix, i  ,:]=0
                count0 +=1
        #--------------------------------------------------------------
        # make xy point data
        count2= int(count0 * 2) 
        self.xys= np.zeros( [count2 ,2])  # available area per x-aix, initial value -1
        c0=0
        for i,line_pos in enumerate (all_line_pos):
            if ixs[i][0] == ixs[i][1]:  # there is no available
                pass
            else:
                self.xys[c0][0]= i # x
                self.xys[c0][1]= ixs[i][0]  # y
                self.xys[count2 -1 - c0][0]= i # x
                self.xys[count2 -1 - c0][1]=ixs[i][1]  # y
                c0+=1
        
    def add_patch1(self,xy):
        #
        # xy is a numpy array with shape Nx2.
        self.ax2.add_patch( patches.Polygon( xy, linewidth=1, edgecolor='green', fill=None)) 
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        
    def save_temp(self,):
        self.temp_file= os.path.splitext(self.file_path)[0]+ '_temp' + '.bmp'
        self.unmask_file= os.path.splitext(self.file_path)[0] + '_temp' + '.npy'
        
        mpimg.imsave(self.temp_file, self.fig_image2)
        print ('save  ', self.temp_file)
        np.save(self.unmask_file, self.unmask)
        print ('save  ', self.unmask_file)

def get_mask_image_files( dir_path, ext='*_MASK_[0-9].bmp'):  
    # get mask file list
    # mask define file name format is ...._MASK_digit.bmp 
    return glob.glob( dir_path +  ext)


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='load mask define BMP file and make mask on template image ')
    parser.add_argument('--jpg_dir', '-w', default='./result_out', help='output directory')
    args = parser.parse_args()
    
    path0= args.jpg_dir + '/'
    
    #  save BMP via "PAINT".  JPG will become out of gray scale data. 
    template_files_list= get_mask_image_files( path0)
    print ('number of files', len(template_files_list))
    
    
    for i,f in enumerate(template_files_list): # template_files_list[0]]):
        # create
        img1= Class_Show2(f)
        # save unmask fig and ixs
        img1.save_temp()
        


   