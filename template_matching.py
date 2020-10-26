#coding:utf-8

#  Detect parts by template matching with mask method. 
#  
#  detection is based on
#  (a)local peaks on template matching score heat map
#  (b)voice/instrument-sound quality, synchronize change along temporal axis
#  
#  テンプレートとマスクデータを読み込んで
#  テンプレートマッチングを計算する *For文を使つかっているので計算時間が掛かる
#  テンプレートマッチングのスコアのヒートマップから(極大値)ピーク位置を見つける
#  最大スコアを基準にして、それと時間軸上、同じ様に変換するものを選ぶ。
#　マッチングした部分を表示する


import sys
import os
import argparse
import pathlib
import glob


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg



from peak_det import *

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.18.4
#  matplotlib  3.3.1



def template_matching_mask(base_img, temp_img, unmask, ZeroMean=True):
    # template matching with mask
    #  a) Normalized Cross-Correlation 正規化相互相関数　1～0    TM_CCORR_NORMED
    #  b) Zero-mean Normalized Cross-Correlation 平均値を引いた後に正規化相互相関数（相関係数） TM_CCOEFF_NORMED
    #                                            正規化相互相関数（相関係数)範囲の1～ -1を１～0に変換して出力する。
    
    # set only one element due to gray sclae fig
    base= np.array(base_img[:,:,0],dtype=np.float)
    temp= np.array(temp_img[:,:,0],dtype=np.float)
    y1y2= np.array( unmask, dtype=np.int)
    
    hy,wx= base.shape
    iy,ix= temp.shape
    # check dimension
    if y1y2.shape[0] != ix:
        print ('error: missmatch ix')
    # check sum value
    temp_val2=0.0
    temp_val1=0.0
    temp_val0=0.0
    for l in range(ix):
        y1= y1y2[l][0]
        y2= y1y2[l][1]
        if y1 != y2: #there is available
            temp2= temp[y1:y2+1,l]
            temp_val2 += np.sum( temp2 **2 )
            temp_val1 += np.sum( temp2 )
            temp_val0 += len(temp2)
    if np.sqrt(temp_val2) != np.sqrt(np.sum(temp ** 2)):
        print ('warning: compute template sqrt sum value is not same')
    
    if ZeroMean:
        # compute temp mean  and (temp-mean) **2 again
        temp_mean= temp_val1/temp_val0
        temp_val2=0.0
        for l in range(ix):
            y1= y1y2[l][0]
            y2= y1y2[l][1]
            if y1 != y2: #if there is available
                temp2= temp[y1:y2+1,l]- temp_mean
                temp_val2 += np.sum( temp2 **2 )
    else:
        temp_mean=0.0
    
    #
    score = np.zeros((hy-iy, wx-ix))
    
    # search y-axis
    for i in range(hy-iy):
        # serach x-axis
        for j in range( wx-ix):
            if ZeroMean:
                # compute mean
                base_val1=0.0
                base_val0=0.0
                for l in range(ix):
                    y1= y1y2[l][0]
                    y2= y1y2[l][1]
                    if y1 != y2: #if there is available
                        base2= base[i+y1:i+y2+1,j+l]
                        base_val1 += np.sum(base2)
                        base_val0 += len(base2)
                        base_mean= base_val1 / base_val0
            else:
                base_mean=0.0
            # -- end of if ZerosMean --
        	
            # comupte per x-axis
            base_val2=0.0
            cc_val=0.0
            for l in range(ix):
            	#
                y1= y1y2[l][0]
                y2= y1y2[l][1]
                if y1 != y2: #if there is available
                    base2= base[i+y1:i+y2+1,j+l] - base_mean
                    temp2= temp[y1:y2+1,l] - temp_mean
                    cc_val += np.sum(base2 * temp2)
                    base_val2 += np.sum( base2 **2 )
                    
            normalize_val= np.sqrt( base_val2 * temp_val2)
            if normalize_val > 0.:
                score[i,j]= cc_val / normalize_val
            
            if 1: # print out every scroe value
                print ( i,j,score[i,j], end='\r', flush=True)
                
    #print (score)
    
    if ZeroMean:
        # 正規化相互相関数（相関係数)範囲の1～ -1を１～0に変換して出力する。
        # Change score range from 1..-1 to 1..0 if ZeroMean is True
        score = (score + 1.0)/2.0
        print ('score range was changed to 1..0 from 1..-1')
    
    # (1) get max arg
    pos = np.unravel_index(score.argmax(), score.shape)
    print ('max position',  pos, score[ pos[0], pos[1]] )
    """
    # (2) sort
    score_sorted=np.sort(np.ravel(score))[::-1]
    c=np.argsort(np.ravel(score))[::-1]
    pos_sorted=np.stack( [c// score.shape[1], c% score.shape[1]]).T
    for i in range(len(score_sorted)):
        print (i, score_sorted[i], pos_sorted[i])
    """
    
    return score, pos


def get_pos_in_accept_range(score, pos, peaks_index, filter_size, accept_x_range_ratio, ignore_y_edge=True):
    # get position list in accept x-axis range around maximum position
    peaks_index_accepted=[]  # position list in accept range around maximum position
    peaks_index_accepted_wo_maxpos=[]  # position list without maximum position
    x_max_pos= pos[1]  # maximum position x
    y_max_pos= pos[0]  # maximum position y
    y_edge_upper=0     # y_edge
    y_edge_lower= score.shape[0]  # y_edge
    # compute accept_x_range  score width * accept_x_range_ratio, Or filter_size
    accept_x_range = int( max([score.shape[1] * accept_x_range_ratio, filter_size]))
    print('accept_x_range ', accept_x_range)
    for i in range( len(peaks_index[0])):
        y=peaks_index[0][i]
        x=peaks_index[1][i]
        #print (x,y, score[y,x])
        
        # ignore y edge position if ignore_y_edge is True
        if ignore_y_edge and ( y == y_edge_upper or y == y_edge_lower):
            continue
        # get only position in accept range around maximum position
        if abs( x - x_max_pos) <= accept_x_range:
            peaks_index_accepted.append([y,x])
    #print(peaks_index_accepted)
    
    for v in peaks_index_accepted:
        if abs(v[1] - x_max_pos) < filter_size  and  abs(v[0] - y_max_pos) < filter_size:
            pass
        else:
            peaks_index_accepted_wo_maxpos.append(v)
            print ( v[0],v[1], score[v[0],v[1]] )
            
    #print(peaks_index_accepted_wo_maxpos)
    
    return peaks_index_accepted_wo_maxpos, peaks_index_accepted



#----------------------------------------------------------------------------------
# CV2 のtemplate_matching のmaskを使ったものを追加。

import cv2  # version > 4.4
#  opencv-python (4.4.0.44)

def template_matching_mask_cv2(base_img, temp_img, unmask, ZeroMean=False):
    # template matching with mask
    #  a) Normalized Cross-Correlation 正規化相互相関数　1～0    TM_CCORR_NORMED　計算時間は早い。
    #  b) Zero-mean Normalized Cross-Correlation 平均値を引いた後に正規化相互相関数（相関係数） TM_CCOEFF_NORMED
    #     When opencv-python version was 3.4.6,  error: (-213)  in function cv::matchTemplateMask
    base= np.array(base_img,dtype=np.float32) / 255.
    temp= np.array(temp_img,dtype=np.float32) / 255.
    y1y2= np.array( unmask, dtype=np.int)
    mask= np.zeros( temp.shape, dtype=np.float32)
    iy,ix,_= temp.shape
    for l in range(ix):
        y1= y1y2[l][0]
        y2= y1y2[l][1]
        if y1 != y2: #there is available
            mask[y1:y2+1,l,:]=1.0
    
    if ZeroMean:
        score = cv2.matchTemplate(base, temp, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        score = cv2.matchTemplate(base, temp, cv2.TM_CCORR_NORMED, mask=mask)
    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(score)
    pos= [maxLoc[1], maxLoc[0]]
    print ('score.shape', score.shape)
    print ('max position',  pos , score[ pos[0], pos[1]] )
    
    if ZeroMean:
        # 正規化相互相関数（相関係数)範囲の1～ -1を１～0に変換して出力する。
        # Change score range from 1..-1 to 1..0 if ZeroMean is True
        score = (score + 1.0)/2.0
        print ('score range was changed to 1..0 from 1..-1')
    
    return score, pos
#----------------------------------------------------------------------------------


class Class_Show3(object):
    def __init__(self, file_path, temp_path, filter_size=3, accept_x_range_ratio=0.05):
        #
        self.file_path= file_path
        print(self.file_path)  # print file name
        self.temp_path= temp_path[0]
        print(self.temp_path) 
        self.unmask_npy_path= temp_path[1]
        print(self.unmask_npy_path)
        self.result_path= os.path.splitext(temp_path[0])[0] + '_template_matching_with_mask.png'
        # get base image
        self.base_img= mpimg.imread( self.file_path)
        print ('base_img.shape', self.base_img.shape)
        # get template
        self.temp_img= mpimg.imread( self.temp_path)
        print ('temp_img.shape', self.temp_img.shape)
        # get unmask npy 
        self.unmask = np.load(self.unmask_npy_path)
        print (self.unmask.shape)
        
        # template matching with mask
        ##self.score, self.pos= template_matching_mask(self.base_img, self.temp_img, self.unmask, ZeroMean=True)
        self.score, self.pos= template_matching_mask_cv2(self.base_img, self.temp_img, self.unmask, ZeroMean=True): #False)
        # get local maximum postions
        self.filter_size= filter_size
        self.peaks_index= detect_peaks(self.score, filter_size=self.filter_size, order=0.6) # adjust order 0.5? 0.6? 0.7? as effective peak theshold ratio to maximum peak
        # get positions in accept range around maximum position
        self.accept_x_range_ratio= accept_x_range_ratio
        self.peaks_index_accepted_wo,_ = get_pos_in_accept_range(self.score, self.pos, self.peaks_index, self.filter_size, self.accept_x_range_ratio)
        
        # start to draw whole image
        self.make_xy()   # make template patch
        self.plot_image()
    
    
    def plot_image(self, rt_without_plotshow=False):
        
        self.fig_image= self.base_img.copy()
        
    	#
        fig, (ax1, ax2) = plt.subplots(1,2)
        self.fig= fig
        self.ax1=ax1
        self.ax2=ax2
        
        #ax.set_axis_off()
        
        self.img0= ax1.imshow( self.fig_image, aspect='equal', origin='upper')
        
        
        heatmap = ax2.pcolor(self.score, cmap=plt.cm.Reds)
        ax2.set_title('heat map of score, local maximum positions', fontsize=8)
        ax2.set_ylim(self.score.shape[0],0)
        
        # show local maximum positions
        ax2.scatter( self.peaks_index[1], self.peaks_index[0], color='black')
        # show argmax position and patch as green
        ax2.scatter( self.pos[1], self.pos[0], color='green')
        self.add_patch1(self.pos[1], self.pos[0], color='green')
        # show other position in accept range
        for v in self.peaks_index_accepted_wo:
            x= v[1]  # x
            y= v[0]  # y
            ax2.scatter( x, y, color='blue')
            self.add_patch1( x, y, color='blue')
        
        plt.tight_layout()
        
        plt.savefig(self.result_path)
        print ('save to ',  self.result_path)
        
        if not rt_without_plotshow:
            plt.show()
    
    
    
    def make_xy(self,):
        # make xy point data
        count0=0
        for i in range( self.unmask.shape[0]):
            #y1= self.unmask[i][0]
            #y2= self.unmask[i][1]
            #print ( y1,y2)
            if self.unmask[i][0] ==  self.unmask[i][1]:  # there is no available
                pass
            else:
                count0 +=1
        
        count2= int(count0 * 2) 
        self.xys= np.zeros( [count2 ,2])  # available area per x-aix, initial value -1
        c0=0
        for i in range ( self.unmask.shape[0]):
            if  self.unmask[i][0] ==  self.unmask[i][1]:  # there is no available
                pass
            else:
                self.xys[c0][0]= i # x
                self.xys[c0][1]= self.unmask[i][0]  # y
                self.xys[count2 -1 - c0][0]= i # x
                self.xys[count2 -1 - c0][1]=self.unmask[i][1]  # y
                c0+=1
        
        
    def add_patch1(self,p_x,p_y, color=None):
        # 
        xy = self.xys + [ p_x, p_y]
        if color is not None:
            edgecolor= color
        else:
            edgecolor='red'
        # xy is a numpy array with shape Nx2.
        self.ax1.add_patch( patches.Polygon( xy, linewidth=1, edgecolor=edgecolor, fill=None)) 
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

def get_jpg_files( dir_path ):
    # get jpg file list
    return glob.glob( dir_path + "*.jpg")
    

def get_temp_files( dir_path):
    # get template file and unmask npy file
    flist= glob.glob( os.path.splitext(dir_path)[0]+ '*_temp' + '.bmp')
    temp_list=[]
    for f in flist:
        n = os.path.splitext(f)[0] + '.npy'
        if os.path.exists( n ):
            temp_list.append([f,n] )
    
    return temp_list


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='template matching with mask')
    parser.add_argument('--jpg_dir', '-w', default='./result_out', help='image, template, and mask directory')
    args = parser.parse_args()
    
    path0= args.jpg_dir + '/'
    
    #
    jpg_files_list= get_jpg_files(path0)
    print ('number of jpg files', len(jpg_files_list))
    #print (jpg_files_list)
    
    
    for i,f in enumerate(jpg_files_list): #[jpg_files_list[0]]): 
        # get template list for f 
        temp_list = get_temp_files(f)
        
        for j, t in enumerate(temp_list): # [temp_list[0]]): 
            img1= Class_Show3(f, t)
        


   