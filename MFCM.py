# =============================================================================
# Modified Fuzzy C-means algorithm 
# (Z. Chen and R. Zwiggelaar "A Modified Fuzzy C Means Algorithm for Breast Tissue Density Segmentation in Mammograms."
# IEEE/Information Technology and Applications in Biomedicine (ITAB) 2010.)
# =============================================================================

import sys
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import time
from utils import makedirs

def make_spatial_distance_window(windowSize, center_coordi):    
    center_y = center_coordi[1]
    center_x = center_coordi[0]
    
    distance_window = np.zeros(windowSize)
    for y in range(windowSize[1]):
        for x in range(windowSize[0]):           
            distance_window[y][x] = np.sqrt((y-center_y)**2+(x-center_x)**2) # Euclidean distance
    return distance_window

def find_reliable_set(window, windowSize):
    median = np.median(window)    
    median_window = np.full((windowSize[1],windowSize[0]), median) # shape:(windowSize[1],windowSize[0]), value=median
        
    difference_nei_med = window-median_window # neighbour - med
        
    deviation = np.sqrt((np.mean(pow(difference_nei_med,2))))
            
    reliable_mat = (difference_nei_med<=deviation).astype(int) # 1: reliable, 0: unreliable
    return reliable_mat
         
 
def sliding_window(image, neighbour_effect, stepSize, windowSize):
    '''slide a window across the image'''     
     
    center_coordi = (int(windowSize[1]/2),int(windowSize[0]/2)) # center coordination of the window
    distance_window = make_spatial_distance_window(windowSize, center_coordi) # Euclidean distance matrix between center coordination and neighbor cordination
    
    filtered_image = []
    
    start = time.time()
    for y in range(0, image.shape[0]-int(windowSize[0]/2), stepSize):
        for x in range(0, image.shape[1]-int(windowSize[1]/2), stepSize):            
            cur_window = image[y:y + windowSize[1], x:x + windowSize[0]] # neighbour window 
            
            if cur_window.shape[0]<windowSize[0] or cur_window.shape[1]<windowSize[1]:
                continue
                
            if np.count_nonzero(cur_window)==0: # Pass if all values in window are zero
                filtered_image.append(0)
                continue
            
            #--------------Find reliable set matrix--------------        
            reliable_mat = find_reliable_set(cur_window, windowSize)
      
            #--------------Weighting coefficients about window--------------           
            center_window = np.full((windowSize[1],windowSize[0]), cur_window[center_coordi[1],center_coordi[0]])
            difference_nei_cent = cur_window - center_window
            
            gray_deviation = np.sqrt(np.sum(pow(difference_nei_cent*reliable_mat,2))/np.count_nonzero(reliable_mat))             
            if gray_deviation==0: gray_deviation = sys.float_info.epsilon
            
            coeff_gray = np.exp(-(pow(difference_nei_cent,2)*reliable_mat)/(neighbour_effect*gray_deviation))
            coeff_distance = np.exp(-distance_window*reliable_mat)
            
            coeff_window = coeff_gray*coeff_distance
                   
            #--------------New intensity value of the center point in the window--------------
            new_gray = np.sum(coeff_window*cur_window)/np.sum(coeff_window)               
            filtered_image.append(round(new_gray))
            
    filtered_image = np.array(filtered_image)
    print("Time :", time.time() - start)
    return filtered_image
    

class MFCM():
    def __init__(self, image, image_bit, n_clusters, m, neighbour_effect, epsilon, max_iter, kernel_size):
        '''Modified Fuzzy C-means clustering

        <image>: 2D array, grey scale image.
        <n_clusters>: int, number of clusters/segments to create.
        <m>: float > 1, fuzziness parameter. A large <m> results in smaller
             membership values and fuzzier clusters. Commonly set to 2.
        <kernel_size>: int >= 1, size of neighborhood.
        <neighbour_effect>: float, parameter which controls the influence extent of neighbouring pixels.
        <max_iter>: int, max number of iterations.
        '''

        #-------------------Check inputs-------------------
        if np.ndim(image) != 2:
            raise Exception("<image> needs to be 2D (gray scale image).")
        if n_clusters <= 0 or n_clusters != int(n_clusters):
            raise Exception("<n_clusters> needs to be positive integer.")
        if m < 1:
            raise Exception("<m> needs to be >= 1.")
        if kernel_size <=0 or kernel_size != int(kernel_size):
            raise Exception("<kernel_size> needs to be positive integer.")
        if epsilon <= 0:
            raise Exception("<epsilon> needs to be > 0")

        self.image = image
        self.image_bit = image_bit
        self.n_clusters = n_clusters
        self.m = m
        self.neighbour_effect = neighbour_effect
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.kernel_size = kernel_size

        self.shape = image.shape # image shape
        self.X = image.flatten().astype('float') # flatted image shape: (number of pixels,1) 
        self.numPixels = image.size
        
    def initial_U(self):
        U=np.zeros((self.num_gray, self.n_clusters))
        idx = np.arange(self.num_gray)
        for ii in range(self.n_clusters):
            idxii = idx%self.n_clusters==ii
            U[idxii,ii] = 1        
        return U
    
    def update_U(self):
        '''Compute weights'''
        idx = np.arange(self.num_gray)
        c_mesh,idx_mesh = np.meshgrid(self.C,idx)
        power = -2./(self.m-1)
        numerator = abs(idx_mesh-c_mesh)**power
        denominator = np.sum(abs(idx_mesh-c_mesh)**power,axis=1)
        return numerator/denominator[:,None]

    def update_C(self):
        '''Compute centroid of clusters'''
        idx = np.arange(self.num_gray)
        idx_reshape = idx.reshape(len(idx),1)
        numerator = np.sum(self.histogram*idx_reshape*pow(self.U,self.m),axis=0)
        denominator = np.sum(self.histogram*pow(self.U,self.m),axis=0)
        return numerator/denominator
   
    def get_filtered_image(self):
         # Create padding image
        print("Getting filtered image..(This process can be time consuming.)") 
        pad_size_y = int(self.kernel_size/2)
        pad_size_x = int(self.kernel_size/2)   
        pad_img = cv2.copyMakeBorder(self.image, pad_size_y, pad_size_y, pad_size_x, pad_size_x, cv2.BORDER_CONSTANT, value=0 ) # zero padding
             
        filtered_image = sliding_window(pad_img, self.neighbour_effect, stepSize=1, windowSize=(self.kernel_size,self.kernel_size))
        dtype = self.image.dtype
        self.filtered_image = filtered_image.reshape(self.shape).astype(dtype)
    
    def calculate_histogram(self):     
        hist_max_value = (1 << self.image_bit)
        hist = cv2.calcHist([self.filtered_image],[0],None,[hist_max_value],[0,hist_max_value])
        self.num_gray = len(hist)
        self.histogram = hist
                       
    def form_clusters(self):
        self.get_filtered_image() 
        self.calculate_histogram()
        
        '''Iterative training'''        
        d = 100
        self.U = self.initial_U()
        if self.max_iter != -1:
            i = 0
            while True:             
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" %(i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i+=1
        else:
            i = 0
            while d > self.epsilon:
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" %(i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i+=1
        self.segmentImage()


    def deFuzzify(self):
        return np.argmax(self.U, axis = 1)

    def segmentImage(self):
        '''Segment image based on max weights'''

        result = self.deFuzzify()
        
        self.result = np.array(self.image, copy=True)
        for i in range(len(result)):
            self.result[self.result==i]=result[i]
            
        self.result = self.result.reshape(self.shape).astype('int')

        return self.result
    
    
def main(DIRECTORY, args):
    IMG_PATH = DIRECTORY['IMG_PATH']
    OUTPUT_PATH = DIRECTORY['OUTPUT_PATH']
    OUTPUT_FILT_IMG_PATH = os.path.join(OUTPUT_PATH,'filtered_img') # path for output (filtered image directory)
    OUTPUT_PLOT_PATH = os.path.join(OUTPUT_PATH,'segmentation') # path for output (plot directory)

    IS_PLOT = args.plot_show 
    IS_SAVE = args.plot_save
    
    files = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))] # read all files in IMG_PATH
    
    for file in files:
        target_img_path = os.path.join(IMG_PATH,file)
        try:
            #--------------Lord image file--------------  
            img= cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE) # cf. 8bit image-> 0~255

            #--------------Clustering--------------  
            cluster = MFCM(img, image_bit=args.num_bit, n_clusters=args.num_cluster, m=args.fuzziness, neighbour_effect=args.neighbour_effect, epsilon=args.epsilon, max_iter=args.max_iteration, kernel_size=args.win_size)
            cluster.form_clusters()
            result=cluster.result
                     
            #-------------------Plot and save result------------------------
            if IS_PLOT:      
                
                fig=plt.figure(figsize=(12,8),dpi=100)
            
                ax1=fig.add_subplot(1,2,1)
                ax1.imshow(img,cmap='gray')
                ax1.set_title('image')
            
                ax2=fig.add_subplot(1,2,2)
                ax2.imshow(result)
                ax2.set_title('segmentation')
                
                plt.show(block=False)
                plt.close()
                
            if IS_SAVE:
                makedirs(OUTPUT_PLOT_PATH)             
                seg_result_path = os.path.join(OUTPUT_PLOT_PATH,"%s.png"%(os.path.splitext(file)[0]))
                plt.imshow(result)
                plt.savefig(seg_result_path, dpi=300)
                plt.close()
                
                makedirs(OUTPUT_FILT_IMG_PATH) 
                filtered_img_path = os.path.join(OUTPUT_FILT_IMG_PATH,"%s.png"%(os.path.splitext(file)[0]))
                plt.imshow(cluster.filtered_image,cmap='gray')
                plt.savefig(filtered_img_path, dpi=300)
                plt.close()
            
        except IOError:
            print("Error")

if __name__ == '__main__':
    main()