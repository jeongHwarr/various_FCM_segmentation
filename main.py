import os
import argparse

from utils import makedirs
import FCM, EnFCM, MFCM

WORKSPACE = os.path.split(os.getcwd())[-1] # current workspace 
IMG_PATH = os.path.join(os.getcwd(),"img") # path for input image directory
OUTPUT_PATH = os.path.join(os.getcwd(),"output") # path for output directory
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_PATH,'segmentation') # path for output (plot directory)
OUTPUT_FILT_IMG_PATH = os.path.join(OUTPUT_PATH,'filtered_img')
    
def get_args():
    parser = argparse.ArgumentParser(description="Tissue Segmentation Using Modified Fuzzy C-Means Algorithm on Mammography.")
    #-----------------Algorithm-----------------
    parser.add_argument('-a', '--algorithm', default='FCM', choices=['FCM', 'EnFCM', 'MFCM'], type=str,
                         help="Choose a fuzzy c-means clustering algorithm. (FCM, EnFCM, MFCM)")
    parser.add_argument('--num_bit', default=8, type=int,
                         help="number of bits of input images")
    #-----------------Fundamental parameters-----------------
    parser.add_argument('-c', '--num_cluster', default='4', type=int,
                         help="Number of cluster")
    parser.add_argument('-m', '--fuzziness', default='2', type=int,
                         help="fuzziness degree")
    parser.add_argument('-i', '--max_iteration', default='100', type=int,
                         help="max number of iterations.")
    parser.add_argument('-e', '--epsilon', default='0.05', type=float,
                         help="threshold to check convergence.")
    #-----------------User parameters-----------------
    parser.add_argument('--plot_show', default=1, choices=[0,1],
                         help="Show plot about result")
    parser.add_argument('--plot_save', default=1, choices=[0,1],
                         help="Save plot about result")
     #-----------------Parametesr for MFCM/EnFCM-----------------
    parser.add_argument('-w', '--win_size', default='5', type=int,
                         help="Window size of MFCM/EnFCM algorithm")
    parser.add_argument('-n', '--neighbour_effect', default='3', type=float,
                         help="Effect factor of the graylevel which controls the influence extent of neighbouring pixels.")
    
    args = parser.parse_args() 
    return args

if __name__ == '__main__':
  
    args = get_args()
    algorithm = args.algorithm
    
    # Subfolders are automatically created with the algorithm name.
    OUTPUT_PATH = os.path.join(OUTPUT_PATH,"%s"%(algorithm)) 
    makedirs(OUTPUT_PATH)
    
    DIRECTORY = {}   
    DIRECTORY['WORKSPACE'] = WORKSPACE
    DIRECTORY['IMG_PATH'] = IMG_PATH
    DIRECTORY['OUTPUT_PATH'] = OUTPUT_PATH
    DIRECTORY['OUTPUT_PLOT_PATH'] = OUTPUT_PLOT_PATH
    DIRECTORY['OUTPUT_FILT_IMG_PATH'] = OUTPUT_FILT_IMG_PATH
    
    
    if algorithm =='FCM':
        FCM.main(DIRECTORY, args)
        
    if algorithm=='EnFCM':
       EnFCM.main(DIRECTORY, args)
    
    if algorithm=='MFCM':
        MFCM.main(DIRECTORY, args)