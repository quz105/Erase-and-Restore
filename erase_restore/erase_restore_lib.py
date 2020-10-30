import numpy as np
import pandas as pd
import os
import random
import shutil
import pickle
import time
import keras
import copy
import cv2

from skimage import img_as_ubyte
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from sklearn.decomposition import PCA

# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')

# Global variables
# num_classes = 1000
try_times = 11

pca = PCA(n_components=10, whiten=True)

#Random mask generation
#There exist various ways to create random mask, what we show here is one sample function.
def get_single_mask(num):
    x_b = np.zeros((224,224))

    for i in range(num):
        x_axis = random.randint(1, 219)
        y_axis = random.randint(1, 218)
        case_radom = random.randint(1, 5)
        
        if (case_radom == 1) :
            x_b[x_axis][y_axis] = 1
            x_b[x_axis][y_axis+1] = 1
            x_b[x_axis][y_axis+2] = 1
            x_b[x_axis][y_axis+3] = 1
            x_b[x_axis][y_axis+4] = 1
            x_b[x_axis+1][y_axis] = 1
            x_b[x_axis+1][y_axis+1] = 1
            x_b[x_axis+1][y_axis+2] = 1
            x_b[x_axis+1][y_axis+3] = 1
            x_b[x_axis+1][y_axis+4] = 1 
        
        elif (case_radom == 2) :
            x_b[x_axis][y_axis] = 1
            x_b[x_axis][y_axis+1] = 1
            x_b[x_axis][y_axis+2] = 1
            x_b[x_axis+1][y_axis] = 1
            x_b[x_axis+1][y_axis+1] = 1
            x_b[x_axis+1][y_axis+2] = 1
            x_b[x_axis+1][y_axis+3] = 1
            x_b[x_axis+1][y_axis+4] = 1
            x_b[x_axis+2][y_axis+3] = 1
            x_b[x_axis+2][y_axis+4] = 1
        
        elif (case_radom == 3) :
            x_b[x_axis][y_axis] = 1
            x_b[x_axis][y_axis+1] = 1
            x_b[x_axis+1][y_axis] = 1
            x_b[x_axis+1][y_axis+1] = 1
            x_b[x_axis+1][y_axis+2] = 1
            x_b[x_axis+1][y_axis+3] = 1
            x_b[x_axis+1][y_axis+4] = 1
            x_b[x_axis+2][y_axis+2] = 1
            x_b[x_axis+2][y_axis+3] = 1
            x_b[x_axis+2][y_axis+4] = 1
        
        elif (case_radom == 4) :
            x_b[x_axis][y_axis] = 1
            x_b[x_axis+1][y_axis] = 1
            x_b[x_axis+1][y_axis+1] = 1
            x_b[x_axis+1][y_axis+2] = 1
            x_b[x_axis+1][y_axis+3] = 1
            x_b[x_axis+1][y_axis+4] = 1
            x_b[x_axis+2][y_axis+1] = 1
            x_b[x_axis+2][y_axis+2] = 1
            x_b[x_axis+2][y_axis+3] = 1
            x_b[x_axis+2][y_axis+4] = 1
        
        elif (case_radom == 5) :
            x_b[x_axis][y_axis] = 1
            x_b[x_axis][y_axis+1] = 1
            x_b[x_axis][y_axis+2] = 1
            x_b[x_axis][y_axis+3] = 1
            x_b[x_axis+1][y_axis] = 1
            x_b[x_axis+1][y_axis+1] = 1
            x_b[x_axis+1][y_axis+2] = 1
            x_b[x_axis+1][y_axis+3] = 1
            x_b[x_axis+1][y_axis+4] = 1
            x_b[x_axis+2][y_axis+4] = 1
    
    x_b_msk = img_as_ubyte(x_b)
    return x_b_msk


#To improve the efficiency of our implementation, we do not generate each mask in realtime. 
#Instead, we generate a group of masks in advance.
def get_masks(num):

    mask = np.expand_dims(get_single_mask(num), axis = 0)
    for k in range(try_times-1):
        mask2 = np.expand_dims(get_single_mask(num), axis = 0)
        mask = np.concatenate((mask, mask2))
    
    # print(mask.shape) ->  (11, 224, 224)
    return mask

# The array mask will be used in the erase_and_restore function
# The parameter (eg. 300) should be arranged accordingly.
mask = get_masks(300)

#This function reads image one by one, then carries out 'erase and restore' on each.
#The prediction result is saved as a single pickle file.
#The directory names should be arranged according to your environment configuration!
def erase_and_restore(begin_index, end_index):
    
    for index in range(begin_index, end_index):
        index_s = '{:0>8d}'.format(index)
        try:
            with open('/adv/adv_imgNet/success_adv_img_resnet50_'+index_s+'.pkl', 'rb') as handler:
                adv_test = pickle.load(handler)
        except IOError:
            continue
        else:
            ## Image (benign or adversarial)
            img_adv1 = copy.deepcopy(adv_test)
        
            img_adv1 = preprocess_input(img_adv1)
            x_image_adv1 = np.expand_dims(img_adv1, axis=0)
            yhat_adv1 = kmodel.predict(x_image_adv1)
            pred_item_adv = [np.argmax(yhat_adv1)]
        
            for k in range(try_times):
                img_adv2 = copy.deepcopy(adv_test)
                x_src_adv = np.ubyte(img_adv2)
            
                ## Inpaint
                dst_adv = cv2.inpaint(x_src_adv, mask[k], 3, cv2.INPAINT_TELEA)
                img_adv22 = preprocess_input(dst_adv)
                x_image_adv2 = np.expand_dims(img_adv22, axis=0)
                yhat_adv2 = kmodel.predict(x_image_adv2)

                pred_item_adv.append(np.argmax(yhat_adv2))
                yhat_adv1 = np.concatenate((yhat_adv1, yhat_adv2))

            with open('/adv/adv_imgNet_vecs/adv_img_resnet50_'+index_s+'_vec.pkl','wb') as handler:
                pickle.dump(yhat_adv1, handler)
            with open('/adv/adv_imgNet_argmax/adv_img_resnet50_'+index_s+'_argmax.pkl','wb') as handler:
                pickle.dump(pred_item_adv, handler)


#The prediction result of each image from a NN-based classifier is a real vector, 
#which is saved as a single pickle file in the folder.
#This function reads those vector files one by one, and carry out PCA on each of them.
#The directory names should be arranged according to your environment configuration!
def read_vecs_save_pca(second_index, last_index):

    with open('/adv/adv_imgNet_vecs/adv_img_resnet50_00000001_vec.pkl', 'rb') as handler:
        adv_matrix = pickle.load(handler)

    rd_adv_matrix = pca.fit_transform(adv_matrix)
    rd_adv_matrix = rd_adv_matrix.reshape(1,120)

    for index in range(second_index, last_index):
        index_s = '{:0>8d}'.format(index)
        try:
            with open('/adv/adv_imgNet_vecs/adv_img_resnet50_'+index_s+'_vec.pkl', 'rb') as handler:
                adv_matrix = pickle.load(handler)
        except IOError:
            continue
        else:
            rd_adv_matrix_tmp = pca.fit_transform(adv_matrix)
            rd_adv_matrix_tmp = rd_adv_matrix_tmp.reshape(1,120)
        
            rd_adv_matrix = np.concatenate((rd_adv_matrix, rd_adv_matrix_tmp))

    #print(rd_adv_matrix.shape)
    with open('rd_by_pca_adv_matrix_120_5000.pkl','wb') as handler:
        pickle.dump(rd_adv_matrix, handler)

