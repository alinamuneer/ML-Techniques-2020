import matplotlib.image as mpimg
from PIL import Image
import os
import glob
import numpy as np
from numpy.linalg import inv as inv
from numpy.lib import scimath as SM



def load_image(filename):
    im=Image.open(filename) #one image to feature extraction
    #loading image
    #im = Image.open(image_path,'r')
    #getting RGB values of each pixel for that image and storing in 2d array
    pix_val = list(im.getdata())
    feature_vector=[np.array(RGB_vectors) for RGB_vectors in pix_val ]
    #shows total_red=total_green=total_blue value of RBG_vector array 24*24*1 of image 
    count = [(count) for count,RGB_vectors in enumerate(pix_val, 1)][-1]
    features=feature_extraction_function(feature_vector,count)
    return np.mat(features)



def feature_extraction_function(feature_vector,count):
    #sum vector 1*3 of RGB values divided by total values of red=green=blue gives us avg of R,G,B 
    feature_avg= (sum(feature_vector))/count
    #minimum value of RED
    red_min=min(a for [a,b,c] in feature_vector)
    #minimum value of GREEN
    green_min=min(b for [a,b,c] in feature_vector)
    #minimum value of BLUE
    blue_min=min(c for [a,b,c] in feature_vector)
    return [[feature_avg[0]],[feature_avg[1]],[feature_avg[2]],[red_min],[green_min],[blue_min]]
    
    
 
def images_into_features_conversion_function():
    positive_features_list = []
    negative_features_list=[]
    negative_examples=0
    for filename in glob.glob('positives/*.png'): #loading all images with for loop from a folder
        negative_examples+=1
        features=load_image(filename)
        positive_features_list.append(features)
    positive_examples=0    
    for filename in glob.glob('negatives/*.png'): #loading all images with for loop from a folder
        positive_examples+=1    
        features=load_image(filename)
        negative_features_list.append(features) 
    total_examples=positive_examples+negative_examples
    
    return np.array(positive_features_list),np.array(negative_features_list),positive_examples,negative_examples,total_examples


def vec_to_matrix_function(feature,mean_vector):
    
    temp_vector=np.mat(feature-mean_vector)
    temp_matrix=np.mat(np.matmul(temp_vector,(np.transpose(temp_vector))))
    
    return temp_matrix

    
def GDA_parameters_function(positive_features_list,negative_features_list,positive_examples,negative_examples,total_examples):
    final_matrix=[]
    phi=positive_examples/total_examples
    mean0_negative_vector=np.mat((sum(negative_features_list))/negative_examples)
    mean1_positive_vector=np.mat((sum(positive_features_list))/positive_examples)
    #finding combined covariance of both positive and negavtive examples together 
    for positive_feature in positive_features_list:
        final_matrix.append(vec_to_matrix_function(positive_feature,mean1_positive_vector))

        
    for negative_feature in negative_features_list:
        final_matrix.append(vec_to_matrix_function(negative_feature,mean0_negative_vector))    
    
    covariance_matrix=np.mat((sum(final_matrix))/total_examples)

    return phi,mean0_negative_vector,mean1_positive_vector,covariance_matrix

def prob_x_for_y_function(testing_features,mean_vector,covariance_matrix):
    prob_x_for_y = (1/((pow((2*np.pi),(len(testing_features)/2)))*(SM.sqrt(np.linalg.det(covariance_matrix)))))*(np.exp(-0.5*float((np.matmul(np.matmul((np.transpose(testing_features-mean_vector)),(inv(covariance_matrix))),(testing_features-mean_vector))))))
    return prob_x_for_y
    
def testing(test_image_path,mean0_negative_vector,mean1_positive_vector,covariance_matrix,phi):
    testing_features=load_image(test_image_path) #take testing image feature vector x
    #calculate p(x|y=0) for x is testing example
    prob_x_for_y_negative=prob_x_for_y_function(testing_features,mean0_negative_vector,covariance_matrix)
    prob_x_for_y_positive=prob_x_for_y_function(testing_features,mean1_positive_vector,covariance_matrix)
    prob_y_negative=1-phi
    prob_y_positive=phi
    
    prob_y_negative_for_x = (prob_x_for_y_negative*prob_y_negative)/((prob_x_for_y_negative*prob_y_negative)+(prob_x_for_y_positive*prob_y_positive))
    prob_y_positive_for_x = (prob_x_for_y_positive*prob_y_positive)/((prob_x_for_y_negative*prob_y_negative)+(prob_x_for_y_positive*prob_y_positive))
    
    if prob_y_positive_for_x > prob_y_negative_for_x:
        result=1
    elif prob_y_positive_for_x < prob_y_negative_for_x:
        result=0

    
    return result
    
    
    
    
    
#TRAINING:
    
#extract features from image
positive_features_list,negative_features_list,positive_examples,negative_examples,total_examples=images_into_features_conversion_function()

#let's find phi,mean0_vector,mean1_vector,covariance_matrix
phi,mean0_negative_vector,mean1_positive_vector,covariance_matrix=GDA_parameters_function(positive_features_list,negative_features_list,positive_examples,negative_examples,total_examples)
    

#TESTING:

result=[]
for filename in glob.glob('positives/*.png'): #loading all images with for loop from a folder

    prediction=testing(filename,mean0_negative_vector,mean1_positive_vector,covariance_matrix,phi)
    if prediction==1:
        result.append(1)
    else:
        result.append(0)
        print('error occured in ' + str(filename)) 
    
    
   
for filename in glob.glob('negatives/*.png'): #loading all images with for loop from a folder
    
    prediction=testing(filename,mean0_negative_vector,mean1_positive_vector,covariance_matrix,phi)
    if prediction==0:
        result.append(1)
    else:
        result.append(0)
        print('error occured in ' + str(filename)) 


    
    