import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


# Loading Dataset
dataset = pd.read_csv('Dataset_1_Team_41.csv')
column = ['X_1', 'X_2', 'Class_value']
mean_file  = open("mean_file.txt","a+")
varaince_file = open("varaince_file.txt","w+")


# #Plotting data points
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# z = dataset['Class_label']
# x1 = dataset['x_1']
# x2 = dataset['x_2']
# ax.scatter(x1, x2, z, c='r', marker='o')
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Y')
# plt.show()

# Number of class in the dataset
Data_class = dataset.iloc[:, -1].unique()  # classes in dataset
Data_class = np.sort(Data_class)
no_of_class = len(Data_class)  # number of class
no_of_feature = dataset.shape[1]-1  # -1 as last column is outcome

# Number of elemets in same class and minimum elements in the calss
class_sample_count = pd.DataFrame(dataset.Class_label.value_counts())
# min_data_in_class = class_sample_count.min() #Class of each class is not same


#####################################################
# Dataset sepearation of class based on training set#
#####################################################

def dataset_seperation(dataset, Data_class):
    '''
    Seperation of dataset based on class value 
    '''
    class_dataset = pd.DataFrame([])
    for i in range(len(dataset)):
        if (dataset.iloc[:, -1][i] == Data_class):
            class_dataset = class_dataset.append(dataset.iloc[i, :])
    return class_dataset

########################
# Splitting the dataset#
########################
def split_dataframe(df, p=[0.65, 0.35]):
    '''
    It will split the dataset into training, validation and testing set based on the given fractin value
    of p. By default p=[0.7,0.15,0.15]
    '''

    train,test = np.split(df.sample(frac=1,replace=False,random_state = 41), [int(p[0]*len(df))])  # split by [0-.7,.7-.85,.85-1]
    return train,test

training_set ,testing_set = split_dataframe(dataset)

####################
# Intial Conditions#
####################
ita = 0.01 #learning rate
no_of_iterarion = 0

##Class wise intail weights 
Intial_weights= []
Intial_wei = []
for i in range(no_of_class):
    Intial_wei = np.linspace(i+3,i+7,no_of_feature+1) #INTIAL WEIGHTS DEFINED
    Intial_weights.append(Intial_wei) #Intial_weights Array 

def sigmoid(feature_vector,weights):
    '''
    It calculates the sigmoid output y(k) of feature vector
    '''
    t = np.dot(feature_vector.T, weights)
    


print('Pragnesh Work!')



