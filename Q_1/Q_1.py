import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


# Loading Dataset
dataset = pd.read_csv('Dataset_4_Team_41.csv')
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
learning_rate = 0.00001 

##Class wise intail weights 
weights = np.linspace(3.5,7.9,no_of_feature) #INTIAL WEIGHTS DEFINED

################
# Batch Graient#
################
def sigmoid_probabilty(feature_matrix,weights):
        '''
        It calculates the sigmoid output y(k) of feature vector
        feature_vector : All feature vector matrix 
        weights : weights vector 
        '''
        t = np.dot(feature_matrix, weights)   #(n*p) (p*1) = (n*1)
        probability = []
        for i in range(len(t)):
                probability.append(1 / (1+np.exp(-t[i])))    #predicted value 
        return probability

def predicted_y(probability):
        prediction = []
        for i in range(len(probability)):
                if(probbility[i] > 0.5):
                        prediction.append(1)
                else:
                        prediction.append(0)
        return prediction


def Graient_descent(weights,actual_y,learning_rate,features):
        m = len(actual_y)
        no_of_iterarion = 0
        accuracy = 0
        while (accuracy != 1):
                predicted_y = sigmoid_probabilty(features,weights)
                weights -= (learning_rate/m) * np.dot(features.T,(predicted_y - actual_y))
                accuracy = accuracy_result(actual_y,predicted_y)
        return weights,no_of_iterarion


def accuracy_result(y_predicted,y_actaul):
        y_predicted = list(y_predicted)
        correct_prediction = 0        
        wrong_prediction = 0
        count = 0
        for i in range(len(y_actaul)):
                count += 1 
                if(y_predicted[i] == y_actaul[i]):
                        correct_prediction += 1
                else:
                        wrong_prediction += 1
                
        accu = correct_prediction / (correct_prediction+wrong_prediction)
        print('accu: ', accu)
        return accu


               
feature_matrix = training_set.iloc[:,:-1]
y = training_set.iloc[:,-1]
Weight_result , iteration  = Graient_descent(weights,y,learning_rate,feature_matrix)

    


print('Pragnesh Work!')



