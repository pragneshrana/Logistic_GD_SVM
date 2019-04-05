import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


# Loading Dataset
dataset = pd.read_csv('Dataset_4_Team_41.csv')

def Normalized_Dataset(dataset):
        '''
        Normalize the dataset 
        Make sure last column is result
        z_i=\frac{x_i-\min(x)}{\max(x)-\min(x)}
        '''
        update_col =[]
        for i in range(dataset.shape[1]-1):
                max = dataset.iloc[:,i].max()
                min = dataset.iloc[:,i].min()
                dataset.iloc[:,i] = ( dataset.iloc[:,i] - min)/(max - min)
        return dataset
dataset = Normalized_Dataset(dataset)

column = ['X_1', 'X_2', 'Class_value']

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
        train = train.reset_index(drop = True) 
        test = test.reset_index(drop = True) 
        return train,test

training_set ,testing_set = split_dataframe(dataset)

####################
# Intial Conditions#
####################
learning_rate = 0.9
no_of_iterarion = 350
Intial_weights = np.array([])

##Class wise intail weights 
for i in range(no_of_class):
        Intial_weights = np.append(Intial_weights,np.linspace(-3*i+21,1*i+58,no_of_feature+1),axis=0) #INTIAL WEIGHTS DEFINED For Two class
Intial_weights = Intial_weights.reshape(no_of_feature+1,no_of_class)
# weights_matrix = np.random.random((N,N))

################
# Batch Graient#
################
def sigmoid_probabilty(feature_matrix,weights):
        '''
        It calculates the sigmoid output y(k) of feature vector
        feature_vector : All feature vector matrix 
        weights : weights vector 
        '''
        t = np.dot(feature_matrix, weights)   #(n*p) (p*1) = (n*1)  or ()
        probability = []

        # if(t.size == 1):
        #         t =np.reshape(t, (1,1))
        for i in range(t.size):
                probability.append(1 / (1+np.exp(-t[i])))    #predicted value 
        return probability

def softmax_probability(feature_matrix,weights):
        '''
        It calculates the softmax output y(k) of feature vector for all class
        feature_vector : All feature vector matrix 
        weights : weights vector 
        '''
        t = np.dot(feature_matrix, weights)   #(n*p) (p*k) = (n*k)  
        exp_t = np.exp(t)       #into exp list
        probability = []        #Final probability of all data classwise
        # if(t.size == 1):
        # t =np.reshape(t, (1,1))
        for i in range(len(exp_t)):
                data_all_class_probability = [] #classwise probability of alll data 
                sum_list_data= 0
                sum_list_data = sum(exp_t[i])   #denomminator sum
                for j in range(len(exp_t[i])):
                        data_all_class_probability.append(exp_t[i][j] / sum_list_data)   #predicted value 
                probability.append(data_all_class_probability)
        return probability      #return probablity of all class for all data 

def one_hot_encoding(y_actual):
        # Number of class in the dataset
        classes = y_actual.unique()  # classes in dataset
        classes = np.sort(classes)
        no_of_class = len(classes)  # number of class
        OHE_Matrix = np.zeros([len(y_actual),no_of_class])
        for i in range(len(y_actual)):
                OHE_Matrix[i,y_actual[i]] = 1
        return OHE_Matrix
        #Evey row represents one hot encoder 
t_nk = one_hot_encoding(training_set.iloc[:,-1])
        

def predicted_y(probability):
        '''
        It will classify the data based on the probability array 
        probability : Array 
        '''
        prediction = []

        # if(t.size == 1):
        #         t =np.reshape(t, (1,1))
        for i in range(len(probability)):
                for j in range(len(probability[i])):
                        maximum = max(probability[i])
                        if(probability[i][j] == maximum):
                                prediction.append(j)
        return prediction


def Gradient_Descent(weights,actual_y,one_hot_encoding,learning_rate,features,no_of_iterarion = 500):
        '''
        It wil run gradient descent till specified no_of_iterarion.
        It calculates the weights and return the weights and no of iterations performed.
        '''
        m = len(actual_y)
        iterated = 0
        # accuracy = 0            #Accuracy initialized 
        # error = 1               #Error intialized
        # error_array = list(np.zeros(2))
        # error_diff = 1         #Error Diff Intialied 
        while (iterated < no_of_iterarion):     #Tolerance
                iterated += 1 
                probability = softmax_probability(features,weights)
                weights-= (learning_rate/m) *np.dot(features.T,(probability-one_hot_encoding))
                prediction =predicted_y(probability)
                accuracy,error = accuracy_result(actual_y,prediction)
        print('error: ', error)
        print('accuracy: ', accuracy)

                # #Error _Diff adjustment 
                # new_error = error
                # last_error  = error_array[0]    #previous Iteration Array
                # error_diff = abs(last_error - new_error)

                # #shifting array from first place to zero 
                # error_array.insert(0,error_array[1])    
                # error_array.insert(1,error)
        print('prediction , actual_y: ', prediction ,actual_y)
        return weights,no_of_iterarion


def accuracy_result(y_predicted,y_actaul):
        '''
        It calculates the Accuracy and Error by given Predicted and actual values.
        accu = correct_prediction / (correct_prediction+wrong_prediction)
        error = wrong_prediction / (correct_prediction+wrong_prediction)
        '''
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
        error = wrong_prediction / (correct_prediction+wrong_prediction)
        return accu,error 

def processable_feature_matrix(feature_matrix):
        '''
        Adds ones in front of matrix
        '''
        ones_array = pd.Series(np.ones(len(feature_matrix)))
        feature_matrix.insert(loc=0, column='Ones', value=ones_array)
        return feature_matrix


####################
# Confusion Matrix #
####################
def confusion_matrix(actual_result, predicted_result):
    '''
    Plots the confusion matrix. For that you have to pass the two array actual result(given classes) and
    Predicted result 
    '''
    import pandas as pd
    import numpy as np
    classes_name = list(set(actual_result))
    no_of_classes = len(classes_name)
    conf_matrix = pd.DataFrame(np.zeros((no_of_classes, no_of_classes)))
    combined_array_dataset = pd.DataFrame()
    combined_array_dataset[0] = actual_result
    combined_array_dataset[1] = predicted_result
    for i in range(no_of_classes):
        subdataset = combined_array_dataset[combined_array_dataset[0]
                                            == classes_name[i]]
        # print('subdataset: ', subdataset)
        val = subdataset[1].value_counts()
        val.sort_index(inplace=True)  # sorting values by index
        # print('val: ', val.shape)
        indexList = val.index
        # print('indexList: ', indexList)

        if (len(indexList) == no_of_classes):
            for j in range(no_of_classes):
                # If there is no mis classification
                if(indexList[j] == j):
                    # print('j: ', j)
                    # print('indexList[j]', indexList[j])
                    try:
                        conf_matrix.iloc[i, j] = val.iloc[j]
                    except IndexError:
                        conf_matrix.iloc[i, j] = 0
                    # print('conf_matrix: ', conf_matrix)
                else:
                    conf_matrix.iloc[i, j] = 0
                    # print('conf_matrix: ', conf_matrix)
        else:
            # In case if class has less misclassification than total class
            m = 0
            for j in range(no_of_classes):
                # If there is no mis classification
                try:
                    if(indexList[m] == j):
                        try:
                            conf_matrix.iloc[i, j] = val.iloc[m]
                        except IndexError:
                            conf_matrix.iloc[i, j] = 0
                        # print('conf_matrix: ', conf_matrix)
                        m += 1
                    else:
                        continue
                except IndexError:
                    continue
    # print('conf_matrix: ', conf_matrix)
    return conf_matrix



#Calcualtion of Training 
feature_matrix_training = training_set.iloc[:,:-1]       #feture matrix of data
feature_matrix_training = processable_feature_matrix(feature_matrix_training)
y_train = training_set.iloc[:,-1]
Weight_result , iteration  = Gradient_Descent(Intial_weights,y_train,t_nk,learning_rate,feature_matrix_training,no_of_iterarion)
print()
#Result Writing to file 
file = open('Result.txt','+a')
file.write('\n \n Weight_result: '+ str(Weight_result))
file.write('\n iteration: '+ str(iteration))

#Testing 
feature_matrix_testing = testing_set.iloc[:,:-1]
feature_matrix_testing = processable_feature_matrix(feature_matrix_testing)
y_actaul_test =testing_set.iloc[:,-1]   #dataset
y_probability = softmax_probability(feature_matrix_testing,Weight_result) #obtained Probability 
y_predicted_test = predicted_y(y_probability)   #Classified Results
test_accu,test_error = accuracy_result(y_predicted_test,y_actaul_test)  
file.write('\n test_error: '+ str(test_error))
file.write('\n test_accu: '+ str(test_accu))

############################################
# Creation of confusion matrix and plotting#
############################################

confu_matrix = confusion_matrix(y_predicted_test,y_actaul_test)
plt.figure(figsize=(10, 7))
sn.heatmap(confu_matrix, annot=True)
plt.title('Confusion Matrix of Test class Dataset-2', fontsize=20)
plt.xlabel('Class', fontsize=18)
plt.ylabel('Class', fontsize=16)
plt.savefig('confusion.png')

file.close()
print('Pragnesh Work!')


############################
# Decision Surface Plotting#
############################
n=50
x1 = list(np.linspace(training_set['x_1'].min(), training_set['x_1'].max(), n))
x2 = list(np.linspace(training_set['x_2'].min(), training_set['x_2'].max(), n))
dataframe = pd.DataFrame([])

#Meshgrid
for i in range(len(x1)):
    for j in range(len(x2)):
        data_vector = [x1[i], x2[j]]
        data_vector = pd.Series(data_vector)
        dataframe = dataframe.append(data_vector, ignore_index=True)

feature_matrix_boundary = processable_feature_matrix(dataframe)
y_probability = softmax_probability(feature_matrix_boundary,Weight_result) #obtained Probability 
y_predicted_boundary = predicted_y(y_probability)   #Classified Results

plt.xlim(training_set['x_1'].min(), training_set['x_1'].max())
plt.ylim(training_set['x_2'].min(), training_set['x_2'].max())
plt.title('Desicion Boundary and Surface for dataset-2', fontsize=20)
plt.xlabel('x_1', fontsize=18)
plt.ylabel('x_2', fontsize=16)
for i in range(len(dataframe)):
        if(y_predicted_boundary[i] == 0):
                plt.scatter(dataframe.iloc[i,1], dataframe.iloc[i,2], marker='*',c='pink')
                continue
        elif(y_predicted_boundary[i] == 1):
                plt.scatter(dataframe.iloc[i,1], dataframe.iloc[i,2], marker='*',c='yellow')
                continue
        else:
                plt.scatter(dataframe.iloc[i,1], dataframe.iloc[i,2], marker='*',c='orange')


#############################
# Given Data Points Plotting#
#############################
x1 = training_set['x_1']
x2 = training_set['x_2']
plt.xlim(training_set['x_1'].min(), training_set['x_1'].max())
plt.ylim(training_set['x_2'].min(), training_set['x_2'].max())
for i in range(len(x1)):
        if(y_train[i] == 0):
                plt.scatter(x1[i], x2[i], label='Scatter plot of Data', marker='*',c='red')
                continue
        elif(y_train[i] == 1):
                plt.scatter(x1[i], x2[i], label='Scatter plot of Data', marker='*',c='blue')
                continue
        else:
                plt.scatter(x1[i], x2[i], label='Scatter plot of Data', marker='*',c='black')
plt.savefig('decision.png')
plt.show()


