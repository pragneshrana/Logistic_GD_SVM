import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


# Loading Dataset
dataset = pd.read_csv('Dataset_2_Team_41.csv')
column = ['X_1', 'X_2', 'Class_value']

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


##################
###Feature Map####
##################
def factorial(n):
        '''
        Calculates the factorial
        '''
        multiplication = 1
        for i in range(1,n+1):
                multiplication = multiplication * i
        return multiplication 

# ans_fac = factorial(5)


def choose(n,r):
        '''
        calculates the nCr
        '''
        n_c_r = factorial(n) / (factorial(r)*factorial(n-r))
        return n_c_r
# ans =choose(15,5)

def mapFeature(dataset,degree):
        '''
        Append the polynomial feature.
        Dataset with feature(X) and result (y).
        Works only for two feature 
        '''
        database = pd.DataFrame([])
        n= degree+1
        no_of_feature = dataset.shape[1]-1  # -1 as last column is outcome
        x1=dataset.iloc[:,0] 
        x2=dataset.iloc[:,1]
        for j in range(n): #for cal power
                for r in range(j+1): #Nowards Loop calculates power
                        colm_list = []
                        for i in range(len(dataset)):
                                colm_list.append(choose(j,r) * pow(x1[i],j-r) * pow(x2[i],r))
                        database[str(j) + str(',') +str(r)] = colm_list
        return database

# ##Testing Of Mapping 
# data = [[1,2],[2,3],[3,4]]
# dataset = pd.DataFrame(data,columns=['Name','Age'])
# dataset = mapFeature(dataset,3)
# dataset.to_csv('file.csv')

###############
# Processing  #
###############
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
        X_train = train.iloc[:,:-1]       #feture matrix of data
        X_train = mapFeature(X_train,degree)
        y_train = train.iloc[:,-1]

        test = test.reset_index(drop = True) 
        X_test = test.iloc[:,:-1]       #feture matrix of data
        X_test = mapFeature(X_test,degree)
        y_test = test.iloc[:,-1]

        return X_train,X_test,y_train,y_test


####################
# Intial Conditions#
####################
learning_rate = 0.5
no_of_iterarion = 1000
regularized_term = 0
degree = 2
no_of_terms = 0
for i in range(degree+1):
        no_of_terms += (i+1)


##Class wise intail weights 
weights = np.linspace(-0.2,0.2,no_of_terms) #INTIAL WEIGHTS DEFINED


################
# Batch Gradient#
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

def log_loss_function(weights, X_t, y_t,lambda_t=0):
        m = len(y_t)
        ones_list = np.ones(len(y_t))
        J = (-1/m) * (np.dot(y_t.T,np.log(sigmoid_probabilty(X_t,weights))) + np.dot((ones_list - y_t).T,np.log(ones_list - sigmoid_probabilty(X_t,weights))))
        reg = (lambda_t/(2*m)) * (weights.T @ weights)
        J = J + reg
        return J

def predicted_y(probability):
        '''
        It will classify the data based on the probability array 
        probability : Array 
        '''
        prediction = []

        # if(t.size == 1):
        #         t =np.reshape(t, (1,1))
        for i in range(len(probability)):
                if(probability[i] > 0.5):
                        prediction.append(1)
                else:
                        prediction.append(0)
        return prediction


def Graient_descent(weights,actual_y,learning_rate,features,regularized_term,no_of_iterarion = 500):
        '''
        It wil run gradient descent till specified no_of_iterarion.
        It calculates the weights and return the weights and no of iterations performed.
        '''
        m = len(actual_y)
        iterated = 0
        accuracy = 0            #Accuracy initialized 
        error = 1               #Error intialized
        error_array = list(np.zeros(2))
        error_diff = 1         #Error Diff Intialied 
        log_loss = []
        accuracy_list = []
        error_list =[]
        while (iterated < no_of_iterarion):     #Tolerance
                iterated += 1 
                probability = sigmoid_probabilty(features,weights)
                weights -= (learning_rate/m) * (np.dot(features.T,(probability - actual_y)) + weights)
                log_loss.append(log_loss_function(weights,features,actual_y))
                prediction_y = predicted_y(probability)
                accuracy,error = accuracy_result(actual_y,prediction_y)
                accuracy_list.append(accuracy)
                error_list.append(error)

                # #Error _Diff adjustment 
                # new_error = error
                # last_error  = error_array[0]    #previous Iteration Array
                # error_diff = abs(last_error - new_error)

                # #shifting array from first place to zero 
                # error_array.insert(0,error_array[1])    
                # error_array.insert(1,error)

        return weights,no_of_iterarion,log_loss,accuracy_list,error_list

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


feature_matrix_training, feature_matrix_testing, y_train, y_test = split_dataframe(dataset)

#Calcualtion of Training     
Weight_result , iteration,log_loss,probability_list,error_list   = Graient_descent(weights,y_train,learning_rate,feature_matrix_training,regularized_term,no_of_iterarion)

#Result Writing to file 
file = open('Result.txt','+a')
file.write('\n \n Weight_result: '+ str(Weight_result))
file.write('\n iteration: '+ str(iteration))

#Testing 
y_probability = sigmoid_probabilty(feature_matrix_testing,weights) #obtained Probability 
y_predicted_test = predicted_y(y_probability)   #Classified Results
test_accu,test_error = accuracy_result(y_predicted_test,y_test)  
file.write('\n test_error: '+ str(test_error))
file.write('\n test_accu: '+ str(test_accu))

############################################
#           log_loss plotting              #
############################################

plt.title('Log Loss', fontsize=20)
plt.xlabel('Error / Accuracy ', fontsize=18)
plt.ylabel('log_loss', fontsize=16)
plt.plot(probability_list,log_loss )
plt.plot(error_list,log_loss)
plt.savefig('Error_Accuracy_Plot')


############################################
# Creation of confusion matrix and plotting#
############################################

confu_matrix = confusion_matrix(y_predicted_test,y_test)
plt.figure(figsize=(10, 7))
sn.heatmap(confu_matrix, annot=True)
plt.title('Confusion Matrix of Test class Dataset-2', fontsize=20)
plt.xlabel('Class', fontsize=18)
plt.ylabel('Class', fontsize=16)
plt.savefig('confusion.png')


############################
# Decision Surface Plotting#
############################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
w = weights
n=100
#Mesh GEneration 
x= list(np.linspace(feature_matrix_training.iloc[:,1].min(), feature_matrix_training.iloc[:,1].max(), n))
y = list(np.linspace(feature_matrix_training.iloc[:,2].min(), feature_matrix_training.iloc[:,2].max(), n))
X, Y = np.meshgrid(x, y)


def pascal_triangle(degree):
    '''
    Gives row and column wise enrtry for given degree
    '''
    Pascal_list =[[1]]   #FIrst entry Defined to start 
    for i in range(1,degree+1): #+1 As we are starting from 1
        temp_list =[]
        for j in range(i+1):  #+1 As we are considering last element
            if(j==0):#First Element = 1
                temp_list.append(1)
                continue
            elif(j == i):#Last Element = 1
                temp_list.append(1)
                continue
            else:
                temp_list.append(Pascal_list[i-1][j]+Pascal_list[i-1][j-1]) # Addition of Upper Two Elements 
        Pascal_list.append(temp_list)
    return Pascal_list

Pascal_Triangle = pascal_triangle(degree)


#F calculation based on pascal
F= 0
counter = 0
for i in range(degree+1):
    for j in range(i+1):  
        print('i-j: ', i-j)      
        print('j: ', j)
        F += w[counter] * pow(X,i-j) * pow(Y,j) * Pascal_Triangle[i][j]
        counter += 1


fig = plt.figure()
ax = Axes3D(fig)
# plt.xlim(feature_matrix_training.iloc[:,1].min(), feature_matrix_training.iloc[:,1].max())
# plt.ylim(feature_matrix_training.iloc[:,2].min(), feature_matrix_training.iloc[:,2].max())
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, F)

#############################
# Given Data Points Plotting#
#############################
x1 = feature_matrix_training.iloc[:,1]
x2 = feature_matrix_training.iloc[:,2]
# ax.xlim(feature_matrix_training.iloc[:,1].min(), feature_matrix_training.iloc[:,1].max())
# ax.ylim(feature_matrix_training.iloc[:,2].min(), feature_matrix_training.iloc[:,2].max())
# ax.title('Decision Boundary')
for i in range(len(x1)):
        if(y_train[i] == 0):
                ax.scatter(x1[i], x2[i], label='Scatter plot of Data', marker='*',c='red')
        else:
                ax.scatter(x1[i], x2[i], label='Scatter plot of Data', marker='*',c='green')
plt.savefig('decision.png')
plt.show()


############################
# Decision Boundary Plotting#
############################
n = 50
x1= list(np.linspace(feature_matrix_training.iloc[:,1].min(), feature_matrix_training.iloc[:,1].max(), n))
x2 = list(np.linspace(feature_matrix_training.iloc[:,2].min(), feature_matrix_training.iloc[:,2].max(), n))
# X, Y = np.meshgrid(x, y)
dataframe = pd.DataFrame([])

#Meshgrid
for i in range(len(x1)):
    for j in range(len(x2)):
        data_vector = [x1[i], x2[j]]
        data_vector = pd.Series(data_vector)
        dataframe = dataframe.append(data_vector, ignore_index=True)

dataframe = mapFeature(dataframe,degree)
y_dataframe = sigmoid_probabilty(dataframe,weights) #obtained Probability 
Y_predicted_dataframe = predicted_y(y_dataframe)   #Classified Results

x1 = dataframe.iloc[:,1]
x2 = dataframe.iloc[:,2]
plt.xlim(dataframe.iloc[:,1].min(), dataframe.iloc[:,1].max())
plt.ylim(dataframe.iloc[:,2].min(), dataframe.iloc[:,2].max())
for i in range(len(x1)):
        if(Y_predicted_dataframe[i] == 0):
                plt.scatter(x1[i], x2[i], label='Scatter plot of Data', marker='o',c='pink')
        else:
                plt.scatter(x1[i], x2[i], label='Scatter plot of Data', marker='o',c='orange')



#############################
# Given Data Points Plotting#
#############################
x1 = feature_matrix_training.iloc[:,1]
x2 = feature_matrix_training.iloc[:,2]
plt.xlim(feature_matrix_training.iloc[:,1].min(), feature_matrix_training.iloc[:,1].max())
plt.ylim(feature_matrix_training.iloc[:,2].min(), feature_matrix_training.iloc[:,2].max())
plt.title('Decision Boundary')
for i in range(len(x1)):
        if(y_train[i] == 0):
                plt.scatter(x1[i], x2[i], label='Scatter plot of Data', marker='*',c='red')
        else:
                plt.scatter(x1[i], x2[i], label='Scatter plot of Data', marker='*',c='green')
plt.savefig('decision.png')
plt.show()
file.close()


