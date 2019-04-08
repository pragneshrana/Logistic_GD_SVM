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

##Testing Of Mapping 
import pandas as pd
data = [[1,2],[2,3],[3,4]]
dataset = pd.DataFrame(data,columns=['Name','Age'])
dataset = mapFeature(dataset,3)
dataset.to_csv('file.csv')
