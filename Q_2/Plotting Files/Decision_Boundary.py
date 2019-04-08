############################
# Decision Surface Plotting#
############################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x=[0,1,2,3,4,5,6,5,9,8,4,5,3,1,8,5,8,9,3,6,5,4]
y=[-2,5,6,-8,9,-7,4,-9,3,-5,1,1,-9,6,2,-8,6,9,-4,1,2,5]
w =[2,-55,2,1,6,5,0,3,4,-20,-8,5,-9,52,4,-485,2,6,5,4,-44,51,-1,451,-1651]


n=50
x= list(np.linspace(min(x), max(x), n))
y = list(np.linspace(min(y), max(y), n))
X, Y = np.meshgrid(x, y)
degree = 3


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
print('Pascal_Triangle: ', Pascal_Triangle)


F= 0
# w =[1,1,1,1,1,1,1,1,1,1]
counter = 0
#No of terms in Binomial 
no_of_terms = 0
for i in range(degree+1):
        no_of_terms += (i+1)

#F calculation based on pascal
for i in range(degree+1):
    for j in range(i+1):  
        print('i-j: ', i-j)      
        print('j: ', j)
        F += w[counter] * pow(X,i-j) * pow(Y,j) * Pascal_Triangle[i][j]
        counter += 1


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(X, Y, F)


##for Cross check 
F = w[0]+w[1]*X+w[2]*Y+w[3]*pow(X,2)+w[4]*2*X*Y + \
w[5]*pow(Y,2) +w[6]* pow(X,3)+w[7]* 3*pow(X,2) *Y +w[8]*3*X*pow(Y,2) +w[9]*pow(Y,3) 
ax = fig.add_subplot(1,2,2, projection='3d')
ax.plot_surface(X, Y, F)
plt.show()