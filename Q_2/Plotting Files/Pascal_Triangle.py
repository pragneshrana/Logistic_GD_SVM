def pascal_triangle(degree):
    '''
    Gives row and column wise enrtry for given degree
    '''
    Pascal_list =[[1]]   #FIrst entry Defined to start 
    print(Pascal_list[0] ,'\n')
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
        print(Pascal_list[i] ,'\n')

    return Pascal_list

Pascal_Triangle = pascal_triangle(4)
print('Pascal_Triangle: ', Pascal_Triangle)