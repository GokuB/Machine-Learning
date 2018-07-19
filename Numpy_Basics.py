#Python for Data Science
#Numpy Basics

"""
Created on Mon Jul 16 13:27:37 2018

@author: Gokul_Balaji
"""
#Numpy
'''Numpy is a core library for scientific computing in python. '''


#Importing Numpy Library 
import numpy as np

#Array Types
'''
1D Array
2D Array (Axis 0-> Across Columns), (Axis 1-> Across Row)
3D Array (Axis 0-> Across Row-Length), (Axis 1-> Across Column-Width), (Axis 2-> Across Depth-Height)
'''
#Creating Arrays using Numpy
A=np.array([1, 2, 3], dtype=int) #1D Array a.k.a Series
B=np.array([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)], dtype=float) #2D Array with Rows and Columns
C=np.array([[(1, 2, 3, 4), (5, 6, 7, 8)], [('9', '10', 11, 12), (13, 14, 15, 16)],[(9, 10, 11, 12), (13, 14, 15, 16)]])

#Initial PlaceHolders

Zeros=np.zeros((5, 5)) #Creates Zero Matrix with 5 rows and 5 columns
Ones=np.ones((10, 5), dtype=np.int8) #Creates Unity Matrix with 10 rows and 10 columns
Identity=np.identity(5, dtype=np.float64) #Creates Identity Matrix with 5 rows and 5 columns
Arange=np.arange(1, 100, 0.1, dtype=np.float) #Creates a Step Matrix given Initial, Final and Step Value
Linespace=np.linspace(20, 30, num=100) #Linespace acts like a scale and returns evenly spaced values
Full=np.full((10, 10), fill_value=10)#Creates a Array given the size and value to be filled
Eye=np.eye(5, dtype=int) #Gives Array with Ones in the Diagonal and Zeros Elsewhere
Random=np.random.random((10,))-50 #Creates Random Matrix given the size and number of values
Empty=np.empty((2, 5)) #Returns new array given size

#Input and Output Operations

np.save('Linespace', Linespace) #Creates .npy file in the current directory
np.savez('Placeholders', Zeros,Ones,Identity,Arange,Full) #Creates .npz file with several files in the current directory
Load=np.load('Placeholders.npz') #Loads the file from directory
print (Load)

String=str('Python Numpy Basics').split()

FromText=np.loadtxt('New Text Document.txt', dtype=str) #Creates Array with the data in the textfile
X=np.genfromtxt('test.csv', dtype=[('x', np.float64), ('y', np.float64)], delimiter=",")
np.savetxt("new.txt", Arange, delimiter=" ") #Saves a new file in the current folder


#Datatypes
'''
Can be used when initializing the dtypes in the methods
np.int64
np.float32
np.complex
np.bool
np.object
np.string_
np.unicode_
'''

#Inspecting your Array

print (Linespace.shape) #Shape of the Array-returns Rows and Columns
print (len(Linespace)) #Returns the Length of the Array
print (Linespace.ndim) #Returns the Dimension of the Array
print (Identity.ndim)
print (C.ndim)
print (C.dtype)
print (Zeros.dtype.name)
New_Mat=Empty.astype(dtype=np.string_) #Converts the Data type of the Object

#Help

print (np.info(np.polyval))

#Artithmetic Operations

One=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
Two=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print (One.dtype)

Add=np.add(One, Two)
print (Add)

Sub=np.subtract(One, Two)
print (Sub)

Mul=np.multiply(One, Two)
print (Mul)

Div=np.divide(One, Two)
print (Div)

Exp=np.exp(Add)
print (Exp)

Sqrt=np.sqrt(np.power(Two,2))
print (Sqrt)

Sine=np.sin(One)
print (Sine)

Cose=np.cos(Two)
print (Cose)

Log=np.log(Exp)
print (Log)

Dot=np.dot(Two, One) 
print (Dot)

#Comparison

print (Eye==Identity)
print ((Add/10)==Two)

print (Two>=Add/10)

print (np.array_equal(Two, Add))
print (np.array_equal(Eye, Identity))

#Aggregate Functions

print (Add.sum())
print (Two.min())
print (Identity)
print (Identity.max(axis=0))
print (Empty)
print (Empty.max(axis=1))
print (Empty.cumsum(axis=0)) #Cumulates the sum across columns
print (Empty.cumsum(axis=1)) #Cumulates the sum across rows
print (Empty.cumprod(axis=0)) #Cumulates the product across columns
print (Empty.cumprod(axis=1)) #Cumulates the product across rows
print (Identity.mean())
print (Add.median())
print (Linespace.corrcoef())
print (Random)
print (np.std(Random))

print (Linespace)
print (Linespace.std())
print (np.std(Linespace))

#Copying Arrays

LineView=Linespace.view
print (Linespace)
print (LineView)

Copy=np.copy(Empty)
Cop=Empty
print (Empty)
print (Copy)
print (Cop)
Deep=Copy.copy()

#Sorting Array
print (Random)
print (Random.sort())

Add.sort(axis=0)

#Subsetting/Slicing/Indexing


Rex=[10,13,1,2,135,178,190,2001010, 2038920]
print (Rex[5])

import pandas as pd
Data=pd.read_csv('Iris.csv')

Iris_Array=np.array(Data)

print (Iris_Array.shape)
print (Iris_Array.dtype)
Iris_Array.dtype.name
Iris_Array.min(axis=0)
Iris_Array.max(axis=0)

#Subsetting 
print (Data.columns.values)
print (Iris_Array[:, 2:])
print (Iris_Array[:, 0:2])
print (Iris_Array[1:10, 1:])
print (Iris_Array[:, -1])
print (Iris_Array[:, 0:2])
print (Iris_Array[:1, 1])
print (Iris_Array[-1 : : ])

#Fancy Indexing

print (Iris_Array[[1, 0, 3, 1], [1, 2, 3, 5]]) #Gets you  (1, 1) (0, 2) (3, 3) (1, 5) elements
print (Iris_Array[[1, 2, 3, 4]][-1:, [0, 0, 0, 1]])


#Array Manipulation
Inverse=np.transpose(Iris_Array)
print (Iris_Array.shape)
print (Inverse.shape)

print (Inverse.ravel())
New=np.append(Add, np.transpose(Add))
Insert=np.insert(New, 2, 10000)
print (Insert)

print (np.delete(Iris_Array, [0], axis=1))

#Combining Arrays
print (Add)
print (New[:10])

np.vsplit(Identity, 5)
np.vstack(Empty, New)










