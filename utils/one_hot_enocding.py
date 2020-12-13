import numpy as np
#######################Code to implement one hot vector from scratch###################
def one_hot_encoding(y):
  # y should be of shape (row,)
  # and it should be numpy.ndarray
  y = np.array(y)

  # fetch total number of outcomes in vector y.
  row = y.shape[0]

  # calculate unique outcomes and count the frequency of each unique outcome
  unique,count = np.unique(y,return_counts=True)
  # make index variable and this index variable is the column of one hot vector 
  index = unique
  # total columns in one hot vector
  col = len(index)

  # initialize one hot vector
  one_hot_vector = []
  # iterate over row and col
  for i in range(row):
    for j in range(col):
      # check the condition to make one hot vector
      # take an e.g y = [1,2,3,2,1,5,4,3,2,4,5,5]
      # index = [1,2,3,4,5]
      # col = 5, index[1] = 2
      # y[1] == index[1] ==2 append 1 to row1 and col1, and other places append 0.
      if index[j] == y[i]:
        one_hot_vector.append(float(1))
      else:
        one_hot_vector.append(float(0))
  # convert the list into numpy.ndarray
  one_hot_vector = np.array(one_hot_vector)
  # reshape the vector into its respective shape i.e., (row,col) y(12,5)
  one_hot_vector = one_hot_vector.reshape(row,col)
  
  # return one hot vector
  return one_hot_vector