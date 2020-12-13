import numpy as np
#####################Code to implement max pooling layer from scratch#######################
  def max_pool(x,window_size):
    row,col = x.shape
    output_array = []
    for i in range(int(row/window_size)):
        for j in range(int(col/window_size)):
            window = x[ window_size*i : (window_size * i) + window_size , window_size*j : (window_size * j) + window_size ]
            output_array.append(np.max(window))
    output_array = np.array(output_array)
    shape = int(len(output_array))
    output_array = output_array.reshape(int(row/window_size),int(col/window_size))
    return output_array


#####################Code to implement average pooling layer from scratch#######################
def avg_pool(x,window_size):
    row,col = x.shape
    output_array = []
    for i in range(int(row/window_size)):
        for j in range(int(col/window_size)):
            window = x[ window_size*i : (window_size * i) + window_size , window_size*j : (window_size * j) + window_size ]
            output_array.append(np.mean(window))
    output_array = np.array(output_array)
    output_array = output_array.reshape(int(row/window_size),int(col/window_size))
    return output_array