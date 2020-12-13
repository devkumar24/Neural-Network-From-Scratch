import numpy as np

def softmax(y):
    """
    softmax = e^y(i)/sum(e^y(i))
    """
    
    ea = np.exp(y)
    
    total = ea/np.sum(ea,axis = 1,keepdims=True)
        
    return total

def identity(y):
    return y

def sigmoid(y):
    sigma = 1 + np.exp(-x)
    return float(1/sigma)

def tanh(y):
    e_pos = np.exp(y)
    e_neg = np.exp(-x)
    
    num = e_pos - e_neg
    den = e_pos + e_neg
    
    return float(num/den)


def ReLu(Z):
    return np.maximum(0,Z)

def gelu(y):
    return (math.erf(y/(2**0.5)) + 1)*(float(x/2))



def softplus(y):
    ey = np.exp(y)
    log = np.log(1+ey)
    return log

def elu(y,alpha = 0.01):
    if y <= 0:
        ey = np.exp(y)
        return alpha*(ey-1)
    else:
        return y

def selu(y,lam = 1.0507,alpha = 1.67326):
    if y<= 0:
        ey = np.exp(y)
        return alpha*(ey-1)*lam
    else:
        return lam*y
def leaky_relu(y):
    if y < 0 :
        return 0.01*y
    else:
        return y

def prelu(y,alpha):
    if y < 0 :
        return alpha*y
    else:
        return y
def softsign(y):
    mod = abs(y)
    return y/(1+mod)
def sqnl(y):
    if y > 2.0:
        return 1
    elif 0 <= y <= 2.0:
        return y - (y**2)/4
    elif 0 > y >= 2.0:
        return y + (y**2)/4
    else:
        return -1
def srelu(y,t_l,t_r,alpha_l,alpha_r):
    if y <= t_l:
        return t_l + (alpha_l*(y - t_l))
    elif t_l < y < t_r:
        return y
    elif y >= t_r:
        return t_r + (alpha_l*(y - t_r))
def bent_identity(y):
    a = (((x**2) + 1)**0.5) - 1
    
    return float(a/2) + y

def silu(y):
    return y/(1 + np.exp(-y))

def gaussian(y):
    x = y**2
    return np.exp(-x)

def sqrbf(y):
    if abs(y) <= 1.0:
        return 1 - (y**2)/2
    elif 1.0 < abs(y) < 2.0:
        return 0.5 * ((2 - abs(x))**2)
    else:
        return 0