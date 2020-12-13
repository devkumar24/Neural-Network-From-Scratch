import math
import random

def identity(y):
    return 1
def step(y):
    if y != 0:
        return 0
    else:
        return None
def sigmoid(y):
    return sigmoid(y)*(1 - sigmoid(y))
def tanh(y):
    return 1 - (tanh(y)**2)
def relu(y):
    if y < 0:
        return 0
    elif y > 0:
        return 1
    else:
        return None
def gelu(y):
    return float(gelu(y)/y) + gelu(y)
def softplus(y):
    return sigmoid(y)
def elu(y,alpha):
    if y < 0 :
        return alpha*(np.exp(y))
    elif y == 1 and alpha == 1:
        return 1
    else:
        return 1
def selu(y,lam = 1.0507,alpha = 1.67326):
    if y< 0:
        ey = np.exp(y)
        return alpha*(ey)*lam
    else:
        return lam

def leaky_relu(y):
    if y< 0:
        return 0.01
    else:
        return 1
def prelu(y,alpha):
    if y <0:
        return alpha
    else:
        return 1
def softsign(y):
    return 1/(1 + abs(y))**2
def sqnl(y):
    a = random.choice(1+y/2,1-y/2)
    return a
def srelu(t_l,y,t_r,a_l,a_r):
    if y <= t_l:
        return a_l
    elif t_l < y < t_r:
        return 1
    else:
        return a_r
def bent_identity(y):
    return x/(2* (((x**2) + 1)**0.5)) + 1
def silu(y):
    num = 1 + np.exp(-y) + y*np.exp(-y)
    den = 1 + np.exp(-y)
    
    return num/(den**2)
def gaussian(y):
    return -2*y*np.exp(-y**2)
def sqrbf(y):
    if abs(y) <= 1:
        return -y
    elif 1 < abs(y) < 2:
        return x - 2
    else:
        return 0