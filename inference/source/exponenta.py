import numpy as np
import math
from scipy import stats


'''
Оценка параметров exp(ax+b) и построение доверительных интервалов

from conf_int import get_stat

b  = get_stat(x,y) 
b - параметры в exp(b[0]x+b[1]) 

b, yhat, dylo, dyhi, X, C = get_stat(x,y,num_points=100,ci = 95) 
yhat - модель экспоненты построенная по 100 точкам (num_points) в диаразоне от x.min() до x.max()
dylo, dyhi - верхняя и нижняя 95% дов граница модели
X - предиктор, size = num_points
C - матрица ковариации парамтеров модели

plt.plot(x,y,'.')
        
plt.plot(X,yhat)
plt.plot(X,yhat-dylo)
plt.plot(X,yhat+dyhi)
'''


def wfit(z,x,sw):
    
    yw = z * sw;
    SW = np.vstack([sw,sw]).T

    xw = x * SW

    Q,R = np.linalg.qr(xw)
    b= np.linalg.inv(R.T@R)@R.T@(Q.T@yw)
    return b, R


def get_stat(x,y,num_points = 400,ci=None):
    # x,y - prices and orders
    # num_points - количество точек разбиения исходного диапазона цен для построения модели экспоненты и дов интервалов
    # ci - доверительный интервал в % (90, 95, 99)
    N = np.ones_like(y)
    x = np.vstack([N,x]).T
    mu = y + 0.25
    eta = np.log(mu)
    it = 0
    #seps = np.sqrt(np.finfo(float).eps)
    b = np.zeros([2,1])
    iterLim = 100
    #convcrit = 10**-6
    
    while it <= iterLim:
        deta = 1/mu
        z = eta + (y - mu) * deta # np.log(mu) + (y-mu)/mu
        sqrtirls = np.abs(deta) * np.sqrt(mu) # 1/np.sqrt(mu)
        
        sqrtw = 1/sqrtirls # = np.sqrt(mu)
        b_old = b
        b,R = wfit(z, x, sqrtw)
        eta = x@b
        mu = np.exp(eta)
        it += 1
        eps = np.abs((b - b_old)/b)
        eps = eps.max()
        if eps<10**-14:
            break
        
    if ci is None:
        return b
    else:
        RI = np.linalg.inv(R.T@R)@R.T@np.eye(2)
        C = RI@RI.T
        se = np.sqrt(np.diag(C))
        SE = np.vstack([se.T*se[0],se.T*se[1]])
        C = C/ SE
    
        V = SE*C
        X = np.linspace(x[:,1].min(),x[:,1].max(),num_points)
        #%
        N = np.ones_like(X)
        X = np.vstack([N,X]).T
        eta = X@b
        vxb = X@V*X
    
    
        vxb = vxb.sum(axis=1)
        yhat =  np.exp(eta)
    
        inf = math.inf
    
        #ci = 80
        crit = stats.t.ppf(1- ((100-ci)/2/100), inf-1) 
    
        dxb = crit * np.sqrt(vxb)
    
        dyhilo = np.vstack([np.exp(eta-dxb) ,np.exp(eta+dxb)]).T
    
        dylo = yhat - dyhilo.min(axis=1)
        dyhi = dyhilo.max(axis=1) - yhat
    
        return b, yhat, dylo, dyhi, X[:,1], C

