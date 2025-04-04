import numpy as np

def lin_exp(x,a,b,c):
    return a * (x- b) * np.exp(-c * (x - b))


def d2b(a,b,c,x,t):
    f = a*c**2*(b-x)*np.exp(c*(b-x)) + 2*a*c*np.exp(c*(b-x)) - t/((b-x)**2)
    return f.sum()

def d2c(a,b,c,x):
    f = a*(b-x)**3*np.exp(c*(b-x))
    return f.sum()

def lin_exponenta(x,t):

    try:
        X = np.vstack([np.ones_like(x),x,x**2,x**3]).T
        a = np.linalg.inv(X.T@X)@X.T@t
    # x_m1 = (-2*a[2]+np.sqrt(4*a[2]**2-12*a[1]*a[3]))/6/a[3]
        x_m2 = (-2*a[2]-np.sqrt(4*a[2]**2-12*a[1]*a[3]))/6/a[3]
        x_inf = -a[2]/3/a[3]

        c0 = 1/(x_inf - x_m2)
        b0 = x_inf-2/c0

        a0=1
        eps = 10**-4
        N = 10000
        for i in range(N):
            a_old = a0
            a0 = t.sum()/((x-b0)*np.exp(-c0*(x-b0))).sum()
            b_old = b0
            b0 = b0 - (a0*((1-c0*(x - b0))*np.exp(-c0*(x-b0))).sum() + c0*t.sum() - (t/(x-b0)).sum())/d2b(a0,b0,c0,x,t)
            c_old = c0
            c0 = c0  - (a0*((x-b0)**2*np.exp(-c0*(x-b0))).sum() - (t*(x-b0)).sum())/d2c(a0,b0,c0,x)

            if np.sqrt(((a_old-a0)/a0)**2+((b_old-b0)/b0)**2+((c_old-c0)/c0)**2)<eps:
                break
                
        out = True
        if i < N-1:
            I = True
        else:
            I = False
    except:
        a0,b0,c0, I = 0,0,0,False
        out = False

    #print(np.sqrt(((a_old-a0)/a0)**2+((b_old-b0)/b0)**2+((c_old-c0)/c0)**2),i)

    return a0,b0,c0,I, out

