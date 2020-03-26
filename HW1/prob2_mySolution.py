#!/usr/local/bin/python3
#

import matplotlib.pyplot as plt
import numpy as np


class GenData:
    """
    This class generates random regression data

    Attributes:
            x: n x d matrix of input data
            y: n dimensional vector of response
    """
    def __init__(self):
        """
        The constructor, with fixed data generation
        """
        np.random.seed(12345)
        d=1000
        ntrn=100
        sigma=1
        temp=np.ones((d,1))/np.linspace(1,500,d).reshape((d,1))
        wtrue=np.sqrt(temp)
        xtrn = np.random.randn(ntrn,d).dot(np.diagflat(temp))
        ytrn = xtrn.dot(wtrue) + sigma* np.random.randn(ntrn,1)
        self.x=xtrn
        self.y=ytrn
    

class RidgeObj:
    """
    This class provides an interface to the ridge regression objective function
    f(w) = 0.5 [\| x* w - y\|_2^2 + lam * w *w]
    we normalize x and y by dividing sqrt{n}

    Attributes:
        x: n x d matrix of normalized input data
        y: n dimensional vector of normalized response
        lam: regularization parameter
        wstar: closed form solution
        fstar: optimal objective function value at wstar
    """

    def __init__(self,data,lam):
        """
        The constructor
        :param data: generated data
        :param lam: regularization parameter
        """
        n=np.size(data.y)
        self.x=data.x/np.sqrt(n)
        self.y=data.y/np.sqrt(n)
        self.lam=lam
        self.__solve__()

    def L(self):
        """
        This function compute the smoothness parameter of the objective
        :return: the smoothness parameter
        """
        # implement
        x_trans = np.transpose(self.x)
        vals, vects = np.linalg.eig(np.dot(x_trans, self.x))
        l_max = np.max(vals)
        l = float(l_max) + self.lam
        return l

    def __solve__(self):
        """
        This function computes the closed form solution of the ridge regression problem
        It then sets self.wstar and self.fstar
        :return:
        """
        # implement
        x_trans = np.transpose(self.x)
        I = np.diagflat(np.ones((1000,1)))
        self.wstar = np.dot(np.linalg.inv(np.dot(x_trans,self.x)+self.lam*I),x_trans).dot(self.y)
        self.fstar = self.obj(self.wstar)

    def obj(self,w):
        """
        This function computes the objective function value
        :param w: parameter at which to compute f(w)
        :return: f(w)
        """
        # implement
        x_trans = np.transpose(self.x)
        y_trans = np.transpose(self.y)
        w_trans = np.transpose(w)
        f_w = 0.5 * (np.dot(y_trans, self.y)-2*np.dot(y_trans, self.x).dot(w) +np.dot(w_trans, x_trans).dot(self.x).dot(w)+self.lam*np.dot(w_trans, w))
        return f_w

    def grad(self,w):
        """
        This function computes the gradient of the objective function
        :param w: parameter at which to compute gradient
        :return: gradient of f(w)
        """
        #implement
        x_trans = np.transpose(self.x)
        w_trans = np.transpose(w)
        df_w = np.dot(x_trans, self.x).dot(w)-np.dot(x_trans, self.y)+self.lam*w
        
        return df_w
        


def gd(ridge, w0, eta, t):
    """
    This function performs gradient descent for t iterations
    :param ridge: ridge objective function class
    :param w0: initial parameter
    :param eta: learning rate
    :param t: number of iterations
    :return: t+1 function values evaluated at the intermediate solutions
    """
    #implement
    subopt = []
    w = w0
    f = ridge.obj(w)
    subopt.append(f)
    for i in range(t):
        w = w - eta*ridge.grad(w)
        f = ridge.obj(w)
        subopt.append(f)
    return subopt
        
    


def main():
    # generate data
    data=GenData()
    

    # experiments with different regularization parameters
    lam_arr=[1e-4,1e-2,1,1e1]
    for lam in lam_arr:
        ridge=RidgeObj(data,lam)
        # trying different learning rate at 0.1/L L 2/L
        eta_arr=np.array([0.1,1,2])/ridge.L()
     

        plt.xlabel('iterations')
        plt.ylabel('primal-suboptimality')

        for eta in eta_arr:
            w0=np.zeros((np.size(ridge.wstar,0),1))
            t=100
            # perform gradient descent and return function values, compute primal suboptimality
            subopt=gd(ridge,w0,eta,t)-ridge.fstar
            leg = 'eta={}'.format(eta)
            subopt = np.squeeze(subopt)
            plt.plot(np.arange(t+1),subopt,label=leg)
        plt.legend()
        plt.yscale('log')
        filename='plot-lam={}.pdf'.format(lam,eta)
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":
    main()

    
 
