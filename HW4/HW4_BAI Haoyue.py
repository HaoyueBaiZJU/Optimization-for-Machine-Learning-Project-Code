#!/usr/local/bin/python3
# HKUST CSE COMP6211 Homework 4
# author: BAI Haoyue

import matplotlib.pyplot as plt
import numpy as np


class BinaryClassificationData:
    """
    This class loads binary classification data from file

    Attributes:
            x: n x d matrix of input data
            y: n dimensional vector of [+-1] response
    """

    def __init__(self, file):
        """
        load binary classification data from file
        :param file: data file (with +1 and -1 binary labels)
               format: label (+1/-1) followed by features
               it assumes the features in [0,255] and scales it to [0,1]
        """
        data_txt = np.loadtxt(file, delimiter=",")
        self.y = np.asfarray(data_txt[:, :1])
        self.x = np.asfarray(data_txt[:, 1:]) * (0.99 / 255) + 0.01


class BinaryLinearClassifier:
    """
    This class is linear classifier
    Attributes:
        w: linear weights
    """

    def __init__(self, w):
        """
        :param w: linear weights
        """
        self.w= w.reshape((-1,1))

    def classify(self,x):
        """
        classify data x
        :param x: data matrix to be classified
        :return: class labels
        """
        yp=x.dot(self.w)
        return (yp>=0)*2-1

    def test_error_rate(self,data):
        """
        compute test error rate on data
        :param data: data to be evaluated
        :return: test error rate
        """
        yp=self.classify(data.x)
        return np.sum(data.y*yp<=0)/np.size(data.y)

    def nnz(self):
        """
        sparsity
        :return: number of nonzero weights
        """
        return sum(np.abs(self.w)>1e-10)

class RegL1L2:
    """
    Implement the L1-L2 regularizer g(w) = 0.5 lam * \|w\|_2^2 + mu * \|w\|_1
    """
    def __init__(self,lam,mu):
        """
        initialization
        :param lam: L2-regularization parameter
        :param mu: L1-regularization parameter
        """
        self.lam=lam
        self.mu=mu

    def param_smoothness(self):
        """
        smoothness parameter
        :return: smoothness
        """
        return self.lam + (self.mu>0)*1e5

    def param_strongconvex(self):
        """
        strong convexity parameter
        :return: self.lam
        """
        return self.lam

    def obj(self,w):
        """
        This function computes g(w) : the objective value of the regularizer
        :param w: model parameter
        :return: g(w)
        """
        return 0.5 * self.lam * (w.transpose().dot(w)) + self.mu * np.linalg.norm(w, 1)

    def grad(self,w):
        """
        This function computes the gradient of the regularizer g(w)
        :param w: model parameter
        :return: gradient of g(w)
        """
        return self.lam * w + self.mu * np.sign(w)

    def prox_map(self, eta, w, i=-1):
        """
        compute the proximal mapping \arg\min_u [ (0.5/eta) * \|u-w\|_2^2 + g_i(w) ]
        :param eta: learning rate
        :param w: parameter to compute proximal mapping
        :param i: the i-th component of g() (-1 is all, not used in the current implementation)
        :return: proximal_map
        """
        wp = w / (1 + self.lam * eta)
        etap = eta / (1 + self.lam * eta)
        u = np.maximum(0, np.abs(wp) - etap * self.mu) * np.sign(wp)
        return u

    def obj_dual(self,alpha):
        """
        compute dual objective - g^*(alpha)= - max_w [w*alpha-g(w)]
        :param alpha: dual parameter
        :return: dual objective
        """
        # implement
      
        #n=np.size(alpha)
        #temp=np.zeros((n,1))
        #for i in range(n):
        #    if alpha[i] > self.mu:
        #        temp[i] = (alpha[i] - self.mu)**2 / (2 * self.lam)
        #    elif alpha[i] < -self.mu:
        #        temp[i] = (alpha[i] + self.mu)**2 / (2 * self.lam)
        #    else:
        #        temp[i] = 0 
        temp = self.grad_dual(alpha)
        #temp_objdual=-temp.transpose().dot(alpha)+self.obj(temp)  
        temp_objdual = - 0.5*self.lam*np.linalg.norm(temp,2)**2
        return temp_objdual



    def grad_dual(self, alpha):
        """
        solve min_w [ -w*alpha + g(w)]
        the solution is gradient of the dual regularizer g^*(alpha)
        :param alpha: dual parameter
        :return:  nabla g^*(alpha)
        """
        u = np.maximum(0, np.abs(alpha) - self.mu) * np.sign(alpha)
        try:
            return (1.0 / self.lam) * u
        except ZeroDivisionError:
            print("Dual gradient not uniquely defined for non-strongly convex L1-L2 regularization")
            raise

class LossHinge:
    """
    Implement the Smoothed Hinge-Loss (SVM) objective function
    f(w) = smoothed-hinge(1-w*x*y)

           smoothed-hinge(z) =  0                 if z>1
                                1-z-gamma/2       if z < 1-gamma
                                (z-1)^2/(2gamma)  otherwise
    Attributes:
	    gamma: smoothing parameter
    """

    def __init__(self,gamma):
        self.gamma=gamma

    def param_gamma(self):
        """
        hinge loss smoothness parameter gamma
        """
        return 1e-5+self.gamma


    def param_smoothness(self, x, y,quick=True):
        """
        This function compute the smoothness parameter of f(w,x,y)
         :param x: feature matrix
        :param y: target response
        :param quick: quick estimation
        :return: the smoothness parameter
        """
        myL=1/self.param_gamma()
        if (quick):
            myL = myL*np.mean(np.sum(x * x, axis=1))
        else:
            u, s, vh = np.linalg.svd(x)
            s = np.amax(s)
            myL=(myL / np.size(x,0)) * s * s
        return myL

    def solve_dual(self,i,alpha,p,xnorm2,lam_n,y):
        """
        solve the dual coordinate ascent at a data point (x,y)
             max_dalpha [-f_i^*(-alpha-dalpha) - (w*x) *dalpha - (0.5/lam_n)*(x*x) * dalpha^2]
        ;param i: the i-the data point
        :param alpha: current dual variable
        :param p:   w*x
        :param xnorm2:  x*x
        :param y:  target
        :return: dalpha
        """
        # implement
        
        #n = np.size(alpha)
        #temp1 = y/n - self.gamma/n*alpha - p
        #temp2 = self.gamma/n + xnorm2/lam_n
        #temp = temp1 / temp2
        #if temp < -1*(y==-1)-alpha:
        #    solvdual = -1*(y==-1)-alpha
        #elif temp > (y==1)-alpha:
        #    solvdual = (y==1)-alpha
        #else:
        #    solvdual = temp 
        dalpha = (lam_n / (lam_n*self.gamma + xnorm2))*(1-p*y-self.gamma*alpha*y)
        alpha2 = alpha*y + dalpha
        alpha2 = max(0,min(1,alpha2))
        solvdual = alpha2*y-alpha
        return solvdual
    

    def obj(self, w, x, y):
        """
        This function compute the loss (1/n) sum_i f_i(w,x_i,y_i)
        :param x: feature matrix
        :param y: target response
        :param w: model parameter
        :return: f(w,x,y)
        """
        wp=w.reshape(np.size(w),1)
        z=1-x.dot(wp)*y
        if (self.gamma>0):
            loss= (z>self.gamma)*(z-self.gamma/2)+(z<=self.gamma)*(z>0)*(0.5/self.gamma)*z**2
        else:
            loss=(z>0)*(z)
        return np.mean(loss)

    def obj_dual(self,alpha,y):#### 
        """
        compute the dual objective (1/n) sum_i -f_i^*(-alpha_i)
        :param alpha: dual parameter
        ;param y: label y
        :return:  dual objective
        """
        # implements
        
        #alpha = alpha.reshape(-1,1)
        #n=np.size(alpha)
        #objdual=0
        #for i in range(n):
        #    objdual += y[i]*alpha[i]-0.5*alpha[i]*alpha[i]*self.gamma
        #objdual = (1 / n) * objdual
        objdual = np.mean(alpha.reshape(-1)*(y.reshape(-1)))-0.5*self.gamma *np.mean(np.square(alpha))
        
        return objdual
    
    

    def grad(self, w, x, y):
        """
        This function computes the gradient of f(w,x,y)
        :param w: parameter at which to compute gradient
        :return: gradient of f(w,x,y)
        """
        wp = w.reshape(np.size(w), 1)
        v=x.dot(wp)
        return self.grad_i(v,wp,x,y,-1)

    def grad_i(self,v,w,x,y,i):
        """
        assume that f(w) = ff(v):  v= X w
        :param v: X w
        :param i: the i-th component of w
        :return:  [nabla ff(w)]_i
        """
        yp=y.reshape(np.size(v),1)
        z = 1 - v.reshape(np.size(v),1) * yp
        if (self.gamma > 0):
            dloss = -1 * (z > self.gamma) + (z <= self.gamma) * (z > 0) * z * (-1 / self.gamma)
        else:
            dloss = -1 * (z > 0)
        if (i<0):
            return x.transpose().dot(dloss * yp) / np.size(dloss)
        return x[:,i].reshape(-1).dot(dloss *yp)/np.size(dloss)

class RegularizedLoss:
    """
    Implement regularized loss of the form:
       phi(w)= f(w) + g(w)
               loss + regularizer
    Atributes:
            data:  (data.x data.y)
            f:    loss f(w)
            g:    regularizer g(w)= sum_i g_i(w_i)
    """
    def __init__(self,data,loss,reg):
        """
        init
        :param data: training data
        :param loss: loss function
        :param reg: regularizer
        """
        self.data=data
        self.f=loss
        self.g=reg
        self.__solve__()

    def __solve__(self):
        """
        This function computes an approximate solution of the objective
        It then sets self.wstar
        :return:
        """
        d=np.size(self.data.x,1)
        w0=np.zeros((d,1))
        t=10000
        alpha=self.set_learning_rate()
        ww = ProxACCL.solve_adaptive_AG(self, w0, alpha, t)
        self.wstar= ww[:,t].reshape(-1,1)
        print ('norm of final prox-gradient={:.2g}'.format(np.linalg.norm(self.grad_prox(alpha,self.wstar),2)))
        return

    def set_learning_rate(self):
        """"
        set learning rate to be 1/f-smoothness
        """
        return 1.0/self.f.param_smoothness(self.data.x,self.data.y,True)

    def obj_f(self,w):
        """
        compute loss objective f(w)
        :param w: model parameter
        :return: f(w)
        """
        return self.f.obj(w,self.data.x,self.data.y)

    def obj(self,w):
        """
        compute regularized-loss objective f(w) + g(w)
        :param w: model parameter
        :return: f(w) + g(w)
        """
        return self.obj_f(w)+self.g.obj(w)

    def obj_dual(self,alpha):
        """
        compute the dual objective (1/n) sum_i -f_i^*(-alpha_i) - g^*( -(1/n)sum_i alpha_i x_i )
        :param alpha: dual parameter
        :return: dual objective
        """
        v= self.data.x.transpose().dot(alpha.reshape(-1,1))/np.size(alpha)
        return self.f.obj_dual(alpha,self.data.y) + self.g.obj_dual(v)

    def grad_f(self,w):
        """
        compute gradient of loss nabla f(w)
        :param w: model parameter
        :return: nabla f(w)
        """
        return self.f.grad(w,self.data.x,self.data.y)

    def grad(self,w):
        """
         compute gradient of rgularized-loss f(w) + g(w)
         :param w: model parameter
         :return: nabla [ f(w) + g(w) ]
         """
        return self.grad_f(w) + self.g.grad(w)

    def grad_prox(self, eta, w):
        """
        This function computes prox gradient of the objective f(w) + g(w)
        :param eta: learning rate
        :param w: parameter at which to compute proximal gradient
        :return: prox_grad(w) = (w- prox(w- eta* nabla f(w)))/eta
        """
        wt = w - eta * self.grad_f(w)
        ww = self.prox_map(eta, wt)
        return (w - ww) / eta

    def prox_map(self,eta,w,i=-1):
        """
        compute proximal map of g
        :param eta: learning rate
        :param w: model parameter
        :param i: the i-th component (-1 means all components)
        :return: argmin_u [ (0.5/eta)* || u-w||_2^2 + g_i(u) ]
        """
        return self.g.prox_map(eta,w,i)

    def grad_gstar(self,alpha):
        """
        compute gradient of g^*(alpha)
        :param alpha: dual parameter
        :return: nabla g^*(alpha)
        """
        return self.g.grad_dual(alpha)

class CD:
    """
    Implement Primal proximal coordinate descent of Lecture 18
    """
    @staticmethod
    def solve(phi,data,t,eta,order='perm'):
        """
        solve min_x phi(x) = f(x) + g(x)   using primal proximal CD
        :param phi: f(x) + g(x)  objective function (require phi.f.grad_i()  phi.prox_map())
        :param data: training data: data.x  and data.y
        :param t: number of iterations
        :param eta: learning rate
        :param order: order of picking coordinate (random perm cyclic)
        :return: the t+1 iterates found by the method from x_0 to x_t
        """
        # implement
        # randomized proximal coordinate descent
        
        n = np.size(data.x, 0)
        d = np.size(data.x, 1)
        w = np.zeros((d))  
        #u = np.dot(data.x, w.transpose())
        u = np.zeros((n))
        result = np.zeros((d, t + 1))
        ii = np.random.permutation(d)
        xnorm2 = np.zeros((d))
        for i in range(d):
            xnorm2[i] = np.linalg.norm(data.x[:,i],2)**2+1e-10
        for ti in range(t):
            if (order == 'random'):
                ii = np.random.randint(d, size=d)
            if (order == 'perm'):
                ii = np.random.permutation(d)
            for iter in range(d):
                i = ii[iter]
                etap = n*eta/xnorm2[i]
                wt = w[i] - etap * phi.f.grad_i(u,w,data.x,data.y,i)
                wp = phi.prox_map(etap, wt, i)
                u += data.x[:,i].reshape(-1)*(wp-w[i])
                w[i] = wp
            result[:, ti + 1] = w.transpose()
        return result

     
        
class SDCA:
    """
    Implement SDCA of Lecture 17
    """
    @staticmethod
    def solve(phi,data,t,order='perm'):
        """
        solve min_x phi(x) = f(x) + g(x)   using SDCA
        :param phi: f(x) + g(x) objective function (phi.f.solve_dual()  phi.grad_gstar() )
        :param data: training data: data.x  and data.y
        :param t: number of iterations
        :param order: order of picking coordinate (random perm cyclic)
        :return: the primal solutions and duality gaps of t+1 iterates found by the method from x_0 to x_t
        """
        n = np.size(data.x, 0)
        d = np.size(data.x, 1)
        alpha = np.zeros((n))
        w = np.zeros((d))
        v = w
        result = np.zeros((d, t + 1))
        duality_gap = np.zeros((t+1))
        duality_gap[0]=phi.obj(w)-phi.obj_dual(alpha)
        xnorm2=np.zeros((n))
        for i in range(n):
            xnorm2[i]=np.linalg.norm(data.x[i,:],2)**2
        ii = np.random.permutation(n)
        for ti in range(t):
            if (order == 'random'):
                ii = np.random.randint(n, size=n)
            if (order == 'perm'):
                ii = np.random.permutation(n)
            lam_n = phi.g.param_strongconvex() * n
            for iter in range(n):
                i = ii[iter]
                dalpha=phi.f.solve_dual(i,alpha[i],w.dot(data.x[i,:]),xnorm2[i],lam_n,data.y[i])
                alpha[i] = alpha[i] + dalpha
                v = v + (dalpha / n) * data.x[i, :]
                w = phi.grad_gstar(v)
            result[:, ti + 1] = w.transpose()
            duality_gap[ti+1]=phi.obj(w)-phi.obj_dual(alpha)
        return result, duality_gap ;

class SPDC:
    """"
    Implement SPDC of Lecture 18
    """
    @staticmethod
    def solve(phi,data,t,order='perm'):
        """
        solve min_x phi(x) = f(x) + g(x)   using SPDC
        :param phi: f(x) + g(x) f(x) + g(x) objective function (phi.f.solve_dual()  phi.prox_map() )
        :param data: training data: data.x  and data.y
        :param t: number of iterations
        :param order: order of picking coordinate (random perm cyclic)
        :return: the t+1 iterates found by the method from x_0 to x_t
        """
        # implement     
        # stochastic primal-dual coordinate method 
        n = np.size(data.x, 0)
        d = np.size(data.x, 1)
        alpha = np.zeros((n))
        xnorm2=np.zeros((n))
        r = np.max(xnorm2)
        for i in range(n):
            xnorm2[i]=np.linalg.norm(data.x[i,:],2)**2
        ii = np.random.permutation(n)
        tau = 1 / (2 * r * np.sqrt(phi.g.lam * phi.f.param_smoothness(data.x,data.y)))
        sigma = np.sqrt(n * phi.g.lam * phi.f.param_smoothness(data.x,data.y)) / (2 * r)
        theta = 1 - 1 / (n + r * np.sqrt(n * phi.f.param_smoothness(data.x,data.y)) / phi.g.lam) 
        u=np.zeros((d))
        w = phi.g.grad_dual(u)
        wbar = w
        result = np.zeros((d, t + 1))
        duality_gap = np.zeros((t+1))
        duality_gap[0]=phi.obj(w)-phi.obj_dual(alpha)
        ii = np.random.permutation(n)
        for ti in range(t):
            if (order == 'random'):
                ii = np.random.randint(n, size=n)
            if (order == 'perm'):
                ii = np.random.permutation(n)
            for iter in range(n):
                i = ii[iter]
                dalpha=phi.f.solve_dual(i,alpha[i],wbar.dot(data.x[i,:]),1,2*sigma,data.y[i])
                alpha[i] = alpha[i] + dalpha
                dw = w + tau*(u + data.x[i, :]*dalpha)
                tw = w
                w = phi.prox_map(tau, dw)
                u = u + (dalpha / n) * data.x[i, :]
                wbar = w + theta*(w - tw)
            result[:, ti + 1] = w.transpose()
            duality_gap[ti+1]=phi.obj(w)-phi.obj_dual(alpha)
        return result, duality_gap ;
 


class ADMM:
    """
    Implement ADMM and accelerated linearized ADMM of Lecture 16
    """
    @staticmethod
    def solve_accl_linear(phi,alpha0,eta,beta,t):
        """
         solve min_x phi(x) = f(x) + g(z)   x = z using accelerated linearized ADMM
        :param phi: composite objective function to be optimized (require phi.grad_f() and phi.prox_map())
        :param alpha0: initial dual variable
        :param eta: primal learning rate for f(x)
        :param beta: momentum parameter
        :param t: number of iterations
        :return: the t+1 iterates found by the method from z_0 to z_t
        """
        # implement
        """
        n = np.size(alpha0)
        d = np.size(eta)
        
        xp = np.zeros((n))
        zp = np.zeros((n))
        xbarp = xp
        zbarp = zp
        alpha = alpha0

        result=np.zeros((d,t+1))
        result[:,0]=zp.transpose()
        for ti in range(t):
            z = phi.prox_map(phi.grad_f(alpha), xbarp, zbarp)
            x = phi.prox_map(phi.grad_g(alpha), xbarp, zp)
            alpha = alpha + (1/eta) *(1-beta)*(x + z)
            zbar = z + beta*(z - zp)
            xbar = x + beta*(x - xp)
            zbarp = zbar
            xbarp = xbar
            zp = z
            xp = x
            result[:,ti+1]=z.transpose()
        return result
        """
        d = np.size(alpha0)
        alpha = alpha0.copy()
        z=phi.grad_gstar(alpha)
        x=z.copy()
        xp=x
        zp=z
        barx=x
        barz=z
        result = np.zeros((d, t + 1))
        result[:, 0] = z.transpose()
        rho=1/eta
        for ti in range(t):
            zt= barz + eta*(alpha+rho*(barx-barz))
            z = phi.prox_map(eta,zt)
            x= barx - eta*(alpha+rho*(barx-z))- eta * phi.grad_f(barx)
            alpha= alpha + rho*(1-beta) * (x-z)
            barx=x+beta*(x-xp)
            xp=x
            barz=z+beta*(z-zp)
            zp=z
            result[:, ti + 1] = z.transpose()
        return result
    
        
class RDA:
    """
    Implementing RDA of Lecture 13
    """
    @staticmethod
    def solve(phi,x0,eta,t):
        """
        solve min_x f(x) + g(x) using RDA
        :param phi(x)=f(x)+g(x): objective function to be minimized (require phi.grad_f() and phi.prox_map())
        :param x0: initial point
        :param eta: learning rate
        :param t: number of iterations
        :return: the t+1 iterates found by the method from x_0 to x_t
        """
        xp = x0
        d = np.size(x0)
        result = np.zeros((d, t + 1))
        result[:, 0] = x0.transpose()
        xt=x0
        etat=0
        for ti in range(t):
            etat=etat+eta
            xt = xt - eta * phi.grad_f(xp)
            x = phi.prox_map(etat, xt)
            xp = x
            result[:, ti + 1] = x.transpose()
        return result

class ProxGD:
    """
    Implementing Proximal Gradient Descent Algorithm
    """
    @staticmethod
    def solve(phi,x0,eta,t):
        """
        solve min_x f(x)+g(x) using proximal gradient descent
        :param phi(x)=f(x)+g(x): objective function to be minimized (require phi.grad_f() and phi.prox_map())
        :param x0: initial point
        :param eta: learning rate
        :param t: number of iterations
        :return: the t+1 iterates found by GD from x_0 to x_t
        """
        xp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        for ti in range(t):
            xt=xp-eta*phi.grad_f(xp)
            x= phi.prox_map(eta,xt)
            xp=x
            result[:,ti+1]=x.transpose()
        return result

    @staticmethod
    def solve_AG(phi, x0, eta0, t):
        """
        solve min_x f(x)+g(x) using proximal gradient descent
        :param phi(x)=f(x)+g(x): objective function to be minimized (require phi.grad_f() and phi.prox_map())
        :param x0: initial point
        :param eta0: initial learning rate
        :param t: number of iterations
        :return: the t+1 iterates found by GD from x_0 to x_t
        """
        tau=0.8
        xp = x0
        d = np.size(x0)
        result = np.zeros((d, t + 1))
        result[:, 0] = x0.transpose()
        eta=eta0
        f0=phi.obj_f(xp)
        for ti in range(t):
            g = phi.grad_f(xp)
            while True:
                xt = xp - eta * g
                x = phi.prox_map(eta, xt)
                f1=phi.obj_f(x)
                if (f1<=f0+ g.transpose().dot(x-xp) + (0.5/eta)*np.linalg.norm(x-xp,2)**2+1e-10):
                    break
                eta=eta*tau
            if (f1<f0 + g.transpose().dot(x-xp) + (0.5*tau/eta)*np.linalg.norm(x-xp,2)**2-1e-10):
                eta=eta/np.sqrt(tau)
            xp = x
            f0=f1
            result[:, ti + 1] = x.transpose()
        return result

class ProxACCL:
    """
    Implement Nesterov's Accelerated Proximal Gradient Algorithm
    """
    @staticmethod
    def solve(phi,x0,alpha,beta,t):
        """
        solve min_x phi(x) := f(x)+g(x) using Nesterov's Acceleration
        :param phi: objective function to be minimized (require phi.grad_f() and phi.prox_map())
        :param x0: initial point
        :param alpha:  learning rate
        :param beta: momentum parameter
        :param t: number of iterations
        :return: the t+1 iterates found by ACCL from x_0 to x_t
        """
        xp=x0
        xpp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        for ti in range(t):
            y=xp+beta*(xp-xpp)
            xt=y-alpha*phi.grad_f(y)
            x=phi.prox_map(alpha,xt)
            xpp=xp
            xp=x
            result[:,ti+1]=x.transpose()
        return result

    @staticmethod
    def solve_adaptive_AG(phi,x0,alpha0,t,eps=1e-16):
        """
        solve min_x phi(x) := f(x)+g(x) using Nesterov's Acceleration
        :param phi: objective function to be minimized (require phi.grad_f() and phi.prox_map())
        :param x0: initial point
        :param alpha0:  initial learning rate
        :param t: number of iterations
        :param eps: stopping criterion of gradient
        :return: the t+1 iterates found by ACCL from x_0 to x_t
        """
        tau=0.8
        c=0.5
        xp=x0
        xpp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        gp=1e-10
        lrate=-2
        alpha=alpha0
        fcc=phi.obj(xp)
        for ti in range(t):
            beta=min(1.0, np.exp(lrate))

            y=xp+beta*(xp-xpp)
            p=phi.grad_f(y)
            xt=y-alpha*p
            x=phi.prox_map(alpha, xt)
            g=np.linalg.norm((y-x)/alpha,2)
            if (g<=eps):
                for tp in range(ti,t):
                    result[:,tp+1]=x.transpose()
                break
            lrate = 0.8 * lrate + 0.2 * 2 * np.log(min(1.0, g / gp))
            gp=g
            fp=phi.obj(y)
            fc=phi.obj(x)
            if (fcc<fc):
                lrate=lrate-1
            if (g**2>1e-16):
                etap=(fp-fc)/g**2
                while(etap<=c*alpha and alpha>=1e-4*alpha0):
                    alpha=tau*alpha
                    xt=y-alpha*p
                    x=phi.prox_map(alpha,xt)
                    g=np.linalg.norm((y-x)/alpha,2)
                    fc=phi.obj(x)
                    etap=(fp-fc)/g**2
                if(etap>=c*alpha/tau):
                    alpha=alpha/np.sqrt(tau)
            xpp=xp
            xp=x
            fcc=fc
            result[:,ti+1]=x.transpose()
        return result



class MyFigure:

    def __init__(self,lam,mu,filename):
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplots(2, 2,figsize=(12,8))
        fig.suptitle(r'iterations ($\lambda$={:.2g} $\mu$={:.2g})'.format(lam, mu))

        axs[0, 0].set_ylabel('primal-suboptimality')
        axs[0, 1].set_ylabel('duality-gap')
        axs[1, 0].set_ylabel('training error rate')
        axs[1, 1].set_ylabel('test error rate')
        self.fig=fig
        self.axs=axs
        self.filename=filename

    def finish(self):
        self.axs[0, 0].legend()
        self.axs[0, 1].legend()
        self.axs[1, 0].legend()
        self.axs[1, 1].legend()
        self.axs[0, 0].set_yscale('log')
        self.axs[0,1].set_yscale('log')
        self.axs[1, 0].set_yscale('log')
        self.axs[1, 1].set_yscale('log')
        self.fig.savefig(self.filename + '.pdf')
        plt.close(self.fig)

    def plot(self,phi, result, col, lab,train_data,test_data, duality_gap=np.zeros((0))):
        """
        plot the convergence result for a method
        :param phi: function to be evaluated
        :param result:  iterates generated by optimization algorithm from 0 to t-1
        :param col: plot color
        :param lab: plot label
        ;param train_data: training data
        ;param test_data: test data
        ;param duality_gap: duality gap
        :return: none
        """

        t = np.size(result, 1)
        xx = np.arange(t)
        yy = np.zeros((t, 1))
        trnerr= np.zeros((t,1))
        tsterr = np.zeros((t, 1))
        nnz= np.zeros((t,1))
        w = np.zeros((np.size(result, 0), 1))
        phi_star=phi.obj(phi.wstar)
        for ti in range(t):
            w[:, 0] = result[:, ti]
            yy[ti] = np.maximum(phi.obj(w) - phi_star, 1e-16)
            lc=BinaryLinearClassifier(w)
            trnerr[ti]=lc.test_error_rate(train_data)
            tsterr[ti] = lc.test_error_rate(test_data)
        if (np.size(xx)==np.size(duality_gap)):
            self.axs[0, 1].plot(xx, yy, linestyle='dotted', color=col, label=lab+' (primal suboptimality)')
            self.axs[0,1].plot(xx, duality_gap, linestyle='dashed', color=col, label=lab+' (duality-gap)')
        self.axs[0,0].plot(xx, yy, linestyle='dashed', color=col, label=lab)
        self.axs[1,0].plot(xx, trnerr, linestyle='dashed', color=col, label=lab)
        self.axs[1,1].plot(xx, tsterr, linestyle='dashed', color=col, label=lab)

def do_experiment(filename,gamma,lam,mu,train_data,test_data):

    print("solving L1-L2 regularized logistic regression with lambda={:.2g} mu={:.2g}".format(lam,mu))

    loss=LossHinge(gamma)
    reg=RegL1L2(lam,mu)
    phi=RegularizedLoss(train_data,loss,reg)

    w0 = np.zeros((np.size(phi.wstar, 0), 1))

    # compare ProxGD proxACCL to GD ACCL
    #
    t=100
    eta=phi.set_learning_rate()
    myfig=MyFigure(lam,mu,filename+'a')

    result=ProxACCL.solve_adaptive_AG(phi,w0,eta,t)
    myfig.plot(phi,result,'black','ProxACCL-AG',train_data,test_data)

    alpha0 = w0
    beta = 1 - 2 * np.sqrt(eta*lam)
    ADMM.solve_accl_linear(phi, alpha0, eta, beta, t)
    myfig.plot(phi,result,'green','ADMM-ACCL-Linear',train_data,test_data)

    eta2=1/gamma
    result = CD.solve(phi,train_data,t,eta2,'perm')
    myfig.plot(phi, result, 'blue', 'CD-perm', train_data,test_data)

    result, duality_gap = SDCA.solve(phi,train_data,t,'perm')
    myfig.plot(phi, result, 'orange', 'SDCA-perm',train_data,test_data,duality_gap)

    result, duality_gap = SPDC.solve(phi,train_data,t,'perm')
    myfig.plot(phi, result, 'red', 'SPDC-perm', train_data, test_data,duality_gap)

    myfig.finish()

    # compare ProxGD proxACCL to ProxGD-BB
    #
    t=100
    alpha = 1.0
    myfig=MyFigure(lam,mu,filename+'b')

    result=ProxACCL.solve_adaptive_AG(phi,w0,alpha,t)
    myfig.plot(phi,result,'black','ProxACCL-AG',train_data,test_data)

    result,duality_gap = SDCA.solve(phi,train_data,t,'cyclic')
    myfig.plot(phi, result, 'blue', 'SDCA-cyclic',train_data,test_data,duality_gap)

    result,duality_gap = SDCA.solve(phi,train_data,t,'random')
    myfig.plot(phi, result, 'orange', 'SDCA-random',train_data,test_data,duality_gap)

    result,duality_gap = SDCA.solve(phi,train_data,t,'perm')
    myfig.plot(phi, result, 'red', 'SDCA-perm',train_data,test_data,duality_gap)

    myfig.finish()



def main():
    train_data=BinaryClassificationData("mnist/mnist_train_binary.csv")
    test_data=BinaryClassificationData("mnist/mnist_test_binary.csv")

    gamma=1
    lam=1e-3
    mu=1e-2
    filename="fig-1"
    do_experiment(filename,gamma,lam,mu,train_data,test_data)

    lam=1e-3
    mu=1e-4
    filename="fig-2"
    do_experiment(filename,gamma,lam,mu,train_data,test_data)

if __name__ == "__main__":
    main()
