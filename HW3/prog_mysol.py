#!/usr/local/bin/python3
#

import matplotlib.pyplot as plt
import numpy as np
import math


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


class LogisticObj:
    """
    This class provides an interface to the L1-L2 regularized logistic regression objective function
    phi(w) = f(w) + g(w)
    f(w) = log(1+exp(w*x*y)
    g(w) = 0.5* lam * w *w + mu * ||w||_1

    Attributes:
        x: n x d matrix of normalized input data
        y: n dimensional binary classification responses {-1 +1} values
        lam: L2-regularization parameter
        mu: L1-regularization parameter
        wstar: approximate solution
    """

    def __init__(self, data, lam, mu):
        """
        The constructor
        :param data: generated data
        :param lam: regularization parameter
        """
        n = np.size(data.y)
        self.x = data.x
        self.y = data.y
        self.lam = lam
        self.mu = mu
        self.__solve__()

    def L(self):
        """
        This function compute the smoothness parameter of the objective
        :return: the smoothness parameter
        """
        n = np.size(self.x, 0)
        u, s, vh = np.linalg.svd(self.x)
        s = np.amax(s)
        return (0.25 / n) * s * s

    def __solve__(self):
        """
        This function computes an approximate solution of the objective
        It then sets self.wstar
        :return:
        """
        d = np.size(self.x, 1)
        w0 = np.zeros((d, 1))
        t = 10000
        alpha = 0.5 / self.L()
        ww = ProxACCL.solve_adaptive_AG(self, w0, alpha, t)
        self.wstar = ww[:, t].reshape(-1, 1)
        print('approximate optimal solution: norm of final prox-gradient={:.2g}'.format(np.linalg.norm(self.grad_prox(alpha, self.wstar), 2)))
        return

    def obj(self, w):
        """
        This function computes the objective function value
        :param w: parameter at which to compute f(w)+g(w)
        :return: phi(w)=f(w)+g(w)
        """
        obj_g= 0.5 * self.lam * (w.transpose().dot(w)) + self.mu * np.linalg.norm(w, 1)
        return self.obj_f(w) + obj_g

    def grad(self, w):
        """
        This function computes the gradient of the objective f(w)+g(w)
        :param w: parameter at which to compute gradient
        :return: gradient of f(w) + g(w)
        """
        grad_g=self.lam * w + self.mu * np.sign(w)
        return self.grad_f(w) + grad_g

    def grad_prox(self, alpha, w):
        """
        This function computes prox gradient of the objective f(w) + g(w)
        :param alpha: learning rate
        :param w: parameter at which to compute proximal gradient
        :return: prox_grad(w) = (w- prox(w- alpha* nabla f(w)))/alpha
        """
        # implement
        temp = w - alpha * self.grad_f(w)
        
        prox_g = (w-self.prox_map(alpha, temp))/alpha
        
        #prox_g = self.prox_map(alpha, temp)
        return prox_g


    def obj_f(self, w):
        """
        This function computes the objective function value of f(x)
        :param w: parameter at which to compute f(w)
        :return: f(w)
        """
        wp = w.reshape(np.size(w), 1)
        loss = np.log(1 + np.exp(-self.x.dot(wp) * self.y))
        return np.mean(loss)

    def grad_f(self, w):
        """
        This function computes the gradient of the objective function
        :param w: parameter at which to compute gradient
        :return: gradient of f(w)
        """
        wp = w.reshape(np.size(w), 1)
        dloss = -1 / (1 + np.exp(self.x.dot(wp) * self.y))
        return self.x.transpose().dot(dloss * self.y) / np.size(dloss)

    def prox_map(self, eta, w):
        """
        compute the proximal mapping \arg\min_u [ (0.5/eta) * ||u-w||_2^2 + g(w) ]
        :param eta: learning rate
        :param w: parameter to compute proximal mapping
        :return: proximal_map
        """
        # implement
        n = np.size(w)
        prox_m=np.zeros((n,1))
        for i in range(n):
            if w[i] > eta*self.mu:
                #prox_m[i] = w[i] - eta*self.mu
                prox_m[i] = (w[i] - eta*self.mu)/(1+eta*self.lam)
            elif w[i] < -eta*self.mu:
                #prox_m[i] = w[i] + eta*self.mu
                prox_m[i] = (w[i] + eta*self.mu)/(1+eta*self.lam)
            else:
                prox_m[i] = 0
        return prox_m
        
    def grad_gstar(self,alpha):
        """
        solve min_w [ -w*alpha + g(w)]
        the solution is gradient of the dual regularizer g^*(alpha)
        :param alpha: dual parameter
        :return:  gradient of g^*(alpha)
        """
        # implement
        #grad_gs = self.lam * alpha
        #return grad_gs
    
        d=np.size(alpha)
        grad_gs=np.zeros((d,1))
        for i in range(d):
            alphai=alpha[i][0]
            if alphai>self.mu:
                grad_gs[i]=(alphai-self.mu)/self.lam
            elif alphai<-self.mu:
                grad_gs[i]=(alphai+self.mu)/self.lam

        return grad_gs



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
        # implement
        xp=x0
        etap = eta
        alphap = xp
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        for ti in range(t):
            alpha = alphap - eta * phi.grad_f(xp)
            etac = etap + eta
            x=phi.prox_map(etac, alpha)
            xp=x
            alphap = alpha
            etap = etac
            result[:,ti+1]=x.transpose()
        return result
        

class PrimalDualAscent:
    """
    Implementing Primal Dual Ascent of Lecture 14
    """
    @staticmethod
    def solve(phi,alpha0,eta,t):
        """
        solve min_x f(x) + g(x) using RDA
        :param phi(x)=f(x)+g(x): objective function to be minimized (require phi.grad_f() and phi.grad_gstar())
        :param alpha0: initial dual point
        :param eta: learning rate in (0,1)
        :param t: number of iterations
        :return: the t+1 iterates found by the method from x_0 to x_t
        """
        # implement
        xp = alpha0
        alphap = alpha0
        d=np.size(xp)
        result=np.zeros((d,t+1))
        result[:,0]=xp.transpose()
        for ti in range(t):
            alpha = (1 - eta) * alphap - eta*phi.grad_f(xp)
            x=phi.grad_gstar(alpha)
            xp=x
            alphap = alpha
            result[:,ti+1]=x.transpose()
        return result

class ProxGD:
    """
    Implementing Gradient Descent Algorithm
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
        # implement
        xp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        for ti in range(t):
            tempx = xp - eta*phi.grad_f(xp)
            x = phi.prox_map(eta, tempx)
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
        # implement
        xp=x0
        etap = eta0
        tau = 0.8
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        for ti in range(t):
            eta = etap
            while True: 
                tempx = xp - eta*phi.grad_f(xp)
                x = phi.prox_map(eta, tempx)
                left = phi.obj_f(x)
                m = np.linalg.norm(x-xp,2)
                right = phi.obj_f(xp) + np.transpose(phi.grad_f(xp)).dot(x-xp) + (1/(2*eta))*m**2
                if left <= right:
                    break
                eta = tau * eta
            left = phi.obj_f(x)
            m = np.linalg.norm(x-xp,2)
            right = phi.obj_f(xp) + np.transpose(phi.grad_f(xp)).dot(x-xp) + (tau/(2*eta))*m**2
            if left <= right:
                eta = eta/tau**(0.5)
            xp=x
            etap = eta
            result[:,ti+1]=x.transpose()
        return result

    @staticmethod
    def solve_BB(phi,x0,eta0,t):
        """
        solve min_x phi(x) := f(x)+g(x) using gradient descent
        :param phi: objective function to be minimized (require phi.grad_f() phi.prox_map())
        :param x0: initial point
        :param eta: learning rate
        :param t: number of iterations
        :return: the t+1 iterates found by GD from x_0 to x_t
        """
        xp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        eta=eta0
        x=xp
        gp=0
        r1=0
        r2=0
        for ti in range(t):
            g=phi.grad_f(x)
            r1 = r1 * 0.2 + np.linalg.norm(x - xp, 2) ** 2
            r2 = r2 * 0.2 + np.dot((x - xp).transpose(), g - gp)[0, 0]
            if (r1 > 1e-16 and r2 > 1e-16):
                eta = max(1e-4 * eta0, min(1e4 * eta0, r1 / r2))
            xp=x
            gp=g
            xt=x-eta*g
            x= phi.prox_map(eta,xt)
            result[:,ti+1]=x.transpose()
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
        # implement
        xp = x0
        xpp = x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        for ti in range(t):
            y = xp + beta*(xp - xpp)
            tempx = y - alpha*phi.grad_f(y)
            x = phi.prox_map(alpha, tempx)
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
        :param beta: momentum parameter
        :param t: number of iterations
        :param eps: stopping criterion of gradient
        :return: the t+1 iterates found by ACCL from x_0 to x_t
        """
        # implement
        tau=0.8
        c=0.5
        xp=x0
        xpp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        lrate=0
        alphap=alpha0
        yp = x0
        fcc=phi.obj_f(xp)
        for ti in range(t):
            beta=min(1.0, np.exp(lrate))
            y=xp+beta*(xp-xpp)
            alpha = alphap
            p=phi.grad_f(y)
            temp = y - alpha*p
            x = phi.prox_map(alpha, temp)
            g=np.linalg.norm(p,2)
            m = np.linalg.norm((x-y)/alpha, 2)
            m2 = np.linalg.norm((xp-yp)/alphap,2)
            lrate=0.8*lrate + 0.2*2*np.log( min( 1.0, m/m2))
            if (g<eps):
                for tp in range(ti,t):
                    result[:,ti+1]=x.transpose()
                break
            fc=phi.obj_f(x)
            if (fcc<fc):
                lrate=lrate-1
            if (g**2>1e-16):
                etap = (phi.obj_f(x)-phi.obj_f(y))/m**2
                while(etap<=c*alpha and alpha>=1e-4*alpha0):
                    alpha=tau*alpha
                    temp=y-alpha*p
                    x=phi.prox_map(alpha, temp)
                    m = np.linalg.norm((x-y)/alpha, 2)
                    etap = (phi.obj_f(y)-phi.obj_f(x))/m**2         
                if(etap>=c*alpha/tau):
                    alpha=alpha/np.sqrt(tau)
            #n1 = np.linalg.norm((x-y)/alpha,2)
            #n2 = np.linalg.norm((xp-yp)/alphap,2)
            #lrate = 0.8*lrate + 0.2*math.log((n1**2)/(n2**2))
            xpp=xp
            xp=x
            yp = y
            alphap = alpha
            fcc = fc
            result[:,ti+1]=x.transpose()
        return result
    
class GD:
    """
    Implementing Gradient Descent Algorithm
    """

    @staticmethod
    def solve(f,x0,eta,t):
        """
        solve min_x f(x) using gradient descent
        :param f: objective function to be minimized (require f.grad())
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
            x=xp-eta*f.grad(xp)
            xp=x
            result[:,ti+1]=x.transpose()
        return result

    @staticmethod
    def solve_AG(f,x0,eta0,t):
        """
        solve min_x f(x) using gradient descent
        :param f: objective function to be minimized (require f.grad())
        :param x0: initial point
        :param eta0: initial learning rate
        :param t: number of iterations
        :return: the t+1 iterates found by GD from x_0 to x_t
        """
        tau=0.8
        c=0.5
        xp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        fp=f.obj(xp)
        eta=eta0
        for ti in range(t):
            g=f.grad(xp)
            g2=np.linalg.norm(g,2)
            x=xp-eta*g
            fc=f.obj(x)
            if (g2>1e-8):
                etap=(fp-fc)/g2**2
                while (etap<=c*eta and eta >1e-4*eta0):
                    eta=eta*tau
                    x=xp-eta*g
                    fc=f.obj(x)
                    etap=(fp-fc)/g2**2
                if (etap>=c*eta/tau):
                    eta=eta/np.sqrt(tau)
            fp=fc
            xp=x
            result[:,ti+1]=x.transpose()
        return result

    @staticmethod
    def solve_BB(f, x0, eta0, t):
        """
        solve min_x f(x) using gradient descent
        :param f: objective function to be minimized (require f.grad())
        :param x0: initial point
        :param eta0: initial learning rate
        :param t: number of iterations
        :return: the t+1 iterates found by GD from x_0 to x_t
        """
        tau = 0.8
        c = 0.5
        xp = x0
        d = np.size(x0)
        result = np.zeros((d, t + 1))
        result[:, 0] = x0.transpose()
        eta = eta0
        x=xp
        gp=0
        r1=0
        r2=0
        for ti in range(t):
            g=f.grad(x)
            r1=r1*0.2+np.linalg.norm(x-xp,2)**2
            r2=r2*0.2+np.dot((x-xp).transpose(),g-gp)[0,0]
            if (r1>1e-16 and r2>1e-16):
                eta = max(1e-4 * eta0, min(1e4 * eta0, r1/r2))
            xp=x
            x = xp - eta * g
            gp=g
            result[:, ti + 1] = x.transpose()
        return result


class ACCL:
    """
    Implement Nesterov's Acceleration Algorithm
    """

    @staticmethod
    def solve(f,x0,alpha,beta,t):
        """
        solve min_x f(x) using Nesterov's Acceleration
        :param f: objective function to be minimized (require f.grad())
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
            x=y-alpha*f.grad(y)
            xpp=xp
            xp=x
            result[:,ti+1]=x.transpose()
        return result

    @staticmethod
    def solve_adaptive_AG(f,x0,alpha0,t,eps=1e-16):
        """
        solve min_x f(x) using Nesterov's Acceleration
        :param f: objective function to be minimized (require f.grad())
        :param x0: initial point
        :param alpha0:  initial learning rate
        :param beta: momentum parameter
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
        gp=np.linalg.norm(f.grad(xp),2)
        lrate=-2
        alpha=alpha0
        fcc=f.obj(xp)
        for ti in range(t):
            beta=min(1.0, np.exp(lrate))

            y=xp+beta*(xp-xpp)
            p=f.grad(y)
            g=np.linalg.norm(p,2)
            if (g<eps):
                for tp in range(ti,t):
                    result[:,ti+1]=x.transpose()
                break
            lrate=0.8*lrate+0.2*2*np.log(min(1.0,g/gp))
            gp=g
            x=y-alpha*p
            fp=f.obj(y)
            fc=f.obj(x)
            if (fcc<fc):
                lrate=lrate-1
            if (g**2>1e-16):
                etap=(fp-fc)/g**2
                while(etap<=c*alpha and alpha>=1e-4*alpha0):
                    alpha=tau*alpha
                    x=y-alpha*p
                    fc=f.obj(x)
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
        axs[0, 1].set_ylabel('weight sparsity')
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
        self.axs[1, 0].set_yscale('log')
        self.axs[1, 1].set_yscale('log')
        self.fig.savefig(self.filename + '.pdf')
        plt.close(self.fig)

    def plot(self,phi, result, col, lab,train_data,test_data):
        """
        plot the convergence result for a method
        :param phi: function to be evaluated
        :param result:  iterates generated by optimization algorithm from 0 to t-1
        :param col: plot color
        :param lab: plot label
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
            nnz[ti]=lc.nnz()
        self.axs[0,1].plot(xx, nnz, linestyle='dashed', color=col, label=lab)
        self.axs[0,0].plot(xx, yy, linestyle='dashed', color=col, label=lab)
        self.axs[1,0].plot(xx, trnerr, linestyle='dashed', color=col, label=lab)
        self.axs[1,1].plot(xx, tsterr, linestyle='dashed', color=col, label=lab)

def do_experiment(filename,lam,mu,train_data,test_data):

    print("solving L1-L2 regularized logistic regression with lambda={:.2g} mu={:.2g}".format(lam,mu))

    phi=LogisticObj(train_data,lam,mu)
    #print(bbb)

    w0 = np.zeros((np.size(phi.wstar, 0), 1))

    # compare ProxGD proxACCL to GD ACCL
    #
    t=100
    alpha = 1.0
    myfig=MyFigure(lam,mu,filename+'a')
    #print(aaa)

    result = ProxGD.solve_AG(phi, w0, alpha, t)
    myfig.plot(phi, result, 'black', 'ProxGD-AG',train_data,test_data)

    result=ProxACCL.solve_adaptive_AG(phi,w0,alpha,t)
    myfig.plot(phi,result,'blue','ProxACCL-AG',train_data,test_data)
    #print(ccc)

    result = GD.solve_AG(phi, w0, alpha, t)
    myfig.plot(phi, result, 'orange', 'GD-AG',train_data,test_data)
    #print(ddd)

    result=ACCL.solve_adaptive_AG(phi,w0,alpha,t)
    myfig.plot(phi,result,'yellow','ACCL-AG',train_data,test_data)

    myfig.finish()

    # compare ProxGD proxACCL to ProxGD-BB
    #
    t=100
    alpha = 1.0
    myfig=MyFigure(lam,mu,filename+'b')

    result = ProxGD.solve_AG(phi, w0, alpha, t)
    myfig.plot(phi, result, 'black', 'ProxGD-AG',train_data,test_data)

    result=ProxACCL.solve_adaptive_AG(phi,w0,alpha,t)
    myfig.plot(phi,result,'blue','ProxACCL-AG',train_data,test_data)

    result = ProxGD.solve_BB(phi, w0, alpha, t)
    myfig.plot(phi, result, 'orange', 'ProxGD-BB',train_data,test_data)

    myfig.finish()

    # compare GD to ProxGD to RDA to Primal Dual Ascent
    #
    t=500
    alpha=1.0
    myfig = MyFigure(lam, mu,filename+'c')

    result = GD.solve(phi, w0, alpha, t)
    myfig.plot(phi, result, 'black', 'GD', train_data, test_data)

    result = ProxGD.solve(phi, w0, alpha, t)
    myfig.plot(phi, result, 'blue', 'ProxGD', train_data, test_data)

    result = RDA.solve(phi, w0, alpha, t)
    myfig.plot(phi, result, 'orange', 'RDA', train_data, test_data)

    alpha0=w0
    eta=min(0.5,20*phi.lam)
    result = PrimalDualAscent.solve(phi, alpha0, eta, t)
    myfig.plot(phi, result, 'red', 'Primal-Dual Ascent', train_data, test_data)

    myfig.finish()


def main():
    train_data=BinaryClassificationData("mnist/mnist_train_binary.csv")
    test_data=BinaryClassificationData("mnist/mnist_test_binary.csv")

    lam=1e-4
    mu=1e-2
    filename="fig-1"
    do_experiment(filename,lam,mu,train_data,test_data)

    lam=1e-4
    mu=1e-4
    filename="fig-2"
    do_experiment(filename,lam,mu,train_data,test_data)

if __name__ == "__main__":
    main()
