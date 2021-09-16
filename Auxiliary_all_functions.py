import numpy as np

from scipy.optimize import minimize
def line_search_golden(x,p,Vold,fun,alphaold,k):
    '''
α can be fixed (inefficient) or can be obtained by an additional algorithm.
The golden-section search is a technique for finding an extremum (minimum or maximum) of a function inside a specified interval.
The method operates by successively narrowing the range of values on the specified interval, which makes it relatively slow, but very robust. 
'''
    #We start with a random point on the function and move in the
    #negative direction of the gradient of the function to reach the local/global minimum.
    N=len(x)/2.
    r=(np.sqrt(5)-1.)/2.   #The golden ratio
    p_norm=np.max(abs(p))
    alpha_2=(1./p_norm)*N/10.
    alpha_2=min(alpha_2,1E12)
    alpha_1=max(alphaold*.01,alpha_2*1E-6)
    
    alpha_min=alpha_1
    Vmin=Vold
    alpha=alpha_1
    its=0
    
    while (alpha<alpha_2): # increasing alpha
        its+=1
        V=fun(x+alpha*p,k)
        if(V<Vmin):
            alpha_min=alpha
        alpha/=r    
#    if abs(alpha_min-alpha_1)/alpha_1<1E-10: # decrasing alpha
#        print('alpha_1 original',alpha_1) 
#        alpha_2=alpha_1
#        alpha_1=alpha_1/1000.
#        alpha=alpha_1
#        r=0.9
#        while (alpha<alpha_2):
#            its+=1
#            V=fun(x+alpha*p,k)
#            if(V<Vmin):
#                alpha_min=alpha
#            alpha/=r   
#            print ('new alphas in LS',alpha_min,V,Vold,Vmin)
    return alpha_min



def nonlinear_conjugate_gradient(dfun,fun,x0,tol,k):
    x=np.copy(x0)
    res=-1.*dfun(x,k) 
    p=np.copy(res)
    V=fun(x,k)
    All_E=np.array([])
    all_res=np.array([])
    iter_n=np.array([])
    res_scalar=np.linalg.norm(res)
    iter=0
    alpha=1
    while(res_scalar>tol):

        iter=iter+1
        p_old=np.copy(p)      
        res_old=np.copy(res)
        V_old=V
        alpha=line_search_golden(x,p,V_old,fun,alpha,k)
        x = x+alpha*p_old    
        res=-1.*dfun(x,k) 
        V=fun(x,k) 
        res_scalar=np.linalg.norm(res)
        all_res=np.append(all_res,res_scalar)
        curr_E= fun(x,k)
        All_E=np.append(All_E,curr_E)
        iter_n=np.append(iter_n,iter)
        print ('NLCG, iter,res, energy',iter,V,res_scalar)
        if(res_scalar<tol ):
            break
# Several options to choose beta:        
#        beta=np.dot(res,res)/np.dot(res_old,res_old)   
        beta=(np.dot(res,res)-np.dot(res,res_old))/np.dot(res_old,res_old) 
        p=res+beta*p_old  
    return x,all_res,All_E,iter_n

def steepest_descent(dfun,fun,x0,tol,k):
    x=np.copy(x0)
    All_E=np.array([])
    all_res=np.array([])
    iter_n=np.array([])
    res=-1.*dfun(x,k)
    res_scalar=np.linalg.norm(res)
    iter=0.
    alpha=1
    while(res_scalar>tol):
         iter=iter+1
         p=res
         V_old=fun(x,k)
         alpha=line_search_golden(x,p,V_old,fun,alpha,k)
         x = x+ alpha*p
         res=-1.*dfun(x,k)
         res_scalar=np.linalg.norm(res)
         all_res=np.append(all_res,res_scalar)
         curr_E= fun(x,k)
         All_E=np.append(All_E,curr_E)
         iter_n=np.append(iter_n,iter)
         print ('Steepest, iter, energy,res',iter,fun(x,k),res_scalar)
    return x,all_res,All_E,iter_n
  
