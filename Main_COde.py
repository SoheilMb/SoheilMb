import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patxi
import Presentation_all_functions
import pandas as pd
RED=[255,0,0]
BLUE=[0,255,0]
GREEN=[0,0,255]

'''
Given parameters of the problem:
D: Diameter of the discs
R: Radius of the discs
mg : mass * gravity acceleration
L= Width of the container holding the discs
'''
D=2.
R=D/2.
mg =9.8
L=10*D

N=int(input("Enter number of discs: "))
tol= N*0.01*mg

'''
Coordinates of discs. We use sets so that they don't overlap
'''
xo=set()
while len(xo)<N:
    x=np.random.randint(1,L)
    y=np.random.randint(1,L)
    temp=set((x,y))
    if temp not in xo:
        xo.add((x,y))
        
'''
Array to store the coordinates of the discs
Remember that x0=[x1,y1,x2,y2,x3,y3,...]
'''
x0=np.array([])        

for pair in xo:
    x0=np.append(x0,pair[0])
    x0=np.append(x0,pair[1])



def V_floor(y_pos):    # we pass y_pos as the parameter
    if y_pos<R:         #Apply a penalty in case the disc touches the floor
        return 0.5*mg*((y_pos-R)**2)
    else:
        return 0        # No penalty
  
def dV_floor(y_pos):   # we pass y_pos as the parameter
    
    if y_pos<R:          #Apply a penalty in case the disc touches the floor
        return mg*((y_pos-R))
    else:
        return 0

def V(r):  # we pass the normal of the distance vector between two discs as
    #the parameter
    if r<2*R:         # Apply a penalty in case the distance between the discs
                # is less than the size of a disc
        return 0.5*mg*((r-2*R))**2
    else:
        return 0
#######################################################################################
def dV(r):
    ## Derivate of the contact penalty function
    if r<2*R:
        return mg*((r-2*R))
    else:
        return 0
############################################################################        
def V_wall(x_pos):     ## we pass x_pos as the parameter
    if x_pos<R:       # Apply a penalty in case the disc touches the LEFT
                        #side of the container
        return 0.5*mg*((x_pos-R)**2)
    elif x_pos>L-R:      #Apply a penalty in case the disc touches the RIGHT
                        #side of the container
        return 0.5*mg*((x_pos+R-L)**2)
    else:
        return 0
###################################################################################    
def dV_wall(x_pos):       # we pass x_pos as the parameter
                        # Derivate of the wall penalty function
    if x_pos<R:
        return mg*((x_pos-R))
    elif x_pos>L-R:
        return mg*((x_pos+R-L))
    else:
        return 0 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def energy(x,penal):
    import numpy as np
    n=len(x)
    energy=0.
    for k in range(0,n//2):     # Iterate through all discs
                                #and save the x,y coordinates in xk
        xk=np.array([x[2*k],x[2*k+1]])
        for i in range(k+1,n//2):
            # Iterate through the rest of discs and store their x,y in xi
            # This loop does not check the already checked ones to avoid
            #repetitiveness
            xi=np.array([x[2*i],x[2*i+1]])
            xki=xi-xk
            rki=np.linalg.norm(xki)
            energy+=penal*V(rki)  # Add penalty to the energy in case of contact

        energy+=mg*xk[1]         # Add potential energy mg*y to total energy
        energy+=penal*V_floor(xk[1])   #Add contribution of contact with floor
                                        #(by using y_position of discs)
        
        energy+=penal*V_wall(xk[0])   #Add contribution of contact with walls
                                        #(by using x_position of discs)
        
    return energy          # returns an scalar value of energy (value to be minimized)

"""
The gradient of the energy is the vector used for gradient methods to descent
in the energy landscape and finding a minimum. Indeed, the negative of the
derivative of the energy respect the position of a particle k represents the
force acting on that particle.
"""

def d_energy(x,penal):   
    import numpy as np
    n=len(x)
    denergy=np.zeros(n)
   
    for k in range(0,n//2):    # Iterate through all discs and save x,y in xk
        xk=np.array([x[2*k],x[2*k+1]]) 
        for i in range(0,n//2):  
            if (k!=i):      # A disc can't touch itself
                            #otherwise the value of rki would be 0
                xi=np.array([x[2*i],x[2*i+1]])                           
                xki=xi-xk                  
                rki=np.linalg.norm(xki)    
                ddV=dV(rki)
                if(ddV!=0):        #In case two discs touch the contribution
                                    # is added to the vertical and horizontal unit vectors
                    ddV=ddV/rki                                            
                    denergy[2*k]+=-penal*ddV*xki[0] #Add to the unit vector in horizontal
                    denergy[2*k+1]+=-penal*ddV*xki[1]   #Add to the unit vector in vertical


        denergy[2*k+1]+=mg    # The  potential energy mg*y 
        
        
        denergy[2*k]+= penal*dV_wall(xk[0])   #add wall contributions to dericative of energy
        denergy[2*k+1]+= penal*dV_floor(xk[1])  # add floor contribution to dericative of energy

    return denergy    # Returns a non_scalar value

'''
Function to draw Circle Patches
'''

def draw_positions(X,methodname):
    n=len(X)
    print(X)

    x=np.array([])
    y=np.array([])

    for k in range(0,n//2):
        x=np.append(x,X[2*k])
        y=np.append(y,X[2*k+1])
    plt.scatter(x,y,s=15,c="red")

    plt.grid()
    title= "N= "+ str(N) +"  K= "+str(K)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    circles=[]
    '''
This loop creates circles with random colors and then creates
patch.Circle objects and draws them
'''
    for coord in range(len(x)):
        clor=random.choice(["red","blue","green"])
        c1=patxi.Circle((x[coord],y[coord]),radius=1,color=clor)
    
        ax.add_patch(c1)


    plt.xlim([0,20]) 
    plt.ylim([0,20])
    title= methodname+ "   N= "+ str(N) +"  K= "+str(K)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
This function saves the number in iterations, E* values and residual values
in each iteration for the conjugate and steepest methods.
Two storage values are defined, one using pandas dataframes
and then saving to csv files and the other is saving in text files.
To save computational costs, the second method is commented and it does not run 
'''
def Save_iteration_Data(file_prefix,all_E,total_iter,all_residual):
    filename=file_prefix+" N= "+ str(N) +"  K= "+str(K)+".csv"
    columns=['Iterations','E*','Residual']
    df=pd.DataFrame(columns=columns)
    '''
    with open(filename,"w") as myfile:
        for idx in range(len(all_E)):
            df.loc[idx]=[total_iter[idx],all_E[idx],all_residual[idx]]
            myfile.write(str(total_iter[idx])+','+str(all_E[idx])+','+str(all_residual[idx])+'\n')
    '''
    for idx in range(len(all_E)):
            df.loc[idx]=[total_iter[idx],all_E[idx],all_residual[idx]]
    df.to_csv(filename)

'''
Same as above but for the Scipy method Although it can be
done with the function mentioned above
'''
def save_important_values(sol,all_E,all_residual):
    #residual=np.linalg.norm(sol.hess_inv)
    #iters=sol.nit
    #Energy=sol.fun
    all_iters=np.array([])
    for itr in range(len(all_E)):
        all_iters=np.append(all_iters,itr)
    columns=['Iterations','E*','Residual']
    df=pd.DataFrame(columns=columns)
    for idx in range(len(all_E)):
            df.loc[idx]=[all_iters[idx],all_E[idx],all_residual[idx]]
    
    filename= "Scipy method__N= "+str(N)+" K "+str(K)+" .csv"
    df.to_csv(filename)
'''
#EXTRA
def callback(x):
    fobj = energy(x)
    history.append(fobj)
'''

"""""""""""""""""""""
Solve the problem
"""""""""""""""""""""


print("Solve Minimization problem using:\n\
1- Non Linear Conjugate Gradient\n \
2- Steepest Descent \n \
3- Scipy built in method\n")
      
method=int(input("Enter the number of the desired method: "))

'''
Depending on the chosen method by the user we only import certain methods
from the auxiliary module containing all methods (i.e.Presentation_all_functions)
'''

if method==1:
    from Presentation_all_functions import nonlinear_conjugate_gradient 
    from Presentation_all_functions import  line_search_golden
elif method==2:
    from Presentation_all_functions import steepest_descent
    from Presentation_all_functions import  line_search_golden
    
elif method==3:
    #from Presentation_all_functions import built_in
    from scipy.optimize import minimize
     
    
else:
    print("Number not valid.Enter an integer value between 1 and 3")



r=8 # It can be changed by the user
K=1 # initial K
all_E=np.array([])
all_residual= np.array([])
total_iter=np.array([])
history_E = []
history_res=[]
while K<500:
    if method==1:
        X,all_residual,all_E,total_iter = nonlinear_conjugate_gradient(d_energy,energy,x0,tol,K)
        draw_positions(X,"Conjugate Gradient")
        Save_iteration_Data("Conjugate Gradient",all_E,total_iter,all_residual)
        K*=r
    elif method==2:
        X,all_residual,all_E,total_iter = steepest_descent(d_energy,energy,x0,tol,K)
        draw_positions(X,"Steepest Descent")
        Save_iteration_Data("Steepest Descent",all_E,total_iter,all_residual)
        K*=r
    elif method==3:
        #sol= built_in(d_energy,energy,x0,tol,K)
        #X=sol.x
        '''
Callback method is used to store the values of E* and residual in each
iteration
'''
        def callback(x0):
            fobj = energy(x0,K)
            res=-1.*d_energy(x0,K)
            res_scalar=np.linalg.norm(res)
            history_E.append(fobj)
            history_res.append(res_scalar)
        '''
If we didnt need to store the values of E* and residual in each iteration
we wouldnt need to call scipy.optimize.minimize here and would be imported from the
Presentation_all_functions module. But the callback function makes things a bit more
complicated
'''
        sol=minimize(energy, x0, method='BFGS', jac=d_energy,\
             options={'disp': True},args=(K),callback=callback)
        X=sol.x  
        draw_positions(X,"Scipy Method")
        #print("history_E: ", history_E)
        
        save_important_values(sol,history_E,history_res)
        print(sol)
        K*=r
        
        
    
        
