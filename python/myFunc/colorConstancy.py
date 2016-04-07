import numpy as np
from numpy import linalg as LA


def mycliplims(im, lo, hi):
    imout = im
    if imout < lo:
        imout = lo
    if imout > hi:
        imout = hi

    return imout


def multiangle(approx, actual):
    ss1=np.shape(approx)
    ss2=np.shape(actual)
    if ss1 != ss2: 
        print('error in multiangle: dim"s don"t match.')
        return
    
    N = ss1[0]
    angles = np.zeros([N,1])
    for i in range(N):
        #approx[i,:] = approx[i,:]/np.sum(approx[i,:])
        nn1 = LA.norm(actual[i,:])
        nn2 = LA.norm(approx[i,:])
        if not (nn1==0 or nn2==0):
            tt = np.sum(actual[i,:]*approx[i,:])/(nn1*nn2)
            #print("tt"+str(tt))
            if tt<-1.0001 or tt> 1.0001:
                angles[i] = None
            else:
                tt = mycliplims(tt,-1.0,1.0)
                angles[i] = (180/np.pi)*np.arccos(tt) #Error
    return angles

def deMultiangle2(approx, actual):
    ss1=np.shape(actual)
    ss2=np.shape(approx)
    if ss1 != ss2: 
        print('error in multiangle: dim"s don"t match.')
        return
    
    N = ss1[0]
    dim = ss1[1]
    deApprox = np.zeros([N,dim])

    for i in range(N):
        nn2 = LA.norm(actual[i,:])
        nn1 = LA.norm(approx[i,:])
        if not (nn1==0 or nn2==0):
            tt = np.sum(actual[i,:]*approx[i,:])/(nn1*nn2)

            if tt<-1.0001 or tt> 1.0001:
                pass
            else:
                tt = mycliplims(tt,-1.0,1.0)
                arccos_tt = np.arccos(tt)

                f = np.sum(actual[i,:]*approx[i,:])
                g = nn1*nn2
                d_arccos_tt = (180/np.pi) #(-180./np.pi)/(arccos_tt**2)
                #print d_arccos_tt
                d_tt = (-1)/np.sqrt(1-tt**2)
                #print d_tt
                d_x = (actual[i,:]/g) -  (f/(g**2))*approx[i,:]*nn2/nn1
                #print d_x
                deApprox[i,:] = d_x * d_arccos_tt * d_tt
                #print d_tt

    return deApprox

def deMultiangle(approx, actual):
    ss1=np.shape(actual)
    ss2=np.shape(approx)
    if ss1 != ss2: 
        print('error in multiangle: dim"s don"t match.')
        return
    
    N = ss1[0]
    dim = ss1[1]
    deApprox = np.zeros([N,dim])

    for i in range(N):
        nn1 = LA.norm(actual[i,:])
        nn2 = LA.norm(approx[i,:])
        if not (nn1==0 or nn2==0):
            f = np.sum(actual[i,:]*approx[i,:])
            g = nn1*nn2
            M = f/g

            if M<-1.0001 or M> 1.0001:
                deApprox[i,:] = None
            else:      
                K = np.pi*np.arccos(M)
                #Derivative Caculation
                phiL_K = -180*(K**2)
                phiL_M = phiL_K*(np.pi*(-1)/(np.sqrt(1-M**2)))

                phiM_x = actual[i,:]/g
                phiM_x += ((-f)/(g**2))*(nn2/nn1)*approx[i,:]

                deApprox[i,:] = phiL_M*phiM_x
            
    return deApprox

def numericalGradCheck(approx, actual, h):
    analyticalGrad = deMultiangle2(approx, actual)
    #print analyticalGrad

    im1 = approx - [h,h,h]
    im2 = approx + [h,h,h]

    loss1 = multiangle(im1, actual)
    loss2 = multiangle(im2, actual)

    numericalGrad = ((loss2 - loss1)/2.)/([h,h,h])

    #print analyticalGrad
    # print numericalGrad
    # print np.vstack((analyticalGrad, numericalGrad))
    error = abs(analyticalGrad - numericalGrad)/ np.amax(abs(np.vstack((analyticalGrad, numericalGrad))), axis=0)
    #print np.amax(abs(np.vstack((analyticalGrad, numericalGrad))), axis=0)
    print error

if __name__ == '__main__':
    
    #approx =  abs(np.random.randn(5,3))
    #actual = approx + 0.001

    #print approx, actual
    #h = 0.00001
    #numericalGradCheck(approx, actual, h)
    approx = np.array([[0.20954534, -0.13629462,  0.18868887]])
    actual = np.array([[0.36758   ,  0.324884  ,  0.30753601]])
    h = 0.0001
    approx1 = np.array([[1,2+h,3]])
    approx2 = np.array([[1,2-h,3]])
    print (multiangle(approx1, actual) - multiangle(approx2, actual))/(2*h)
    print deMultiangle(approx, actual)
    print deMultiangle2(approx, actual)
    print multiangle(approx, actual)
