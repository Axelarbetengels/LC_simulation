import numpy as np

def mfvf(xyf, minlength=1):
   beginning=np.amin(xyf[:,0])
   xyf[:,0]=xyf[:,0]-beginning
   length=np.amax(xyf[:,0])
   if length > minlength :
      result=np.array([length,np.std(xyf[:,1],ddof=1)**2.0])
      xyfpart1_indices=np.where(xyf[:,0] < length/2.0)
      xyfpart2_indices=np.where(xyf[:,0] >= length/2.0)
      xyfpart1=xyf[xyfpart1_indices]
      xyfpart2=xyf[xyfpart2_indices]
      if xyfpart1[:,0].size > 1 :
         a=mfvf(xyfpart1, minlength)
         if (a != -1).all() :
            result=np.vstack((result, a))
      if xyfpart2[:,0].size > 1 :
         a=mfvf(xyfpart2, minlength)
         if (a != -1).all() : 
            result=np.vstack((result, a))
      return result
   else :
      return np.array(-1)