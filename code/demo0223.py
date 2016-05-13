#!/usr/bin/env python
# encoding=utf-8




import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
import cPickle
import pylab as plt

 

def eval_wrapper(yhat, y):  
    y = np.array(y);print y[:10]
    y = y.astype(int);print yhat[:10]
    #yhat = np.array(yhat)
    #yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)  
    err=np.sum((y-yhat)*(y-yhat))/float(y.shape[0])
    #return quadratic_weighted_kappa(yhat, y)
    return err



def get_params(maxDepth):
    
    plst={ 
  	"objective": "binary:logistic",
   	"booster": "gbtree",
   	"eval_metric": "auc",
  	"eta": 0.01, # 0.06, #0.01,
  	#"min_child_weight": 240,
	"silent":1,
   	"subsample": 0.75,
   	"colsample_bytree": 0.68,
   	"max_depth": maxDepth
	}

    return plst


if __name__=='__main__':
	 


	 

	




	#####################
	#xgboost
	###################
	
 
	 

	###cross valid split trainset
	x,y=np.arange(10).reshape((5,2)),range(5)
	print x,y
	from sklearn.cross_validation import train_test_split
	for r in range(3):
		x1,x2,y1,y2=train_test_split(x,y,test_size=0.3)
		print x1,x2
		 
		 

 
	
	
	  
		
		
	
	
	

 	
 

	 




	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



