#!/usr/bin/env python
# encoding=utf-8




import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
import cPickle
import pylab as plt

dataPath='/home/yr/accelerate-management-process/'

def eval_wrapper(yhat, y):  #pred true
    y = np.array(y);print y[:10]
    y = y.astype(int);print yhat[:10]
    yhat = np.array(yhat)
    #yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)  
    #####accuracy
    #err=np.sum((y-yhat)*(y-yhat))/float(y.shape[0])
    #return err
    #######-loglikely
    return np.mean(-np.log(yhat+0.00001)*y-(1.-y)*np.log(1.-yhat+0.00001) )
    
    #return quadratic_weighted_kappa(yhat, y)



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


def save2pickle(c,name):
    write_file=open(dataPath+str(name),'wb')
    cPickle.dump(c,write_file,-1)#[ (timestamp,[motion,x,y,z]),...]
    write_file.close()
 
def load_pickle(path_i):
    f=open(path_i,'rb')
    data=cPickle.load(f)#[ [time,[xyz],y] ,[],[]...]
    f.close()
    #print data.__len__(),data[0]
    return data	


if __name__=='__main__':

	 
	# XGBoost params:
	

	print('Load data...')
	train = pd.read_csv("../input/train.csv")
	target = train['target']
	train = train.drop(['ID','target'],axis=1)
	test = pd.read_csv("../input/test.csv")
	ids = test['ID'].values
	test = test.drop(['ID'],axis=1)
	#
	print('Clearing...')
	for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems())[:]:
	    # each columns
	    
	    if train_series.dtype == 'O':
		#for objects: factorize
		train[train_name], tmp_indexer = pd.factorize(train[train_name])
		test[test_name] = tmp_indexer.get_indexer(test[test_name])
		 
		#but now we have -1 values (NaN)
	    else:
		#for int or float: fill NaN
		tmp_len = len(train[train_series.isnull()]); 
		if tmp_len>0:
		    train.loc[train_series.isnull(), train_name] = train_series.mean()
		#and Test
		tmp_len = len(test[test_series.isnull()])
		if tmp_len>0:
		    test.loc[test_series.isnull(), test_name] = train_series.mean()  #TODO



	 

	




	#####################
	#xgboost
	###################
	missing_indicator=-1000
 	xgtrainAll = xgb.DMatrix(train.values, target.values,missing=missing_indicator);
	 

	###cross valid split trainset
	trainAllPredList=[]
	from sklearn.cross_validation import train_test_split
	for r in range(1):
		xtrain,xtest,ytrain,ytest=train_test_split(train.values,target.values,test_size=0.3)
		# convert data to xgb data structure
		
		xgtrain = xgb.DMatrix(xtrain, ytrain,missing=missing_indicator);
		print 'x y',xtrain.shape,ytrain.shape
		xgtest = xgb.DMatrix(xtest,missing=missing_indicator)
	 
	     

		# train model
		print('Fit different model...')
		for boost_round in [500,500][:1]:
		
		 
			for maxDepth in [64,14][:1]:#7  14
				xgboost_params = get_params(maxDepth)
			 
				# train model
			 
			
				#clf = xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round,verbose_eval=True,maximize=False)
				clf=xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round)

				# train test  error
				train_preds = clf.predict(xgtrain, ntree_limit=clf.best_iteration)
				test_preds=clf.predict(xgtest,ntree_limit=clf.best_iteration)
				print 'depth round',maxDepth,boost_round
				print('Train err is:', eval_wrapper(train_preds, ytrain))# 32 100  err0.10
				print('Test err is(see whether overfit:', eval_wrapper(test_preds, ytest))# 32 100  err0.10
				#for bagging
				trainAll_preds=clf.predict(xgtrainAll)
				trainAllPredList.append(trainAll_preds)
			 	#turn out overfitting, trainloss0.09 testloss 0.47
			 



	#####
	save2pickle([np.array(trainAllPredList),target.values],'baggingPred')
	 


	##bagging
	trainAllPred,target=load_pickle(dataPath+'baggingPred')
	print 'bagging',trainAllPred.shape,target.shape#[3,n]
	pred=np.sum(trainAllPred,axis=0)/float(trainAllPred.shape[0])
	print('bagging train err is:', eval_wrapper(pred, target))
	
	

	""" 
	#test predict
	print('Predict...')
	test_preds = clf.predict(xgtest, ntree_limit=clf.best_iteration)
	# Save results
	#
	preds_out = pd.DataFrame({"ID": ids, "PredictedProb": test_preds})
	preds_out.to_csv("../acc_process_submission.csv")
	#
	"""
	 
	 
	

	
	
	  
		
		
	
	
	

 	
 

	 




	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



