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


def pad(train):
	train.v22.fillna('',inplace=True)
	padded=train.v22.str.pad(4)
	spadded=sorted(np.unique(padded))
	v22_map={}
	c=0
	for i in spadded:
		v22_map[i]=c
		c+=1
	train.v22=padded.replace(v22_map,inplace=False)
	return train


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

def calcDist(dataC,xi,xiObsInd,xiMissInd):
	n=dataC.shape[0]
	xObs=xi[xiObsInd]
	xMiss=xi[xiMissInd]
	dist=np.tile(xObs.reshape((1,-1)),(n,1)) - dataC[:,xiObsInd]#[n,d]
	dist=np.sum(dist*dist,axis=1)
	
	distInd=np.argsort(dist)[:10]
	return distInd



def vote(vec):
	dic={}
	for v in vec:
		if v not in dic:
			dic[v]=1
		else:dic[v]+=1
	ll=sorted(dic.iteritems(),key=lambda asd:asd[1],reverse=False)
	#print dic
	return ll[-1][0]

if __name__=='__main__':
	""" 
	###
	trainset fill nan with knn
	######3
	missFea,completeFea,train,test,dataComplete=load_pickle(dataPath+'midData')
	print len(missFea),len(completeFea),train.shape,test.shape,dataComplete.shape
	num,dim=train.shape
	for i in range(num)[:]:
		#for each xi
		xi=train[i,:]
		#get missing value xi
		if -1 in xi or -1000 in xi:#discrete continuous value
			xiMissInd=list(np.where(xi==-1)[0]) + list(np.where(xi==-1000)[0]);#print 'miss',xiMissInd
			xiObsInd=list(np.where(xi>0)[0])
			xObs=xi[xiObsInd]
			xMiss=xi[xiMissInd]
			## get candidate 10
			distListInd=calcDist(dataComplete,xi,xiObsInd,xiMissInd)#[n,131]->[10,]candidate
			candidate=dataComplete[distListInd,:]#[10,131]
			for d in xiMissInd:
				if xi[d]==-1:#discrete
					majority=vote(candidate[:,d])
					xi[d]=majority;#print majority
				if xi[d]==-1000:#continuous
					xi[d]=np.mean(candidate[:,d]);#print xi[d]
		#
		train[i,:]=xi

	print train.shape
	save2pickle(train,'missHandled_train')
	 


	##########3
	#test set
	missFea,completeFea,train,test,dataComplete=load_pickle(dataPath+'midData')
	print len(missFea),len(completeFea),train.shape,test.shape,dataComplete.shape
	num,dim=test.shape
	for i in range(num)[:]:
		#for each xi
		xi=test[i,:]
		#get missing value xi
		if -1 in xi or -1000 in xi:#discrete continuous value
			xiMissInd=list(np.where(xi==-1)[0]) + list(np.where(xi==-1000)[0]);#print 'miss',xiMissInd
			xiObsInd=list(np.where(xi>0)[0])
			xObs=xi[xiObsInd]
			xMiss=xi[xiMissInd]
			## get candidate 10
			distListInd=calcDist(dataComplete,xi,xiObsInd,xiMissInd)#[n,131]->[10,]candidate
			candidate=dataComplete[distListInd,:]#[10,131]
			for d in xiMissInd:
				if xi[d]==-1:#discrete
					majority=vote(candidate[:,d])
					xi[d]=majority;#print majority
				if xi[d]==-1000:#continuous
					xi[d]=np.mean(candidate[:,d]);#print xi[d]
		#
		test[i,:]=xi

	print test.shape
	save2pickle(test,'missHandled_test')
	"""





	 


	#####################
	#xgboost
	###################
	# convert data to xgb data structure
	train=load_pickle(dataPath+'missHandled_train');print train.shape
	test=load_pickle(dataPath+'missHandled_test')
	target=load_pickle(dataPath+'target')
	missing_indicator=-1
	xgtrain = xgb.DMatrix(train, target);
	
	xgtest = xgb.DMatrix(test)
 
	 

	 	
	 
	# train model
	print('Fit different model...')
	for boost_round in [50,100][:1]:
		
		 
		for maxDepth in [7,14][:1]:#7  14
			xgboost_params = get_params(maxDepth)
			 
			# train model
			 
			
			#clf = xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round,verbose_eval=True,maximize=False)
			clf=xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round)

			# train error
			train_preds = clf.predict(xgtrain, ntree_limit=clf.best_iteration)
			print maxDepth,boost_round
			print('Train err is:', eval_wrapper(train_preds, target))# 50 7 0.19
				
				
			
			
		




	""" 
	################
	#test predict
	print('Predict...')
	test_preds = clf.predict(xgtest, ntree_limit=clf.best_iteration)
	# Save results
	#
	#preds_out = pd.DataFrame({"ID": ids, "PredictedProb": test_preds})
	preds_out = pd.DataFrame({"PredictedProb": test_preds})
	preds_out.to_csv("../acc_process_submission0223.csv")	
	""" 
	
	 


	 
	 
	 
	

	
	
	  
		
		
	
	
	

 	
 

	 




	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



