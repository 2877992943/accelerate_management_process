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



def pca(x):
	x=x.T#[n,131]->[131,n]
	m=np.mean(x,axis=1);print 'mean',m.shape #[131,]
	removeMean=x-np.tile(m.reshape((-1,1)),(1,x.shape[1]))  #[131,1]->[131,n]
	cov=np.dot(removeMean,removeMean.T);print 'cov',cov.shape
	Sigma,U=np.linalg.eig(np.mat(cov));print 'u sigma',U.shape,Sigma.shape#[d,d][d,]
	SigmaInd=np.argsort(Sigma)[:-100000:-1];print SigmaInd.shape
	Sigma=Sigma[SigmaInd]
	U=U[:,SigmaInd]

	#energy of sigma
	Sigma2=Sigma*Sigma
	energy=0;tt=np.dot(Sigma,Sigma)
	for i in range(Sigma.shape[0]):
		energy+=Sigma2[i]
		#print energy/tt
		if energy/tt>=0.99:
			print '0.99',i
			break

	##
	U=U[:,:i]#[d,30]
	xrot=U.T*removeMean #[30,d][d,n] ->[30,n]
	xrot=xrot.T #[n,30]
	return xrot



def dummy(x):#[n,d]
	#df=pd.DataFrame(x)#array->dataFrame
	for d in range(x.shape[1]):
		serial=x[:,d]
		
		mat=pd.get_dummies(serial)#[n,di]
		if d==0:
			dmMat=mat
		else:dmMat=np.concatenate((dmMat,mat),axis=1)
	return dmMat



def get_testID():
	test = pd.read_csv("../input/test.csv")
	ids = test['ID'].values
	return ids	

if __name__=='__main__':
	""" 
	###
	#trainset fill nan with knn
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
	#test set fill nan with knn
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





	 

	train=load_pickle(dataPath+'missHandled_train');num_train=train.shape[0];print train.shape
	test=load_pickle(dataPath+'missHandled_test');print test.shape
	target=load_pickle(dataPath+'target')
	splitFea=load_pickle(dataPath+'splitFea')#[n,d]
	v22_int1,v22_int2,v22_01dim26,v56_int1,v56_int2,v56_01dim26,v125_int1,v125_int2,v125_01dim26=load_pickle(dataPath+'v22v56v125_newFea')
	delInd=load_pickle(dataPath+'DelInd')
	#
	#v22_int=np.concatenate((v22_int1.reshape((-1,1)),v22_int2.reshape((-1,1)),v56_int1.reshape((-1,1)),v56_int2.reshape((-1,1)),v125_int1.reshape((-1,1)),v125_int2.reshape((-1,1)) ),axis=1)#[n,d]
	v22_int=np.concatenate((v22_int1.reshape((-1,1)),v56_int1.reshape((-1,1)),v125_int1.reshape((-1,1)) ),axis=1)#[n,d]
	v22_01dim26=np.concatenate((v22_01dim26,v56_01dim26,v125_01dim26),axis=1)#[n,d]
	print v22_int.shape,v22_01dim26.shape
	v22Freq,v56Freq,v125Freq=load_pickle(dataPath+'v22v56v125Freq')
	v22Freq=np.concatenate((v22Freq.reshape((-1,1)),v56Freq.reshape((-1,1)),v125Freq.reshape((-1,1)) ),axis=1)
	""" 
	########
	##svd memory error
	from numpy import linalg
	U,Sigma,VT=linalg.svd(train[:,:10])
	save2pickle([U,Sigma,VT],'svd')
	print U.shape,Sigma.shape,VT.shape
	#energy of sigma
	Sigma2=Sigma*Sigma
	energy=0;tt=np.dot(Sigma,Sigma)
	for i in range(Sigma.shape[0]):
		energy+=Sigma2[i]
		if energy/tt>=0.99:
			print i
			break
	"""

	############3
	#continue variable pca
	print 'continuous variable...'
	continuInd=[0,1,3,22,72,110]+range(4,21)+range(24,29)+range(31,37)+range(38,46)+range(47,51)+range(52,55)+range(56,61)+range(62,65)+range(66,70)+range(75,78)+range(79,90)+range(91,106)+range(107,109)+range(113,124)+range(125,128)+range(129,131);print continuInd.__len__()

	####remove corr close dimension
	#continuInd1=list(set(continuInd)-delInd)
	#continuInd=continuInd1 
	xc=train[:,continuInd];print 'x',xc.shape #[n,131]->[131,n]
	xc1=test[:,continuInd];print xc1.shape
	 
	
	#xcrot=pca(xc)#[n,131]->[n,30]
	#polynomial feature
	from sklearn.preprocessing import PolynomialFeatures
	#poly=PolynomialFeatures(degree=2,interaction_only=True)
	#xcrot=poly.fit_transform(xcrot);print xcrot.shape
	
	
	###############
	#discrete 2 dummy
	print 'discrete variable...'
	discreteInd=[2,23,29,30,37,46,51,61,65,70,71,73,74,78,90,106,128,109,111,112]
	#discreteInd=[2,23,29,30,37,46,51,55,61,65,70,71,74,78,90,106,128,112]
	discreteInd2=[21,55,124]#v22 v56 v125
	xd=train[:,discreteInd];print 'x',xd.shape
	xd1=test[:,discreteInd];print xd1.shape
	xd2=np.concatenate((xd,xd1),axis=0)
	for col in range(xd.shape[1]):
		print  discreteInd[col],np.unique(xd2[:,col]).shape

	xdd=dummy(xd2) 
	print 'dummy',xdd.shape#[n,d]->[n,dd]
	 
	#xddrot=pca(xdd);print xddrot.shape
	#xddrot=poly.fit_transform(xddrot);print xddrot.shape
	 
		
	
	
	 

	

	 
	 
	#####
	#not pca   	
	## 
	
	#train=np.concatenate((xdd,xc),axis=1);print 'tt dim',train.shape#loss 0.5718 0.5739
	#train=np.concatenate((xd,xc),axis=1);print 'tt dim',train.shape#loss 0.5706 0.5752 
	#train=np.concatenate((xdd,xc,train[:,21].reshape((-1,1)),splitFea),axis=1);print 'tt dim',train.shape#0.5708 0.5725
	train=np.concatenate((xdd[:num_train,:],xc,v22_01dim26[:num_train,:],v22_int[:num_train,:],train[:,21].reshape((-1,1)),v22Freq[:num_train,:] ),axis=1) #v22_int.reshape((-1,1)),v22_01dim26,splitFea,train[:,21].reshape((-1,1))
	test=np.concatenate((xdd[num_train:,:],xc1,v22_01dim26[num_train:,:],v22_int[num_train:,:],test[:,21].reshape((-1,1)),v22Freq[num_train:,:]),axis=1)
 	
	###########
	#pca
	##
	#train=np.concatenate((xddrot,xcrot,train[:,21].reshape((-1,1)) ),axis=1);print 'tt dim',train.shape#.5773.5840
	#train=np.concatenate((xddrot,xcrot ),axis=1);print 'tt dim',train.shape#loss 0.5780 0.5842
	#####################
	#xgboost
	###################
	# convert data to xgb data structure
	from sklearn.cross_validation import train_test_split
	missing_indicator=-1
	xtrain,xtest,ytrain,ytest=train_test_split(train,target,test_size=0.3)
	xgtrain = xgb.DMatrix(xtrain, ytrain,missing=missing_indicator);
	print 'x y',xtrain.shape,ytrain.shape,test.shape
	xgtest = xgb.DMatrix(xtest,missing=missing_indicator)
	
	xgtest1 = xgb.DMatrix(test,missing=missing_indicator)# no label
 
	 

	 	
	 
	# train model
	minValidLoss=10
	print('Fit different model...')
	for boost_round in [500,500][:1]:
		
		 
		for maxDepth in [10,14][:1]:#7  14
			xgboost_params = get_params(maxDepth)
			 
			# train model
			 
			
			#clf = xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round,verbose_eval=True,maximize=False)
			clf=xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round)

			# train test  error
			train_preds = clf.predict(xgtrain, ntree_limit=clf.best_iteration)
			test_preds=clf.predict(xgtest,ntree_limit=clf.best_iteration)
			print 'depth round',maxDepth,boost_round
			print('Train err is:', eval_wrapper(train_preds, ytrain))# 32 100  err0.10
			print('Test err is(see whether overfit:', eval_wrapper(test_preds, ytest))
			#update min loss
			#validLoss=eval_wrapper(test_preds, ytest)
			#if validLoss<minValidLoss:
				#minValidLoss=validLoss
				#print('Predict...')
				#test_preds1 = clf.predict(xgtest1, ntree_limit=clf.best_iteration)
				 
	  
	 
	  
				
				
			
			
		




	  
 	 
	###########
	#Save results
	#
	
	print('Predict...')
	test_preds1 = clf.predict(xgtest1, ntree_limit=clf.best_iteration)

	ids=get_testID()
	preds_out = pd.DataFrame({"ID": ids, "PredictedProb": test_preds1})
	 
	#
	
	preds_out.to_csv("../acc_process_submission0307.csv")	
	 
	 


	 


#####
#pca continuVariable  loss 0.5899
#continuVarible loss 0.5813	 
	 
##not pca continuVariable only trainLoss 0.5789  testLoss0.5841	
##pca continuVariable only     trainLoss 0.5891  testLoss 0.5928
#polynomial continuVariable
# dummy(discreteVariable)_notPca + continuVariable_pca  trainloss 0.5780  testloss 0.5817

	
	
	  
		
		
	
	
	

 	
 

	 




	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



