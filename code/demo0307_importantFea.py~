#!/usr/bin/env python
# encoding=utf-8




import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
import cPickle
import pylab as plt
from sklearn.ensemble import RandomForestClassifier



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
	 
	train=load_pickle(dataPath+'missHandled_train');num_train=train.shape[0];print train.shape
	test=load_pickle(dataPath+'missHandled_test');print test.shape
	target=load_pickle(dataPath+'target')
	splitFea=load_pickle(dataPath+'splitFea')#[n,d]
	v22_int,v22_01dim26,v56_int,v56_01dim26,v125_int,v125_01dim26=load_pickle(dataPath+'v22v56v125_newFea')
	delInd=load_pickle(dataPath+'DelInd')
	#
	v22_int=np.concatenate((v22_int.reshape((-1,1)),v56_int.reshape((-1,1)),v125_int.reshape((-1,1)) ),axis=1)#[n,d]
	v22_01dim26=np.concatenate((v22_01dim26,v56_01dim26,v125_01dim26),axis=1)#[n,d]
	print v22_int.shape,v22_01dim26.shape
	v22Freq,v56Freq,v125Freq=load_pickle(dataPath+'v22v56v125Freq')
	v22Freq=np.concatenate((v22Freq.reshape((-1,1)),v56Freq.reshape((-1,1)),v125Freq.reshape((-1,1)) ),axis=1)
	 

	############3
	#continue variable pca
	print 'continuous variable...'
	continuInd=[0,1,3,22,72,110]+range(4,21)+range(24,29)+range(31,37)+range(38,46)+range(47,51)+range(52,55)+range(56,61)+range(62,65)+range(66,70)+range(75,78)+range(79,90)+range(91,106)+range(107,109)+range(113,124)+range(125,128)+range(129,131);print continuInd.__len__()

	 
	xc=train[:,continuInd];print 'xc',xc.shape #[n,131]->[131,n]
	#xc1=test[:,continuInd];print xc1.shape
	 
	 
	
	
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
	 
	 
		
	
	
	 

	

	 
	#dummy 195 continue108  26 3 3
	train=np.concatenate((xdd[:num_train,:],xc,v22_01dim26[:num_train,:],v22_int[:num_train,:],v22Freq[:num_train,:] ),axis=1) #v22_int.reshape((-1,1)),v22_01dim26,splitFea,train[:,21].reshape((-1,1))
	 

	 

	X=train;y=target
	rf=RandomForestClassifier(n_estimators=100)
	rf.fit(X,y)
	feat_imp=pd.Series(rf.feature_importances_,index=range(train.shape[1]) )
	feat_imp.sort_values(inplace=True)
	ax=feat_imp.tail(50).plot(kind='barh',figsize=(10,7),title='feature importance')

	plt.show()
	
	
	  
		
		
	
	
	

 	
 

	 




	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



