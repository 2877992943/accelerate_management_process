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


def str2IntSer26(strI):
	#if strI!=nan:
	#print '26',strI
	if isinstance(strI,str) and len(strI)>=1:
		n=len(strI);strI=strI.lower()
		#initial dic
		charList='a b c d e f g h i j k l m n o p q r s t u v w x y z'
		charList=charList.split(' ');freqList=[0]*len(charList)#indList=range(len(charList))
		strIndDic=dict(zip(charList,freqList))
		#
		for i in range(n):#'abd'
			strIndDic[strI[i] ]+=1
		return strIndDic.values()
	else:
		return np.zeros((26,))
		


def feaInt(v22,v22UniqueSort):
	v22UniqueSort=list(v22UniqueSort)
	dic=dict(zip(v22UniqueSort,range(len(v22UniqueSort))  ))	#{'abd':4....}
	return np.array([dic[strI] for strI in v22])
		
		
			
def getFreq(arr):
	##get dic
	dic={}
	for strI in arr:
		if strI not in dic:
			dic[strI]=1
		else:dic[strI]+=1
	#####
	freqList=[dic[strI] for strI in arr]
	return np.array(freqList)


if __name__=='__main__':
	# XGBoost params:
	

	print('Load data...')
	train = pd.read_csv("../input/train.csv");num_train=train.values.shape[0]
	target = train['target'];save2pickle(target.values,'target')
	train = train.drop(['ID','target'],axis=1)
	test = pd.read_csv("../input/test.csv")
	ids = test['ID'].values
	test = test.drop(['ID'],axis=1)
	#
	"""
	print('Clearing...')
	###############
	#v22 v56 v125 splitFea 'bcn'remain,add more variable,err not decrease
	#############3 
	train['v22_0']=train.v22.str[0]; 
	train['v22_1']=train.v22.str[:2];# train['v22_1']=train.v22.str[1]; 
	train['v22_2']=train.v22.str[:3]; #train['v22_2']=train.v22.str[2]; 
	train['v22_3']=train.v22.str[:4]; #train['v22_3']=train.v22.str[3]; 
	train['v22_4']=train.v22.str[-1:]
	train['v22_5']=train.v22.str[-2:]
	train['v56_0']=train.v56.str[0]; 
	train['v56_1']=train.v56.str[1];
	train['v125_0']=train.v125.str[0]; 
	train['v125_1']=train.v125.str[1];
	train['v113_0']=train.v113.str[0]
	train['v113_1']=train.v113.str[1]
	strList=['v22','v56','125','113']
	newfea=[]
	for strI in strList:
		for col in train.columns:
			if col.find(strI+'_')!=-1:
				print col
				serial=train[col].values
				print np.unique(serial).shape
				print np.unique(serial)[:50]
				#
				s, tmp_indexer = pd.factorize(train[col])
				print s.shape
				newfea.append(s)
	newfea=np.array(newfea).T#[d,n]	->[n,d]
	print newfea.shape#[n,10]
	save2pickle(newfea,'splitFea_1')
	"""
	

	"""
	###########
	#value count,split data by fea, fail
	##############
	discreteInd=[2,21,23,29,30,37,46,51,55,61,65,70,71,73,74,78,90,106,128,109,111,112,124]
	att='v'+str(discreteInd[2]+1)#v22 is 21
	series=train[att]
	print att,series.value_counts()	
	#######
	#v24-E,v47-C,v110-A  get index
	ind10=train[train['v24']=='E'].index;print ind10[:10],ind10.shape
	ind11=train[train['v24']!='E'].index
	ind20=train[train['v47']=='C'].index;print ind20[:10],ind20.shape
	ind21=train[train['v47']!='C'].index
	ind30=train[train['v110']=='A'].index;print ind30[:10],ind30.shape
	ind31=train[train['v110']!='A'].index
	save2pickle([ind10,ind11,ind20,ind21,ind30,ind31],'splitDataByFea')
	"""



	  
	############
	#encode v22 abdv v56 v125
	###############3
	v22=train.append(test).v22.values;print v22.shape,v22[0]
	#
	v22UniqueSort=np.sort(np.unique(v22))[::-1];print v22UniqueSort[-10:],v22UniqueSort.shape
	v22_int1=feaInt(v22,v22UniqueSort);#print v22_int.shape
	v22_int2=feaInt(v22,v22UniqueSort[::-1])
	v22_01dim26=np.array([str2IntSer26(strI) for strI in v22[:]]) ;print v22_01dim26.shape#'abd'->[26,]01
	##
	v56=train.append(test)['v56'].values;print v56.shape,v56[0]
	v56UniqueSort=np.sort(np.unique(v56))[::-1];print v56UniqueSort[-10:],v56UniqueSort.shape
	v56_int1=feaInt(v56,v56UniqueSort);#print v56_int.shape
	v56_int2=feaInt(v56,v56UniqueSort[::-1])
	v56_01dim26=np.array([str2IntSer26(strI) for strI in v56[:]]) ;print v56_01dim26.shape#'abd'->[26,]01
	##
	v125=train.append(test)['v125'].values;print v125.shape,v125[0]
	v125UniqueSort=np.sort(np.unique(v125))[::-1];print v125UniqueSort[-10:],v125UniqueSort.shape
	v125_int1=feaInt(v125,v125UniqueSort);#print v125_int.shape
	v125_int2=feaInt(v125,v125UniqueSort[::-1])
	v125_01dim26=np.array([str2IntSer26(strI) for strI in v125[:]]) ;print v125_01dim26.shape#'abd'->[26,]01


	save2pickle([v22_int1,v22_int2,v22_01dim26,v56_int1,v56_int2,v56_01dim26,v125_int1,v125_int2,v125_01dim26],'v22v56v125_newFea')
	 


	"""
	########
	#v22 count
	v22=train.append(test).v22 
	#print v22.value_counts();
	v22Freq=getFreq(v22.values)
	#
	v56=train.append(test).v56 
	v56Freq=getFreq(v56.values)
	#
	v125=train.append(test).v125 
	v125Freq=getFreq(v125.values)
	#
	save2pickle([v22Freq,v56Freq,v125Freq],'v22v56v125Freq')
	"""
	 
	
	 


	"""
	################3
	#corr
	################33
	continuInd=[0,1,3,22,72,110]+range(4,21)+range(24,29)+range(31,37)+range(38,46)+range(47,51)+range(52,55)+range(56,61)+range(62,65)+range(66,70)+range(75,78)+range(79,90)+range(91,106)+range(107,109)+range(113,124)+range(125,128)+range(129,131)
	continuInd1=['v'+str(i+1) for i in continuInd]
	df_c=train[continuInd1] 
	corrDF=df_c.corr()
	corrDF.to_csv('../corr_continu.csv')
	#closeEnoughVariable0=corrDF[corrDF>=0.9 and corrDF<1].columns
	#closeEnoughVariable1=corrDF[corrDF>=0.9 and corrDF<1].index
	corr= corrDF.values
	position=[]
	for row in range(corr.shape[0]):
		for col in range(corr.shape[1])[row:]:
			if row!=col and corr[row,col]>=0.95:
				print corr[row,col]
				print row,col
				position.append([row,col])
	#9-41-54  12-46 18-55-61 23-41-47-54-73-88 24-52  27-79 29-67  36-79 38-94  41-54 47-73  55-61 67-99 75-78 80-96 90-105
	delIndList=[continuInd[ pair[0] ] for pair in position]
	print set(delIndList)
	
	save2pickle(set(delIndList),'DelInd')
	"""
	
		
		
	
	


	#pad v22
	#train=pad(train)
	#

	"""
	test['v22_0']=test.v22.str[0]; 
	test['v22_1']=test.v22.str[1]; 
	test['v22_2']=test.v22.str[2]; 
	test['v22_3']=test.v22.str[3]; 
	test['v56_0']=test.v56.str[0]; 
	test['v56_1']=test.v56.str[1];
	test['v125_0']=test.v125.str[0]; 
	test['v125_1']=test.v125.str[1];
	""" 




	""" 
	#dropna not factorized
	train1=train.dropna(axis=1,how='any')#12 fea with all value
	train2=train.dropna(axis=0,how='any');print 'complete data',train2.values.shape #complete fea data
	test2=test.dropna(axis=0,how='any')
	train2test2=np.concatenate((train2.values,test2.values),axis=0);print train2test2.shape#not factorized
	print 'all value fea',train1.columns
	test1=test[train1.columns]

	#train=train1;test=test1 

	
	
	#
	  
	# fill na ,factorize str feature
	missFea=[];completeFea=[]
	feaInd=-1
	for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems())[:]:
	    feaInd+=1
	    # each columns,fea
	    valuePercnt_train=train[train_name].count()/float(train.values.shape[0])
	    valuePercnt_test=test[test_name].count()/float(test.values.shape[0])
	    #print 'non-nan value fea',train_name,train_series.dtype,valuePercnt_train,valuePercnt_test
	    ##
	    if train_series.dtype == 'O':
		#for objects: factorize
		
		
		train[train_name], tmp_indexer = pd.factorize(train[train_name]);
		#print np.unique(tmp_indexer).shape
		test[test_name] = tmp_indexer.get_indexer(test[test_name])
		if valuePercnt_test+valuePercnt_train<2.:missFea.append(feaInd)
		else:completeFea.append(feaInd)
		
		 
		#but now we have -1 values (NaN)
	    else:
		#print train_name,np.unique(train_series).shape
		 
		#for int or float: fill NaN with mean
		if valuePercnt_test+valuePercnt_train<2.:
			missFea.append(feaInd)
			tmp_len = len(train[train_series.isnull()]); 
			if tmp_len>0:
		    		train.loc[train_series.isnull(), train_name] = -1000
			#and Test
			tmp_len = len(test[test_series.isnull()])
			if tmp_len>0:
		    		test.loc[test_series.isnull(), test_name] = -1000   

			
		else:
			completeFea.append(feaInd)
			tmp_len = len(train[train_series.isnull()]); 
			if tmp_len>0:
		    		train.loc[train_series.isnull(), train_name] = train_series.mean()
			#and Test
			tmp_len = len(test[test_series.isnull()])
			if tmp_len>0:
		    		test.loc[test_series.isnull(), test_name] = train_series.mean()  #TODO

	"""



	"""
	print len(missFea),len(completeFea)
	##
	missInd=list(np.where(train.values==-1)[0])+list(np.where(train.values==-1000)[0])
	train1=train.drop(missInd,axis=0,inplace=False)
	missInd=list(np.where(test.values==-1)[0])+list(np.where(test.values==-1000)[0])
	test1=test.drop(missInd,axis=0,inplace=False)
	train2test2=np.concatenate((train1,test1),axis=0);print 'complete data',train2test2.shape
	save2pickle([missFea,completeFea,train.values,test.values,train2test2],'midData')
	"""  
	 


	"""


	 
	 
	#####################
	#xgboost
	###################
	# convert data to xgb data structure
	missing_indicator=-1000
	xgtrain = xgb.DMatrix(train.values, target.values,missing=missing_indicator);
	
	#xgtest = xgb.DMatrix(test,missing=missing_indicator)
 
	 

	 
	 
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
			print('Train err is:', eval_wrapper(train_preds, target.values))# 50 7 0.19
			 
	 		 
	 
	"""
	



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
	 
	 
	

	
	
	  
		
		
	
	
	

 	
 

	 




	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



