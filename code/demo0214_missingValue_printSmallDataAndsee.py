#!/usr/bin/env python
# encoding=utf-8






 
import pandas as pd
import xgboost as xgb

from ml_metrics import quadratic_weighted_kappa

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)
    
def get_params(maxDepth):
    
    plst={ 
  	"objective": "binary:logistic",
   	"booster": "gbtree",
   	"eval_metric": "auc",
  	"eta": 0.01, # 0.06, #0.01,
  	#"min_child_weight": 240,
   	"subsample": 0.75,
   	"colsample_bytree": 0.68,
   	"max_depth": maxDepth
	}

    return plst


if __name__=='__main__':
	# XGBoost params:
	xgboost_params = get_params(7)

	print('Load data...')
	train = pd.read_csv("../input/train1.csv")
	target = train['target']
	train = train.drop(['ID','target'],axis=1)
	test = pd.read_csv("../input/test1.csv")
	ids = test['ID'].values
	test = test.drop(['ID'],axis=1)
	#
	print('Clearing...')
	#split str 'be'->b e
	train['v22_0']=train.v22.str[0];print train['v22_0']
	train['v22_1']=train.v22.str[1];print train['v22_1']
	train['v22_2']=train.v22.str[2];print train['v22_2'] 
	train['v22_3']=train.v22.str[3];print train['v22_3']
	#train['v22_0']=pd.factorize(train['v22_0'])[0];print train['v22_0']
	#train['v22_1']=pd.factorize(train['v22_1'])[0];print train['v22_1']
	#train['v22_2']=pd.factorize(train['v22_2'])[0];print train['v22_2'] 
	#train['v22_3']=pd.factorize(train['v22_3'])[0];print train['v22_3']
	 
	"""
	#
	for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems())[:]:
	    # each columns
	    print '1 series type  train_name',train_series.dtype,train_name
	    if train_series.dtype == 'O':
		#for objects: factorize
		train[train_name], tmp_indexer = pd.factorize(train[train_name])
		test[test_name] = tmp_indexer.get_indexer(test[test_name])
		print 'o',train[train_name].values
		#but now we have -1 values (NaN)
	    else:
		#for int or float: fill NaN
		tmp_len = len(train[train_series.isnull()]);print 'null len:',tmp_len,'null:',train_series.isnull()
		if tmp_len>0:
		    train.loc[train_series.isnull(), train_name] = train_series.mean()
		#and Test
		tmp_len = len(test[test_series.isnull()])
		if tmp_len>0:
		    test.loc[test_series.isnull(), test_name] = train_series.mean()  #TODO
	"""







	"""
	xgtrain = xgb.DMatrix(train.values, target.values)
	xgtest = xgb.DMatrix(test.values)

	#train
	print('Fit the model...')
	boost_round = 5 #1800 CHANGE THIS BEFORE START
	clf = xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round,verbose_eval=True,maximize=False)

	# train error
	train_preds = clf.predict(xgtest, ntree_limit=clf.best_iteration)
	print('Train score is:', eval_wrapper(train_preds, target.values))

	#test predict
	print('Predict...')
	test_preds = clf.predict(xgtest, ntree_limit=clf.best_iteration)
	# Save results
	#
	preds_out = pd.DataFrame({"ID": ids, "PredictedProb": test_preds})
	preds_out.to_csv('xgb_offset_submission.csv')
	#
	"""
	

	
	
	  
		
		
	
	
	

 	
 

	 




	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



