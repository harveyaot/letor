#! /usr/bin/env python
import numpy
import sys
import cPickle
import argparse
import time
import logging


import making
import config

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report

params = {'n_estimators': 300, 'max_depth':5, 'loss':"deviance"}
params = {'max_features':"auto",'learning_rate':0.5,'n_estimators': 500, 'max_leaf_nodes':50,'max_depth':10, 'loss':"deviance",'verbose':3}

clf  = GradientBoostingClassifier(**params)

def load_labeled_data(train_data):
   X,Y =  making.extract_feature_and_label(train_data)
   return X, Y
    
def load_weights(weight_data):
    weights = []
    for line in open(weight_data,'w'):
        g = line.strip()
        weights.append(flaot(g))
    return weights
    
def dump_model(model,dest):
    with open(dest, 'wb') as fid:
        cPickle.dump(model, fid)
    
def load_model(src):
    with open(src,'rb') as fid:
        model_loaded = cPickle.load(fid)
    return model_loaded

def get_weights(y):
    weights = []
    for i in y:
        if i == 1:
            weights.append(5)
        elif i == 0:
            weights.append(2)
        else:
            weights.append(1)
    return numpy.array(weights)
    
def run_cv():
    args = parser.parse_args()
    if args.train_data is not None:
        X,Y = load_labeled_data(args.train_data);
    else:
        print "please set train_data"
        return 
    X,Y = load_labeled_data(args.train_data)
    X_train, X_test, y_train, y_test = \
                    cross_validation.train_test_split(X,Y, \
                            test_size=0.6, random_state=0)
#   weights = get_weights(y_train);
    weights = None
    clf.fit(X_train,y_train,weights)
    model_out = args.model_out;
    dump_model(clf,model_out);
    y_predict = clf.predict(X_test)
    print classification_report(y_test,y_predict);
    
def run_train(args):
    if args.train_data is not None:
        logging.info("Begin Loading training data...")
        X,Y = load_labeled_data(args.train_data);
        logging.info("Finish Loading training data...")
    else:
        print "please set train_data"
        return 

    logging.info("Begin training ...")
    clf.fit(X,Y)
    logging.info("Finish training ...")
    model_out = args.model_out;
    dump_model(clf,model_out);
    logging.info("training over dump model to %s"%(model_out))

def run_report():
    args = parser.parse_args();
    model_in = args.model_in
    if args.predict_data is not None:
        X_p,Y_t = load_labeled_data(args.predict_data)
    else:
        print "please set predict_data"
        return 
    clf = load_model(model_in)
    Y_p = clf.predict(X_p)
    print
    print classification_report(Y_t,Y_p);

    
def run_predict(args):
    model_in = args.model_in
    if args.predict_data is not None:
        X_p,Y_t = load_labeled_data(args.predict_data)
    else:
        print "please set predict_data"
        return 
    clf = load_model(model_in)
    Y_pp = clf.predict_proba(X_p)
    Y_pc = clf.predict(X_p)
    print Y_pp, Y_pc

def run_predict_realtime():
    args = parser.parse_args();
    model_in = args.model_in
    X_p,Y_t = load_from_stdin()
    clf = load_model(model_in)
    Y_pp = clf.predict_proba(X_p)
    Y_pc = clf.predict(X_p)
    print Y_pp
    print Y_pc
    
def run_analyse(args):
    model_in = args.model_in
    
    clf = load_model(model_in)
    print clf.feature_importances_
    
def run_debug():
    import debug_user
    args = parser.parse_args();
    model_in = args.model_in
    if args.predict_data is not None:
       users,X_p,Y_t,lls,debugs = debug_user.load_labeled_data(args.predict_data)
    else:
        print "please set predict_data"
        return 
    clf = load_model(model_in)
    Y_p = clf.predict(X_p)
    Y_proba = clf.predict_proba(X_p)
    debug_user.debug_for_user(users,X_p,Y_p,Y_t,lls,debugs,Y_proba)

def run_user_report():
    import debug_user
    args = parser.parse_args();
    model_in = args.model_in
    clf = load_model(model_in)
    if args.predict_data is not None:
       users,X_p,Y_t,lls,debugs = debug_user.load_labeled_data(args.predict_data)
    Y_p = debug_user.predict_for_user(users,X_p,clf) 
    print
    print classification_report(Y_t,Y_p);
    
def run(locals):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--task",help="The tasks are:train,predict,cv")
    parser.add_argument("-tf","--train_data")
    parser.add_argument("-pf","--predict_data")
    parser.add_argument("-wf","--weight_data")
    parser.add_argument("-mi","--model_in",default='ml_tmp.mod')
    parser.add_argument("-mo","--model_out",default='ml_tmp.mod')

    args = parser.parse_args();

    if args.task is not None:
        task = args.task.lower()
        if task in ['train','predict','cv','analyse','report','debug','user_report']:
            locals['run_%s'%task](args);
        else:
            print >> sys.stderr,"Please Make sure task type are:train,predict,cv,analyse,report,debug"
    else:
        print >> sys.stderr, "Please set the task type."
        sys.exit(-1)

if __name__ == "__main__":
    run(locals());
