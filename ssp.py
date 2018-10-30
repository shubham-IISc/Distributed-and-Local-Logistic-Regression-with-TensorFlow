
# coding: utf-8

# In[1]:



import time
import re
import string
import collections
import tensorflow as tf
import numpy as np
import nltk
from collections import Counter,defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse import load_npz
import argparse 
import sys 
import random
import os
from tensorflow.python import debug as tf_debug
import h5py 


# In[2]:


#test accuracy
def test_accuracy(Y_test,prediction_te):
    count=0
    for i in range(prediction_te[0].shape[0]):
        if(Y_test[i][prediction_te[0][i]]==1):
            count=count+1
    return count/np.float(prediction_te[0].shape[0])





#training accuracy
def train_accuracy(Y_train,prediction_tr):
    count=0
    for i in range(len(prediction_tr)):
        if(Y_train[i][prediction_tr[i]]>0):
            count=count+1
    print(count/np.float(len(prediction_tr)))



#reading data
X_train=load_npz('X_train_sparse.npz');
X_test=load_npz('X_test_sparse.npz');
Y_train=np.load('Y_train.npy');
Y_test=np.load('Y_test.npy')


# In[ ]:


#server settings
parameter_servers=["10.1.1.252:2225"];
workers=["10.1.1.253:2223","10.1.1.251:2224"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers,"worker":workers})
tf.app.flags.DEFINE_string("job_name","","'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index",0,"Index of task within the job")
FLAGS=tf.app.flags.FLAGS

#cluster settings
config=tf.ConfigProto();
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
config.log_device_placement=True

server=tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index,config=config)


# In[ ]:


if FLAGS.job_name=='ps':
    server.join()
elif FLAGS.job_name=='worker':
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
        stale_parameter=8
        reg_constant =0.001
        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale = reg_constant) 
        num_epochs = 50
        batch_size = 5000
        num_batches_per_epoch=int(X_train.shape[0]/batch_size)
        num_batches_per_epoch_test=int(X_test.shape[0]/batch_size)
        num_examples_test=X_test.shape[0]
        num_examples=X_train.shape[0]
        num_features=X_train.shape[1]
        num_classes=Y_train.shape[1]
        inputs = tf.placeholder(shape=[None, num_features],dtype=tf.float32)
        outputs= tf.placeholder(shape=[None, num_classes],dtype=tf.float32)

        W=tf.get_variable(name = "W", shape = [num_features,num_classes],initializer=initializer,regularizer=regularizer) 
        b=tf.get_variable(name = "b", shape = [num_classes],initializer=initializer)
        logits = tf.add(tf.matmul(inputs, W),b)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=outputs))


        logits_pr=tf.nn.softmax(logits)
        pred=tf.argmax(logits,axis=1)

        #regularization losses
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        starter_learning_rate = 0.002
        global_step = tf.get_variable('global_step', [],initializer = tf.constant_initializer(0),trainable = False,dtype=tf.int32)
        # final loss and optimization
        final_cost=cost+reg_constant*sum(reg_losses)
        
        #optimization
        lr= tf.train.exponential_decay(starter_learning_rate, global_step,
                                               num_batches_per_epoch, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        
        replicated_opt=tf.contrib.opt.DropStaleGradientOptimizer(opt=optimizer,staleness=stale_parameter)
        
        optimizer_final = replicated_opt.minimize(final_cost, global_step=global_step)
        
        init = tf.global_variables_initializer()
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),global_step=global_step,init_op=init)
    
    
    with sv.prepare_or_wait_for_session(server.target) as session:
        
        #iterating over total number of epochs
        begin=time.time()
        for curr_epoch in range(num_epochs):
            start=time.time()
            train_cost=0
            for batch in range(num_batches_per_epoch):

                # Getting the index
                indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
                batch_train_inputs = X_train.tocsr()[indexes].toarray()
                batch_train_targets= Y_train[indexes]
                # feed dictionay for training
                feed = {inputs: batch_train_inputs,
                        outputs: batch_train_targets
                        }

               
                batch_cost, _ = session.run([final_cost, optimizer_final], feed)
                train_cost += batch_cost*batch_size
            log = "Epoch {}/{}, train_cost = {:.3f}, test_accuracy = {:.3f}, epoch time = {:.3f}s, total time = {:.3f}s"
            test_inputs = X_test.tocsr().toarray()
            feed = {inputs: test_inputs}
            prediction_te = session.run([pred], feed)
            acc=test_accuracy(Y_test,prediction_te)
            print(log.format(curr_epoch+1, num_epochs, train_cost/(num_batches_per_epoch*batch_size),acc,time.time()-start,time.time()-begin))
    
            
    sv.stop()
    print('Model has been trained ')

        

