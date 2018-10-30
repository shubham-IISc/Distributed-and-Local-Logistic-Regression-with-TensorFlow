
# coding: utf-8

# In[67]:


import re
import string
import collections
import tensorflow as tf
import numpy as np
import nltk
import time
from collections import Counter,defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse import load_npz



# In[68]:


#test accuracy
def test_accuracy(Y_test,prediction_te):
    count=0
    for i in range(prediction_te[0].shape[0]):
        #print(prediction[0][i])
        if(Y_test[i][prediction_te[0][i]]==1):
            count=count+1
    return count/np.float(prediction_te[0].shape[0])


# In[69]:


#training accuracy
def train_accuracy(Y_train,prediction_tr):
    count=0
    for i in range(len(prediction_tr)):
        if(Y_train[i][prediction_tr[i]]>0):
            count=count+1
    return count/np.float(len(prediction_tr))


# In[70]:


X_train=load_npz('X_train_sparse.npz');
X_testthree_w_1[i]=load_npz('X_test_sparse.npz');
Y_train=np.load('Y_train.npy');
Y_test=np.load('Y_test.npy')


# In[72]:


#with regularization and constant learning rate

reg_constant =0.001
initializer = tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer(scale = reg_constant) 
num_epochs = 100
batch_size = 5000
num_batches_per_epoch=int(X_train.shape[0]/batch_size)
num_batches_per_epoch_test=int(X_test.shape[0]/batch_size)
num_examples_test=X_test.shape[0]
num_examples=X_train.shape[0]
num_features=X_train.shape[1]
num_classes=Y_train.shape[1]

graph = tf.Graph()
with graph.as_default():
   
    inputs = tf.placeholder(shape=[None, num_features],dtype=tf.float32)
    outputs= tf.placeholder(shape=[None, num_classes],dtype=tf.float32)
    
    W=tf.get_variable(name = "W", shape = [num_features,num_classes],initializer=initializer,regularizer=regularizer) 
    b=tf.get_variable(name = "b", shape = [num_classes],initializer=initializer)
    logits = tf.add(tf.matmul(inputs, W),b)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=outputs))


    logits_pr=tf.nn.softmax(logits)
    pred=tf.argmax(logits,axis=1)
    
    #regularization losses
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    # final loss and optimization
    final_cost=cost+reg_constant*sum(reg_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(final_cost) 
test_acc_c=[]
loss_constant_lr=[]
with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()
    begin=time.time()
    #iterating over total number of epochs
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
           
            batch_cost, _ = session.run([final_cost, optimizer], feed)
            train_cost += batch_cost*batch_size
    
        log = "Epoch {}/{}, train_cost = {:.3f}, test_accuracy = {:.3f}, epoch time = {:.3f}s, total time = {:.3f}s"
        
        loss_constant_lr.append(train_cost/(num_batches_per_epoch*batch_size))
        
        test_inputs = X_test.tocsr().toarray()
        feed = {inputs: test_inputs}
        prediction_te = session.run([pred], feed)
        acc=test_accuracy(Y_test,prediction_te)
        test_acc_c.append(acc)
        print(log.format(curr_epoch+1, num_epochs, train_cost/(num_batches_per_epoch*batch_size),acc,time.time()-start,time.time()-begin))

    prediction_tr=[]
    for batch in range(num_batches_per_epoch):
                                                                   
        # Getting the index
        indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
        batch_train_inputs = X_train.tocsr()[indexes].toarray()
        feed = {inputs: batch_train_inputs}
        prediction = session.run([pred], feed)
        prediction_tr.extend(list(prediction[0]))
    test_inputs = X_test.tocsr().toarray()
    feed = {inputs: test_inputs}
    prediction_te = session.run([pred], feed)
    


# In[73]:


tr_acc=train_accuracy(Y_train,prediction_tr)
print("accuracy on training data with constant learning rate is ",tr_acc)
te_acc=test_accuracy(Y_test,prediction_te)
print("accuracy on test data with constant learning rate is ",te_acc)


# In[74]:


#exponential decay

reg_constant =0.001
initializer = tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer(scale = reg_constant) 
num_epochs = 100
batch_size = 5000
num_batches_per_epoch=int(X_train.shape[0]/batch_size)
num_batches_per_epoch_test=int(X_test.shape[0]/batch_size)
num_examples_test=X_test.shape[0]
num_examples=X_train.shape[0]
num_features=X_train.shape[1]
num_classes=Y_train.shape[1]

graph = tf.Graph()
with graph.as_default():
    starter_learning_rate = 0.002
    global_step = tf.get_variable('global_step', [],initializer = tf.constant_initializer(0),trainable = False)
    inputs = tf.placeholder(shape=[None, num_features],dtype=tf.float32)
    outputs= tf.placeholder(shape=[None, num_classes],dtype=tf.float32)
    
    W=tf.get_variable(name = "W", shape = [num_features,num_classes],initializer=initializer,regularizer=regularizer) 
    b=tf.get_variable(name = "b", shape = [num_classes],initializer=initializer)
    logits = tf.add(tf.matmul(inputs, W),b)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=outputs))


    logits_pr=tf.nn.softmax(logits)
    pred=tf.argmax(logits,axis=1)
    
    #regularization losses
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    # final loss and optimization
    final_cost=cost+reg_constant*sum(reg_losses)
    lr= tf.train.exponential_decay(starter_learning_rate, global_step,
                                           num_batches_per_epoch, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(final_cost,global_step=global_step) 
test_acc_d=[]
loss_decay_lr=[]
with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()
    begin=time.time()
    #iterating over total number of epochs
    for curr_epoch in range(num_epochs):
        train_cost=0
        start = time.time()
        for batch in range(num_batches_per_epoch):

            # Getting the index
            indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
            batch_train_inputs = X_train.tocsr()[indexes].toarray()
            batch_train_targets= Y_train[indexes]
            # feed dictionay for training
            feed = {inputs: batch_train_inputs,
                    outputs: batch_train_targets
                    }
           
            batch_cost, _ = session.run([final_cost, optimizer], feed)
            train_cost += batch_cost*batch_size
        log = "Epoch {}/{}, train_cost = {:.3f}, test_accuracy = {:.3f}, epoch time = {:.3f}s, total time = {:.3f}s"
        loss_decay_lr.append(train_cost/(num_batches_per_epoch*batch_size))
        test_inputs = X_test.tocsr().toarray()
        feed = {inputs: test_inputs}
        prediction_te = session.run([pred], feed)
        acc=test_accuracy(Y_test,prediction_te)
        test_acc_d.append(acc)
        print(log.format(curr_epoch+1, num_epochs, train_cost/(num_batches_per_epoch*batch_size),acc,time.time()-start,time.time()-begin))

    prediction_tr=[]
    for batch in range(num_batches_per_epoch):
                                                                   
        # Getting the index
        indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
        batch_train_inputs = X_train.tocsr()[indexes].toarray()
        feed = {inputs: batch_train_inputs}
        prediction = session.run([pred], feed)
        prediction_tr.extend(list(prediction[0]))
    test_inputs = X_test.tocsr().toarray()
    feed = {inputs: test_inputs}
    prediction_te = session.run([pred], feed)
    


# In[75]:


tr_acc=train_accuracy(Y_train,prediction_tr)
print("accuracy on training data with exponential decay learning rate is ",tr_acc*100.0,"%")
te_acc=test_accuracy(Y_test,prediction_te)
print("accuracy on test data with exponential decay learning rate is ",te_acc*100.0,"%")


# In[76]:


#exponential increase
reg_constant =0.001
initializer = tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer(scale = reg_constant) 
num_epochs = 100
batch_size = 5000
num_batches_per_epoch=int(X_train.shape[0]/batch_size)
num_batches_per_epoch_test=int(X_test.shape[0]/batch_size)
num_examples_test=X_test.shape[0]
num_examples=X_train.shape[0]
num_features=X_train.shape[1]
num_classes=Y_train.shape[1]

graph = tf.Graph()
with graph.as_default():
    starter_learning_rate = 0.002
    global_step = tf.get_variable('global_step', [],initializer = tf.constant_initializer(0),trainable = False)
    inputs = tf.placeholder(shape=[None, num_features],dtype=tf.float32)
    outputs= tf.placeholder(shape=[None, num_classes],dtype=tf.float32)
    
    W=tf.get_variable(name = "W", shape = [num_features,num_classes],initializer=initializer,regularizer=regularizer) 
    b=tf.get_variable(name = "b", shape = [num_classes],initializer=initializer)
    logits = tf.add(tf.matmul(inputs, W),b)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=outputs))


    logits_pr=tf.nn.softmax(logits)
    pred=tf.argmax(logits,axis=1)
    
    #regularization losses
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    # final loss and optimization
    final_cost=cost+reg_constant*sum(reg_losses)
    lr= tf.train.exponential_decay(starter_learning_rate, global_step,
                                           num_batches_per_epoch, 1.04, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(final_cost,global_step=global_step) 
test_acc_i=[]
loss_increasing_lr=[]
with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()
    begin=time.time()
    #iterating over total number of epochs
    for curr_epoch in range(num_epochs):
        train_cost=0
        start = time.time()
        for batch in range(num_batches_per_epoch):

            # Getting the index
            indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
            batch_train_inputs = X_train.tocsr()[indexes].toarray()
            batch_train_targets= Y_train[indexes]
            # feed dictionay for training
            feed = {inputs: batch_train_inputs,
                    outputs: batch_train_targets
                    }
           
            batch_cost, _ = session.run([final_cost, optimizer], feed)
            train_cost += batch_cost*batch_size
        log = "Epoch {}/{}, train_cost = {:.3f}, test_accuracy = {:.3f}, epoch time = {:.3f}s, total time = {:.3f}s"
        loss_increasing_lr.append(train_cost/(num_batches_per_epoch*batch_size))
        test_inputs = X_test.tocsr().toarray()
        feed = {inputs: test_inputs}
        prediction_te = session.run([pred], feed)
        acc=test_accuracy(Y_test,prediction_te)
        test_acc_i.append(acc)
        print(log.format(curr_epoch+1, num_epochs, train_cost/(num_batches_per_epoch*batch_size),acc,time.time()-start,time.time()-begin))

    prediction_tr=[]
    for batch in range(num_batches_per_epoch):
                                                                   
        # Getting the index
        indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
        batch_train_inputs = X_train.tocsr()[indexes].toarray()
        feed = {inputs: batch_train_inputs}
        prediction = session.run([pred], feed)
        prediction_tr.extend(list(prediction[0]))
    test_inputs = X_test.tocsr().toarray()
    feed = {inputs: test_inputs}
    prediction_te = session.run([pred], feed)
    


# In[77]:


tr_acc=train_accuracy(Y_train,prediction_tr)
print("accuracy on training data with exponential increasing learning rate is ",tr_acc*100,"%")
te_acc=test_accuracy(Y_test,prediction_te)
print("accuracy on test data with exponential increasing learning rate is ",te_acc*100,"%")


# In[79]:


import matplotlib.pyplot as plt

x=[i for i in range(1,101)]
plt.plot(x,loss_constant_lr , 'b', label='constant_learning_rate')
plt.plot(x,loss_decay_lr, 'r', label='decreasing_learning_rate')
plt.plot(x, loss_increasing_lr, 'g', label='increasing_learning_rate')
plt.ylabel('cross entropy loss')
plt.xlabel('number of epochs')
plt.grid()
plt.legend() # add legend based on line labels
plt.show()


# In[80]:


x=[i for i in range(1,101)]
plt.plot(x, test_acc_c , 'b', label='constant_learning_rate')
plt.plot(x,test_acc_d, 'r', label='decreasing_learning_rate')
plt.plot(x, test_acc_i, 'g', label='increasing_learning_rate')
plt.ylabel('test accuracy')
plt.xlabel('number of epochs')
plt.grid()
plt.legend() # add legend based on line labels
plt.show()

