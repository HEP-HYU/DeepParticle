from __future__ import division
import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DIVICES"] = "1"

import numpy as np
import csv
from sklearn.utils import shuffle
import re
import string
import math
from ROOT import TFile, TTree
from ROOT import *
import ROOT
from array import array
import pandas as pd
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape) :
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

df = pd.read_hdf("output_ttbb.h5")
df = df.filter(['signal',
    'dR','dEta','dPhi',
    'nuPt','nuEta','nuPhi','nuMass',
    'lbPt','lbEta','lbPhi','lbMass',
    'lb1Pt','lb1Eta','lb1Phi','lb1Mass',
    'lb2Pt','lb2Eta','lb2Phi','lb2Mass',
    'diPt','diEta','diPhi','diMass',
    'csv1','csv2','pt1','pt2','eta1','eta2','phi1','phi2','e1','e2'
])
train_input = df.values
#np.load(df.values).astyle(np.float32)
train_out = train_input[:,0]
train_data = train_input[:,1:]

numbertr=len(train_out)

order=shuffle(range(numbertr),random_state=200)
train_out = train_out[order]
train_data = train_data[order,0::]
train_out = train_out.reshape((numbertr,1))
trainnb = 0.9

valid_data = train_data[int(trainnb*numbertr):numbertr,0::]
valid_data_out = train_out[int(trainnb*numbertr):numbertr]

train_data_out = train_out[0:int(trainnb*numbertr)]
train_data = train_data[0:int(trainnb*numbertr),0::]

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None,33])
y_ = tf.placeholder(tf.float32, shape=[None,1])

W1 = weight_variable([33,300])
b1 = bias_variable([300])
A1 = tf.nn.relu(tf.matmul(x,W1)+b1)
W2 = weight_variable([300,300])
b2 = bias_variable([300])
A2 = tf.nn.relu(tf.matmul(A1,W2)+b2)
W3 = weight_variable([300,300])
b3 = bias_variable([300])
A3 = tf.nn.relu(tf.matmul(A2,W3)+b3)
W4 = weight_variable([300,300])
b4 = bias_variable([300])
A4 = tf.nn.relu(tf.matmul(A3,W4)+b4)
W5 = weight_variable([300,300])
b5 = bias_variable([300])
A5 = tf.nn.relu(tf.matmul(A4,W5)+b5)
W6 = weight_variable([300,300])
b6 = bias_variable([300])
A6 = tf.nn.relu(tf.matmul(A5,W6)+b6)
W7 = weight_variable([300,300])
b7 = bias_variable([300])
A7 = tf.nn.relu(tf.matmul(A6,W7)+b7)
W8 = weight_variable([300,300])
b8 = bias_variable([300])
A8 = tf.nn.relu(tf.matmul(A7,W8)+b8)
W9 = weight_variable([300,1])
b9 = bias_variable([1])
y = tf.matmul(A8,W9)+b9

cross_entropy = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

ntrain = len(train_data)
batch_size = 128
cur_id = 0
cur_id_p = 0
cur_id_n = 0
epoch = 0

saver = tf.train.Saver()
model_output_name = "33v_300n_layer_9"

tmpout=''
with tf.Session() as sess :
  sess.run(tf.global_variables_initializer())
  if os.path.exists('models/'+model_output_name+'/model_out.meta'):
    print "Model file already exists!"
    saver.restore(sess, 'models/'+model_output_name+'/model_out')
  else :
    for i in range(5000) :
      batch_data = train_data[cur_id:cur_id+batch_size]
      batch_data_out = train_data_out[cur_id:cur_id+batch_size]
      cur_id = cur_id + batch_size
      if cur_id > ntrain :
        cur_id = 0
        epoch += 1
        tmpout = str(epoch) + "epoch passed"
        print tmpout

      train_step.run(feed_dict={x:batch_data, y_:batch_data_out})
      
    saver.save(sess, 'models/'+model_output_name+'/model_out')
    print "Model saved!"

  prediction = tf.nn.sigmoid(y)
  pred = prediction.eval(feed_dict={x:valid_data})

  with open('models/'+model_output_name+'/output.csv','wb') as f :
    writer = csv.writer(f, delimiter=" ")
    for i in range(len(valid_data)) :
      val_x = valid_data_out[i]
      val_y = pred[i]
      writer.writerows(zip(val_y,val_x))

with open("var.txt","w") as f :
  f.write("directory models/"+model_output_name+"/model_out\n")
  
