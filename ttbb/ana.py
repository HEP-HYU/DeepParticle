from __future__ import division
import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import csv
from sklearn.utils import shuffle
import re
import string
import math
from ROOT import TFile, TTree
from ROOT import *
import ROOT
import numpy as np
from array import array

timer = ROOT.TStopwatch()
timer.Start()

train_input = np.load('output_ttbb.npy').astype(np.float32)

train_out = train_input[:,0]
train_data = train_input[:,1:]

numbertr=len(train_out)

#Shuffling
order=shuffle(range(numbertr),random_state=100)
train_out=train_out[order]
train_data=train_data[order,0::]

train_out = train_out.reshape( (numbertr, 1) )
trainnb=0.9 # Fraction used for training

#Splitting between training set and cross-validation set
valid_data=train_data[int(trainnb*numbertr):numbertr,0::]
valid_data_out=train_out[int(trainnb*numbertr):numbertr]

train_data_out=train_out[0:int(trainnb*numbertr)]
train_data=train_data[0:int(trainnb*numbertr),0::]

import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model_output_name = "keras"

if os.path.exists('models/'+model_output_name+'/model.h5'):
    print "Model file exists already!"
    from tensorflow import keras
    model = keras.models.load_model('models/'+model_output_name+'/model.h5')
else:
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'binary_crossentropy'])
    history = model.fit(train_data, train_data_out, batch_size=1024, epochs=200, validation_data=(valid_data,valid_data_out))

    model.save('models/'+model_output_name+'/model.h5')

    evaluate = model.predict( valid_data ) 
    with open('models/'+model_output_name+'/output.csv', 'wb') as f:
      writer = csv.writer(f, delimiter=" ")
      for i in range(len(valid_data)):
        val_x = valid_data_out[i] 
        val_y = evaluate[i]
        writer.writerows( zip(val_y,val_x) )

model.summary()

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('models/'+ model_output_name+'/loss.pdf')
plt.show()

genchain = TChain("ttbbLepJets/gentree");
genchain.Add("TT_powheg_ttbb.root")
chain = TChain("ttbbLepJets/tree");
chain.Add("TT_powheg_ttbb.root")

f = TFile( 'output_ttbb_new.root', 'RECREATE')
t = TTree( 'tree', 'tree for ttbb')

signal = array( 'i', [0] )
dR = array( 'd', [0] )
dEta = array( 'd', [0] )
dPhi = array( 'd', [0] )
nuPt = array( 'd', [0] )
nuEta = array( 'd', [0] )
nuPhi = array( 'd', [0] )
nuMass = array( 'd', [0] )
lbPt = array( 'd', [0] )
lbEta = array( 'd', [0] )
lbPhi = array( 'd', [0] )
lbMass = array( 'd', [0] )
lb1Pt = array( 'd', [0] )
lb1Eta = array( 'd', [0] )
lb1Phi = array( 'd', [0] )
lb1Mass = array( 'd', [0] )
lb2Pt = array( 'd', [0] )
lb2Eta = array( 'd', [0] )
lb2Phi = array( 'd', [0] )
lb2Mass = array( 'd', [0] )
lb1nuPt = array( 'd', [0] )
lb1nuEta = array( 'd', [0] )
lb1nuPhi = array( 'd', [0] )
lb1nuMass = array( 'd', [0] )
lb2nuPt = array( 'd', [0] )
lb2nuEta = array( 'd', [0] )
lb2nuPhi = array( 'd', [0] )
lb2nuMass = array( 'd', [0] )
diPt = array( 'd', [0] )
diEta = array( 'd', [0] )
diPhi = array( 'd', [0] )
diMass = array( 'd', [0] )
csv1 = array( 'd', [0] )
csv2 = array( 'd', [0] )
pt1 = array( 'd', [0] )
pt2 = array( 'd', [0] )
eta1 = array( 'd', [0] )
eta2 = array( 'd', [0] )
phi1 = array( 'd', [0] )
phi2 = array( 'd', [0] )
e1 = array( 'd', [0] )
e2 = array( 'd', [0] )

t.Branch('signal', signal, 'signal/I')
t.Branch('dR', dR, 'dR/D')
t.Branch('dEta', dEta, 'dEta/D')
t.Branch('dPhi', dPhi, 'dPhi/D')
t.Branch('nuPt', nuPt, 'nuPt/D')
t.Branch('nuEta', nuEta, 'nuEta/D')
t.Branch('nuPhi', nuPhi, 'nuPhi/D')
t.Branch('nuMass', nuMass, 'nuMass/D')
t.Branch('lbPt', lbPt, 'lbPt/D')
t.Branch('lbEta', lbEta, 'lbEta/D')
t.Branch('lbPhi', lbPhi, 'lbPhi/D')
t.Branch('lbMass', lbMass, 'lbMass/D')
t.Branch('lb1Pt', lb1Pt, 'lb1Pt/D')
t.Branch('lb1Eta', lb1Eta, 'lb1Eta/D')
t.Branch('lb1Phi', lb1Phi, 'lb1Phi/D')
t.Branch('lb1Mass', lb1Mass, 'lb1Mass/D')
t.Branch('lb2Pt', lb2Pt, 'lb2Pt/D')
t.Branch('lb2Eta', lb2Eta, 'lb2Eta/D')
t.Branch('lb2Phi', lb2Phi, 'lb2Phi/D')
t.Branch('lb2Mass', lb2Mass, 'lb2Mass/D')
t.Branch('lb1nuPt', lb1nuPt, 'lb1nuPt/D')
t.Branch('lb1nuEta', lb1nuEta, 'lb1nuEta/D')
t.Branch('lb1nuPhi', lb1nuPhi, 'lb1nuPhi/D')
t.Branch('lb1nuMass', lb1nuMass, 'lb1nuMass/D')
t.Branch('lb2nuPt', lb2nuPt, 'lb2nuPt/D')
t.Branch('lb2nuEta', lb2nuEta, 'lb2nuEta/D')
t.Branch('lb2nuPhi', lb2nuPhi, 'lb2nuPhi/D')
t.Branch('lb2nuMass', lb2nuMass, 'lb2nuMass/D')
t.Branch('diPt', diPt, 'diPt/D')
t.Branch('diEta', diEta, 'diEta/D')
t.Branch('diPhi', diPhi, 'diPhi/D')
t.Branch('diMass', diMass, 'di<ass/D')
t.Branch('csv1', csv1, 'csv1/D')
t.Branch('csv2', csv2, 'csv2/D')
t.Branch('pt1', pt1, 'pt1/D')
t.Branch('pt2', pt2, 'pt2/D')
t.Branch('eta1', eta1, 'eta1/D')
t.Branch('eta2', eta2, 'eta2/D')
t.Branch('phi1', phi1, 'phi1/D')
t.Branch('phi2', phi2, 'phi2/D')
t.Branch('e1', e1, 'e1/D')
t.Branch('e2', e2, 'e2/D')

# Create the output file. 
f_hist = ROOT.TFile("hist_ttbb.root", "recreate")
f_hist.cd()
edgesfordR = [0.4, 0.6, 1.0, 2.0, 4.0]
edgesforMass = [0, 60, 100, 170, 400]

h_dR_ch0 = ROOT.TH1F('h_dR_ch0','dR between two addtional b jets', 4, array('d',edgesfordR) )
h_Mass_ch0 = ROOT.TH1F('h_Mass_ch0','Invariant mass of two addtional b jets', 4, array('d',edgesforMass))
h_dR_ch1 = ROOT.TH1F('h_dR_ch1','dR between two addtional b jets', 4, array('d',edgesfordR) )
h_Mass_ch1 = ROOT.TH1F('h_Mass_ch1','Invariant mass of two addtional b jets', 4, array('d',edgesforMass))

h_dR_dR_ch0 = ROOT.TH1F('h_dR_dR_ch0','dR between two addtional b jets', 4, array('d',edgesfordR) )
h_Mass_dR_ch0 = ROOT.TH1F('h_Mass_dR_ch0','Invariant mass of two addtional b jets', 4, array('d',edgesforMass))
h_dR_dR_ch1 = ROOT.TH1F('h_dR_dR_ch1','dR between two addtional b jets', 4, array('d',edgesfordR) )
h_Mass_dR_ch1 = ROOT.TH1F('h_Mass_dR_ch1','Invariant mass of two addtional b jets', 4, array('d',edgesforMass))

h_dR_Gen_Den = ROOT.TH1F('h_dR_Gen_Den','dR between two addtional b jets', 4, array('d',edgesfordR) )
h_Mass_Gen_Den = ROOT.TH1F('h_Mass_Gen_Den','Invariant mass of two addtional b jets', 4, array('d',edgesforMass))

h_dR_Gen_ch0 = ROOT.TH1F('h_dR_Gen_ch0','dR between two addtional b jets', 4, array('d',edgesfordR) )
h_Mass_Gen_ch0 = ROOT.TH1F('h_Mass_Gen_ch0','Invariant mass of two addtional b jets', 4, array('d',edgesforMass))
h_dR_Gen_ch1 = ROOT.TH1F('h_dR_Gen_ch1','dR between two addtional b jets', 4, array('d',edgesfordR) )
h_Mass_Gen_ch1 = ROOT.TH1F('h_Mass_Gen_ch1','Invariant mass of two addtional b jets', 4, array('d',edgesforMass))

h_dR_fine_Gen_ch0 = ROOT.TH1F('h_dR_fine_Gen_ch0','dR between two addtional b jets', 20, 0, 5 )
h_Mass_fine_Gen_ch0 = ROOT.TH1F('h_Mass_fine_Gen_ch0','Invariant mass of two addtional b jets', 40, 0, 400)
h_dR_fine_Gen_ch1 = ROOT.TH1F('h_dR_fine_Gen_ch1','dR between two addtional b jets', 20, 0, 5)
h_Mass_fine_Gen_ch1 = ROOT.TH1F('h_Mass_fine_Gen_ch1','Invariant mass of two addtional b jets', 40, 0, 400)

h2_dR_Response_ch0 = ROOT.TH2F('h2_dR_Response_ch0','dR between two addtional b jets', 4, array('d',edgesfordR), 4, array('d',edgesfordR))
h2_Mass_Response_ch0 = ROOT.TH2F('h2_Mass_Response_ch0','Invariant mass of two addtional b jets', 4, array('d',edgesforMass), 4, array('d',edgesforMass) )
h2_dR_Response_ch1 = ROOT.TH2F('h2_dR_Response_ch1','dR between two addtional b jets', 4, array('d',edgesfordR), 4, array('d',edgesfordR) )
h2_Mass_Response_ch1 = ROOT.TH2F('h2_Mass_Response_ch1','Invariant mass of two addtional b jets', 4, array('d',edgesforMass), 4, array('d',edgesforMass)) 

genentries = genchain.GetEntries()
for i in xrange(genentries):
  genchain.GetEntry(i)

  addbjet1 = TLorentzVector()
  addbjet2 = TLorentzVector()
  addbjet1.SetPtEtaPhiE( genchain.addbjet1_pt, genchain.addbjet1_eta, genchain.addbjet1_phi, genchain.addbjet1_e)
  addbjet2.SetPtEtaPhiE( genchain.addbjet2_pt, genchain.addbjet2_eta, genchain.addbjet2_phi, genchain.addbjet2_e)

  dibdR_Gen = addbjet1.DeltaR( addbjet2 )
  dibMass_Gen = (addbjet1 + addbjet2).M()

  h_dR_Gen_Den.Fill(dibdR_Gen, genchain.genweight)
  h_Mass_Gen_Den.Fill(dibMass_Gen, genchain.genweight)

entries = chain.GetEntries() 

jetindex = open("jetindex.txt", 'w')

nEvents = 0
nEvents_matchable = 0
nEvents_match = 0
nEvents_match_dR = 0

for i in xrange(entries):
  chain.GetEntry(i)

  eventweight = chain.PUWeight[0]*chain.genweight*chain.lepton_SF[0]*chain.jet_SF_CSV[0]

  MET_px = chain.MET * math.cos( chain.MET_phi )
  MET_py = chain.MET * math.sin( chain.MET_phi )
  nu = TLorentzVector( MET_px, MET_py, 0, chain.MET)
  lep = TLorentzVector() 
  lep.SetPtEtaPhiE( chain.lepton_pT, chain.lepton_eta, chain.lepton_phi, chain.lepton_E)
  passmu = False
  passel = False
  passmu = chain.channel == 0 and lep.Pt() > 30 and abs(lep.Eta()) < 2.1
  passel = chain.channel == 1 and lep.Pt() > 35 and abs(lep.Eta()) < 2.1
  if passmu == False and passel == False:
    continue

  addbjet1 = TLorentzVector()
  addbjet1.SetPtEtaPhiE(chain.addbjet1_pt, chain.addbjet1_eta, chain.addbjet1_phi, chain.addbjet1_e);
  addbjet2 = TLorentzVector()
  addbjet2.SetPtEtaPhiE(chain.addbjet2_pt, chain.addbjet2_eta, chain.addbjet2_phi, chain.addbjet2_e);

  addbjet1_matched = TLorentzVector(0,0,0,0)
  addbjet2_matched = TLorentzVector(0,0,0,0)

  njets = 0
  nbjets = 0 
  for j in range( len(chain.jet_pT) ):
    jet = TLorentzVector()
    jet.SetPtEtaPhiE(chain.jet_pT[j], chain.jet_eta[j], chain.jet_phi[j], chain.jet_E[j])
    jet *= chain.jet_JER_Nom[j]
    if jet.Pt() < 30 or abs(jet.Eta()) > 2.4 :
      continue
    njets = njets + 1
    btag = 0.9535 #tight working point
    #btag = 0.8484 #medium working point
    #btag = 0.54264 #loose working point
    if chain.jet_CSV[j] > btag:
      nbjets = nbjets+1

  if njets < 6:  
    continue
  if nbjets < 3:
    continue
 
  for i in range( len(chain.jet_pT) ):
    tmp = TLorentzVector()
    tmp.SetPtEtaPhiE( chain.jet_pT[i], chain.jet_eta[i], chain.jet_phi[i], chain.jet_E[i] )
    #when finding matched jet, not use pt and eta
    #tmp *= chain.jet_JER_Nom[i]
    #if tmp.Pt() > 20 and abs(tmp.Eta()) < 2.4: 
    if addbjet1.DeltaR( tmp ) < 0.4:
      addbjet1_matched = tmp;
    if addbjet2.DeltaR( tmp ) < 0.4:
      addbjet2_matched = tmp;

  if addbjet1_matched.Pt() > 0 and  addbjet2_matched.Pt() > 0 :
    nEvents_matchable =  nEvents_matchable + 1 

  small_dR = 999
  test_data = []
  sel = []
  indexSmalldR = []
  for j in range( len(chain.jet_pT) - 1):
    for k in range( j+1, len(chain.jet_pT) ):
      jet1 = TLorentzVector()
      jet1.SetPtEtaPhiE( chain.jet_pT[j], chain.jet_eta[j], chain.jet_phi[j], chain.jet_E[j] )
      jet1 *= chain.jet_JER_Nom[j]
      jet2 = TLorentzVector()
      jet2.SetPtEtaPhiE( chain.jet_pT[k], chain.jet_eta[k], chain.jet_phi[k], chain.jet_E[k] )
      jet2 *= chain.jet_JER_Nom[k]

      #when training, not applying jet pt and eta for signal sample
      #acceptanceCutOnJets = jet1.Pt() > 20 and abs(jet1.Eta()) < 2.4 and jet2.Pt() > 20 and abs(jet2.Eta()) < 2.4
      if chain.jet_CSV[j] > btag and chain.jet_CSV[k] > btag:
	b1 = TLorentzVector()
	b2 = TLorentzVector()
	b1.SetPtEtaPhiE( chain.jet_pT[j], chain.jet_eta[j], chain.jet_phi[j], chain.jet_E[j]) 
	b2.SetPtEtaPhiE( chain.jet_pT[k], chain.jet_eta[k], chain.jet_phi[k], chain.jet_E[k]) 
	dR[0] = b1.DeltaR(b2)
	dEta[0] = abs( b1.Eta() - b2.Eta())
	dPhi[0] = b1.DeltaPhi(b2)
	nuPt[0] = (b1+b2+nu).Pt()
	nuEta[0] = (b1+b2+nu).Eta()
	nuPhi[0] = (b1+b2+nu).Phi()
	nuMass[0] = (b1+b2+nu).M()
	lbPt[0] = (b1+b2+lep).Pt()
	lbEta[0] = (b1+b2+lep).Eta()
	lbPhi[0] = (b1+b2+lep).Phi()
	lbMass[0] = (b1+b2+lep).M()
	lb1Pt[0] = (b1+lep).Pt()
	lb1Eta[0] = (b1+lep).Eta()
	lb1Phi[0] = (b1+lep).Phi()
	lb1Mass[0] = (b1+lep).M()
	lb2Pt[0] = (b2+lep).Pt()
	lb2Eta[0] = (b2+lep).Eta()
	lb2Phi[0] = (b2+lep).Phi()
	lb2Mass[0] = (b2+lep).M()
	lb1nuPt[0] = (b1+lep+nu).Pt()
	lb1nuEta[0] = (b1+lep+nu).Eta()
	lb1nuPhi[0] = (b1+lep+nu).Phi()
	lb1nuMass[0] = (b1+lep+nu).M()
	lb2nuPt[0] = (b2+lep+nu).Pt()
	lb2nuEta[0] = (b2+lep+nu).Eta()
	lb2nuPhi[0] = (b2+lep+nu).Phi()
	lb2nuMass[0] = (b2+lep+nu).M()
	diPt[0] = (b1+b2).Pt()
	diEta[0] = (b1+b2).Eta()
	diPhi[0] = (b1+b2).Phi()
	diMass[0] = (b1+b2).M()
	csv1[0] = chain.jet_CSV[j] 
	csv2[0] = chain.jet_CSV[k]
	pt1[0] = b1.Pt()
	pt2[0] = b2.Pt()
	eta1[0] = b1.Eta()
	eta2[0] = b2.Eta()
	phi1[0] = b1.Phi()
	phi2[0] = b2.Phi()
	e1[0] = b1.E()
	e2[0] = b2.E()
	data = []
	data.append(dR[0])
	data.append(dEta[0])
	data.append(dPhi[0])
	data.append(nuPt[0])
	data.append(nuEta[0])
	data.append(nuPhi[0])
	data.append(nuMass[0])
	data.append(lbPt[0])
	data.append(lbEta[0])
	data.append(lbPhi[0])
	data.append(lbMass[0])
	data.append(lb1Pt[0])
	data.append(lb1Eta[0])
	data.append(lb1Phi[0])
	data.append(lb1Mass[0])
	data.append(lb2Pt[0])
	data.append(lb2Eta[0])
	data.append(lb2Phi[0])
	data.append(lb2Mass[0])
	#data.append(lb1nuPt[0])
	#data.append(lb1nuEta[0])
	#data.append(lb1nuPhi[0])
	#data.append(lb1nuMass[0])
	#data.append(lb2nuPt[0])
	#data.append(lb2nuEta[0])
	#data.append(lb2nuPhi[0])
	#data.append(lb2nuMass[0])
	data.append(diPt[0])
	data.append(diEta[0])
	data.append(diPhi[0])
	data.append(diMass[0])
	data.append(csv1[0])
	data.append(csv2[0])
	data.append(pt1[0])
	data.append(pt2[0])
	data.append(eta1[0])
	data.append(eta2[0])
	data.append(phi1[0])
	data.append(phi2[0])
	data.append(e1[0])
	data.append(e2[0])
	test_data.append(data)
	tmp = []
	tmp.append(j)
	tmp.append(k)
	sel.append(tmp)
	if dR[0] < small_dR:
	  small_dR = dR[0]
	  indexSmalldR = tmp

	if ( addbjet1_matched.DeltaR( b1 ) == 0 and addbjet2_matched.DeltaR( b2 ) ==0 ) or ( addbjet2_matched.DeltaR( b1) == 0  and addbjet1_matched.DeltaR( b2) == 0 ):
	  signal[0] = 1
	else:
	  signal[0] = 0

	t.Fill()


  input_data = np.asarray( test_data )

  pred = model.predict( input_data )
  maxval = pred.argmax()

  #print sel from DNN
  addbjet1_reco = TLorentzVector()
  addbjet2_reco = TLorentzVector()
  addbjet1_reco.SetPtEtaPhiE( chain.jet_pT[ sel[maxval][0] ], chain.jet_eta[ sel[maxval][0] ], chain.jet_phi[ sel[maxval][0] ], chain.jet_E[ sel[maxval][0] ])  
  addbjet2_reco.SetPtEtaPhiE( chain.jet_pT[ sel[maxval][1] ], chain.jet_eta[ sel[maxval][1] ], chain.jet_phi[ sel[maxval][1] ], chain.jet_E[ sel[maxval][1] ])  

  dibdR = addbjet1_reco.DeltaR( addbjet2_reco )
  dibMass = ( addbjet1_reco + addbjet2_reco).M()

  addbjet1_reco_dR = TLorentzVector()
  addbjet2_reco_dR = TLorentzVector()
  addbjet1_reco_dR.SetPtEtaPhiE( chain.jet_pT[ indexSmalldR[0] ], chain.jet_eta[ indexSmalldR[0] ], chain.jet_phi[ indexSmalldR[0] ], chain.jet_E[ indexSmalldR[0] ])
  addbjet2_reco_dR.SetPtEtaPhiE( chain.jet_pT[ indexSmalldR[1] ], chain.jet_eta[ indexSmalldR[1] ], chain.jet_phi[ indexSmalldR[1] ], chain.jet_E[ indexSmalldR[1] ])

  dibdR_dR = addbjet1_reco_dR.DeltaR( addbjet2_reco_dR )
  dibMass_dR = ( addbjet1_reco_dR + addbjet2_reco_dR).M()

  addbjet1 = TLorentzVector()
  addbjet2 = TLorentzVector()
  addbjet1.SetPtEtaPhiE( chain.addbjet1_pt, chain.addbjet1_eta, chain.addbjet1_phi, chain.addbjet1_e)
  addbjet2.SetPtEtaPhiE( chain.addbjet2_pt, chain.addbjet2_eta, chain.addbjet2_phi, chain.addbjet2_e)

  dibdR_Gen = addbjet1.DeltaR( addbjet2 )
  dibMass_Gen = (addbjet1 + addbjet2).M()

  Split = False

  if Split:
    if i%2 == 0:
      if chain.channel == 0:
	h2_dR_Response_ch0.Fill( dibdR, dibdR_Gen, eventweight)
	h2_Mass_Response_ch0.Fill( dibMass, dibMass_Gen, eventweight)
      elif chain.channel == 1:
	h2_dR_Response_ch1.Fill( dibdR, dibdR_Gen, eventweight)
	h2_Mass_Response_ch1.Fill( dibMass, dibMass_Gen, eventweight)
      else:
	print "Error! --> No channel information"
    elif i%2 == 1:
      if chain.channel == 0:
	h_dR_ch0.Fill( dibdR, eventweight )
	h_Mass_ch0.Fill(dibMass, eventweight)
	h_dR_dR_ch0.Fill( dibdR_dR, eventweight)
	h_Mass_dR_ch0.Fill(dibMass_dR, eventweight)
	h_dR_Gen_ch0.Fill( dibdR_Gen, eventweight )
	h_Mass_Gen_ch0.Fill(dibMass_Gen, eventweight)
	h_dR_fine_Gen_ch0.Fill( dibdR_Gen, eventweight )
	h_Mass_fine_Gen_ch0.Fill(dibMass_Gen, eventweight)
      elif chain.channel ==1:
	h_dR_ch1.Fill( dibdR, eventweight )
	h_Mass_ch1.Fill(dibMass, eventweight)
	h_dR_dR_ch1.Fill( dibdR, eventweight )
	h_Mass_dR_ch1.Fill(dibMass, eventweight)
	h_dR_Gen_ch1.Fill( dibdR_Gen, eventweight )
	h_Mass_Gen_ch1.Fill(dibMass_Gen, eventweight)
	h_dR_fine_Gen_ch1.Fill( dibdR_Gen, eventweight )
	h_Mass_fine_Gen_ch1.Fill(dibMass_Gen, eventweight)
      else:
	print "Error!  --> No channel information"
  else:
    if chain.channel == 0:
      h2_dR_Response_ch0.Fill( dibdR, dibdR_Gen, eventweight)
      h2_Mass_Response_ch0.Fill( dibMass, dibMass_Gen, eventweight)
      h_dR_ch0.Fill( dibdR, eventweight )
      h_Mass_ch0.Fill(dibMass, eventweight)
      h_dR_dR_ch0.Fill( dibdR_dR, eventweight)
      h_Mass_dR_ch0.Fill(dibMass_dR, eventweight)
      h_dR_Gen_ch0.Fill( dibdR_Gen, eventweight )
      h_Mass_Gen_ch0.Fill(dibMass_Gen, eventweight)
      h_dR_fine_Gen_ch0.Fill( dibdR_Gen, eventweight )
      h_Mass_fine_Gen_ch0.Fill(dibMass_Gen, eventweight)
    elif chain.channel == 1:
      h2_dR_Response_ch1.Fill( dibdR, dibdR_Gen, eventweight)
      h2_Mass_Response_ch1.Fill( dibMass, dibMass_Gen, eventweight)
      h_dR_ch1.Fill( dibdR, eventweight )
      h_Mass_ch1.Fill(dibMass, eventweight)
      h_dR_dR_ch1.Fill( dibdR, eventweight )
      h_Mass_dR_ch1.Fill(dibMass, eventweight)
      h_dR_Gen_ch1.Fill( dibdR_Gen, eventweight )
      h_Mass_Gen_ch1.Fill(dibMass_Gen, eventweight)
      h_dR_fine_Gen_ch1.Fill( dibdR_Gen, eventweight )
      h_Mass_fine_Gen_ch1.Fill(dibMass_Gen, eventweight)
    else:
      print "Error!  --> No channel information"
  #f_hist.Write(h_dR)
  #f_hist.Write(h_Mass) 

  match1 = addbjet1_reco.DeltaR( addbjet1 ) < 0.5 or addbjet1_reco.DeltaR( addbjet2 )  < 0.5 
  match2 = addbjet2_reco.DeltaR( addbjet1 ) < 0.5 or addbjet2_reco.DeltaR( addbjet2 )  < 0.5 

  match1_dR = addbjet1_reco_dR.DeltaR( addbjet1 ) < 0.5 or addbjet1_reco_dR.DeltaR( addbjet2 )  < 0.5
  match2_dR = addbjet2_reco_dR.DeltaR( addbjet1 ) < 0.5 or addbjet2_reco_dR.DeltaR( addbjet2 )  < 0.5

  nEvents = nEvents + 1
  if match1 and match2:
    nEvents_match = nEvents_match + 1   
  if match1_dR and match2_dR:
    nEvents_match_dR = nEvents_match_dR + 1
  

  #print match1 , " ", match2
  if nEvents%1000 == 0:
    print "nEvents (matchable) = ", nEvents, "nEvents_matchable = ", nEvents_matchable
    print "nEvents (DNN) = ", nEvents, "nEvents_match = ", nEvents_match 
    print "nEvents (dR)  = ", nEvents, "nEvents_match = ", nEvents_match_dR 
  c = str(chain.event) + ' , ' + str(sel[maxval][0]) + ' , ' +  str(sel[maxval][1]) + '\n'
  #print c
  jetindex.write( c )

f_hist.Write()
f_hist.cd()
f_hist.Close()

jetindex.close()
f.Write()
f.Close()

match_matchable = nEvents_matchable / nEvents
match_rate = nEvents_match / nEvents
match_rate_dR = nEvents_match_dR / nEvents
print "matable rate = ", match_matchable, "matching rate from DNN = " , match_rate , "matching rate from dR =",  match_rate_dR

timer.Stop()
rtime = timer.RealTime(); # Real time (or "wall time")
ctime = timer.CpuTime(); # CPU time
print("RealTime={0:6.2f} seconds, CpuTime={1:6.2f} seconds").format(rtime,ctime)
print("{0:4.2f} events / RealTime second .").format( nEvents/rtime)
print("{0:4.2f} events / CpuTime second .").format( nEvents/ctime)
