import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import csv
import math
import numpy as np
from array import array

from ROOT import *
import ROOT

import tensorflow as tf
import csv
from sklearn.utils import shuffle
import re
import string
import math

from tqdm import tqdm

def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
  nEvent = str(iteration) + '/' + str(total)
  formatStr = "{0:." + str(decimals) + "f}"
  percent = formatStr.format(100*(iteration/float(total)))
  filledLength = int(round(barLength * iteration/float(total)))
  bar = '#'*filledLength + '-'*(barLength-filledLength)
  sys.stdout.write('\r%s |%s| %s%s %s %s' % (prefix, bar, percent, '%', suffix, nEvent)),
  if iteration == total:
    sys.stdout.write('\n')
  sys.stdout.flush()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def ana(directory, inputFile, process) :
  timer = ROOT.TStopwatch()
  timer.Start()

  df = pd.read_hdf("output_ttbb.h5")
  df = df.filter(['signal',
      'dR','dEta','dPhi',
      'nuPt','nuEta','nuPhi','nuMass',
      'lbPt','lbEta','lbPhi','lbMass',
      'lb1Pt','lb1Eta','lb1Phi','lb1Mass',
      'lb2Pt','lb2Eta','lb2Phi','lb2Mass',
      'diPt','diEta','diPhi','diMass',
      'csv1','csv2','pt1','pt2','eta1','eta2','phi1','phi2','e1','e2'])
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
  batch_size = 1024 
  cur_id = 0
  cur_id_p = 0
  cur_id_n = 0
  epoch = 0

  saver = tf.train.Saver()
  model_output_name = "33v_300n_layer_9"

  tmpout = ''
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
  
    data = False
    if 'Data' in process:
      data = True

    ttbb = False 
    if 'ttbb' in process:
      ttbb = True

    closureTest = False

    muon_ch = 0
    muon_pt = 30.0
    muon_eta = 2.1
    electron_ch = 1
    electron_pt = 35.0
    electron_eta = 2.1
    jet_pt = 30.0
    jet_eta = 2.4
    jet_CSV_tight = 0.9535
    jet_CSV_medium = 0.8484
    number_of_jets = 6
    number_of_bjets = 3
    nChannel = 2
    nStep = 4

    f_out = ROOT.TFile("output/hist_"+process+".root", "recreate")

    RECO_LEPTON_PT_ = "LeptonPt"
    RECO_LEPTON_ETA_ = "LeptonEta"
    RECO_NUMBER_OF_JETS_ = "nJets"
    RECO_NUMBER_OF_BJETS_ = "nBjets"
    RECO_ADDJETS_DELTAR_ = "RecoJetDeltaR"
    RECO_ADDJETS_INVARIANT_MASS_ = "RecoJetInvMass"
    GEN_ADDBJETS_DELTAR_ = "GenbJetDeltaR"
    GEN_ADDBJETS_INVARIANT_MASS_ = "GenbJetInvMass"
    RESPONSE_MATRIX_DELTAR_ = "ResponseMatrixDeltaR"
    RESPONSE_MATRIX_INVARIANT_MASS_ = "ResponseMatrixInvMass"
   
    xNbins_reco_addjets_dR = 4
    reco_addjets_dR_width = [0.4,0.6,1.0,2.0,4.0]

    xNbins_reco_addjets_M = 4
    reco_addjets_M_width = [0.0,60.0,100.0,170.0,400.0]

    xNbins_gen_addbjets_dR = 4
    gen_addbjets_dR_width = [0.4,0.6,1.0,2.0,4.0]

    xNbins_gen_addbjets_M = 4
    gen_addbjets_M_width = [0.0,60.0,100.0,170.0,400.0]

    h_gen_addbjets_deltaR_nosel = []
    h_gen_addbjets_invMass_nosel = []
    h_lepton_pt = [[0]*nStep for i in range(nChannel)]
    h_lepton_eta = [[0]*nStep for i in range(nChannel)]
    h_njets = [[0]*nStep for i in range(nChannel)]
    h_nbjets = [[0]*nStep for i in range(nChannel)]
    h_reco_addjets_deltaR = [[0]*nStep for i in range(nChannel)]
    h_reco_addjets_invMass = [[0]*nStep for i in range(nChannel)]
    h_gen_addbjets_deltaR = [[0]*nStep for i in range(nChannel)]
    h_gen_addbjets_invMass = [[0]*nStep for i in range(nChannel)]
    h_respMatrix_deltaR = [[0]*nStep for i in range(nChannel)]
    h_respMatrix_invMass = [[0]*nStep for i in range(nChannel)]

    for iChannel in range(0,nChannel) :
      h_gen_addbjets_deltaR_nosel.append(
	ROOT.TH1D(
	  "h_" + GEN_ADDBJETS_DELTAR_ + "_Ch" + str(iChannel) + "_nosel_" + process, "",
	  xNbins_gen_addbjets_dR, array('d', gen_addbjets_dR_width)
	)
      )
      h_gen_addbjets_invMass_nosel.append(
	ROOT.TH1D(
	  "h_" + GEN_ADDBJETS_INVARIANT_MASS_ + "_Ch" + str(iChannel) + "_nosel_" + process, "",
	  xNbins_gen_addbjets_M, array('d', gen_addbjets_M_width)
	)
      )
      for iStep in range(0,nStep) :
	h_lepton_pt[iChannel][iStep] = ROOT.TH1D(
	  "h_" + RECO_LEPTON_PT_ + "_Ch" + str(iChannel) + "_S" + str(iStep) + "_" + process, "",
	  20, 0, 400
	)
	h_lepton_pt[iChannel][iStep].GetXaxis().SetTitle("p_{T}(GeV)")
	h_lepton_pt[iChannel][iStep].GetYaxis().SetTitle("Entries")
	h_lepton_pt[iChannel][iStep].Sumw2()

	h_lepton_eta[iChannel][iStep] = ROOT.TH1D(
	  "h_" + RECO_LEPTON_ETA_ + "_Ch" + str(iChannel) + "_S" + str(iStep) + "_" + process, "",
	  20, 0, 2.5
	)
	h_lepton_eta[iChannel][iStep].GetXaxis().SetTitle("#eta")
	h_lepton_eta[iChannel][iStep].GetYaxis().SetTitle("Entries")
	h_lepton_eta[iChannel][iStep].Sumw2()

	h_njets[iChannel][iStep] = ROOT.TH1D(
	  "h_" + RECO_NUMBER_OF_JETS_ + "_Ch" + str(iChannel) + "_S" + str(iStep) + "_" + process, "",
	  10, 0, 10
	)
	h_njets[iChannel][iStep].GetXaxis().SetTitle("Jet multiplicity")
	h_njets[iChannel][iStep].GetYaxis().SetTitle("Entries")
	h_njets[iChannel][iStep].Sumw2()

	h_nbjets[iChannel][iStep] = ROOT.TH1D(
	  "h_" + RECO_NUMBER_OF_BJETS_ + "_Ch" + str(iChannel) + "_S" + str(iStep) + "_" + process, "",
	  10, 0, 10
	)
	h_nbjets[iChannel][iStep].GetXaxis().SetTitle("bJet multiplicity")
	h_nbjets[iChannel][iStep].GetYaxis().SetTitle("Entries")
	h_nbjets[iChannel][iStep].Sumw2()

	h_reco_addjets_deltaR[iChannel][iStep] = ROOT.TH1D(
	  "h_" + RECO_ADDJETS_DELTAR_ + "_Ch" + str(iChannel) + "_S" + str(iStep) + "_" + process, "",
	  xNbins_reco_addjets_dR, array('d', reco_addjets_dR_width)
	)
	h_reco_addjets_deltaR[iChannel][iStep].GetXaxis().SetTitle("#DeltaR_{b#bar{b}}")
	h_reco_addjets_deltaR[iChannel][iStep].GetYaxis().SetTitle("Entries")
	h_reco_addjets_deltaR[iChannel][iStep].Sumw2()

	h_reco_addjets_invMass[iChannel][iStep] = ROOT.TH1D(
	  "h_" + RECO_ADDJETS_INVARIANT_MASS_ + "_Ch" + str(iChannel) + "_S" + str(iStep) + "_" + process, "",
	  xNbins_reco_addjets_M, array('d', reco_addjets_M_width)
	)
	h_reco_addjets_invMass[iChannel][iStep].GetXaxis().SetTitle("M_{b#bar{b}}(GeV)")
	h_reco_addjets_invMass[iChannel][iStep].GetYaxis().SetTitle("Entries")
	h_reco_addjets_invMass[iChannel][iStep].Sumw2()

	h_gen_addbjets_deltaR[iChannel][iStep] = ROOT.TH1D(
	  "h_" + GEN_ADDBJETS_DELTAR_ + "_Ch" + str(iChannel) + "_S" + str(iStep) + "_" + process, "",
	  xNbins_gen_addbjets_dR, array('d', gen_addbjets_dR_width)
	)
	h_gen_addbjets_deltaR[iChannel][iStep].GetXaxis().SetTitle("#DeltaR_{b#bar{b}}")
	h_gen_addbjets_deltaR[iChannel][iStep].GetYaxis().SetTitle("Entries")
	h_gen_addbjets_deltaR[iChannel][iStep].Sumw2()

	h_gen_addbjets_invMass[iChannel][iStep] = ROOT.TH1D(
	  "h_" + GEN_ADDBJETS_INVARIANT_MASS_ + "_Ch" + str(iChannel) + "_S" + str(iStep) + "_" + process, "",
	  xNbins_gen_addbjets_M, array('d', gen_addbjets_M_width)
	)
	h_gen_addbjets_invMass[iChannel][iStep].GetXaxis().SetTitle("M_{b#bar{b}}(GeV)")
	h_gen_addbjets_invMass[iChannel][iStep].GetYaxis().SetTitle("Entries")
	h_gen_addbjets_invMass[iChannel][iStep].Sumw2()

	h_respMatrix_deltaR[iChannel][iStep] = ROOT.TH2D(
	  "h_" + RESPONSE_MATRIX_DELTAR_ + "_Ch" + str(iChannel) + "_S" + str(iStep) + "_" + process, "",
	  xNbins_reco_addjets_dR, array('d', reco_addjets_dR_width),
	  xNbins_gen_addbjets_dR, array('d', gen_addbjets_dR_width)
	)
	h_respMatrix_deltaR[iChannel][iStep].GetXaxis().SetTitle("Reco. #DeltaR_{b#bar{b}}")
	h_respMatrix_deltaR[iChannel][iStep].GetYaxis().SetTitle("Gen. #DeltaR_{b#bar{b}}")
	h_respMatrix_deltaR[iChannel][iStep].Sumw2()

	h_respMatrix_invMass[iChannel][iStep] = ROOT.TH2D(
	  "h_" + RESPONSE_MATRIX_INVARIANT_MASS_ + "_Ch" + str(iChannel) + "_S" + str(iStep) + "_" + process, "",
	  xNbins_reco_addjets_M, array('d', reco_addjets_M_width),
	  xNbins_gen_addbjets_M, array('d', gen_addbjets_M_width)
	)
	h_respMatrix_invMass[iChannel][iStep].GetXaxis().SetTitle("Reco. M_{b#bar{b}}(GeV)")
	h_respMatrix_invMass[iChannel][iStep].GetYaxis().SetTitle("Gen. M_{b#bar{b}}(GeV)")
	h_respMatrix_invMass[iChannel][iStep].Sumw2()

    if ttbb == True :
      genchain = TChain("ttbbLepJets/gentree")
      genchain.Add(directory+"/"+inputFile+".root")

      print "GENTREE RUN"
      for i in xrange(genchain.GetEntries()):
	printProgress(i, genchain.GetEntries(), 'Progress:', 'Complete', 1, 50)
	genchain.GetEntry(i)
	addbjet1 = TLorentzVector()
	addbjet2 = TLorentzVector()
	addbjet1.SetPtEtaPhiE(genchain.addbjet1_pt, genchain.addbjet1_eta, genchain.addbjet1_phi, genchain.addbjet1_e)
	addbjet2.SetPtEtaPhiE(genchain.addbjet2_pt, genchain.addbjet2_eta, genchain.addbjet2_phi, genchain.addbjet2_e)

	gendR = addbjet1.DeltaR(addbjet2)
	genM = (addbjet1+addbjet2).M()

	if genchain.genchannel == muon_ch:
	  h_gen_addbjets_deltaR_nosel[muon_ch].Fill(gendR,genchain.genweight)
	  h_gen_addbjets_invMass_nosel[muon_ch].Fill(genM,genchain.genweight)
	elif genchain.genchannel == electron_ch:
	  h_gen_addbjets_deltaR_nosel[electron_ch].Fill(gendR,genchain.genweight)
	  h_gen_addbjets_invMass_nosel[electron_ch].Fill(genM,genchain.genweight)
	else:
	  print("Error")
	  
    chain = TChain("ttbbLepJets/tree")
    chain.Add(directory+"/"+inputFile+".root")

    print "TREE RUN"
    nEvents = 0
    nEvt_isMatch_DNN = 0
    nEvt_isMatch_mindR = 0

    for i in xrange(chain.GetEntries()):
      printProgress(i, chain.GetEntries(), 'Progress:', 'Complete', 1, 50)
      chain.GetEntry(i)

      eventweight = chain.PUWeight[0]*chain.genweight
      if not data :
	eventweight *= chain.lepton_SF[0]*chain.jet_SF_CSV[0]

      MET_px = chain.MET*math.cos(chain.MET_phi)
      MET_py = chain.MET*math.sin(chain.MET_phi)
      nu = TLorentzVector(MET_px, MET_py, 0, chain.MET)
	
      lep = TLorentzVector()
      lep.SetPtEtaPhiE(chain.lepton_pT, chain.lepton_eta, chain.lepton_phi, chain.lepton_E)

      passmuon = False
      passelectron = False
      passmuon = chain.channel == muon_ch and lep.Pt() > muon_pt and abs(lep.Eta()) < muon_eta 
      passelectron = chain.channel == electron_ch and lep.Pt() > electron_pt and abs(lep.Eta()) < electron_eta
      if passmuon == False and passelectron == False : continue

      njets = 0
      nbjets = 0

      for iJet in range(len(chain.jet_pT)):
	jet = TLorentzVector()
	jet.SetPtEtaPhiE(chain.jet_pT[iJet],chain.jet_eta[iJet],chain.jet_phi[iJet],chain.jet_E[iJet])

	if not data :
	  jet *= chain.jet_JER_Nom[iJet]

	if jet.Pt() < jet_pt or abs(jet.Eta()) > jet_eta : continue

	njets += 1
	if chain.jet_CSV[iJet] > jet_CSV_tight : nbjets += 1

      mindR_idx = []
      dataset = [] 
      combidR = []
      minimumdR = 9999
      for j in range(0, len(chain.jet_pT)-1):
        if chain.jet_pT[j] < jet_pt or abs(chain.jet_eta[j]) < jet_eta : continue
	for k in range(j+1, len(chain.jet_pT)):
	  if chain.jet_pT[k] < jet_pt or abs(chain.jet_eta[k]) < jet_eta : continue
	  if chain.jet_CSV[j] > jet_CSV_tight and chain.jet_CSV[k] > jet_CSV_tight:
	    b1 = TLorentzVector()
	    b2 = TLorentzVector()
	    b1.SetPtEtaPhiE(chain.jet_pT[j], chain.jet_eta[j], chain.jet_phi[j], chain.jet_E[j])
	    b2.SetPtEtaPhiE(chain.jet_pT[k], chain.jet_eta[k], chain.jet_phi[k], chain.jet_E[k])
#if not data :
#	      b1 *= chain.jet_JER_Nom[j]
#	      b2 *= chain.jet_JER_Nom[k]
	    dR = b1.DeltaR(b2)
	    combidR.append([j,k])
	    
	    #CNN input
	    dataset.append([
	      dR,abs(b1.Eta()-b2.Eta()),b1.DeltaPhi(b2),
	      (b1+b2+nu).Pt(),(b1+b2+nu).Eta(),(b1+b2+nu).Phi(),(b1+b2+nu).M(),
	      (b1+b2+lep).Pt(),(b1+b2+lep).Eta(),(b1+b2+lep).Phi(),(b1+b2+lep).M(),
	      (b1+lep).Pt(),(b1+lep).Eta(),(b1+lep).Phi(),(b1+lep).M(),
	      (b2+lep).Pt(),(b2+lep).Eta(),(b2+lep).Phi(),(b2+lep).M(),
	      (b1+b2).Pt(),(b1+b2).Eta(),(b1+b2).Phi(),(b1+b2).M(),
	      chain.jet_CSV[j],chain.jet_CSV[k],
	      b1.Pt(),b2.Pt(),b1.Eta(),b2.Eta(),b1.Phi(),b2.Phi(),b1.E(),b2.E()
	    ])

	    tmp_idx = []
	    tmp_idx.append(j)
	    tmp_idx.append(k)
	    if dR < minimumdR :
	      minimumdR = dR
	      mindR_idx = tmp_idx

      #gen level additional bjets
      gen_addbjet1 = TLorentzVector()
      gen_addbjet2 = TLorentzVector()
      gen_dR = 999
      gen_M = 999
      if ttbb :
	gen_addbjet1.SetPtEtaPhiE(chain.addbjet1_pt,chain.addbjet1_eta,chain.addbjet1_phi,chain.addbjet1_e)
	gen_addbjet2.SetPtEtaPhiE(chain.addbjet2_pt,chain.addbjet2_eta,chain.addbjet2_phi,chain.addbjet2_e)

	gen_dR = gen_addbjet1.DeltaR(gen_addbjet2)
	gen_M = (gen_addbjet1+gen_addbjet2).M()

      reco_dR = -999
      reco_M = -999
      reco_addbjet1 = TLorentzVector()
      reco_addbjet2 = TLorentzVector()
      reco_addbjet1_mindR = TLorentzVector()
      reco_addbjet2_mindR = TLorentzVector()

      #additional bjets from DNN
      if len(dataset) is not 0 :
	inputset = np.array(dataset)
	pred = prediction.eval(feed_dict={x:inputset}) 
	maxval = pred.argmax()
	
	reco_addbjet1.SetPtEtaPhiE(
	  chain.jet_pT[combidR[maxval][0]],chain.jet_eta[combidR[maxval][0]],
	  chain.jet_phi[combidR[maxval][0]],chain.jet_E[combidR[maxval][0]])
	reco_addbjet2.SetPtEtaPhiE(
	  chain.jet_pT[combidR[maxval][1]],chain.jet_eta[combidR[maxval][1]],
	  chain.jet_phi[combidR[maxval][1]],chain.jet_E[combidR[maxval][1]])

	reco_dR = reco_addbjet1.DeltaR(reco_addbjet2)
	reco_M = (reco_addbjet1+reco_addbjet2).M()

      #additional bjets from minimum delta R
      if len(mindR_idx) is not 0 :
	reco_addbjet1_mindR.SetPtEtaPhiE(
	  chain.jet_pT[mindR_idx[0]],chain.jet_eta[mindR_idx[0]],chain.jet_phi[mindR_idx[0]],chain.jet_E[mindR_idx[0]])
	reco_addbjet2_mindR.SetPtEtaPhiE(
	  chain.jet_pT[mindR_idx[1]],chain.jet_eta[mindR_idx[1]],chain.jet_phi[mindR_idx[1]],chain.jet_E[mindR_idx[1]])
      
      passchannel = -999
      passcut = 0

      #matching ratio
      isMatch_DNN = False
      isMatch_mindR = False
      isMatch_DNN = (reco_addbjet1.DeltaR(gen_addbjet1) < 0.5 and reco_addbjet2.DeltaR(gen_addbjet2) < 0.5) or (reco_addbjet1.DeltaR(gen_addbjet2) < 0.5 and reco_addbjet2.DeltaR(gen_addbjet1) < 0.5)
      isMatch_mindR = (reco_addbjet1_mindR.DeltaR(gen_addbjet1) < 0.5 and reco_addbjet2_mindR.DeltaR(gen_addbjet2) < 0.5) or (reco_addbjet1_mindR.DeltaR(gen_addbjet2) < 0.5 and reco_addbjet2_mindR.DeltaR(gen_addbjet1) < 0.5)
 
      if passmuon == True and passelectron == False :
	passchannel = muon_ch
      elif passmuon == False and passelectron == True :
	passchannel = electron_ch
      else :
	print "Error!"

      if njets >= number_of_jets :
	passcut += 1
	if nbjets >= number_of_bjets-1 :
	  passcut += 1
	  if nbjets >= number_of_bjets :
	    passcut += 1    
	    nEvents += 1
	    if isMatch_DNN : nEvt_isMatch_DNN += 1
	    if isMatch_mindR : nEvt_isMatch_mindR += 1

      if closureTest:
	outputFile = "hist_closureTest.root"
	print "FIJFIJ!"
      else :
	for iStep in range(0,passcut+1) :
	  h_lepton_pt[passchannel][iStep].Fill(lep.Pt(), eventweight)
	  h_lepton_eta[passchannel][iStep].Fill(lep.Eta(), eventweight)
	  h_njets[passchannel][iStep].Fill(njets, eventweight)
	  h_nbjets[passchannel][iStep].Fill(nbjets, eventweight)
	  h_reco_addjets_deltaR[passchannel][iStep].Fill(reco_dR, eventweight)
	  h_reco_addjets_invMass[passchannel][iStep].Fill(reco_M, eventweight)
	  if ttbb :
	    h_gen_addbjets_deltaR[passchannel][iStep].Fill(gen_dR, eventweight)
	    h_gen_addbjets_invMass[passchannel][iStep].Fill(gen_M, eventweight)
	    h_respMatrix_deltaR[passchannel][iStep].Fill(reco_dR, gen_dR, eventweight)
	    h_respMatrix_invMass[passchannel][iStep].Fill(reco_M, gen_M, eventweight)
	  
    matching_DNN = 0.0
    matching_mindR = 0.0
    if nEvents is not 0 :
      matching_DNN = float(nEvt_isMatch_DNN) / float(nEvents)
      matching_mindR = float(nEvt_isMatch_mindR) / float(nEvents)
    print "Selected Events / Total Events : "+str(nEvents)+"/"+str(chain.GetEntries())
    print "Matching Ratio from DNN : "+str(matching_DNN)+"("+str(nEvt_isMatch_DNN)+"/"+str(nEvents)+")"
    print "Matching Ratio from minimun dR : "+str(matching_mindR)+"("+str(nEvt_isMatch_mindR)+"/"+str(nEvents)+")"

    for iChannel in range(0,2) :
      for iStep in range(0,4) :
	h_lepton_pt[iChannel][iStep].AddBinContent(20,h_lepton_pt[iChannel][iStep].GetBinContent(21))
	h_lepton_eta[iChannel][iStep].AddBinContent(20,h_lepton_eta[iChannel][iStep].GetBinContent(21))
	h_njets[iChannel][iStep].AddBinContent(10,h_njets[iChannel][iStep].GetBinContent(11))
	h_nbjets[iChannel][iStep].AddBinContent(10,h_nbjets[iChannel][iStep].GetBinContent(11))
	h_reco_addjets_deltaR[iChannel][iStep].AddBinContent(xNbins_reco_addjets_dR, h_reco_addjets_deltaR[iChannel][iStep].GetBinContent(xNbins_reco_addjets_dR+1))
	h_reco_addjets_invMass[iChannel][iStep].AddBinContent(xNbins_reco_addjets_M, h_reco_addjets_invMass[iChannel][iStep].GetBinContent(xNbins_reco_addjets_M+1))
	h_gen_addbjets_deltaR[iChannel][iStep].AddBinContent(xNbins_gen_addbjets_dR, h_gen_addbjets_deltaR[iChannel][iStep].GetBinContent(xNbins_gen_addbjets_dR+1))
	h_gen_addbjets_invMass[iChannel][iStep].AddBinContent(xNbins_gen_addbjets_M, h_gen_addbjets_invMass[iChannel][iStep].GetBinContent(xNbins_gen_addbjets_M+1))

	for iXaxis in range(1, xNbins_reco_addjets_dR+1) :
	  tmp = h_respMatrix_deltaR[iChannel][iStep].GetBinContent(iXaxis, xNbins_gen_addbjets_dR)+h_respMatrix_deltaR[iChannel][iStep].GetBinContent(iXaxis, xNbins_gen_addbjets_dR+1)
	  h_respMatrix_deltaR[iChannel][iStep].SetBinContent(iXaxis, xNbins_gen_addbjets_dR, tmp)
	for iYaxis in range(1, xNbins_gen_addbjets_dR+1) :
	  tmp = h_respMatrix_deltaR[iChannel][iStep].GetBinContent(xNbins_reco_addjets_dR, iYaxis)+h_respMatrix_deltaR[iChannel][iStep].GetBinContent(xNbins_reco_addjets_dR+1, iYaxis)
	  h_respMatrix_deltaR[iChannel][iStep].SetBinContent(xNbins_reco_addjets_dR, iYaxis, tmp)

	for iXaxis in range(1, xNbins_reco_addjets_M+1) :
	  tmp = h_respMatrix_invMass[iChannel][iStep].GetBinContent(iXaxis, xNbins_gen_addbjets_M)+h_respMatrix_invMass[iChannel][iStep].GetBinContent(iXaxis, xNbins_gen_addbjets_M+1)
	  h_respMatrix_invMass[iChannel][iStep].SetBinContent(iXaxis, xNbins_gen_addbjets_M, tmp)
	for iYaxis in range(1, xNbins_gen_addbjets_M+1) :
	  tmp = h_respMatrix_invMass[iChannel][iStep].GetBinContent(xNbins_reco_addjets_M, iYaxis)+h_respMatrix_invMass[iChannel][iStep].GetBinContent(xNbins_reco_addjets_M+1, iYaxis)
	  h_respMatrix_invMass[iChannel][iStep].SetBinContent(xNbins_reco_addjets_M, iYaxis,tmp)
       
	h_lepton_pt[iChannel][iStep].ClearUnderflowAndOverflow()
	h_lepton_eta[iChannel][iStep].ClearUnderflowAndOverflow()
	h_njets[iChannel][iStep].ClearUnderflowAndOverflow()
	h_nbjets[iChannel][iStep].ClearUnderflowAndOverflow()
	h_reco_addjets_deltaR[iChannel][iStep].ClearUnderflowAndOverflow()
	h_reco_addjets_invMass[iChannel][iStep].ClearUnderflowAndOverflow()
	h_gen_addbjets_deltaR[iChannel][iStep].ClearUnderflowAndOverflow()
	h_gen_addbjets_invMass[iChannel][iStep].ClearUnderflowAndOverflow()
	h_respMatrix_deltaR[iChannel][iStep].ClearUnderflowAndOverflow()
	h_respMatrix_invMass[iChannel][iStep].ClearUnderflowAndOverflow()
    f_out.Write()
    f_out.Close()
