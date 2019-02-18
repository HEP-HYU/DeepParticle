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

genchain = TChain("fcncLepJets/gentree");
genchain.Add("TT_powheg_ttbb.root")
chain = TChain("fcncLepJets/tree");
chain.Add("TT_powheg_ttbb.root")

f = TFile( 'output_ttbb.root', 'RECREATE')
t = TTree( 'tree', 'tree for ttbb')

signal = array( 'i', [0] )
dR = array( 'd', [0] )
dEta = array( 'd', [0] )
dPhi = array( 'd', [0] )
nuPt = array( 'd', [0] )
nuEta = array( 'd', [0] )
nuPhi = array( 'd', [0] )
nuM = array( 'd', [0] )
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
t.Branch('nuM', nuM, 'nuM/D')
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

nEvents = 0
nEvents_matchable = 0

entries = chain.GetEntries()
for i in xrange(entries):
  chain.GetEntry(i)

  eventweight = chain.PUWeight[0]*chain.genweight*chain.lepton_SF[0]*chain.jet_SF_deepCSV_30[0]

  MET_px = chain.MET * math.cos( chain.MET_phi )
  MET_py = chain.MET * math.sin( chain.MET_phi )
  nu = TLorentzVector( MET_px, MET_py, 0, chain.MET)
  lep = TLorentzVector() 
  lep.SetPtEtaPhiE( chain.lepton_pt, chain.lepton_eta, chain.lepton_phi, chain.lepton_e)
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
  for j in range( len(chain.jet_pt) ):
    jet = TLorentzVector()
    jet.SetPtEtaPhiE(chain.jet_pt[j], chain.jet_eta[j], chain.jet_phi[j], chain.jet_e[j])
    jet *= chain.jet_JER_Nom[j]
    if jet.Pt() < 30 or abs(jet.Eta()) > 2.4 :
      continue
    njets = njets + 1
    btag = 0.9535 #tight working point
    #btag = 0.8484 #medium working point
    #btag = 0.54264 #loose working point
    if chain.jet_SF_deepCSV_30[j] > btag:
      nbjets = nbjets+1

  if njets < 6:  
    continue
  if nbjets < 3:
    continue
 
  for i in range( len(chain.jet_pt) ):
    tmp = TLorentzVector()
    tmp.SetPtEtaPhiE( chain.jet_pt[i], chain.jet_eta[i], chain.jet_phi[i], chain.jet_e[i] )
    #when finding matched jet, not use pt and eta
    #tmp *= chain.jet_JER_Nom[i]
    #if tmp.Pt() > 20 and abs(tmp.Eta()) < 2.4: 
    if addbjet1.DeltaR( tmp ) < 0.4:
      addbjet1_matched = tmp;
    if addbjet2.DeltaR( tmp ) < 0.4:
      addbjet2_matched = tmp;

  nEvents = nEvents + 1
  if addbjet1_matched.Pt() > 0 and  addbjet2_matched.Pt() > 0 :
    nEvents_matchable =  nEvents_matchable + 1 

  for j in range( len(chain.jet_pt) - 1):
    for k in range( j+1, len(chain.jet_pt) ):
      jet1 = TLorentzVector()
      jet1.SetPtEtaPhiE( chain.jet_pt[j], chain.jet_eta[j], chain.jet_phi[j], chain.jet_e[j] )
      jet1 *= chain.jet_JER_Nom[j]
      jet2 = TLorentzVector()
      jet2.SetPtEtaPhiE( chain.jet_pt[k], chain.jet_eta[k], chain.jet_phi[k], chain.jet_e[k] )
      jet2 *= chain.jet_JER_Nom[k]

      #when training, not applying jet pt and eta for signal sample
      #acceptanceCutOnJets = jet1.Pt() > 20 and abs(jet1.Eta()) < 2.4 and jet2.Pt() > 20 and abs(jet2.Eta()) < 2.4
      if chain.jet_SF_deepCSV_30[j] > btag and chain.jet_SF_deepCSV_30[k] > btag:
	b1 = TLorentzVector()
	b2 = TLorentzVector()
	b1.SetPtEtaPhiE( chain.jet_pt[j], chain.jet_eta[j], chain.jet_phi[j], chain.jet_e[j]) 
	b2.SetPtEtaPhiE( chain.jet_pt[k], chain.jet_eta[k], chain.jet_phi[k], chain.jet_e[k]) 
	dR[0] = b1.DeltaR(b2)
	dEta[0] = abs( b1.Eta() - b2.Eta())
	dPhi[0] = b1.DeltaPhi(b2)
	nuPt[0] = (b1+b2+nu).Pt()
	nuEta[0] = (b1+b2+nu).Eta()
	nuPhi[0] = (b1+b2+nu).Phi()
	nuM[0] = (b1+b2+nu).M()
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
	csv1[0] = chain.jet_SF_deepCSV_30[j] 
	csv2[0] = chain.jet_SF_deepCSV_30[k]
	pt1[0] = b1.Pt()
	pt2[0] = b2.Pt()
	eta1[0] = b1.Eta()
	eta2[0] = b2.Eta()
	phi1[0] = b1.Phi()
	phi2[0] = b2.Phi()
	e1[0] = b1.E()
	e2[0] = b2.E()

	if ( addbjet1_matched.DeltaR( b1 ) == 0 and addbjet2_matched.DeltaR( b2 ) ==0 ) or ( addbjet2_matched.DeltaR( b1) == 0  and addbjet1_matched.DeltaR( b2) == 0 ):
	  signal[0] = 1
	else:
	  signal[0] = 0

	t.Fill()

  nEvents = nEvents + 1
  #print match1 , " ", match2
  if nEvents%1000 == 0:
    print "nEvents (matchable) = ", nEvents, "nEvents_matchable = ", nEvents_matchable


f.Write()
f.Close()

if nEvents is not 0:
  match_matchable = nEvents_matchable / nEvents
  print "matable rate = ", match_matchable

timer.Stop()
rtime = timer.RealTime(); # Real time (or "wall time")
ctime = timer.CpuTime(); # CPU time
print("RealTime={0:6.2f} seconds, CpuTime={1:6.2f} seconds").format(rtime,ctime)
print("{0:4.2f} events / RealTime second .").format( nEvents/rtime)
print("{0:4.2f} events / CpuTime second .").format( nEvents/ctime)
