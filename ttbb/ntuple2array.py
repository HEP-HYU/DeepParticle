import numpy as np
from numpy.lib.recfunctions import stack_arrays
from ROOT import *
from root_numpy import tree2array
from array import array
import math
import glob
import pandas as pd
import deepdish.io as io
"""
chain = TChain("ttbbLepJets/tree")
chain.Add("TT_powheg_ttbb.root")
f = TFile("output_ttbb.root", "RECREATE")
tree = TTree("tree", "tree for DNN")

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

tree.Branch('signal', signal, 'signal/I')
tree.Branch('dR', dR, 'dR/D')
tree.Branch('dEta', dEta, 'dEta/D')
tree.Branch('dPhi', dPhi, 'dPhi/D')
tree.Branch('nuPt', nuPt, 'nuPt/D')
tree.Branch('nuEta', nuEta, 'nuEta/D')
tree.Branch('nuPhi', nuPhi, 'nuPhi/D')
tree.Branch('nuMass', nuMass, 'nuMass/D')
tree.Branch('lbPt', lbPt, 'lbPt/D')
tree.Branch('lbEta', lbEta, 'lbEta/D')
tree.Branch('lbPhi', lbPhi, 'lbPhi/D')
tree.Branch('lbMass', lbMass, 'lbMass/D')
tree.Branch('lb1Pt', lb1Pt, 'lb1Pt/D')
tree.Branch('lb1Eta', lb1Eta, 'lb1Eta/D')
tree.Branch('lb1Phi', lb1Phi, 'lb1Phi/D')
tree.Branch('lb1Mass', lb1Mass, 'lb1Mass/D')
tree.Branch('lb2Pt', lb2Pt, 'lb2Pt/D')
tree.Branch('lb2Eta', lb2Eta, 'lb2Eta/D')
tree.Branch('lb2Phi', lb2Phi, 'lb2Phi/D')
tree.Branch('lb2Mass', lb2Mass, 'lb2Mass/D')
tree.Branch('lb1nuPt', lb1nuPt, 'lb1nuPt/D')
tree.Branch('lb1nuEta', lb1nuEta, 'lb1nuEta/D')
tree.Branch('lb1nuPhi', lb1nuPhi, 'lb1nuPhi/D')
tree.Branch('lb1nuMass', lb1nuMass, 'lb1nuMass/D')
tree.Branch('lb2nuPt', lb2nuPt, 'lb2nuPt/D')
tree.Branch('lb2nuEta', lb2nuEta, 'lb2nuEta/D')
tree.Branch('lb2nuPhi', lb2nuPhi, 'lb2nuPhi/D')
tree.Branch('lb2nuMass', lb2nuMass, 'lb2nuMass/D')
tree.Branch('diPt', diPt, 'diPt/D')
tree.Branch('diEta', diEta, 'diEta/D')
tree.Branch('diPhi', diPhi, 'diPhi/D')
tree.Branch('diMass', diMass, 'di<ass/D')
tree.Branch('csv1', csv1, 'csv1/D')
tree.Branch('csv2', csv2, 'csv2/D')
tree.Branch('pt1', pt1, 'pt1/D')
tree.Branch('pt2', pt2, 'pt2/D')
tree.Branch('eta1', eta1, 'eta1/D')
tree.Branch('eta2', eta2, 'eta2/D')
tree.Branch('phi1', phi1, 'phi1/D')
tree.Branch('phi2', phi2, 'phi2/D')
tree.Branch('e1', e1, 'e1/D')
tree.Branch('e2', e2, 'e2/D')

entries = chain.GetEntries()
for i in xrange(entries) :
  if i%1000 == 0 : print(i)
  chain.GetEntry(i)

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

  MET_px = chain.MET*math.cos(chain.MET_phi)
  MET_py = chain.MET*math.sin(chain.MET_phi)
  nu = TLorentzVector(MET_px, MET_py, 0, chain.MET)
        
  lep = TLorentzVector()
  lep.SetPtEtaPhiE(chain.lepton_pT, chain.lepton_eta, chain.lepton_phi, chain.lepton_E)

  passmuon = chain.channel == muon_ch and lep.Pt() > muon_pt and abs(lep.Eta()) < muon_eta
  passelectron = chain.channel == electron_ch and lep.Pt() > electron_pt and abs(lep.Eta()) < electron_eta

  if passmuon == False and passelectron == False:
    continue

  njets = 0
  nbjets = 0

  for iJet in range(len(chain.jet_pT)):
    jet = TLorentzVector()
    jet.SetPtEtaPhiE(chain.jet_pT[iJet],chain.jet_eta[iJet],chain.jet_phi[iJet],chain.jet_E[iJet])

    jet *= chain.jet_JER_Nom[iJet]

    if jet.Pt() < 30 or abs(jet.Eta()) > 2.4 : continue

    ++njets

    if chain.jet_CSV[iJet] > jet_CSV_tight : ++nbjets

  if njets < 6 : continue
  if nbjets < 3 : continue
  
  for i in range(len(chain.jetree.pT)):
    tmp = TLorentzVector()
    tmp.SetPtEtaPhiE(chain.jet_pT[i],chain.jet_eta[i],chain.jet_phi[i],chain.jet_E[i])
    tmp *= chian.jet_JER_Nom[i]
    if tmp.Pt() > 20 and tmp.Eta() < 2.5 and tmp.Eta() > -2.5: 
      if addbjet.DeltaR( tmp ) < 0.4:
        addbjet_matched = tmp;
      if addbjet.DeltaR( tmp ) < 0.4:
        addbjet_matched = tmp;

  for j in range(len(chain.jetree.pT)-1):
    for k in range(j+1, len(chain.jetree.pT)):
      if chain.jet_CSV[j] > jet_CSV_tight and chain.jet_CSV[k] > jet_CSV_tight:
        b1 = TLorentree.Vector()
        b2 = TLorentree.Vector()
        b1.SetPtEtaPhiE(chain.jet_pT[j], chain.jet_eta[j], chain.jet_phi[j], chain.jet_E[j])
        b2.SetPtEtaPhiE(chain.jet_pT[k], chain.jet_eta[k], chain.jet_phi[k], chain.jet_E[k])
        
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
        csv1[0] = chain.jetree.CSV[j] 
        csv2[0] = chain.jetree.CSV[k]
        pt1[0] = b1.Pt()
        pt2[0] = b2.Pt()
        eta1[0] = b1.Eta()
        eta2[0] = b2.Eta()
        phi1[0] = b1.Phi()
        phi2[0] = b2.Phi()
        e1[0] = b1.E()
        e2[0] = b2.E()

        if (addbjet1_matched.DeltaR(b1) == 0 and addbjet2_matched.DeltaR(b2) == 0) or (addbjet2_matched.DeltaR(b1) == 0  and addbjet1_matched.DeltaR(b2) == 0) : signal[0] = 1
        else : signal[0] = 0
        
        tree.Fill()

f.Write()
f.Close()
"""
ttbb = TFile.Open("output_ttbb.root")
ttbb_tree = ttbb.Get("tree")
ttbb_array = tree2array(ttbb_tree)
ttbb_df = pd.DataFrame(ttbb_array)
io.save('output_ttbb.h5', ttbb_df)
