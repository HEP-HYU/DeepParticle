import os
import re
import string
from processfiles import processROOTfilelist
from ROOT import TFile 
import ROOT

myfile = TFile("output_ttbb.root")

#variables = ["signal","dR","dEta","dPhi","nuPt","nuEta","nuPhi","nuMass","lbPt","lbEta","lbPhi","lbMass","lb1Pt","lb1Eta","lb1Phi","lb1Mass","lb2Pt","lb2Eta","lb2Phi","lb2Mass","lb1nuPt","lb1nuEta","lb1nuPhi","lb1nuMass","lb2nuPt","lb2nuEta","lb2nuPhi","lb2nuMass","diPt","diEta","diPhi","diMass","csv1","csv2","pt1","pt2","eta1","eta2","phi1","phi2","e1","e2"]
variables = ["signal","dR","dEta","dPhi","nuPt","nuEta","nuPhi","nuM","lbPt","lbEta","lbPhi","lbMass","lb1Pt","lb1Eta","lb1Phi","lb1Mass","lb2Pt","lb2Eta","lb2Phi","lb2Mass","diPt","diEta","diPhi","diMass","csv1","csv2","pt1","pt2","eta1","eta2","phi1","phi2","e1","e2"]

processROOTfilelist("./","tree",variables)

