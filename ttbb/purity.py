from ROOT import TFile, TTree
from ROOT import *
import ROOT
import math

gROOT.SetStyle("Plain")
gStyle.SetOptStat(1110)
gStyle.SetOptFit(1)
gStyle.SetStatW(0.25)
gStyle.SetStatH(0.15)

gStyle.SetCanvasDefH(200)
gStyle.SetCanvasDefW(200)

gStyle.SetAxisColor(1, "XYZ")
gStyle.SetStripDecimals(1)
gStyle.SetTickLength(0.03, "XYZ")
gStyle.SetNdivisions(510, "XYZ")
gStyle.SetPadTickX(1)
gStyle.SetPadTickY(1)

gStyle.SetPadTopMargin(0.1)
gStyle.SetPadBottomMargin(0.15)
gStyle.SetPadLeftMargin(0.15)
gStyle.SetPadRightMargin(0.10)

gStyle.SetTitleColor(1, "XYZ")
gStyle.SetTitleFont(42, "XYZ")
gStyle.SetTitleSize(0.04, "XYZ")
gStyle.SetTitleXOffset(1.1)
gStyle.SetTitleYOffset(1.5)

gStyle.SetLabelColor(1, "XYZ")
gStyle.SetLabelFont(42, "XYZ")
gStyle.SetLabelOffset(0.008, "XYZ")
gStyle.SetLabelSize(0.04, "XYZ")

gROOT.ForceStyle()

def GetPurityStability( h2_dR_Response_ch0 , channel):

  h_purity_dR_ch0 = h2_dR_Response_ch0.ProjectionX()
  h_stability_dR_ch0 = h2_dR_Response_ch0.ProjectionY()
  h_purity_Mass_ch0 = h2_Mass_Response_ch0.ProjectionX()
  h_stability_Mass_ch0 = h2_Mass_Response_ch0.ProjectionY()

  h_dR_ch0_Reco = h_purity_dR_ch0.Clone("h_dR_ch0_Reco")
  h_dR_ch0_Gen = h_stability_dR_ch0.Clone("h_dR_ch0_Gen")
  h_Mass_ch0_Reco = h_purity_Mass_ch0.Clone("h_Mass_ch0_Reco")
  h_Mass_ch0_Gen = h_stability_Mass_ch0.Clone("h_Mass_ch0_Gen")

  for i in range( h_purity_dR_ch0.GetNbinsX() ):
    den = h_purity_dR_ch0.GetBinContent(i+1)
    num = h2_dR_Response_ch0.GetBinContent(i+1, i+1)
    purity = num/den
    h_purity_dR_ch0.SetBinContent(i+1, purity)
    h_purity_dR_ch0.SetBinError(i+1, abs(purity)*math.sqrt(pow(math.sqrt(den)/den,2)+pow(math.sqrt(num)/num,2)))
    
    print i," ", den," ", num," ", purity
  for i in range( h_stability_dR_ch0.GetNbinsX() ):
    den = h_stability_dR_ch0.GetBinContent(i+1)
    num = h2_dR_Response_ch0.GetBinContent(i+1, i+1)
    stability = num/den
    h_stability_dR_ch0.SetBinContent(i+1, stability)

  for i in range( h_purity_Mass_ch0.GetNbinsX() ):
    den = h_purity_Mass_ch0.GetBinContent(i+1)
    num = h2_Mass_Response_ch0.GetBinContent(i+1, i+1)
    purity = num/den
    h_purity_Mass_ch0.SetBinContent(i+1, purity)

  for i in range( h_stability_Mass_ch0.GetNbinsX() ):
    den = h_stability_Mass_ch0.GetBinContent(i+1)
    num = h2_Mass_Response_ch0.GetBinContent(i+1, i+1)
    stability = num/den
    h_stability_Mass_ch0.SetBinContent(i+1, stability)

  c_purity_dR_ch0 = TCanvas( ('c_purity_dR_%s'%channel),"c_purity_dR_ch0",1)
  h_purity_dR_ch0.SetTitle("")
  h_purity_dR_ch0.SetStats(0)
  h_purity_dR_ch0.GetXaxis().SetTitle("dR between two add. b jets")
  h_purity_dR_ch0.GetYaxis().SetTitle("Purity")
  h_purity_dR_ch0.Draw("hist")
  c_purity_dR_ch0.Print( ('c_purity_dR_%s.pdf' % channel) )

  c_stability_dR_ch0 = TCanvas( ('c_stability_dR_%s'%channel),"c_stability_dR_ch0",1)
  h_stability_dR_ch0.SetTitle("")
  h_stability_dR_ch0.SetStats(0)
  h_stability_dR_ch0.GetXaxis().SetTitle("dR between two add. b jets")
  h_stability_dR_ch0.GetYaxis().SetTitle("Stability") 
  h_stability_dR_ch0.Draw("hist")
  c_stability_dR_ch0.Print( ('c_stability_dR_%s.pdf' % channel) )

  c_purity_Mass_ch0 = TCanvas( ('c_purity_Mass_%s'%channel),"c_purity_Mass_ch0",1)
  h_purity_Mass_ch0.SetTitle("")
  h_purity_Mass_ch0.SetStats(0)
  h_purity_Mass_ch0.GetXaxis().SetTitle("Invariant mass (GeV)")
  h_purity_Mass_ch0.GetYaxis().SetTitle("Purity")
  h_purity_Mass_ch0.Draw("hist")
  c_purity_Mass_ch0.Print( ('c_purity_Mass_%s.pdf' % channel) )

  c_stability_Mass_ch0 = TCanvas( ('c_stability_Mass_%s'%channel),"c_stability_Mass_ch0",1)
  h_stability_Mass_ch0.SetTitle("")
  h_stability_Mass_ch0.SetStats(0)
  h_stability_Mass_ch0.GetXaxis().SetTitle("Invariant mass (GeV)")
  h_stability_Mass_ch0.GetYaxis().SetTitle("Stability")
  h_stability_Mass_ch0.Draw("hist")
  c_stability_Mass_ch0.Print( ('c_stability_Mass_%s.pdf' % channel) )

  c_dR_Response_ch0 = TCanvas( ('c_dR_Response_%s'%channel),"c_dR_Response_ch0",1)
  h2_dR_Response_ch0.SetTitle("")
  h2_dR_Response_ch0.SetStats(0)
  h2_dR_Response_ch0.GetXaxis().SetTitle("dR (RECO)")
  h2_dR_Response_ch0.GetYaxis().SetTitle("dR (GEN)")
  h2_dR_Response_ch0.Draw("box")
  c_dR_Response_ch0.Print( ('c_dR_Response_%s.pdf'%channel))

  c_Mass_Response_ch0 = TCanvas( ('c_Mass_Response_%s'%channel),"c_Mass_Response_ch0",1)
  h2_Mass_Response_ch0.SetTitle("")
  h2_Mass_Response_ch0.SetStats(0)
  h2_Mass_Response_ch0.GetXaxis().SetTitle("Mass (RECO)")
  h2_Mass_Response_ch0.GetYaxis().SetTitle("Mass (GEN)")
  h2_Mass_Response_ch0.Draw("box")
  c_Mass_Response_ch0.Print( ('c_Mass_Response_%s.pdf'%channel))

  c_dR_ch0_Reco = TCanvas( ('c_dR_%s_Reco'%channel), "c_dR_ch0_Reco", 2)
  h_dR_ch0_Reco.SetTitle("")
  h_dR_ch0_Reco.SetStats(0)
  h_dR_ch0_Reco.GetXaxis().SetTitle("Reco. dR")
  h_dR_ch0_Reco.Draw("hist")
  c_dR_ch0_Reco.Print( ('c_dR_%s_Reco.pdf'%channel) )

  c_dR_ch0_Gen = TCanvas( ('c_dR_%s_Gen'%channel), "c_dR_ch0_Gen", 2)
  h_dR_ch0_Gen.SetTitle("")
  h_dR_ch0_Gen.SetStats(0)
  h_dR_ch0_Gen.GetXaxis().SetTitle("Gen. dR")
  h_dR_ch0_Gen.Draw("hist")
  c_dR_ch0_Gen.Print( ('c_dR_%s_Gen.pdf'%channel) )

  c_Mass_ch0_Reco = TCanvas( ('c_Mass_%s_Reco'%channel), "c_Mass_ch0_Reco", 2)
  h_Mass_ch0_Reco.SetTitle("")
  h_Mass_ch0_Reco.SetStats(0)
  h_Mass_ch0_Reco.GetXaxis().SetTitle("Reco. Mass (GeV)")
  h_Mass_ch0_Reco.Draw("hist")
  c_Mass_ch0_Reco.Print( ('c_Mass_%s_Reco.pdf'%channel) )

  c_Mass_ch0_Gen = TCanvas( ('c_Mass_%s_Gen'%channel), "c_Mass_ch0_Gen", 2)
  h_Mass_ch0_Gen.SetTitle("")
  h_Mass_ch0_Gen.SetStats(0)
  h_Mass_ch0_Gen.GetXaxis().SetTitle("Gen. Mass (GeV)")
  h_Mass_ch0_Gen.Draw("hist")
  c_Mass_ch0_Gen.Print( ('c_Mass_%s_Gen.pdf'%channel) )

f = ROOT.TFile("hist_ttbb.root")

h2_dR_Response_ch0  = f.Get("h2_dR_Response_ch0")
h2_dR_Response_ch1  = f.Get("h2_dR_Response_ch1")

GetPurityStability( h2_dR_Response_ch0 , "ch0")
GetPurityStability( h2_dR_Response_ch1 , "ch1")

##Acceptance
h_dR_Gen_Den = f.Get("h_dR_Gen_Den")
h_Mass_Gen_Den = f.Get("h_Mass_Gen_Den")
h_dR_Gen_ch0 = f.Get("h_dR_Gen_ch0")
h_Mass_Gen_ch0 = f.Get("h_Mass_Gen_ch0")
h_dR_Gen_ch1 = f.Get("h_dR_Gen_ch1")
h_Mass_Gen_ch1 = f.Get("h_Mass_Gen_ch1")

h_dR_Acc_ch0 = h_dR_Gen_ch0.Clone("h_dR_Acc_ch0")
h_dR_Acc_ch1 = h_dR_Gen_ch1.Clone("h_dR_Acc_ch1")
h_Mass_Acc_ch0 = h_Mass_Gen_ch0.Clone("h_Mass_Acc_ch0")
h_Mass_Acc_ch1 = h_Mass_Gen_ch1.Clone("h_Mass_Acc_ch1")

h_dR_Acc_ch0.Divide( h_dR_Gen_Den )
h_dR_Acc_ch1.Divide( h_dR_Gen_Den )
h_Mass_Acc_ch0.Divide( h_Mass_Gen_Den )
h_Mass_Acc_ch1.Divide( h_Mass_Gen_Den )

c_dR_Acc_ch0 = TCanvas("c_dR_Acc_ch0","c_dR_Acc_ch0",1)
h_dR_Acc_ch0.Draw()
c_dR_Acc_ch0.Print("c_dR_Acc_ch0.pdf")
c_dR_Acc_ch1 = TCanvas("c_dR_Acc_ch1","c_dR_Acc_ch1",1)
h_dR_Acc_ch1.Draw()
c_dR_Acc_ch1.Print("c_dR_Acc_ch1.pdf")
c_Mass_Acc_ch0 = TCanvas("c_Mass_Acc_ch0","c_Mass_Acc_ch0",1)
h_Mass_Acc_ch0.Draw()
c_Mass_Acc_ch0.Print("c_Mass_Acc_ch0.pdf")
c_Mass_Acc_ch1 = TCanvas("c_Mass_Acc_ch1","c_Mass_Acc_ch1",1)
h_Mass_Acc_ch1.Draw()
c_Mass_Acc_ch1.Print("c_Mass_Acc_ch1.pdf")





