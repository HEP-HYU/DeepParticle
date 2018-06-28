# -*- coding: utf-8 -*-
"""
Convert ROOT files into python numpy 
readable files
Author: Suyong Choi (Department of Physics, Korea University suyong@korea.ac.kr)
Created: 2018-Feb-12
"""

from ROOT import TChain, TBranch
import numpy as np
import array
import pickle

def convertROOTtoNumpy(treename, rootfile, numpyfile, branchnamelistfile, branchnamelist=[]):
    # open root file and get list of branches
    tc = TChain(treename)
    tc.Add(rootfile)

    tc.LoadTree(0) # you need to do this to load the file
    nentries = tc.GetEntries()
    
    allbranchlist = tc.GetListOfBranches()
    allbranchnamelist = map(TBranch.GetName, allbranchlist)
#    for abranch in allbranches:
#        allbranchlist.append(abranch.GetName())
    branchindices = []
    branchlist = []       
    tmpvlist=[]
    ndim = len(branchnamelist)
    # if no branchnamelist is given then process all branches
    # could run into problem if a branch is not a number type
    if ndim==0:
        getall = 1
        ndim = len(allbranchnamelist)
        branchnamelist = allbranchnamelist
        branchindices = range(ndim)
    else:
        getall = 0        
        for abranch in branchnamelist:
            index = allbranchnamelist.index(abranch)
            if index>=0:
                branchindices.append(index)
            
    # create raw list of branches to be used
#    for bridx in branchindices:
 #       branchlist.append(allbranches.At(bridx))
    branchlist = map(allbranchlist.At, branchindices)
    valid_branchnamelist=[]
    # create variables that will store 
    for abranch in branchlist:
        leaves = abranch.GetListOfLeaves()
        branchname = abranch.GetName()
        for leaf in leaves:
            leaftype = leaf.GetTypeName()
            print branchname + " : " + leaftype
            validleaf = True
            if leaftype == 'Double_t':
                vtype = 'd'
            elif leaftype == 'Int_t':
                vtype = 'i'
            elif leaftype == 'Float_t':
                vtype = 'f'
            else:
                print abranch.GetName()+' is not a number but of type %s'%leaftype
                print 'This branch will be skipped'
                validleaf = False
            
            if validleaf:
                # must declare array type to store the root leaf value 
                tmpvar = array.array(vtype, [0])
                
                valid_branchnamelist.append(branchname)
                tc.SetBranchStatus(branchname, 1)
                tc.SetBranchAddress(branchname, tmpvar)
                # this list contains the variables that will get updated when GetEntry is called
                tmpvlist.append(tmpvar)
                
    # number of leaves that are really numbers
    ndim = len(tmpvlist)
    
    try:
        assert(ndim == len(tmpvlist))
    except:
        print 'Total branches %d'%ndim
        print 'Allocated variables %d'%(len(tmpvlist))
        raise RuntimeError('variable list does not match the number of branches')

    # create numpy
    convertednumpy = np.zeros(shape=(nentries, ndim), dtype=np.float32)
    
    for iev in xrange(nentries):
        tc.GetEntry(iev, getall)
        for ivar in xrange(ndim):
            convertednumpy[iev, ivar] = tmpvlist[ivar][0]*1.0
        #if iev%1000==0:
        #    print 'Read entry %d'%(iev)
    np.save(numpyfile, convertednumpy)
    pickle.dump(valid_branchnamelist, open(branchnamelistfile, 'wt'))
    print 'Wrote %d entries from %s into %s file'%(nentries, rootfile, numpyfile) 
    
if __name__=='__main__':
    # branches to covert to numpy
    branchlist = ['Npv', 'lep1Pt', 'lep1dxy', 'lep1dz', 'Mtl1'
        , 'lep2Pt', 'lep2dxy', 'lep2dz',  'Mtl2'
        , 'Nlep', 'Mll', 'dRll', 'dEtall', 'dPhill', 'MEt', 'Ht'
        , 'Htb', 'HtRat', 'Mt', 'Njet', 'NbjetL', 'NbjetM', 'NbjetT'
        , 'WNjet', 'Aplanarity', 'Planarity', 'Sphericity', 'Centrality'
        , 'C', 'D', 'Mjj', 'dPhijj', 'Mlj', 'dPhilj', 'jet1Pt', 'jet2Pt'
        , 'Nw']
    
    convertROOTtoNumpy('baselinetree'
        , 'ntuple_nominal/Template_BLSUSY_FourTopSignal_SUSYWP_Nominal.root'
        , 'testoutput.npy'
        , 'testoutput.pkl')
    #    , branchnamelist=branchlist)
    
