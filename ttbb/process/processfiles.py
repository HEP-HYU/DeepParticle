# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:01:46 2018

@author: Suyong Choi (Department of Physics, Korea University suyong@korea.ac.kr)
"""

import os
import re
import string
import convertROOTtoNumpy

def processROOTfilelist(directory, treename, branchnamelist=[]):
    flist = os.listdir(directory)
    for fname in flist:
        fullname = os.path.join(directory, fname)
        if os.path.isfile(fullname) and re.match('.*\.root', fname):
            numpyfname = fullname.replace('.root', '.npy')
            vlistfname = fullname.replace('.root', '.pkl')            
            convertROOTtoNumpy.convertROOTtoNumpy(treename, fullname, numpyfname, vlistfname, branchnamelist)
        elif os.path.isdir(fname):
            fullname = os.path.join(directory, fname)
            processROOTfilelist(fullname, treename, branchnamelist)

if __name__=='__main__':
    processROOTfilelist('./', 'baselinetree')