from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv
import numpy as np

def plot(vsig, vbkg, num_bins, minx, maxx, title, name):
  fig, ax = plt.subplots()
  sn3, sbins, patches = ax.hist(vsig, num_bins, range=[minx, maxx], normed=True, color="red", label="signal")
  bn3, bbins, patches = ax.hist(vbkg, num_bins, range=[minx, maxx], normed=True, color="blue", label="background")

  #print sbins, " ", bbins
  # add a 'best fit' line
  ax.set_ylabel('Entries')
  ax.set_xlabel(title) 
  ax.set_title(title)

  #log scale
  #plt.yscale("log")

  # Tweak spacing to prevent clipping of ylabel
  fig.tight_layout()
  plt.legend(loc='upper right')
  plt.show()

  fig.savefig('models/output_'+name+'.pdf')

  return sn3, bn3

def split_data( data ):
  tmp_sig = []
  tmp_bkg = []
  for row in data:
    x = float(str.split(row[0]," ")[0])
    tag = float(str.split(row[0]," ")[1])
    if tag == 1:
      tmp_sig.append( x )
    elif tag == 0:
      tmp_bkg.append( x )

  return tmp_sig, tmp_bkg

models = ["veto_layer2_400_100", "veto_layer2_800_800", "veto_cnn_layer2_1600_400"]
names = ["dnn_400_100", "dnn_800_800", "cnn_1600_400"]

output_relIso = csv.reader(open('data/veto/rel_iso_with_labels.csv','r'))
output_model0 = csv.reader(open('models/'+models[0]+'/output.csv','r'))
output_model1 = csv.reader(open('models/'+models[1]+'/output.csv','r'))
output_model2 = csv.reader(open('models/'+models[2]+'/output.csv','r'))

value_sig = [[],[],[],[]]
value_bkg = [[],[],[],[]]

value_sig[0], value_bkg[0] = split_data( output_relIso )
value_sig[1], value_bkg[1] = split_data( output_model0 )
value_sig[2], value_bkg[2] = split_data( output_model1 )
value_sig[3], value_bkg[3] = split_data( output_model2)

num_bins = 1000
num_bins_rel = 100
num_bins_layer11 = 1000
sig_eff = [[],[],[],[]]
bkg_eff = [[],[],[],[]]

# the histogram of the data
sn0, bn0 = plot( value_sig[0], value_bkg[0], num_bins_rel, 0.0, 1.0, "Relative Isolation", "RelIso")
sn1, bn1 = plot( value_sig[1], value_bkg[1], num_bins, 0.0, 1.0, "DNN output", models[0])
sn2, bn2 = plot( value_sig[2], value_bkg[2], num_bins, 0.0, 1.0, "DNN output", models[1])
sn3, bn3 = plot( value_sig[3], value_bkg[3], num_bins_layer11, 0.0, 1.0, "DNN output", models[2])

fig, ax = plt.subplots()

# Efficiency
#traditional isolation : signal area should be smaller than threshold
for i in range(0,50):
  sig_eff[0].append(sum(sn0[0:i]) / sum(sn0[0:num_bins_rel]))
  bkg_eff[0].append(sum(bn0[0:i]) / sum(bn0[0:num_bins_rel]))

print "debug = ", sig_eff[0], bkg_eff[0]

#deep isolation : signal area should be above than threshold
for i in range(100,1001):
  sig_eff[1].append(sum(sn1[i:num_bins]) / sum(sn1[0:num_bins]))
  bkg_eff[1].append(sum(bn1[i:num_bins]) / sum(bn1[0:num_bins]))
  sig_eff[2].append(sum(sn2[i:num_bins]) / sum(sn2[0:num_bins]))
  bkg_eff[2].append(sum(bn2[i:num_bins]) / sum(bn2[0:num_bins]))

#for layer11
for i in range(200,2001):
    if i%10 == 0:
      sig_eff[3].append(sum(sn3[i:num_bins_layer11]) / sum(sn3[0:num_bins_layer11]))
      bkg_eff[3].append(sum(bn3[i:num_bins_layer11]) / sum(bn3[0:num_bins_layer11]))

ax = plt.plot(bkg_eff[0], sig_eff[0], label='Relative Iso')
ax = plt.plot(bkg_eff[1], sig_eff[1], label=names[0])
ax = plt.plot(bkg_eff[2], sig_eff[2], label=names[1])
ax = plt.plot(bkg_eff[3], sig_eff[3], label=names[2])

plt.ylabel("Signal efficiency (%)")
plt.xlabel("Background efficiency (%)")
plt.legend(loc='center right')
plt.savefig("models/rocs_with_RelIso.pdf")
plt.show()

