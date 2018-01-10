from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv
import numpy as np

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

Trad = False
model_output_name = 'veto_layer3_800_800_800'
output = csv.reader(open('models/'+ model_output_name+'/output.csv','r'))
value_sig = []
value_bkg = []
value_sig, value_bkg = split_data( output )

num_bins = 100
 
fig, ax = plt.subplots()

sig_eff = []
bkg_eff = []

# the histogram of the data
sn, sbins, patches = ax.hist(value_sig, num_bins, range=[0.0, 1.0], normed=True, color="red", label="signal")
bn, bbins, patches = ax.hist(value_bkg, num_bins, range=[0.0, 1.0], normed=True, color="blue", label="background")

print sbins, " ", bbins
# add a 'best fit' line
ax.set_xlabel('DNN output')
ax.set_ylabel('Entries')
ax.set_title('Deep learning output')

#log scale
#plt.yscale("log")

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.legend(loc='upper right')
plt.show()

fig.savefig('models/'+ model_output_name+'/output.pdf')

# Efficiency
if Trad:
  for i in range(0,50):
    sig_eff.append(sum(sn[0:i]) / sum(sn[0:num_bins]))
    bkg_eff.append(sum(bn[0:i]) / sum(bn[0:num_bins]))
else:
  for i in range(10,100):
    sig_eff.append(sum(sn[i:num_bins]) / sum(sn[0:num_bins]))
    bkg_eff.append(sum(bn[i:num_bins]) / sum(bn[0:num_bins]))

ax = plt.plot(bkg_eff, sig_eff, label='Deep Iso')
plt.ylabel("Signal efficiency (%)")
plt.xlabel("Background efficiency (%)")
plt.legend(loc='center right')
plt.savefig('models/'+ model_output_name+'/roc.pdf')
plt.show()

