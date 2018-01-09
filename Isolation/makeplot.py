from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv
import numpy as np

output_sig = csv.reader(open('signal.csv','r'))
output_bkg = csv.reader(open('background.csv','r'))
value_sig = []
value_bkg = []

signal_selected = 0
background_selected = 0

threshold = 0.5 #for test
for row in output_sig:
    x = float(str.split(row[0]," ")[0])
    value_sig.append( x )
    if x > threshold:
      signal_selected = signal_selected + 1

for row in output_bkg:
    x = float(str.split(row[0]," ")[0])
    value_bkg.append( x )
    if x > threshold:
      background_selected = background_selected + 1

signal_total = len(value_sig)
background_total = len(value_bkg)

signal_eff = float(signal_selected / signal_total) 
background_eff = float(background_selected / background_total) 

print "signal numerator = ", signal_selected, " signal denominator = ", signal_total 
print "signal eff. = ", signal_eff, " background eff. = ", background_eff

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

fig.savefig("output.pdf")

# Efficiency
for i in range(50,64):
  sig_eff.append(sum(sn[i:num_bins]) / sum(sn[0:num_bins]))
  bkg_eff.append(sum(bn[i:num_bins]) / sum(bn[0:num_bins]))

ax = plt.plot(bkg_eff, sig_eff, label='Deep Iso')
plt.ylabel("Signal efficiency (%)")
plt.xlabel("Background efficiency (%)")
plt.legend(loc='center right')
plt.savefig("roc.pdf")
plt.show()

