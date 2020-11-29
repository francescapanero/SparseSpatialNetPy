######################### TO BE REVISED ###################

import matplotlib.pyplot as plt
import numpy as np
import csv
import powerlaw as pl

# Plot adjacency matrix indexed by locations

ind1, ind2 = np.nonzero(np.triu(Z1new, 1)) # returns indices of non zero elements
fig, ax = plt.subplots()
ax.plot(x1[ind1],x1[ind2], 'b.', x1[ind2],x1[ind1], 'b.')
ax.set(xlabel='x_i', ylabel='x_j',
       title='Adjacency matrix')

# Plot degree distribution.

deg = deg[ind]
a = np.sum(deg<=100)
fit = pl.Fit(np.sort(deg)[0:a], discrete=True) # fit power law to low degrees? very empirical
figCCDF = pl.plot_ccdf(deg, label='alpha=10')
figCCDF.set(xlabel='degree', ylabel='distribution',
       title='Double power law degree distribution')

# TO DO: add lines of power law. This does not work.

y = np.linspace(1, 100, 100)
plt.plot(y, y**(-sigma))
y = np.linspace(100, 1000, 1000)
plt.plot(y, y**(-tau))
#fit.plot_ccdf(color='r', linewidth=2, ax=figCCDF)
#fit.power_law.plot_ccdf(color='r', linestyle='--', ax=figCCDF)

# second way: with Poisson
# accept = (np.random.poisson(XYw / (1 + XY ** beta)) > 0)
# accept = accept + accept.T
# accept = (accept > 0)
# Z2 = accept
# deg = np.sum(Z2, axis=1)
# ind = (deg > 0)
# x1 = x[ind]
# Z2 = Z2[np.ix_(ind, ind)]
# ind1, ind2 = np.nonzero(np.triu(Z2, 1))
# plt.plot(x1[ind1],x1[ind2], 'b.', x1[ind2],x1[ind1], 'b.')


# Plot weight layers

w0 = np.min(w)
wstar = [w0 * (2 ** j) for j in range(100)] # layers
fig, ax = plt.subplots()
ax.plot(np.sort(np.log(w)),'bo')
ax.set(xlabel='index of node', ylabel='log sociability',
       title='Layers of sociability weights')
for wj in wstar[0:8]:
    ax.axhline(y=np.log(wj),color='red')

# Save edge list and nodes attributes (locations) to plot on gephi

List = [('Source', 'Target')]
for source in range(np.shape(Z1new)[0]):
    for target in range(np.shape(Z1new)[0]):
        if Z1new[source,target] == 1:
            List.append((target, source))
with open('edge_list.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerows(List)

location = [('Id', 'Label', 'x1', 'x2')]
for i in range(np.shape(Z1new)[0]):
    location.append((i,i,x1[i,0],x1[i,1]))
with open('node_att.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerows(location)
