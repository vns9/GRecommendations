import pylab
fig = pylab.figure()
figlegend = pylab.figure(figsize=(4,2))
ax = fig.add_subplot(111)
lines = ax.plot(range(10), '-b', range(10), '-g')
figlegend.legend(lines, ('Benchmark attention + Additional Information', 'Benchmark Attention'), 'center')
fig.show()
figlegend.show()
figlegend.savefig('legend1.png', dpi=200)