#!/usr/bin/python

#############################################
#ID3 Algorithm For Clemson Parking Selection#
# Author: Nishant Vishwamitra               #
# Email:  nvishwa@clemson.edu               #
#############################################

import math
import operator

# Dataset #1: Clemson Parking dataset
columns = 3
ids = ['permittype','timeofday','building']
data = [['employee','morning','mcadams','c1' ],
        ['commuter','morning','mcadams','c1' ],
        ['commuter','noon'   ,'library','c2' ],
        ['commuter','noon'   ,'riggs'  ,'c2' ],
        ['employee','morning','riggs'  ,'c11']
       ]
############################################################
# Dataset #2: Unequal Entropies (Uncomment to use)
'''columns = 3
ids = ['x1','x2','x3']
data = [
       ['no','no','no','c0',   ],
       ['no','no','yes','c0',  ],
       ['no','yes','no','c0',  ],
       ['no','yes','yes','c0', ],
       ['yes','no','no','c1',  ],
       ['yes','no','yes','c1', ],
       ['yes','yes','no','c2', ],
       ['yes','yes','yes','c3',]
       ]'''
############################################################
# Dataset #3: Equal Entropies dataset (Uncomment to use)
'''columns = 3
ids = ['x1','x2','x3']
data = [
       ['0','0','0','0',],
       ['0','0','1','1',],
       ['0','1','0','1',],
       ['0','1','1','1',],
       ['1','0','0','1',],
       ['1','0','1','1',],
       ['1','1','0','1',],
       ['1','1','1','1',]
       ]'''
############################################################

avgEntropy = dict()
uAnswerCounts = dict()

# Function to calculate the partitions. Returns dict of column and partitions
def getPartitions():
  partitions = dict()
  for c in range(0, columns):
    l = list()
    for d in data:
      l.append(d[c])
    partitions[ids[c]] = l
  return partitions

# answers
def answers():
  u = list()  
  for d in data:
    u.append(d[columns])
  return u

# Set 0 counts
def resetCounts():
  for a in set(answers()):
    uAnswerCounts[a] = 0

# Calculate Entropies for each column
def entropy():
  partitions = getPartitions()
  for id in ids:
    partition = partitions[id]
    sum = 0
    for p in set(partition):
      # Find the count of p in data
      count = 0
      for element in partition:
        if p == element:
          count += 1
      
      resetCounts()
      # Get answers
      ans = answers()
      for i,a in enumerate(ans):
        if(partition[i] == p):
          uAnswerCounts[a] += 1
      # Calculate avg entropy or p
      for k, v in uAnswerCounts.iteritems():
        if v != 0:
          sum += count * (- (v * 1.0)/count * math.log((v * 1.0)/count, 2))

    avgEntropy[id] = sum

# Calc and store avg entropies
entropy()
print "The average entropies are listed below..."
print "  ", avgEntropy, "\n"

# Function to break tie in case of equal entropies.
# For breaking the tie, I am using an FCFS approach. 
# It is not very elegant but it works and gives uniform
# trees for a dataset.
def tieBreaker():
  # First sort by order of appearance in data, then by entropy
  def byindex(lhs, rhs):
    if ids.index(lhs[0]) < ids.index(rhs[0]):
      return -1
    elif ids.index(lhs[0]) == ids.index(rhs[0]):
      return 0
    else:
      return 1
   
  sAvgEnt = sorted(avgEntropy.items(), cmp=byindex) 
  sAvgEnt = sorted(sAvgEnt, key=operator.itemgetter(1)) 
  return sAvgEnt

order = list()
i = 0
nonodes = 0

# Resolve ties
sAvgEnt = tieBreaker()

for j,k in sAvgEnt:
  order.append(j)

print "Tree nodes would be evaluated in the following order..."
print "  ", order, "\n"

def draw(root, f):
  global i
  global nonodes
  # get possible outcomes
  partitions = set(getPartitions()[root])
  for partition in partitions:
    res = list()
    for p,a in zip(getPartitions()[root], answers()):
      if p == partition:
        if a not in res:
          res.append(a)
    
    if len(res) == 1:
      nonodes += 1
      f.write(root)
      f.write(" [shape=box,style=filled,color=\".7 .3 1.0\"];\n")
      f.write("  " + root + "->")
      f.write(res[0])
      f.write("[ label=\"")
      f.write(partition)
      f.write("\" ]")
      f.write("\n")
      # Remove all rows from d where root = partition and ans = res[0]
      removed = list()
      for d in data:
        posOfRoot = ids.index(root)
        
        if d[posOfRoot] == partition and d[columns] == res[0]:
          removed.append(d)
      for r in removed:
        data.remove(r)
    else :
      if len(res) != 0:
        nonodes += 1
        f.write(root)
        f.write(" [shape=box,style=filled,color=\".7 .3 1.0\"];\n")
        f.write("  " + root + "->")
        f.write(order[i+1])
        f.write("[ label=\"")
        f.write(partition)
        f.write("\" ]")
        f.write("\n")
        i += 1
  if nonodes != 0 and nonodes == len(partitions) and i < columns:
    nonodes = 0
    draw(order[i], f)

# Draw the tree for an input
def makeGraph():
  with open("graph.gv","w+") as f:
    f.write("digraph G {\n")
    f.write("node [shape=circle,style=filled,color=\".7 .3 1.0\"];")
    f.write("\n")
    draw(order[i], f) # start with least entropy attribute
    f.write("}")
    f.close()
    print "ID3 Algrithm Successfully Run..."
    print "Generated GraphViz File \"graph.gv\" in cwd."
    print "Please run \"dot -Tpng graph.gv -o graph.png\" to visualize the decision tree (an image called graph.png would be generated). "
    print "If graphviz has not been installed, run \"sudo apt-get install graphviz\"."

makeGraph()

