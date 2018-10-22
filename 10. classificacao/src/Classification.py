from graphviz import Digraph
from Utils import Path
import collections
import numpy as np
import time

class DecisionTree():
    def __init__(self, col=-1, value=None, true_branch=None, false_branch=None, results=None):
        self.col = col
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.results = results

def divideSet(rows, column, value):
	splittingFunction = None
	if (isinstance(value, int) or isinstance(value, float)):
		splittingFunction = lambda row : row[column] >= value
	else: 
		splittingFunction = lambda row : row[column] == value
	list1 = [row for row in rows if splittingFunction(row)]
	list2 = [row for row in rows if not splittingFunction(row)]
	return (list1, list2)

def uniqueCounts(rows):
	results = {}
	for row in rows:
		r = row[-1]
		if (r not in results): results[r] = 0
		results[r] += 1
	return results

def entropy(rows):
	log2 = lambda x: np.log(x)/np.log(2)
	results = uniqueCounts(rows)
	entr = 0.0

	for r in results:
		p = float(results[r])/len(rows)
		entr -= p*log2(p)
	return entr

def gini(rows):
	total = len(rows)
	counts = uniqueCounts(rows)
	imp = 0.0

	for k1 in counts:
		p1 = float(counts[k1])/total  
		for k2 in counts:
			if k1 == k2: continue
			p2 = float(counts[k2])/total
			imp += p1*p2
	return imp

def growTree(rows, evaluationFunction=entropy):
	if len(rows) == 0: return DecisionTree()

	current_score = evaluationFunction(rows)
	best_gain = 0.0
	best_attr = None
	best_sets = None
	column_count = len(rows[0]) - 1

	for col in range(0, column_count):
		columnValues = [row[col] for row in rows]

		for value in columnValues:
			(set1, set2) = divideSet(rows, col, value)

			# Gain -- Entropy or Gini
			p = float(len(set1)) / len(rows)
			gain = current_score - (p*evaluationFunction(set1)) - ((1-p)*evaluationFunction(set2))
			if (gain > best_gain and len(set1) > 0 and len(set2) > 0):
				best_gain = gain
				best_attr = (col, value)
				best_sets = (set1, set2)

	if (best_gain > 0):
		true_branch = growTree(best_sets[0])
		false_branch = growTree(best_sets[1])
		return DecisionTree(col=best_attr[0], value=best_attr[1], true_branch=true_branch, false_branch=false_branch)
	else:
		return DecisionTree(results=uniqueCounts(rows))

def prune(tree, min_gain, evaluationFunction=entropy, notify=False):
    if (tree.true_branch.results == None): 
        prune(tree.true_branch, min_gain, evaluationFunction, notify)
    if (tree.false_branch.results == None): 
        prune(tree.false_branch, min_gain, evaluationFunction, notify)

    if (tree.true_branch.results != None and tree.false_branch.results != None):
        tb, fb = [], []
        for v, c in tree.true_branch.results.items(): tb += [[v]] * c
        for v, c in tree.false_branch.results.items(): fb += [[v]] * c

        p = float(len(tb)) / len(tb + fb)
        delta = evaluationFunction(tb+fb) - p*evaluationFunction(tb) - (1-p)*evaluationFunction(fb)
        if (delta < min_gain):	
            if notify: print('A branch was pruned: gain = %f' % delta, '\n')		
            tree.true_branch, tree.false_branch = None, None
            tree.results = uniqueCounts(tb + fb)

def classify(tree, observations, data_missing=False):
    
    def withData(observations, tree):
        if (tree.results != None):
            return tree.results
        else:
            v = observations[tree.col]
            branch = None
            if (isinstance(v, int) or isinstance(v, float)):
                if (v >= tree.value): branch = tree.true_branch
                else: branch = tree.false_branch
            else:
                if (v == tree.value): branch = tree.true_branch
                else: branch = tree.false_branch
        return withData(observations, branch)

    def withMissingData(observations, tree):
        if (tree.results != None):
            return tree.results
        else:
            v = observations[tree.col]
            if (v == None):
                tr = withMissingData(observations, tree.true_branch)
                fr = withMissingData(observations, tree.false_branch)
                tcount = sum(tr.values())
                fcount = sum(fr.values())
                tw = float(tcount)/(tcount + fcount)
                fw = float(fcount)/(tcount + fcount)
                result = collections.defaultdict(int)
                for k, v in tr.items(): result[k] += v*tw
                for k, v in fr.items(): result[k] += v*fw
                return dict(result)
            else:
                branch = None
                if (isinstance(v, int) or isinstance(v, float)):
                    if (v >= tree.value): branch = tree.true_branch
                    else: branch = tree.false_branch
                else:
                    if (v == tree.value): branch = tree.true_branch
                    else: branch = tree.false_branch
            return withMissingData(observations, branch)

    if data_missing:
        return classesText(withMissingData(observations, tree))
    else: 
        return classesText(withData(observations, tree))

def classesText(object_key, line=True):
    breakline = '\n' if line else ''
    return ''.join(['%s (%s) %s' % (key, object_key[key], breakline) for key in object_key])

def plotText(decisionTree):
    def toString(decisionTree, indent=''):
        if (decisionTree.results != None):
            return classesText(decisionTree.results, line=False)
        else:
            if (isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float)):
                decision = 'Attr %s: x >= %s?' % (decisionTree.col, decisionTree.value)
            else:
                decision = 'Attr %s: x == %s?' % (decisionTree.col, decisionTree.value)
            true_branch = indent + '|--- yes -> ' + toString(decisionTree.true_branch, indent + '\t\t')
            false_branch = indent + '|--- no  -> ' + toString(decisionTree.false_branch, indent + '\t\t')
            return (decision + '\n' + true_branch + '\n' + false_branch)

    print(toString(decisionTree), '\n')

def plotDiagram(decisionTree, extension=None):

    cl_bg = "#d8dff2"
    cl_root = "#f9bd77"
    cl_left = "#fefedf"
    cl_right = "#63a2d8"
    cl_leaf = "#4a6f46"
    cl_border = "#20202050"

    def plotNodes(decisionTree, dot, route, identifier=0, color=cl_root):
        if (decisionTree.results != None):
            dot.node(str(identifier), classesText(decisionTree.results), shape="egg", style="filled", color=cl_border, fillcolor=cl_leaf, fontcolor="white")
        else:
            decision = 'Attr %s: x >= %s ?' % (decisionTree.col, decisionTree.value)
            dot.node(str(identifier), decision, shape="box", style="filled", color=cl_border, fillcolor=color)

            leftID = identifier + 1 + time.time()
            rightID = identifier + 1001 + time.time()
            route.append([False, identifier, leftID])
            route.append([True, identifier, rightID])
            
            plotNodes(decisionTree.false_branch, dot, route, leftID, cl_left)
            plotNodes(decisionTree.true_branch, dot, route, rightID, cl_right)

    def plotEdges(dot, route):
        for x in range(len(route)):
            dot.edge(str(route[x][1]), str(route[x][2]), label=str(route[x][0]))

    dot = Digraph()
    dot.attr(bgcolor=cl_bg)
    dot.format = 'png'
    route = []

    plotNodes(decisionTree, dot, route)
    plotEdges(dot, route)

    name = Path().getNameResult("decision_tree_diagram", extension)
    dot.render(Path().getPathSave(name))