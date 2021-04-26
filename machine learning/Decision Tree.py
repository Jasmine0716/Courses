# This is a project of the class on machine learning
# created by Jasmine, Fu
# 26th, April, 2021

import numpy as np
import pandas as pd


def trainAndTest(fp, type):
    '''
    Args:
        - fp: String, data path
        - type: 'Gain'/'GainRatio'/'Gini'
    Return:
        None
    '''
    df, headers, labels = read_dat(fp)
    n = df.shape[0]
    train = int(0.8*n)
    df_train = df.iloc[0:train]
    df_test = df.iloc[train:]
    tree = buildTree(df_train,headers[0:-1],labels,headers[-1], type)
    # print(df)
    drawTree(tree, 'Tree'+type+'.dot')    
    right, wrong = testAccurancy(df_test, tree)
    print('Accurancy on test set: %f using %s'%(float(right)/float(right+wrong),type))

def MultiTree(fp, type, m):
    '''
    Args:
        - fp: String, data path
        - type: 'Gain'/'GainRatio'/'Gini'
        - m: int, the number of decision tree to train
    Return:
        None
    '''
    df, headers, labels = read_dat(fp)
    n = df.shape[0]
    train = int(0.5*n)
    test = int(0.2*n)
    df_test = df.iloc[np.random.randint(0,n,test)]
    dfs = {}
    trees = {}
    for i in range(m):
        data = df.iloc[np.random.randint(0,n,train)]
        trees['tree'+str(i)] = buildTree(data,headers[0:-1],labels,headers[-1], type)
    right, wrong = MultiTreeTest(trees, df_test)
    print('Accurancy on test set: %f while m = %d'%(float(right)/float(right+wrong),m))

def MultiTreeTest(trees, df):
    '''
    Args:
        - trees: dictionary, contains multiple decision trees
        - df: DataFrame, testing set
    Return:
        - total_right, total_wrong: int, the number of right/wrong labeled examples
    '''
    total_right = 0
    total_wrong = 0
    for i in range(df.shape[0]):
        vote = {'right':0,'wrong':0}
        for key in trees.keys():
            tree = trees[key]
            n = judge(df.iloc[i], tree)
            vote['right'] += n
            vote['wrong'] += (1-n)
        if vote['right']>=vote['wrong']:
            total_right += 1
        else:
            total_wrong += 1
    return total_right, total_wrong

def judge(df,tree):
    '''
    This function is used to judge one data example based on one decision tree
    Args:
        - df: DataFrame, the one example
        - tree: dictionary, the one decision tree
    '''
    n = 0
    for key in tree.keys():
        attr, attr_val = key
        subtree = tree[key]
        if subtree == 'Leaf':
            return int(df[attr]==attr_val)
        if df[attr] != attr_val:
            continue
        n = judge(df, subtree)
    return n

def testAccurancy(df, tree):
    '''
    using the trained tree to test the accurancy of the testing dataset
    '''
    right = 0
    wrong = 0
    for key in tree.keys():
        attr, attr_val = key
        subTree = tree[key]
        # print(subTree)
        if subTree == 'Leaf':
            sub_right = np.sum(df[attr]==attr_val)
            sub_wrong = df.shape[0]-sub_right
            return sub_right,sub_wrong
        idx = np.where(df[attr]==attr_val)
        sub_df = df.iloc[idx]
        sub_right, sub_wrong = testAccurancy(sub_df,subTree)
        right += sub_right
        wrong += sub_wrong
    return right, wrong

def buildTree(df, attrs, labels, target_attr, type):
    '''
    Args:
        - idx: the indexes of data which are used to build a tree 
        - df: the whole dataset
        - attrs: the attributes that have not been used yet
    Returns:
        -  sub_tree: a dictionary of the subtree
    '''
    if len(np.unique(df[target_attr].values))==1:
        return {(target_attr,df[target_attr].values[0]):'Leaf'}
    if attrs == []:
        return {(target_attr,findMostLabel(df, target_attr)):'Leaf'}
    attr_to_split = chooseBestAttr(df, attrs, labels, target_attr, type)
    attr_labels = labels[attr_to_split]
    attrs.remove(attr_to_split)
    attrs_new = attrs.copy()
    node = {}
    for label in attr_labels:
        idx = np.where(df[attr_to_split]==label)
        subTree = buildTree(df.iloc[idx], attrs_new, labels, target_attr, type)
        s = (attr_to_split, label)
        node[s] = subTree
    return node

def findMostLabel(df, target_attr):
    '''
    find out the catagory that happens most frequently in the dataset
    '''
    labels = np.array(df[target_attr])
    uni_labels = np.unique(labels)
    max_count = 0
    max_label = None
    for label in uni_labels:
        count = np.sum(labels == label)
        if count>max_count:
            max_count = count
            max_label = label
    return max_label

def read_dat(fp):
    '''
    This funtion return the data frame of .dat file
    Args:
        - fp: the path of .dat file
    Return:
        - df: DataFrame, the corresponding dataFrame of the file
        - headers: list, the list of attribute names
        - labels: dictionary, {'attribute': attribute value}
    '''
    f = open(fp)
    lines = f.readlines()
    data_n = 0
    headers = []
    data = []
    labels = {}
    for i in range(len(lines)):
        line = lines[i]
        if '@attribute' in line:
            s = line.split(' ')
            headers.append(s[1])
        if '@data' in line:
            data_n = i+1
            break
    for i in range(data_n,len(lines)):
        line = lines[i][0:-1]
        num_s = line.split(',')
        num_data = []
        for s in num_s:
            num_data.append(s)
        data.append(np.array(num_data))
    data = np.array(data)
    df = pd.DataFrame(data, columns = np.array(headers))
    for header in headers:
        types = np.unique(df[header])
        types = np.delete(types, np.where(types=='?'))
        labels[header] = types
    return df, headers, labels

def compute_Ent(dataSet):
    '''
    This function compute the Shannon Ent for the target dataSet
    Args
        - dataSet: numpy array, the values of the target attribute that are used to compute Ent
    Return
        - ent: float
    '''
    count = {}
    for label in dataSet:
        if label not in count.keys():
            count[label]=0
        count[label] += 1
    ent = 0
    for k in count:
        p = float(count[k])/float(len(dataSet))
        ent -= p*np.log2(p)
    return ent

def compute_Gini(dataSet):
    '''
    Arg
        - dataSet: numpy array, the values of the target attribute that are used to compute Gini index
    Return
        - gini: float
    '''
    count = {}
    for label in dataSet:
        if label not in count.keys():
            count[label]=0
        count[label] += 1
    gini = 1
    for k in count:
        p = float(count[k])/float(len(dataSet))
        gini -= p**2
    return gini

def chooseBestAttr(df, attrs, labels, target_attr, type):
    '''
    Choose the best attribute to split which results in the best performance
    Args:
        - df: DataFrame, data to be labled
        - attrs: list, attributes that have not been used yet
        - target_attr: String, the attribute to predict
        - type: 'Gain'/'GainRatio'/'Gini'
    return:
        - attr: string
    '''
    ent_original = compute_Ent(df[target_attr])
    gini_original = compute_Gini(df[target_attr])
    scores = []
    n_all = np.size(df[target_attr])
    for attr in attrs:
        ent_attr = 0
        gini_attr = 0
        split_info = 0
        attr_labels = labels[attr]
        for label in attr_labels:
            idx = np.where(df[attr]==label)
            n = np.sum(df[attr]==label)
            data = df.iloc[idx]
            ent_attr += (n/n_all)*compute_Ent(data[target_attr])
            gini_attr += (n/n_all)*compute_Gini(data[target_attr])
            split_info -= (float(n)/float(n_all))*np.log2((float(n)/float(n_all)))
        if type == 'Gain':
            scores.append(ent_original-ent_attr)
        elif type == 'GainRatio':
            scores.append((ent_original-ent_attr)/split_info)
        elif type == 'Gini':
            scores.append(gini_original-gini_attr)
    scores = np.array(scores)
    idx = np.argmax(scores)
    return attrs[idx]

def drawTree(tree, fp):
    '''
    Draw the decision tree by dot language
    Args:
        - tree: dictionary, the decision tree to be drawn
        - fp: String, the path where the file will be stored
    '''
    f = open(fp,'w')
    f.write('digraph g {\n\t\t'+'node0'+'[label=\"DecisionTree\"];\n')
    s,_ = drawNode(tree, 0)
    f.write(s)
    f.write('}')
    f.close()

def drawNode(subTree, i):
    '''
    Draw detail content
    '''
    n = i
    s = ''
    for key in subTree.keys():
        n = n+1
        attr, attr_val = key
        attr_val=str(attr_val)
        s = s+'\t\tnode'+str(n)+'[label="'+attr+': '+attr_val+'"];\n'
        s = s+'\tnode'+str(i)+'->'+'node'+str(n)+';\n'
        if(subTree[key]=='Leaf'):
            return s, n
        s1, n1 = drawNode(subTree[key],n)
        s = s+s1
        n = n1
    return s, n

# trainAndTest('data/test.txt','Gain')
# trainAndTest('data/test.txt','GainRatio')
# trainAndTest('data/test.txt','Gini')
MultiTree('data/titanic.dat','Gain',3)
MultiTree('data/titanic.dat','Gain',5)
MultiTree('data/titanic.dat','Gain',7)
MultiTree('data/titanic.dat','Gain',8)
MultiTree('data/titanic.dat','Gain',9)
MultiTree('data/titanic.dat','Gain',11)