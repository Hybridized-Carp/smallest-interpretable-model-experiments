from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import numba as nb

#def npclsf(x):
#    lol=np.where(np.all(xhnpy == x,axis=1))[-1][-1]
#    return yhnpy[lol]

@nb.njit(['(bool_[:],int16[:,:])',
          '(uint8[:],int16[:,:])',
          '(uint8[:],int64[:,:])'])
def ApplyTree(x,nodes):
    node=0
    while nodes[node][0] != -1:
        node= nodes[node][1+x[nodes[node][0]]]
    return nodes[node][1]


def BakeTree(nodes,features,size):
    bt=[]
    stack=[]

def TreePath(x,nodes):
    node=0
    path=[0]
    while nodes[node][0] != -1:
        node= nodes[node][1+x[nodes[node][0]]]
        path.append(node)
    return (nodes[node][1],node,path)

def TreeSize(nodes,layer=0,pos=0):
    return len(nodes)
    if len(nodes[pos]) == 1:
        return layer+1
    else:
        return max(
            TreeSize(nodes,layer+1,nodes[pos][1]),
            TreeSize(nodes,layer+1,nodes[pos][2])
        )
    
def FindChecks(nodes, start=0):
    if nodes[start][0] == -1:
        return []
    return [nodes[start][0]] + FindChecks(nodes, nodes[start][1]) + FindChecks(nodes, nodes[start][2])

def CheckBeneathLayer(nodes, start, feature):
    if nodes[start][0] == -1:
        return False
    if nodes[start][0]==feature:
        return True
    return CheckBeneathLayer(nodes, nodes[start][1], feature) or CheckBeneathLayer(nodes, nodes[start][1], feature)

def CheckPath(nodes, path, feature):
    for i in path:
        if nodes[i][0]==feature:
            return True
    return False
    
def FindOptModelStr(classification_instance, size):
    initial_ma_pairs =[]

    nfeatures=len(classification_instance[0][0])
    for i in range(nfeatures):
        #feature 0 mapping
        zi = -1
        oi = -1
        annotations = [0]
        for example, eout in classification_instance:
            if not example[i]:
                zi = eout
                annotations.append(example)
                break
        for example, eout in classification_instance:
            if example[i]:
                oi = eout
                annotations.append(example)
                break
        if (zi != -1) and (oi != -1):
            initial_ma_pairs.append(([[i, 1,2], [-1,zi,zi],[-1,oi,oi]],annotations))

    #for example,eout in classification_instance:
    #    if eout:
    #        initial_ma_pairs.append(([[1]],[example]))
    #        break
    
    #for example,eout in classification_instance:
    #    if not eout:
    #        initial_ma_pairs.append(([[0]],[example]))
    #        break

    for model, annotations in initial_ma_pairs:
        res = FindOptExtStr(classification_instance, size, model, annotations)
        if res != None:
            return res
    
    return None

def FindOptExtStr(classification_instance, size, model=None, annotations=None):
    model_pass=True 
    for example, result in classification_instance:
        if ApplyTree(example, np.array(model,dtype=np.int16)) != result:
            model_pass=False
            break
    if model_pass:
        return model
    #two methods of size
    #if len(model[0])>=size:
    if TreeSize(model) >= size:
        #print("death")
        return None
    strict_exts = FindStrictExtStr(model, annotations, example)
    B = None
    for nModel, nAnnotations in strict_exts:
        if TreeSize(nModel) <= size:
            A = FindOptExtStr(classification_instance, size, nModel, nAnnotations)
            if A != None:
                if B == None or TreeSize(B) > TreeSize(A):
                    B=A
    return B     

def FindStrictExtStr(model, annotations, example):
    #print("!")
    extensions=[]
    lv,ln,lpath=TreePath(example,model)
    CAv= annotations[ln]
    hmd=(example^CAv)
    cur_checks=[model[x][0] for x in lpath[:-1]]
    for findex, feature in enumerate(hmd):
        if feature:
            if not CheckPath(model,lpath, findex):
                tmodel=model.copy()
                nA = annotations.copy()
                nc=len(model)
                tmodel.append(tmodel[ln])
                nA.append(CAv)
                tmodel.append([-1,1-lv,1-lv])
                nA.append(example)
                tmodel[ln]=[findex,nc+1-example[findex],nc+example[findex]]
                extensions.append([tmodel,nA])
                        
    return extensions

ddata=list(range(256))

#classification function based on decision list
def clsf(x):
    if (x & 8) == 0:
        return 1
    elif (x&4) != 0:
        return 0
    elif (x&2) != 0:
        return 1
    else:
        return 0

def dtmtodsmterms(nodes,start=0,fm=0,v=0):
    cnode=nodes[start]
    if len(cnode) == 1:
        return [(fm,v,cnode[0])]
    fm |= cnode[0]
    terms=[]
    terms+= dtmtodsmterms(nodes,cnode[1],fm,v)
    terms+= dtmtodsmterms(nodes,cnode[2],fm,v|cnode[0])
    return terms
    
def dtmtodsm(nodes):
    zm=[[],0]
    om=[[],1]
    terms = dtmtodsmterms(nodes)
    #could prob use list comprehension
    for i in terms:
        if i[2]:
            zm[0].append((i[0],i[1]))
        else:
            om[0].append((i[0],i[1]))
    return (zm,om)

# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 

X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
#print(mushroom.metadata) 
  
# variable information 
#'print(mushroom.variables) 

X_hot = (pd.get_dummies(X))
y_hot = (pd.get_dummies(y))
 
print(X_hot.columns)
xhnpy = X_hot.to_numpy()
yhnpy = y_hot.to_numpy().T[1]
print(xhnpy)

tdata=np.arange(256,dtype=np.uint8)
tout=list(map(clsf, tdata))
tdata=np.unpackbits(tdata.reshape(1,256).T,axis=1)
print(tdata)
a = FindOptModelStr(list(zip(xhnpy,yhnpy)),2**4)
#a = FindOptModelStr(list(zip(tdata,tout)),7)

print(a)
print(X_hot.columns[19])
#if a != None:
#    print(dtmtodsm(a))

