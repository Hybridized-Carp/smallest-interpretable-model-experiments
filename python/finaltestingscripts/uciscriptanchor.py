from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import numba as nb

#def npclsf(x):
#    lol=np.where(np.all(xhnpy == x,axis=1))[-1][-1]
#    return yhnpy[lol]

@nb.njit(['bool_(int8[:,:],bool_[:])',
          'bool_(int16[:,:],bool_[:])',
          'bool_(int16[:,:],uint8[:])',
          'bool_(int64[:,:],uint8[:])'])
def ApplyTree(nodes: np.ndarray,example: np.ndarray):
    node=0
    while nodes[node][0] != -1:
        node= nodes[node][1+example[nodes[node][0]]]
    return nodes[node][1]


def BakeTree(nodes,features,size):
    bt=[]
    stack=[]
@nb.njit(['int8[:,:],bool_[:]',
          'int16[:,:],bool_[:]',
          'int32[:,:],bool_[:]',
          'int16[:,:],uint8[:]',
          'int64[:,:],uint8[:]'])
def TreePath(nodes: np.ndarray,example: np.ndarray):
    node=0
    path=[0]
    while nodes[node][0] != -1:
        node= nodes[node][1+example[nodes[node][0]]]
        path.append(node)
    return (nodes[node][1],node,np.array(path))

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


def FindOptModelStr(classification_instance, msize):
    print("finding opt model")
    initial_ma_pairs = []
    nfeatures=len(classification_instance[0][0])
    if nfeatures <= 127:
        array_dtype= np.int8
    elif nfeatures <= (2**15)-1:
        array_dtype= np.int16
    else:
        array_dtype=np.int32
    example_dtype=classification_instance[0][0].dtype
    for i in range(nfeatures):
        #nested loops to avoid any unneccessary checking
        for example1, eout1 in classification_instance:
            if not example1[i]:
                zi = eout1
                for example2, eout2 in classification_instance:
                    if example2[i]:
                        initial_model = np.full((msize,3),fill_value=-2,dtype=array_dtype)
                        initial_annotations = np.empty((msize,nfeatures),dtype=example_dtype)
                        initial_model[0] = [i,1,2]
                        initial_annotations[0] = example1
                        initial_model[1] = [-1,eout1,eout1]
                        initial_annotations[1] = example1
                        initial_model[2] = [-1,eout2,eout2]
                        initial_annotations[2] = example2
                        initial_ma_pairs.append((initial_model, initial_annotations))
                        break
                break
    
    #for example,eout in classification_instance:
    #    if eout:
    #        initial_ma_pairs.append(([[1]],[example]))
    #        break
    
    #for example,eout in classification_instance:
    #    if not eout:
    #        initial_ma_pairs.append(([[0]],[example]))
    #        break

    for model, annotations in initial_ma_pairs:
        res = FindOptExtStr(classification_instance, msize, 3, model, annotations)
        if res is not None:
            return res
    
    return None

@nb.jit
def FindOptExtStr(classification_instance, msize, csize, model: np.ndarray, annotations: np.ndarray):
    print(csize)
    model_pass=True 
    for example, result in classification_instance:
        if ApplyTree(model,example) != result:
            model_pass=False
            break
    if model_pass:
        return model
    
    if model[-2][0] !=-2:
        return None
    
    strict_exts = FindStrictExtStr(model, annotations, example)
    B = None
    for nModel, nAnnotations in strict_exts:
        A = FindOptExtStr(classification_instance, msize, csize+1, nModel, nAnnotations)
        if A is not None:
            if (B is None) or (np.argmax(B[:,0]==-2) > np.argmax(A[:,0]==-2)):
                B=A
    return B     

@nb.njit(['int8[:,:],bool_[:,:],bool_[:]',
          'int16[:,:],bool_[:,:],bool_[:]',
          'int32[:,:],bool_[:,:],bool_[:]',
          'int16[:,:],uint8[:,:],uint8[:]',
          'int64[:,:],uint8[:,:],uint8[:]'])
def FindStrictExtStr(model: np.ndarray, annotations: np.ndarray, example: np.ndarray):
    extensions=[]
    lv, ln, lpath=TreePath(model,example)
    CAv= annotations[ln]
    hmd=(example^CAv)
    cur_checks=[model[x][0] for x in lpath[:-1]]
    for findex, feature in enumerate(hmd):
        if feature:
            if not feature in model[:,0][lpath]:
                nModel=model.copy()
                nAnnotations=annotations.copy()
                nc=np.argmax(model[:,0]==-2)
                nModel[nc]=(nModel[ln])
                nAnnotations[nc]=(annotations[ln])
                nModel[nc+1]=([-1,1-lv,1-lv])
                nAnnotations[nc+1]=example
                nModel[ln]=[findex,nc+1-example[findex],nc+example[findex]]
                nAnnotations[ln]=example
                extensions.append((nModel,nAnnotations))
                        
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

rt = True
if (rt):
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
    
    for i in X_hot.columns:
        print(i)
    xhnpy = X_hot.to_numpy()
    yhnpy = y_hot.to_numpy().T[1]
    #print(xhnpy)
    a = FindOptModelStr(list(zip(xhnpy,yhnpy)),2**4)
else:
    tdata=np.arange(256,dtype=np.uint8)
    tout=list(map(clsf, tdata))
    tdata=np.unpackbits(tdata.reshape(1,256).T,axis=1).astype(np.bool_)
    print(tdata)
    a = FindOptModelStr(list(zip(tdata,tout)),7)

print(a)
#print(X_hot.columns[19])
#if a != None:
#    print(dtmtodsm(a))

