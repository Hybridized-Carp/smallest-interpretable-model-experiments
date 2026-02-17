from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
  
# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 

X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
#print(mushroom.metadata) 
  
# variable information 
#'print(mushroom.variables) 

X_hot = (pd.get_dummies(X,dtype=int))
y_hot = (pd.get_dummies(y))

print(X_hot.columns)
xhnpy = X_hot.to_numpy()
yhnpy = y_hot.to_numpy().T[1]

def npclsf(x):
    lol=np.where(np.all(xhnpy == x,axis=1))[-1][-1]
    return yhnpy[lol]

def ApplyTree(x,nodes):
    node=0
    while len(nodes[node]) >1:
        node= nodes[node][1+x[nodes[node][0]]]
    return bool(nodes[node][0])

def TreePath(x,nodes):
    node=0
    path=[0]
    while len(nodes[node]) >1:
        node= nodes[node][1+x[nodes[node][0]]]
        path.append(node)
    return (nodes[node][0],node,path)

def TreeSize(nodes,layer=0,pos=0):
    #return len(nodes)
    if len(nodes[pos]) == 1:
        return layer+1
    else:
        return max(
            TreeSize(nodes,layer+1,nodes[pos][1]),
            TreeSize(nodes,layer+1,nodes[pos][2])
        )
    
def FindChecks(nodes, start=0):
    if len(nodes[start]) ==1:
        return []
    return [nodes[start][0]] + FindChecks(nodes, nodes[start][1]) + FindChecks(nodes, nodes[start][2])

def CheckBeneathLayer(nodes, start, feature):
    if len(nodes[start]) ==1:
        return False
    if nodes[start][0]==feature:
        return True
    return CheckBeneathLayer(nodes, nodes[start][1], feature) or CheckBeneathLayer(nodes, nodes[start][1], feature)
    
def FindOptModelStr(classification_instance, size):
    for example,eout in classification_instance:
        if eout:
            model=[[1]]
            A1=[example]
            break
    
    res = FindOptExtStr(classification_instance, size, model, A1)
    if res != None:
        return res
    
    for example,eout in classification_instance:
        if not eout:
            model=[[0]]
            A2=[example]
            break

    return FindOptExtStr(classification_instance, size,model, A2)

def FindOptExtStr(classification_instance, size, model=None, annotations=None):
    model_pass=True
    for example, result in classification_instance:
        if ApplyTree(example, model) != result:
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
            for node in lpath:
                    if not CheckBeneathLayer(model,node,findex):
                        tmodel=model.copy()
                        nA = annotations.copy()
                        nc=len(model)
                        tmodel.append(tmodel[node])
                        nA.append(CAv)
                        tmodel.append([1-lv])
                        nA.append(example)
                        tmodel[node]=[findex,nc+1-example[findex],nc+example[findex]]
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
    
a = FindOptModelStr(list(zip(xhnpy,yhnpy)),10)


print(a)
print(X_hot.columns[19])
#if a != None:
#    print(dtmtodsm(a))

