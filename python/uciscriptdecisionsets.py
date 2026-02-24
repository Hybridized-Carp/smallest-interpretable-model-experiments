from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import numba as nb

#def npclsf(x):
#    lol=np.where(np.all(xhnpy == x,axis=1))[-1][-1]
#    return yhnpy[lol]

def ApplyDecisionSet(model,x):
    for term in model[0]:
        if(x&term[0])== term[1]:
            return 1-model[1]
    return model[1]

def FindOptModelStr(classification_instance, size):
    for example, result in classification_instance:
        if result == 1:
            A1={'default':example}
            break
    
    res = FindOptExtStr(classification_instance, size, [[],1], A1)
    if res != None:
        return res
    
    for example,result in classification_instance:
        if classification_instance[1](example) == 0:
            A2={'default':example}
            break

    return FindOptExtStr(classification_instance, size,[[],0], A2)

def FindOptExtStr(classification_instance, size, model=None, annotations=None):
    if model != None:
        model_pass=True
        for example,result in classification_instance:
            if ApplyDecisionSet(model,example) != result:
                model_pass=False
                break
        if model_pass:
            return model
        #two methods of size
        #if len(model[0])>=size:
        if sum(map(lambda a: a[0].bit_count(), model[0])) >= size:
            #print("death")
            return None
    else:
        example = None
    strict_exts = FindStrictExtStr(classification_instance, size, model, annotations, example, result)
    B = None
    for strict_ext in strict_exts:
        nModel, nAnnotations = strict_ext
        if sum(map(lambda a: a[0].bit_count(), nModel[0])) <= size:
            A = FindOptExtStr(classification_instance, size, nModel, nAnnotations)
            if A != None:
                if B == None:
                    B=A
                elif sum(map(lambda a: a[0].bit_count(), B[0])) >sum(map(lambda a: a[0].bit_count(), A[0])):
                    B=A
    return B
                 

def FindStrictExtStr(classification_instance, size, model, annotations, example , expres):
    #print("!")
    extensions=[]
    if model[1] != expres:
        ex2 = annotations['default']
        hmd = (ex2^example)
        t=1
        while hmd >0:
            if (hmd%2 == 1):
                nA = annotations.copy()
                nA[(t,example&t)]=example
                tmodel=model.copy()
                tmodel[0].append((t,example&t))
                extensions.append([tmodel,nA])
            t<<=1
            hmd>>=1
        return extensions
    #if multiple terms fuck us up how do we know that we chose the right one :(
    term=-1
    for t in model[0]:
        if (example&t[0]) == t[1]:
            term=t
    #if term==-1:
    #    print("poison")
    ex2 = annotations[term]
    hmd = (ex2^example)
    t=1
    while hmd >0:
        if (hmd%2 == 1):
            nA = annotations.copy()
            tmodel=model.copy()
            bitmask=t|term[0]
            for i in range(len(tmodel[0])):
                if tmodel[0][i] == term:
                    #temp=(bitmask,(ex2^example)&bitmask)
                    #HOPE THIS IS RITGHT
                    nA[(bitmask,term[1])] = nA[term]
                    del nA[term]
                    tmodel[0][i]=(bitmask,term[1])
            extensions.append([tmodel,nA])
        t<<=1
        hmd>>=1
    return extensions

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
    
    #print(X_hot.columns)
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

