from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import cupy as cp
import numba as nb
import bitarray as ba
from bitarray import bitarray, frozenbitarray
from bitarray import util as bautil
import datetime
from copium import deepcopy
import time
import random
#def npclsf(x):
#    lol=np.where(np.all(xhnpy == x,axis=1))[-1][-1]
#    return yhnpy[lol]

def concatfba(fbaset):
    r=bitarray()
    for i in fbaset:
        r+=i
    return r

#define our helper functions before we doing anything with the actual algo
def mapnparraytofrozenbitarray(nparry):
    nl=[]
    bv=ba.bitarray()
    for i in nparry:
        bv.pack(i.tobytes())
        nl.append(ba.frozenbitarray(bv))
        bv.clear()
    return nl


#can definitely clean this up
def isolatesetbitsfrombaasfba(orgba):
    baset=[]
    tempba=ba.bitarray(len(orgba))
    for i in orgba.search(1):
        tempba[i]=1
        baset.append(ba.frozenbitarray(tempba))
        tempba[i]=0
    return baset

def dsmsize(model):
    bitsperterm=map(lambda a: a[0].count(1), model[0])
    return sum(bitsperterm)

def ApplyDecisionSet(model,x):
    for term in model[0]:
        if(x&term[0])== term[1]:
            return 1-model[1]
    return model[1]

def FindLargestLitInModel(model):
    max = 0
    if len(model[0]) > 0:
        for term in model[0]:
            *_, last = term[0].search(1)
            if last > max:
                max=last
    else:
        # important otherwise we can't create any initial rules
        max = 4294967296
    return max

def FindOptModelStr(classification_instance, size):
    s=size
    for example, result in classification_instance:
        if result == 0:
            A1={'default':example}
            break
    
    t1=time.time()
    print(t1)
    res = FindOptExtStr(classification_instance, s, [[],0], A1)
    t2=time.time()
    print(t2)
    print(t2-t1)
    if res != None:
        s=dsmsize(res)-1
    for example,result in classification_instance:
        if result==1:
            A2={'default':example}
            break
    m=FindOptExtStr(classification_instance, s,[[],1], A2)
    if m != None:
        if (res == None) or (dsmsize(res) > dsmsize(m)):
            res=m
    t3=time.time()
    print(t3)
    print(t3-t2)
    print(t3-t1)
    print(datetime.timedelta(seconds=(t3-t1)))
    print(datetime.timedelta(seconds=(t3-t2)))
    print(datetime.timedelta(seconds=(t2-t1)))
    return res

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
        if dsmsize(model) >= size:
            #print("death")
            return None
    else:
        example = None
    strict_exts = FindStrictExtStr(classification_instance, size, model, annotations, example, result)
    B = None
    for strict_ext in strict_exts:
        nModel, nAnnotations = strict_ext
        if dsmsize(nModel) <= size:
            A = FindOptExtStr(classification_instance, size, nModel, nAnnotations)
            if A != None:
                if B == None or dsmsize(B) > dsmsize(A):
                    B=A
                    size=dsmsize(B)
    return B

def itentropy(classification_instance, fbaset):
    key=dict()
    ycnpy = cp.array(list(map(lambda a: a[1], classification_instance)))
                 

def FindStrictExtStr(classification_instance, size, model, annotations, example , expres):
    #print("!")
    extensions=[]
    if model[1] != expres:
        ex2 = annotations['default']
        hmd = (ex2^example)
        ifbaset=isolatesetbitsfrombaasfba(hmd)
        key=dict()
        for fba in ifbaset:
            fba.find(1)
        for fba in ifbaset:
            nA = annotations.copy()
            nA[(fba,(fba&example))]=example
            tmodel=deepcopy(model)
            tmodel[0].append((fba,(fba&example)))
            extensions.append([tmodel,nA])  
        return extensions
    #if multiple terms fuck us up how do we know that we chose the right one :(
    t=-1
    tindex=-1
    #currently we go for the last term that is incorrect first
    for index, term in enumerate(model[0]):
        if (example&term[0]) == term[1]:
            t=term
            tindex=index
    #if term==-1:
    #    print("poison")
    ex2 = annotations[t]
    hmd = (ex2^example)
    ifbaset=isolatesetbitsfrombaasfba(hmd) 
    for fba in ifbaset:
        if  fba.find(1) < t[0].find(1):
            nA = annotations.copy()
            tmodel = deepcopy(model)
            featuremask=fba|t[0]
            value=(fba&ex2)|t[1]
            nA[(featuremask,value)] = nA[t]
            del nA[t]
            tmodel[0][tindex]=(featuremask,value)
            extensions.append([tmodel,nA])
    t=1
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
    

def baclsf(x):
    if (x[0]) == 0:
        return (x,1)
    elif (x[2]) != 0:
        return (x,0)
    elif (x[1]) != 0:
        return (x,1)
    else:
        return (x,0)

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

rt = False
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
    bita=mapnparraytofrozenbitarray(xhnpy)
    a = FindOptModelStr(list(zip(bita,yhnpy)),5)
else:
    random.seed("synthetic")
    # mushroom data set stats:
    # 8124 examples
    # median number of 22 bits set
    # 116 bits
    if random.random() != 0.19566309432916196:
        print("psrng is not initialised as expected")
        exit()
    fbaset=[]
    for i in range(8124):
        fbaset.append(ba.frozenbitarray(bautil.random_k(116,22)))
    fbasdci=list(map(baclsf,fbaset))
    #a = FindOptModelStr(fbasdci,6)

def dbadmstoreadable(x):
    print(f"Default rule: {bool(x[1])}")
    for fm, v in x[0]:
        fi=fm.search(1)
        print(f"{*[f"x_{i}={v[i]}" for i in fi],}")
#print(a)
#dbadmstoreadable(a)
#print(X_hot.columns[19])
#if a != None:
#    print(dtmtodsm(a))

