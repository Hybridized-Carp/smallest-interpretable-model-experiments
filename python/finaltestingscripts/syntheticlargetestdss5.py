from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
#import numba as nb
import bitarray as ba
from bitarray import bitarray, frozenbitarray
from bitarray import util as bautil
from copium import deepcopy
import time
import random
import numpy as np
import cProfile, sys
import os
try:
    import mlx.core as mx
    mlx_present = True
except ImportError:
    mlx_present = False
#fuck mlx, no reduce primitive
#im going back to cuda 
mlx_present = False
#def npclsf(x):
#    lol=np.where(np.all(xhnpy == x,axis=1))[-1][-1]
#    return yhnpy[lol]

def dbadmstoreadable(x):
    print(f"Default rule: {bool(x[1])}")
    for fm, v in x[0]:
        fi=fm.search(1)
        print(f"{*[f"x_{i}={v[i]}" for i in fi],}")

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

def bitmask_term_to_numpy_term(term):
    np_term = [[],[]]
    for i in term[0].search(1):
        if term[1][i]:
            np_term[1].append(i)
        else:
            np_term[0].append(i)
    return np_term

def evaluate_numpy_term(t_examples, term):
    z_res = np.bitwise_or.reduce(t_examples[term[0]])
    o_res = np.bitwise_and.reduce(t_examples[term[1]])
    return o_res & ~z_res

#can't be fucked to think of a good name for this rn
#but its basically a helper function where we provide in the base model results
#the base ci results
#the new models and it generates a partial function that returns the score
#we then use that to sort our results based on the score
def party_function(t_examples,t_results, m_results):
    def scorer(model):
        new_bm_term=model[0][0][-1]
        np_term = bitmask_term_to_numpy_term(new_bm_term)
        term_res = evaluate_numpy_term(t_examples, np_term) 
        m_res = m_results | ( ~term_res if model[1] else term_res)
        return int (np.sum(np.bitwise_count(m_results^t_results)))
    return scorer

        


def dsmsize(model):
    bitsperterm=map(lambda a: a[0].count(1), model[0])
    return sum(bitsperterm)

def ApplyDecisionSet(model,x):
    for term in model[0]:
        if(x&term[0])== term[1]:
            return 1-model[1]
    return model[1]

def FindOptModelStr(classification_instance, size, use_heuristics=True):
    if use_heuristics:
        #pad classification instance to 8 bits
        padding=(8-(len(classification_instance)%8))%8
        for i in range(padding):
            classification_instance.append(classification_instance[-1])

        #put examples into temp numpy array
        ex_count = len(classification_instance)
        f_count = len(classification_instance[0][0])
        tnp=np.ndarray(shape=(ex_count,f_count),dtype=bool)
        result_tnp=np.ndarray(shape=ex_count, dtype=bool)
        for i, (example, result) in enumerate(classification_instance):
            tnp[i] = np.frombuffer(example.unpack(), dtype=bool)
            result_tnp[i] = result
        if mlx_present:
            transposed_packed_examples = mx.array(np.packbits(tnp.T, axis=1))
            transposed_packed_results = mx.array(np.packbits(result_tnp.reshape((-1,1)),axis=0))
        else:
            transposed_packed_examples = np.packbits(tnp.T, axis=1)
            transposed_packed_results = np.packbits(result_tnp.reshape((-1,1)),axis=0)
    else:
        transposed_packed_examples=None
        transposed_packed_results=None
    s=size
    for example, result in classification_instance:
        if result == 0:
            A1={'default':example}
            break
    
    t1=time.time()
    #print(t1)
    res = FindOptExtStr(classification_instance, s, [[],0], A1, use_heuristics, transposed_packed_examples, transposed_packed_results)
    t2=time.time()
    #print(t2)
    print(t2-t1)
    if res != None:
        s=dsmsize(res)
    for example,result in classification_instance:
        if result==1:
            A2={'default':example}
            break
    m=FindOptExtStr(classification_instance, s,[[],1], A2, use_heuristics, transposed_packed_examples, transposed_packed_results)
    if m != None:
        if (res == None) or (dsmsize(res) > dsmsize(m)):
            res=m
    t3=time.time()
    #print(t3)
    print(t3-t2)
    print(t3-t1)
    return res

def FindOptExtStr(
        classification_instance, 
        size, 
        model = None, 
        annotations = None, 
        use_heuristics = True, 
        t_examples = None, 
        t_results = None,
        cur_size = 0):
    #print(cur_size)
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
        if cur_size >= size:
            #print("death")
            return None
    else:
        example = None
    strict_exts = FindStrictExtStr(classification_instance, size, model, annotations, example, result, use_heuristics, t_examples, t_results)
    B = None
    for strict_ext in strict_exts:
        nModel, nAnnotations = strict_ext
        if cur_size < size:
            A = FindOptExtStr(classification_instance, size, nModel, nAnnotations, use_heuristics, t_examples, t_results, cur_size=cur_size+1)
            if A != None:
                if B == None:
                    B=A
                elif dsmsize(B) > dsmsize(A):
                    B=A
    return B
                 

def FindStrictExtStr(
        classification_instance, 
        size, 
        model, 
        annotations,
        example, 
        expres, 
        use_heuristics = True,
        t_examples = None, 
        t_results = None):
    #print("!")
    extensions=[]
    if model[1] != expres:
        ex2 = annotations['default']
        hmd = (ex2^example)
        ifbaset=isolatesetbitsfrombaasfba(hmd)
        if use_heuristics:
            #calculate results for current model
            m_res = np.zeros(dtype=np.uint8, shape=t_results.shape)
            for term in model[0]:
                np_term = bitmask_term_to_numpy_term(term)
                term_res = evaluate_numpy_term(t_examples,np_term) 
                m_res = np.bitwise_or(m_res,(~term_res if model[1] else term_res))
        for fba in ifbaset:
            nA = annotations.copy()
            new_term=(fba,(fba&example))
            nA[new_term]=example
            tmodel=deepcopy(model)
            tmodel[0].append(new_term)
            extensions.append([tmodel,nA])
        if use_heuristics:
            sorted(extensions, key=party_function(t_examples, t_results, m_res))
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
    if (x[32]) == 0:
        return (x,1)
    elif (x[4]) != 0:
        return (x,0)
    elif (x[16]) != 0:
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
    a = FindOptModelStr(list(zip(bita,yhnpy)),6)
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
    ex_count=2048
    f_count=48
    padding=(8-(ex_count%8))%8
    tnp=np.ndarray(shape=(ex_count+padding,f_count),dtype=bool)
    result_tnp=np.ndarray(shape=ex_count+padding, dtype=bool)
    fbasdci=[]
    for i in range(ex_count):
        temp_fba=ba.frozenbitarray(bautil.random_k(f_count,12))
        fbasdci.append(baclsf(temp_fba))
    pr = cProfile.Profile()
    pr.enable()
    a = FindOptModelStr(fbasdci,5,True)
    pr.disable()
    # - for text dump
    print(os.curdir)
    with open(f'cpu_{time.time()}.txt', 'w') as output_file:
        sys.stdout = output_file
        pr.print_stats( sort='time' )
        sys.stdout = sys.__stdout__


print(a)
#dbadmstoreadable(a)
#print(X_hot.columns[19])
#if a != None:
#    print(dtmtodsm(a))

