
#class classification_instance

#manual implementation of decision list
def classification_rule(x):
    if (x & 8) == 0:
        return (x,1)
    elif (x&4) != 0:
        return (x,0)
    elif (x&2) != 0:
        return (x,1)
    else:
        return (x,0)

#manual implementation of the decision set
def dcset(x):
    r=[[(8,0),(14,10)],0]
    for i in r[0]:
        if(x&i[0])== i[1]:
            return (x,1-r[1])
    return (x,r[1])

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
    
#decision tree based classification function
def tclsf(x):
    nodes=[
        None,
        None,
        [0b1000,1,3],
        [0b0100,4,0],
        [0b0010,0,1],
    ]
    node=2
    while node > 1:
        node= nodes[node][2] if x&(nodes[node][0]) else nodes[node][1]
    return node

def tclsfr(x):
    return (x,tclsf(x))

def dtmtodsmterms(nodes,start=2,fm=0,v=0):
    cnode=nodes[start]
    fm |= cnode[0]
    terms=[]
    if cnode[1] == 0:
        terms.append((fm,v,0))
    elif cnode[1] == 1:
        terms.append((fm,v,1))
    else:
        terms+= dtmtodsmterms(nodes,cnode[1],fm,v)
    
    if cnode[2] == 0:
        terms.append((fm,v|cnode[0],0))
    elif cnode[2] == 1:
        terms.append((fm,v|cnode[0],1))
    else:
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

def ApplyDecisionSet(x, model):
    for term in model[0]:
        if(x&term[0])== term[1]:
            return 1-model[1]
    return model[1]

# is size the number of terms?
# or is size the sum of the number of literals in each term
# because for binary features unless you have more than 64 features for your examples
# (maybe more with funky vector instructions)
# the amnt of time spent checking features will be less than the jumps
# might be an over fitting thing tho?
def FindOptModelStr(classification_instance, size):
    for example in classification_instance[0]:
        if classification_instance[1](example) == 1:
            A1={'default':example}
            break
    
    res = FindOptExtStr(classification_instance, size, [[],1], A1)
    if res != None:
        return res
    
    for example in classification_instance[0]:
        if classification_instance[1](example) == 0:
            A2={'default':example}
            break

    return FindOptExtStr(classification_instance, size,[[],0], A2)

def FindOptExtStr(classification_instance, size, model=None, annotations=None):
    if model != None:
        model_pass=True
        for example in classification_instance[0]:
            if ApplyDecisionSet(example, model) != classification_instance[1](example):
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
    strict_exts = FindStrictExtStr(classification_instance, size, model, annotations, example)
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
                 

def FindStrictExtStr(classification_instance, size, model, annotations, example):
    #print("!")

    extensions=[]
    if model[1] != classification_instance[1](example):
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
        


nodes=[
        None,
        None,
        [0b1000,1,3],
        [0b0100,4,0],
        [0b0010,0,1],
    ]
sad=dtmtodsm(nodes)
print(sad)

#def FindStrictExtStr(C,e):
ddata=list(range(256))
cdata=list(map(classification_rule, ddata))
dsdata=list(map(tclsfr, ddata))
nodes=[]
nodesl=[]
nodesr=[]
nodesrule=[]
annotation=[]

a = FindOptModelStr((ddata,tclsf),7)
print(a)
a = FindOptModelStr((ddata,clsf),7)
print(a)

for i in ddata:
     if cdata[i] != dsdata[i]:
         print(i)

