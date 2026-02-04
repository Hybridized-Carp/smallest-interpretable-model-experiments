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
    r=[(8,0),(14,10)]
    for i in r:
        if(x&i[0])== i[1]:
            return (x,1)
    return (x,0)

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

#def FindStrictExtStr(C,e):
ddata=list(range(256))
cdata=list(map(classification_rule, ddata))
dsdata=list(map(dcset, ddata))
nodes=[]
nodesl=[]
nodesr=[]
nodesrule=[]
annotation=[]

def ApplyDecisionSet(x, model):
    for term in model[0]:
        if(x&term[0])== term[1]:
            return 1-model[1]
        else:
            return model[1]

# is size the number of terms?
# or is size the sum of the number of literals in each term
# because for binary features unless you have more than 64 features for your examples
# (maybe more with funky vector instructions)
# the amnt of time spent checking features will be less than the jumps
# might be an over fitting thing tho?
def FindOptModelStr(classification_instance, size):
    return FindOptExtStr(classification_instance, size)

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
            print("death")
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
    print("!")
    if model == None and annotations == None:
        for example in classification_instance[0]:
            if classification_instance[1](example) == 1:
                e1= example
                A1={'default':e1}
                break
        for example in classification_instance[0]:
            if classification_instance[1](example) == 0:
                e2= example
                A2={'default':e2}
                break
        return [[[[],e1],A1],[[[],e2],A2]]
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
    for term in model[0]:
        if (example&term[0]) == term[1]:
            break
    ex2 = annotations[term]
    hmd = (ex2^example)
    t=1
    while hmd >0:
        if (hmd%2 == 1):
            nA = annotations.copy()
            tmodel=model.copy()
            bitmask=t|term[0]
            for temp in tmodel[0]:
                if temp == term:
                    temp=(bitmask,ex2&bitmask)
            extensions.append([tmodel,nA])
        t<<=1
        hmd>>=1
    return extensions
        
a = FindOptModelStr((ddata,clsf),7)

for i in ddata:
     if cdata[i] != dsdata[i]:
         print(i)


         
