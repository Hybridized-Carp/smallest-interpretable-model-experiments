def FindOptModelStr(classification_instance, size):
    for example in classification_instance[0]:
        if classification_instance[1](example) == 1:
            model=[[1]]
            A1=[example]
            break
    
    res = FindOptExtStr(classification_instance, size, model, A1)
    if res != None:
        return res
    
    for example in classification_instance[0]:
        if classification_instance[1](example) == 0:
            model=[[0]]
            A2=[example]
            break

    return FindOptExtStr(classification_instance, size,model, A2)

def ApplyTree(x,nodes):
    node=0
    while len(nodes[node]) >1:
        node= nodes[node][2] if x&(nodes[node][0]) else nodes[node][1]
    return nodes[node][0]

def TreePath(x,nodes):
    node=0
    path=[0]
    while len(nodes[node]) >1:
        node= nodes[node][2] if x&(nodes[node][0]) else nodes[node][1]
        path.append(node)
    return (nodes[node][0],node,path)

def TreeSize(nodes,layer=0,pos=0):
    if len(nodes[pos]) == 1:
        return layer+1
    else:
        return max(
            TreeSize(nodes,layer+1,nodes[pos][1]),
            TreeSize(nodes,layer+1,nodes[pos][2])
        )

def FindOptExtStr(classification_instance, size, model=None, annotations=None):
    model_pass=True
    for example in classification_instance[0]:
        if ApplyTree(example, model) != classification_instance[1](example):
            model_pass=False
            break
    if model_pass:
        return model
    #two methods of size
    #if len(model[0])>=size:
    if TreeSize(model) >= size:
        #print("death")
        return None
    strict_exts = FindStrictExtStr(classification_instance, size, model, annotations, example)
    B = None
    for nModel, nAnnotations in strict_exts:
        if TreeSize(nModel) <= size:
            A = FindOptExtStr(classification_instance, size, nModel, nAnnotations)
            if A != None:
                if B == None:
                    B=A
                elif TreeSize(B) > TreeSize(A):
                    B=A
    return B     

def FindStrictExtStr(classification_instance, size, model, annotations, example):
    #print("!")
    extensions=[]
    lv,ln,lpath=TreePath(example,model)
    CAv= annotations[ln]
    hmd=(example^CAv)
    t=1
    while hmd >0:
            if (hmd%2 == 1):
                tmodel=model.copy()
                nA = annotations.copy()
                nc=len(model)
                tmodel.append([lv])
                nA.append(CAv)
                tmodel.append([1-lv])
                nA.append(example)
                if (example & t):
                    tmodel[ln]=[t,nc,nc+1]
                else:
                    tmodel[ln]=[t,nc+1,nc]
                extensions.append([tmodel,nA])
                
            t<<=1
            hmd>>=1
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

ddata=list(range(256))

