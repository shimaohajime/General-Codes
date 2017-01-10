an# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 19:49:54 2016

@author: Hajime
"""
groupid = np.array([0,0,1,1,3,3,3])
dum = CreateDummy(groupid)
x = np.arange(14).reshape([2,7]).T.astype(float)

y = np.arange(7)

#for i in xrange(id_num):
#    a = np.sum(x[groupid==id_list[i],:],axis=0)
#    sums[groupid==id_list[i],:] =a
import numpy as np

def CreateDummy(groupid, sparse=0):
    nobs = groupid.size
    id_list = np.unique(groupid)
    id_num = id_list.size    
    if sparse==0:
        groupid_dum = np.zeros([nobs,id_num])
    elif sparse==1:
        groupid_dum = scipy.sparse.lil_matrix((nobs,id_num))
        
    for i in xrange(id_num):
        a = (groupid==id_list[i]).repeat(id_num).reshape([nobs,id_num])
        b = (id_list==id_list[i]).repeat(nobs).reshape([id_num,nobs]).T
        c = a*b
        groupid_dum[c] = 1        
    return groupid_dum

def DummyToID(dummy):
    a = np.arange(np.shape(dummy)[1])
    b = np.sum( dummy *a, axis=1)
    return b.astype(int)

def ShapeID(groupid):
    #Make ID continuous, start from zero
    nobs = groupid.size
    id_list = np.unique(groupid)
    id_num = id_list.size    
    newid = np.zeros(nobs)
    newid_list = np.arange(id_num)
    for i in xrange(id_num):
        newid[groupid==id_list[i]] = i
    return newid.astype(int)
    
def SumByGroup(groupid,x,shrink=0):
    nobs = groupid.size
    id_list = np.unique(groupid)
    id_num = id_list.size
    if x.ndim==1:
        x = np.array([x]).T
    nchar = x.shape[1]
    if shrink==0:    
        sums = np.zeros([nobs,nchar])
        for i in xrange(id_num):
            a = np.sum(x[groupid==id_list[i],:],axis=0)
            sums[groupid==id_list[i],:] =a
        return sums
    if shrink==1:
        sums = np.zeros([id_num,nchar])
        for i in xrange(id_num):
            a = np.sum(x[groupid==id_list[i],:],axis=0)
            sums[i] = a
        return sums

def SumByGroupDummy(groupid_dummy,x,shrink=0):
    a = np.dot(x.T,groupid_dummy)
    if shrink==0:
        b = np.dot(groupid_dummy,a.T)
    if shrink==1:
        b = a
    return b
    
def MeanByGroup(groupid,x,shrink=0):
    nobs = groupid.size
    id_list = np.unique(groupid)
    id_num = id_list.size
    if x.ndim==1:
        x = np.array([x]).T
    nchar = x.shape[1]
    if shrink==0:    
        ave = np.zeros([nobs,nchar])
        for i in xrange(id_num):
            a = np.mean(x[groupid==id_list[i],:],axis=0)
            ave[groupid==id_list[i],:] =a
        return ave
    if shrink==1:
        ave = np.zeros([id_num,nchar])
        for i in xrange(id_num):
            a = np.mean(x[groupid==id_list[i],:],axis=0)
            ave[i] = a
        return ave

def MergeByID(id1,data1,id2,data2): #merge data2 to data1
    n1 = id1.shape[0]
    n2 = id2.shape[0]    
    
    newid = GenIDByIDs( np.r_[id1,id2] )
    newid_list = np.unique(newid)
    newid1 = newid[:n1]
    newid2 = newid[n1:]
        
    #id_out=np.array([])
    data_out=data1
    for i in newid_list:
        np.c_[data1, ]
        #id_out = np.append( id_out, np.repeat(i, np.max()) )

def GenIDByIDs(id1):
    n = id1.shape[0]
    id1_list = unique_rows(id1)
    u = id1_list.shape[0]
    new_id = np.zeros(n)
    for i in xrange(u):
        new_id[ np.prod(id1==id1_list[i],axis=1).astype(bool) ] = i
    return new_id
        
def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])
    
    
    
def QuadPoly(vec):
    n = vec.shape[0]
    b = np.tile(vec,n).reshape([n,n])
    c = np.repeat(vec,n).reshape([n,n])
    d = b*c
    e = d[np.tril_indices(n)]
    f= np.append(vec, e)
    return f
    
def QuadPoly_vec(vec_vec):
    rep = vec_vec.shape[0]
    n = vec_vec.shape[1]
    i = np.tril_indices(n)[0]
    j = np.tril_indices(n)[1]
    n_out = i.shape[0]    
    ii = np.tile(i,rep)+np.repeat(np.arange(rep)*n, n_out)
    jj = np.tile(j,rep)
    
    b = np.tile(vec_vec, n).reshape([n*rep,n])
    c = np.repeat(vec_vec, n).reshape([n*rep,n])
    d = b*c
    e = d[ii,jj].reshape([rep,n_out])
    f = np.c_[vec_vec, e]
    return f

def ivreg_2sls(x,y,z,invA=None):
    if invA is None:
        N_inst = z.shape[1]
        invA = np.linalg.solve( np.dot(z.T,z), np.identity(N_inst) )
    temp1 = np.dot(x.T,z)
    temp2 = np.dot(y.T,z)
    temp3 = np.dot(np.dot(temp1,invA),temp1.T) #x'z(z'z)^{-1}z'x
    temp4 = np.dot(np.dot(temp1,invA),temp2.T) #x'z(z'z)^{-1}z'y
    bhat = np.linalg.solve(temp3,temp4)
    gmmresid = y - np.dot(x, bhat)
    temp5=np.dot(gmmresid.T, z)
    f=np.dot(np.dot(temp5, invA),temp5.T)
    del x,y,z,invA
    return bhat,f  
    
#Taken from http://stackoverflow.com/questions/18353280/iterator-over-all-partitions-into-k-groups
def neclusters(l, K):
    for c in clusters(l, K):
        if all(x for x in c): yield c   
def clusters(l, K):
    if l.size>0:
        prev = None
        for t in clusters(l[1:], K):
            tup = sorted(t)
            if tup != prev:
                prev = tup
                for i in xrange(K):
                    yield tup[:i] + [[l[0]] + tup[i],] + tup[i+1:]
    else:
        yield [[] for _ in xrange(K)]           

#-------------------------------
#from http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    #dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        #out = np.zeros([n, len(arrays)], dtype=dtype)
        out = np.zeros([n, len(arrays)])

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


#Measure Time------
def measure_time(func,a,rep=1):
    start = time.time()
    for i in xrange(rep):
        func(a)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"



#dictionary <=> combined data.
def MergeData(data_dic):
    n = list(data_dic.values())[0].shape[0]
    data_col={}
    i=0
    data_comb = np.ones(n)
    for item in data_dic.keys():
        data_comb = np.c_[data_comb, data_dic[item]]
        if data_dic[item].ndim==1:
            data_col[item] = np.arange(i, i+1)
            i=i+1
        if data_dic[item].ndim==2:
            data_col[item] = np.arange(i, i+data_dic[item].shape[1])
            i=i+data_dic[item].shape[1]
    data_comb=np.delete(data_comb,0,axis=1)
    return data_comb, data_col
    
def RecoverData(data_comb, data_col):
    data_recovered = {}
    for item in data_col.keys():
        data_recovered[item] = data_comb[:,data_col[item]]
        if data_recovered[item].shape[1]==1:
            data_recovered[item]=data_recovered[item].flatten()
    return data_recovered

