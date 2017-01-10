# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:32:06 2016

@author: Hajime
"""

class LogitDemandEstimation:
    def __init__(datatype, datadir, price_file, x_file, iv_file, share_file, sales_file=None, marketsize_file=None, mktid_file, prodid_file,\
    flag_TitleFE=0):
        
        if datatype=='csv':
            self.price = np.genfromtxt(datadir+price_file, delimiter=',')
            self.x = np.genfromtxt(datadir+x_file, delimiter=',')
            self.iv = np.genfromtxt(datadir+iv_file, delimiter=',')
            if share_file!=None:
                self.share = np.genfromtxt(datadir+share_file, delimiter=',')
            if share_file==None:
                self.sales = np.genfromtxt(datadir+sales_file, delimiter=',')
                self.marketsize = np.genfromtxt(datadir+marketsize_file, delimiter=',')
                self.share = self.sales/self.marketsize
            self.mktid = np.genfromtxt(datadir+mktid_file, delimiter=',').astype(int)
            self.prodid = np.genfromtxt(datadir+prodid_file, delimiter=',').astype(int)
            
        if datatype=='npy':
            self.price = np.load(datadir+price_file)
            self.x = np.load(datadir+x_file)
            self.iv = np.load(datadir+iv_file)
            if share_file!=None:
                self.share = np.load(datadir+share_file)
            if share_file==None:
                self.sales = np.load(datadir+sales_file)
                self.marketsize = np.load(datadir+marketsize_file)
                self.share = self.sales/self.marketsize
            self.mktid = np.load(datadir+mktid_file).astype(int)
            self.prodid = np.load(datadir+prodid_file).astype(int)
            
        #Fixed Effect
        self.flag_TitleFE=flag_TitleFE
        if self.flag_TitleFE==1:
            self.TitleDum = self.CreateDummy(self.prodid, sparse=0)
            self.x = np.c_[self.x, self.TitleDum]
            
        
        #Set Parameters
        self.N_obs = self.x.shape[0]
        MarketIndex, cdindex_temp = np.unique(self.mktid, return_index=True)
        self.cdindex = np.append(cdindex_temp[1:]-1,self.N_obs-1).astype(int)
        self.N_market = np.size(MarketIndex)
        self.N_char = self.x.shape[1]
        if self.flag_TitleFE==1:
            self.N_char = self.N_char + self.N_prod
        self.N_iv = self.iv.shape[1]
        
        #Create IV
        self.IV = self.x
        for i in xrange(self.N_iv):
            if np.linalg.matrix_rank(np.c_[ self.x[:,0], self.iv[:,i] ])>1:
                self.IV = np.c_[self.IV, self.iv[:,i]]
        self.N_IV = self.IV.shape[1]
        
    def EstimateLogit():
        #Logit regression
        self.s_jt = self.share
        outshr=np.zeros(self.N_obs)
        for m in xrange(self.N_market):
            temp = 1.0 - np.sum(self.s_jt[np.where(self.mktid==m)])
            outshr[np.where(self.cdid==m)]=temp
        if np.any(outshr<0.):
            self.outshr = outshr
            sys.exit('outshare negative (logit regression)')            
            
        y=np.log(self.s_jt)-np.log(outshr) #difference to the outside share.
        
        chars = self.x
        IVs = self.IV

        t =self.ivreg_2sls(chars,y,IVs)[0]
        theta1_logit = t
        
        mval = np.dot(chars, t) #mean value (mval)=theta*x
        del chars,IVs
        if self.flag_Display==1:
            print('Logit regression')
            print('coeff:',t)
            print('theta1_logit.shape'+str(theta1_logit.shape))
