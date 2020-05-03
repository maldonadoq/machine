import numpy as np

class Kmeans:
    
    def init(self, x):
        r = np.random.rand(self.k, self.dim)
        c = np.zeros((self.k, self.dim))
        
        for i in range(self.dim):
            m = x[:,i].min()
            l = (x[:,i].max() - m)
            c[:,i] = m + (r[:,i] * l)
            
        return c
    
    def cluster(self):
        cl = []
        for j in range(self.k):
            cl.append([])
        return cl
    
    def distance(self, x1, x2):
        s = 0
        for i in range(self.dim):
            s += np.power(x1[i]-x2[i], 2)
            
        return np.sqrt(s)
    
    def means(self):
        for i in range(len(self.clt)):
            tmp = np.asarray(self.clt[i])
            
            if(tmp.shape[0] > 0):            
                for j in range(self.dim):
                    self.cent[i,j] = tmp[:,j].mean()
            else:
                self.cent[i] = np.zeros(self.dim)
    
    def fit(self, x, k, epochs):
        self.dim = x.shape[1]
        self.k = k
        
        self.cent = self.init(x)
        cd = np.zeros(self.k)
        
        for i in range(epochs):
            self.clt = self.cluster()
            for xp in x:
                for t in range(len(self.cent)):
                    cd[t] = self.distance(xp, self.cent[t])
                cm = np.argmin(cd)
                
                self.clt[cm].append(xp)
                
            self.means()
            
        return (self.cent, self.clt)