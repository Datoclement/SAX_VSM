import numpy as np
import scipy.stats as spst

class Saxvsm :

    num_alphabet = 10
    len_window = 4
    len_word = 2

    def get_distribution(self) :

        cell = 1.0 / (self.num_alphabet * 1.0)

        return spst.norm.ppf(cell * (np.arange(self.num_alphabet) + 1))

    def __init__(self,num_class,len_series) :

        self.num_class = num_class
        self.len_series = len_series
        self.num_word = self.num_alphabet ** self.len_word

        self.total_instance = 0
        self.num_instance_class = np.zeros(shape=(num_class),dtype=int)
        self.tf = np.zeros(shape=(self.num_word,num_class))
        self.idf = np.zeros(shape=(self.num_word))

        assert (self.len_window % self.len_word == 0)
        self.len_piece = self.len_window / self.len_word

        self.breakpoints = self.get_distribution()

    def sax(self,instance):

        def to_letter(mean) :
            h = -1
            t = self.num_alphabet-1
            while(True):
                if(h+1>=t):
                    break
                m = (h+t)/2
                if(self.breakpoints[m]<mean):
                    h=m
                    continue
                if(self.breakpoints[m]>mean):
                    t=m
                    continue
                h=m-1
                t=m
            return t

        def to_word(curmean,curmeans) :
            word = np.array([to_letter(curmeans[i]-curmean) for i in range(self.len_word)])
            wid = 0
            for i in range(self.len_word) :
                wid *= self.num_alphabet
                wid += word[i]
            return wid

        len_series = self.len_series
        len_window = self.len_window
        len_piece = self.len_piece
        len_word = self.len_word
        num_word = self.num_word

        curmean = np.mean(instance[0:len_window])
        curmeans = np.array([np.mean([
                instance[
                i * self.len_piece : (i+1) * self.len_piece ])
                for i in range(len_word)])

        words_vect = np.zeros(shape=(num_word),dtype=float)
        for pos in range(len_series-len_window+1) :
            curword = to_word(curmean,curmeans)
            words_vect[curword] += 1.0
            if(pos < len_series-len_window) :
                curmean += (instance[pos+len_window] - instance[pos]) / (1.0*len_window)
                curmeans += np.array([
                        (instance[pos + (i+1) * self.len_piece]
                        - instance[pos + i * self.len_piece]) / (1.0*len_piece) for i in range(len_word)])

        return words_vect

    def update(self,words,label):

        self.num_instance_class[label] += 1
        n = self.num_instance_class[label]
        self.tf[:,label] = self.tf[:,label] * (n-1)/(1.0*n) + words / (1.0*n)

        self.total_instance += 1
        n = self.total_instance
        self.idf = self.idf * (n-1)/(1.0*n) + words / (1.0*n)

    def add(self,instance,label) :
        words = self.sax(instance)
        self.update(words,label)

    def predict(self,instance) :
        words = self.sax(instance)
        e_r = np.array([i for i in range(self.num_word) if not self.idf[i] == 0])

        # euclidien
        # words = np.expand_dims(words[e_r],axis=1)
        # weights = np.log(1.0+self.tf[e_r])
        # return np.argmin(np.sum((words-weights)**2,axis=0))

        # tfidf cos similarity
        words = words[e_r]
        weights = - np.log(1.0+self.tf[e_r]) \
                * np.log(1/(1.0*self.total_instance)+np.expand_dims(self.idf[e_r],axis=1))

        # print self.tf[e_r]
        # print self.idf[e_r]
        # print (np.expand_dims(self.idf[e_r],axis=1))
        # print weights
        return np.argmax(np.dot(words,weights))

clf = Saxvsm(2,10)

inst = np.array([1,2,3,4,5,4,7,8,9,10])
inst2 = np.array([2,2,3,4,5,4,7,8,9,10])

clf.add(inst,1)
clf.add(inst2,0)

print clf.predict(inst2)
print clf.predict(inst)
