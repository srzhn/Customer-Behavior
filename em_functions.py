import numpy as np
import pandas as pd
import seaborn as sns
import math
import networkx as nx
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import pickle
import time
from functools import reduce
from operator import add
import os

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# print(color.BOLD + 'Hello World !' + color.END +' ssss '+ color.BOLD + 'Hello World !' + color.END)
# print(f" {color.BOLD} fdfdfdf {color.END} dfdfdfdfd {color.BOLD} dfdfdfdfdf {color.END}")

# class DataBase():
#     def __init__(self, weeplace_checkins, city=None):
# #         self.weeplace_checkins = pd.read_csv('../weeplaces/weeplace_checkins.csv', engine='python').dropna()
#         self.df = weeplace_checkins
#         if city is not None:
#             self.df = self.df.query(f"city==\'{city}\'")
#         self.df = (self.df['city'].value_counts()[:5]).index
#         col = ['userid', 'datetime','city','category']
#         self.df = self.df[weeplace_checkins['city'].isin(cities)][col]
#         self.categories = ['Food', 'Home / Work / Other', 'Shops', 'Parks & Outdoors',
#                            'Arts & Entertainment', 'College & Education', 'Nightlife',
#                            'Travel']


    
class CityDataFrame():
    def __init__(self, df, city:str):
        super(CityDataFrame, self).__init__()
        self.city = city
        self.df = self.df_for_certain_city(df, city)
        self.le = LabelEncoder()
        self.le.fit(self.df['category'].values)
        self.le_name_mapping = dict(zip(self.le.classes_, self.le.transform(self.le.classes_))) # можно исправить
        self.categories = self.le.classes_
        
    
    def df_for_certain_city(self, df, city):
        rename_ = {'Colleges & Universities':'College & Education', 
               'Colleges & Universitie':'College & Education',
                'Homes, Work, Others':'Home / Work / Other',
                'Arts & Entertainmen':'Arts & Entertainment',
                'Nightlife Spots':'Nightlife',
                'Travel Spots':'Travel',
                'Travel Spot':'Travel', 
                'Nightlife Spot':'Nightlife',           
              'Great Outdoors':'Parks & Outdoors'}
        new_df = df[df['city']==city]
        new_df = new_df[new_df['userid'].isin((new_df['userid'].value_counts() > 10)
                                           .index[:(new_df['userid'].value_counts() > 10).sum()])]
        new_df = new_df[new_df['datetime'] > '2009-01-01']
        new_df['category'] = new_df['category'].astype('str').apply(lambda x: x[:x.find(':')])
        new_df['category'] = new_df['category'].apply(lambda x: x if x not in rename_ else rename_[x])
        new_df['datetime'] = pd.to_datetime(new_df.datetime)
        return new_df
        
    def generation_of_sequences(self, min_len = 5):
        '''
        Реализовано для недель
        Каждая последовательность не меньше min_len (по умолчанию, 5 и более чекинов за неделю)
        '''
        means_of_checkins_in_week = []
        sequences = []
        for user in self.df['userid'].unique():
            user_df = self.df[self.df['userid']==user][['datetime', 'category']]
    #         user_df['category'] = user_df['category'].map(le_name_mapping)
            user_df['category'] = self.le.transform(user_df['category'])
            user_df['week'] = user_df['datetime'].apply(lambda x: 
                                                        str(x.year)) + ' ' +user_df.datetime.apply(lambda x: str(x.weekofyear))
            user_df['dayofweek'] = user_df['datetime'].apply(lambda x: x.dayofweek)
            user_df = user_df.sort_values(by='datetime', ascending=True)
            means_of_checkins_in_week.append(np.mean(user_df['week'].value_counts()))
            sequences.append([user])
            for week in user_df.week.unique():
                seq = reduce((lambda x, y: str(x) + str(y)), user_df[user_df['week']==week].category.values)
                if  type(seq)==str and len(seq)>=min_len:
                    sequences[-1].append(seq)
        means_of_checkins_in_week = np.array(means_of_checkins_in_week)
        return sequences

    def union_of_seq(self, seqs, full=True, q=0.5):
        """full==False:Берем случайные последовательности, если количество последовательностей больше медианы. 
        Если количество полсдеовательностей для пользователя меньше, то все."""
        s = []
        if full is True:
            for i in seqs:
                if len(i)>1:
                    s.extend(i[1:])
        else: 
    #         med = median_seq(seq)
            med = self.quantile_seq(seqs, q)
            for seq in seqs:
                if len(seq)>1:
                    ss = seq[1:]
                    if len(ss)>med:
                        ss = np.random.choice(ss,int(med),replace=False)
                    s.extend(ss)
        return s
    
    # Количество последовательностей
    def count_seq(self, seqs):
        return reduce(add, list(map(lambda x: len(x)-1, seqs)))

    # Среднее количество последовательностей.
    def mean_seq(self, seqs):
        return np.mean(list(map(lambda x: len(x)-1, seqs))) # -1 - for removing user name

    def median_seq(self, seqs):
        return np.median(list(map(lambda x: len(x)-1, seqs))) # -1 - for removing user name

    def quantile_seq(self, seqs, q=0.5):
        return np.quantile(list(map(lambda x: len(x)-1, seqs)), q) # -1 - for removing user name


class EMClass(CityDataFrame):
    def __init__(self, df, city, C):
        super(EMClass, self).__init__(df, city)
        self.C = C
        self.reset_params()
        
    def reset_params(self):
        self.pi = np.ones(self.C)/self.C
        
        self.f = np.random.rand(self.C, 8)
        self.f = self.f / np.sum(self.f, axis=1).reshape(-1,1)
        
        self.T = np.random.rand(self.C,8,8)
        self.T = np.array([t/np.sum(t, axis=1).reshape(-1,1) for t in self.T])
        
    def read_params(self, **kwargs):
        if 'pi' in kwargs.keys():
            pi = kwargs.get('pi')
            if len(pi)==self.C:
                self.pi = pi
            else:
                assert ValueError, 'shape of pi is not match with C.'
        if 'f' in kwargs.keys():
            f = kwargs.get('f')
            if f.shape==[self.C, 8]:
                self.f = f
            else:
                assert ValueError, f'shape of f is not {[self.C, 8]}.'
        if 'T' in kwargs.keys():
            T = kwargs.get('T')
            if T.shape==[self.C, 8, 8]:
                self.T = T
            else:
                assert ValueError, f'shape of T is not {[self.C, 8, 8]}.'

    def prob(self, seq, c):
        prod = self.f[c][int(seq[0])]
        for i in range(len(seq)-1):
            prod*=self.T[c][int(seq[i]),int(seq[i+1])]
        return prod    
        
    def EMstep(self, seqs):
        #E-step
        #print("Old pi = {}\nOld f for c = 0 is {}\nOld T for c = 0 is {}".format(pi,f[0],T[0]))
        r = []
        for seq in seqs:
            r.append(0)
            r[-1] = np.array([self.pi[c]*self.prob(seq, c) for c in range(self.C)])
            r[-1] /= np.sum(r[-1])
        r = np.array(r)

        #M-step
    #     print(sum([r[i][2] for i in range(len(seq))]))

        m = np.sum(r, axis=0)
        new_pi = m/np.sum(m)
        new_f = [np.zeros(8) for c in range(self.C)]
        new_T = [np.zeros((8,8)) for c in range(self.C)]
        for c in range(self.C):
            for i in range(len(seqs)):
                first_element_of_seq = int(seqs[i][0])
                new_f[c][first_element_of_seq]+=r[i][c]
            new_f[c]/=np.sum(new_f[c])    #из соображений, что сумма должна быть равна 1
            for i in range(len(seqs)):    #sequence
                for j in range(len(seqs[i])-1):    #step of sequence
                    previous_step,next_step = int(seqs[i][j]), int(seqs[i][j+1])
                    new_T[c][previous_step,next_step]+=r[i][c]
            for category in range(8):
                new_T[c][category,:] = (new_T[c][category,:]/
                                        np.sum(new_T[c][category,:]))
        return (new_pi, new_f, new_T)

    def likelyhood(self, seq):
        prod = 1
        for i in seq:
            sum_ = 0
            for c in range(self.C):
                sum_+=pi[c]*self.prob(i,c)
            #print(sum_)
            prod*=sum_
        return prod

    def stop_criterion(self, pi,f,T,pin,fn,Tn):
        a1 = np.linalg.norm(pi-pin)
        a2 = np.linalg.norm(np.reshape(f, (1,-1))-np.reshape(fn, (1,-1)))
        a3 = np.linalg.norm(np.reshape(T, (1,-1))-np.reshape(Tn, (1,-1)))
        return a1+a2+a3
    
    def EM(self, seqs, eps = 0.001):
        new_pi,new_f,new_T = self.EMstep(seqs)
    #     print(new_pi,new_f,new_T)
        i=0
        L=self.stop_criterion(self.pi, self.f, self.T, new_pi, new_f, new_T)
        while (L>eps):
            i+=1
            if i%5==0:
                print("i = {}, L = {}".format(i,L)) 
            self.pi, self.f, self.T = new_pi, new_f, new_T
            new_pi, new_f, new_T = self.EMstep(seqs)
    #         print(new_pi)
            L=self.stop_criterion(self.pi, self.f, self.T, new_pi, new_f, new_T)
        print("Last number of iterations: {}".format(i))    
        self.pi, self.f, self.T = new_pi, new_f, new_T
        #print('\n\nNew values for variables:\n pi = {},\n f[0] = {},\n T[0] = {}'.format(new_pi,new_f,new_T))
        return new_pi, new_f, new_T
    
    def predictions_for_sequences(self, seqs):
        predictions = []
        for seq in seqs:
            user = seq[0]
            predictions.append([user])
            for i in range(1,len(seq)):
                pred = np.array([self.prob(seq[i], c=c)*self.pi[c] for c in range(self.C)])
                pred/=np.sum(pred)
                predictions[-1].append(np.round(pred,3))
        return predictions
        #     print(np.round(pred,3))

    def class_with_max_probability(self, predictions):
        res = []
        for pred in predictions:
            res.append([pred[0]])
            if np.isnan(pred[1][0]):
                res[-1].append(-1)
            else:
                res[-1].append(np.argmax(pred[1]))
        return res

    def prediction_for_user(self, predictions):
        pu = []
        for pred in predictions:
            pu.append([pred[0]])
            res = np.zeros(self.C)
            for j in range(1,len(pred)):
                res+=pred[j]
            res/=np.sum(res)
            pu[-1].append(np.round(res, 3))
        return pu

    def predictions(self, full_seq, show=False):
        p = [self.predictions_for_sequences(full_seq)]
        p.append(self.prediction_for_user(p[-1]))
        p.append(self.class_with_max_probability(p[-1]))
        if show == True:
            for j in range(-1,C):
                print(j,np.sum([i[1]==j for i in p[-1]]))
        return pd.Series([i[1] for i in p[-1]], index=[i[0] for i in p[-1]]),p
    
    def generate_sequence_for_class(self, l):
        """generate sequence of activity by f,T.

        cluster<num_of_clusters (or len(f))"""
        if l>=self.C:
            assert ValueError, "l must be lower then C!"
        seq = [np.random.choice(np.arange(8), 1, p=self.f)[0]]
        for i in range(1,l):
            seq.append(np.random.choice(np.arange(8), 1, p=self.T[seq[-1]])[0])
        return seq
    
    def plot_heatmaps(self, save=False, fig_name=None):
        
        def fsave(params=None):
            if os.path.isdir("plt")==False:
                os.mkdir('plt')
            plt.savefig(f"./plt/{time.time()}_{params}.png")
        print('pi = ', np.round(self.pi, 4))
        fig = plt.figure(figsize=(10,5))
        hm = sns.heatmap(data=self.f, annot=True, vmin=0, vmax=0.5, cmap = 'Greens')
        hm.set_xticklabels(self.le.classes_, rotation=45)
        hm.set_yticklabels(range(1, self.C+1))
        if save:
            fsave(params=None)
        if fig_name:
            plt.savefig(fig_name+'_f.png')
        plt.show()

        fig = plt.figure(figsize=(15,15))
        for i in range(self.C):
            plt.subplot(100*math.ceil(self.C/2) + 21+i)
            plt.title(f"Class {i+1}, pi = {np.round(self.pi[i], 4)}")
            hm = sns.heatmap(data=self.T[i], annot=True, vmin=0, vmax=0.5, cmap = 'Greens')    
            hm.set_xticklabels(self.le.classes_, rotation=45)
            hm.set_yticklabels(self.le.classes_, rotation=45)
        plt.tight_layout()
        if save:
            fsave(params=None)
        if fig_name:
            plt.savefig(fig_name+'_T.png')
        plt.show()


    def main(self, eps=0.01, min_seq_len=5, q=0.5, full=False, save=False): 
        '''
        Input: clean databases of cities.
        Output: (a,b,c)
                a - sequences of each user in city
                b - parameters of model
                c - predictions for each user
        '''
        print(time.ctime())
        print(self.city)
        seqs = self.union_of_seq(self.generation_of_sequences(min_len=min_seq_len), full=full, q=q)
        print(time.ctime())
        print("{} generation of sequences - Done".format(self.city))
        print(f"Total count of sequences for min_len = {min_seq_len}: {color.BOLD}{len(seqs)}{color.END}")
        print(f"Quantile {q} (count of seqs of unique persons): {color.BOLD}{self.quantile_seq(seqs, q)}{color.END}")
        self.EM(seqs, eps=eps)

        print(time.ctime())
        print("{} fitting of model - Done".format(self.city))
        seqs_full = self.generation_of_sequences(min_len=1)
        print(time.ctime())
        print("{} generation of full sequences - Done".format(self.city))
        print(f"Total count of sequences: {color.BOLD}{self.count_seq(seqs_full)}{color.END}")
        predictions_for_city = self.predictions(seqs_full)
        params = (self.pi, self.f, self.T)
        print("{} - Done".format(self.city))

        if save:
            print('Saving...', end=' ')
            SAVE_DIR = "./saves"
            if not os.path.isdir(SAVE_DIR):
                os.mkdir(SAVE_DIR)
            np.savez(os.path.join(SAVE_DIR, self.city +f' minlen={min_seq_len},q={q},eps={eps} parameters'), pi=params[0],*params[1],*params[2])
            with open(os.path.join(SAVE_DIR, self.city+f' minlen={min_seq_len},q={q},eps={eps} seq'), 'wb') as fp:
                pickle.dump(seqs_full, fp)
            print('Done.\n\n')
        return (seqs, seqs_full,params,predictions_for_city)

def read_param(s):
    data = np.load(s+' parameters.npz')
    pi = data['pi']
    f = []
    T = []
    for i in range(4):
        f.append(data['arr_'+str(i)])
        T.append(data['arr_'+str(i+4)])
    return (pi,f,T)

def read_seq(s):
    with open (s+' seq', 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist

def plot_heatmaps(pi, f, T, le, fig_name=None):
    C=len(pi)
    
    print('pi = ', np.round(pi, 4))
    plt.figure(figsize=(10,5))
    plt.gcf().subplots_adjust(bottom=0.35)
    hm = sns.heatmap(data=f, annot=True, vmin=0, vmax=0.5, cmap = 'Greens')
    hm.set_xticklabels(le.classes_, rotation=45)
    hm.set_yticklabels(range(1, C+1))
    if fig_name:
        plt.savefig(fig_name+'_f.png')
        
    plt.show()
    
    plt.figure(figsize=(15,15))
    plt.gcf().subplots_adjust(bottom=0.15)
    for i in range(C):
        plt.subplot(100*math.ceil(C/2) + 21+i)
        plt.title(f"Class {i+1}, pi = {np.round(pi[i], 4)}")
        hm = sns.heatmap(data=T[i], annot=True, vmin=0, vmax=0.5, cmap = 'Greens')    
        hm.set_xticklabels(le.classes_, rotation=45)
        hm.set_yticklabels(le.classes_, rotation=45)

    plt.tight_layout()
    if fig_name:
        plt.savefig(fig_name+'_T.png')
    plt.show()
