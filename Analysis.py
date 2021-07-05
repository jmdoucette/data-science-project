# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import time
from scipy.stats import sem
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from pingouin import partial_corr
import matplotlib.pyplot as plt

df=pd.read_csv("/Users/jamesdoucette/Desktop/Schoolwork/data-science-project/genre+release.csv")
continuous_vars=['acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence']
discrete_vars=['key','mode','time_signature']
#others are popularity,genre,year

def z_test(data1,data2):
    mean1=np.mean(data1)
    mean2=np.mean(data2)
    diff = np.abs(mean1 - mean2)
    se = np.sqrt(sem(data1)**2 + sem(data2)**2 )
    p=(1-norm.cdf(diff/se))
    return [mean1>mean2,mean1,mean2,p]

def pearson(var1,var2):
    return pearsonr(df[var1],df[var2])

def spearman(var1,var2):
    return spearmanr(df[var1],df[var2])

def print_corr(var1,var2):
    p=pearson(var1,var2)
    s=spearman(var1,var2)
    print(var1)
    print(f'Spearmanr {s[0]} with p value {s[1]}\n')
    
def get_genre(i):
    genres=['Movie', 'R&B', 'A Capella', 'Alternative', 'Country', 'Dance', 'Electronic', 'Anime', 'Folk', 'Blues', 'Opera', 'Hip-Hop', 'Children\'s Music', 'Rap', 'Indie', 'Classical', 'Pop', 'Reggae', 'Reggaeton', 'Jazz', 'Rock', 'Ska', 'Comedy', 'Soul', 'Soundtrack', 'World']
    return genres[i]
    

def population_correlations():
    print('Correlation with population, no control')
    for var in continuous_vars:
        print_corr(var,'popularity')
    print_corr('year','popularity')
    
    print('Discrete Variables')
    
def population_partial_correlations():
    print('Correlation with population, controlling for release year')
    for var in continuous_vars:
        p=partial_corr(df,x=var,y='popularity',covar='year',method='spearman')
        print(var)
        print(f"Correlation coefficient {p['r'].values[0]} with p value {p['p-val'].values[0]}\n")
        
def control_genre():
    print('Controlling for release year and numerical genre')
    for var in continuous_vars:
        p=partial_corr(df,x=var,y='popularity',covar=['year','genrenum'],method='spearman')
        print(var)
        print(f"Correlation coefficient {p['r'].values[0]} with p value {p['p-val'].values[0]}\n")
        
def all_genres():
    print('Controlling by release year, separating by genre')
    for i in range(26):
        print(f'Genre {get_genre(i)}')
        for var in continuous_vars:
            this_genre=df[df['genre']==i]
            p=partial_corr(this_genre,x=var,y='popularity',covar='year',method='spearman')
            print(var)
            print(f"Correlation coefficient {p['r'].values[0]} with p value {p['p-val'].values[0]}\n")
        
def year_correlations():
    print('Continuous Variables')
    for var in continuous_vars:
        print_corr(var,'year')
        
def discrete():
    print('Testing discretes')
    for var in discrete_vars:
        vals=df[var].unique()
        print(var)
        for val in vals:
            positive=df[df[var]==val]['popularity']
            negative=df[df[var]!=val]['popularity']
            test=z_test(positive,negative)
            print(val,test)
        print("\n")
        
def average_by_year(var):
    years=np.arange(1960,2020)
    res=[]
    for year in years:
        res.append(df[df['year']==year][var].mean())
    return res

def average_by_genre(var):
    res=[]
    for i in range(26):
        res.append(df[df['genrenum']==i][var].mean())
    return res

def popularity_graphs():
    graph_vars=['duration_ms', 'energy', 'tempo', 'valence', 'popularity']
    years=list(range(1960,2020))
    for var in graph_vars:
        plt.style.use('ggplot')
        plt.plot(years,average_by_year(var))
        plt.xlabel('Year')
        plt.ylabel(var)
        plt.show()
        

def genre_graphs():
    graph_vars=['duration_ms', 'energy', 'tempo', 'valence', 'popularity']
    genre_nums=list(range(26))
    genres=['Movie', 'R&B', 'A Capella', 'Alternative', 'Country', 'Dance', 'Electronic', 'Anime', 'Folk', 'Blues', 'Opera', 'Hip-Hop', 'Children\'s Music', 'Rap', 'Indie', 'Classical', 'Pop', 'Reggae', 'Reggaeton', 'Jazz', 'Rock', 'Ska', 'Comedy', 'Soul', 'Soundtrack', 'World']
    for var in graph_vars:
        plt.style.use('ggplot')
        plt.bar(genre_nums,average_by_genre(var))
        plt.ylabel(var)
        plt.xlabel('Genre')
        plt.ylabel(var)
        plt.xticks(genre_nums, genres,rotation=90)
        plt.show()
        
def year_amt():
    years=list(range(1960,2020))
    plt.style.use('ggplot')
    plt.plot(years,list(map(lambda x: len(df[df['year']==x]),years)))
    plt.ylabel('Number of Songs in Dataset')
    plt.xlabel('Year')
    plt.show()
    

        

            