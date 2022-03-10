import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt   

## sklearn : https://scikit-learn.org/stable/index.html
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

## statsmodels : https://www.statsmodels.org/stable/index.html
import statsmodels.formula.api as smf
import statsmodels.api as sm

## stop warings
import warnings    
warnings.filterwarnings("ignore")



def  visualisation_coefficients_class(Base, Modele, Nom_variable) :  
    
    # table qui contien exp coeff et erreur
    data = pd.DataFrame(pd.DataFrame(Modele.params.apply(math.exp)))
    data = data.reset_index()
    #prendre la colonne [0.025 et supprimer la 1ere ligne
    err = pd.DataFrame(Modele.summary().tables[1].data)[[0,5]]
    err = err.drop(axis=0 , index=0)
    err.columns = ['nom','erreur']
    data = pd.merge(data,err,how='left', right_on='nom' , left_on='index')
    data.columns = ['index','coeff','nom','erreur']
    del data['nom']
    
    #on récupère les indices
    indices = pd.DataFrame(data['index'])
    indices.columns = ['indice']
    #indices propres sous la forme variable_modalité
    data['new'] = ''
    for i in range(len(indices)):
      #extraction 
      data.new[i] = indices.indice[i][indices.indice[i].find('(')+1:indices.indice[i].find(',')]+"_"+indices.indice[i][indices.indice[i].find('.')+1:indices.indice[i].find(']')]
    
    counts = pd.DataFrame(Base[Nom_variable].value_counts().reset_index())
    counts.columns =['value', 'freq']
    #Contient les modalités des variables
    x = counts.value
    counts.value = counts.value.astype(str)
    counts = counts.assign(labels = Nom_variable + "_" + x.apply(str))
    #label dans le code
    data = data[data.new.isin(counts.labels)]

    counts = counts.sort_values(by=['value'])
    
    l = [i for i in range(len(counts))]
    counts = counts.set_index([pd.Index((l))])
    counts.columns = ['value', 'freq','variable']

    #concatenation de 1 avec le reste
    coeff = pd.DataFrame(np.array(data))
    coeff = coeff[[1,2,3]]
    #changer les noms des colonnes
    coeff.columns = ['coefficient','erreur','variable']
    #concaténation counts et les coeff
    data_finale = pd.merge(coeff, counts, how='left')
    #data_finale['err_finale'] = data_finale.coefficient - data_finale.erreur.astype(float) 
    return(data_finale)