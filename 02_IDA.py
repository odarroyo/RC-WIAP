# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:24:59 2022

@author: orlandoaram
"""

# hay que llamar al opensees
import openseespy.opensees as ops
# hay que llamar al opsvis para poder plotear la visualización
#import opsvis as opsv
# esta es una librería estándar para plotear otras cosasy poder crear figuras.
import matplotlib.pyplot as plt

import Lib_analisis as an

import opseestools.analisis as an2

import numpy as np

import multiprocessing

from joblib import Parallel, delayed

import joblib as jb

import os as os

import time

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
ops.wipe()
 
# Importar análisis de gravedad
# Esta rutina hace un IDA de un registro

#%% INFORMACION DE ENTRADA
# ======================================

script_dir = os.path.dirname(os.path.abspath(__file__))

# records es una lista de los registros de los nombres de los archivos txt de los sismos. Deben estar en la misma carpeta de este archivos
# SpectrumFactor incluye la lista de los factores para escalar los registros a un espectro determinado
# NSteps inclute el número de datos de cada terremoto.
# DTs incluye los incrementos de tiempo de cada registros
# nodes incluye el nodo del techo
# SFactor son los factores escalares (tomando como 1.0 el sismo escalado al espectro según SpectrumFactor)

# Create full paths for the GM files
gm_files = ['GM01.txt', "GM02.txt", "GM03.txt", "GM04.txt", "GM05.txt", "GM06.txt", "GM07.txt", "GM08.txt", "GM09.txt", "GM10.txt", "GM11.txt", "GM12.txt", "GM13.txt", "GM14.txt", "GM15.txt",
         "GM16.txt", "GM17.txt", "GM18.txt", "GM19.txt", "GM20.txt", "GM21.txt", "GM22.txt", "GM23.txt", "GM24.txt", "GM25.txt", "GM26.txt", "GM27.txt", "GM28.txt", "GM29.txt", "GM30.txt",
          "GM31.txt", "GM32.txt", "GM33.txt", "GM34.txt", "GM35.txt", "GM36.txt", "GM37.txt", "GM38.txt", "GM39.txt", "GM40.txt", "GM41.txt", "GM42.txt", "GM43.txt", "GM44.txt"]

records = [os.path.join(script_dir, gm_file) for gm_file in gm_files] 
Nsteps= [3000, 3000, 2000, 2000, 5590, 5590, 4535, 4535, 9995, 9995, 7810, 7810, 4100, 4100, 4100, 4100, 5440, 5440, 6000, 6000, 2200, 2200, 11190, 11190, 7995, 7995, 7990, 7990, 2680, 2300, 8000, 8000, 2230, 2230, 1800, 1800, 18000, 18000, 18000, 18000, 2800, 2800, 7270, 7270]
DTs= [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.005,0.005,0.01,0.01,0.01,0.01,0.005,0.005,0.05,0.05,0.02,0.02,0.0025,0.0025,0.005,0.005,0.005,0.005,0.02,0.02,0.005,0.005,0.01,0.01,0.02,0.02,0.005,0.005,0.005,0.005,0.01,0.01,0.005,0.005]
GMcode = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
SFactor = [0.25,0.5,1.0,1.5,2.0,3.0,5.0] # input by the user
Sa_d = 0.5625 
T_scaling = 0.59 # periodo para el escalamiento
T = jb.load('T.pkl')
T.reshape(300,1)
GMSa = jb.load('Sa_FEMA.pkl')
gmt = [np.interp(T_scaling,T,GMSa[i]) for i in range(len(GMSa))]
SpectrumFactor = Sa_d/np.array(gmt)

#%% 

nrecords = len(DTs)
Arq1 = jb.load('model5-L.pkl')
#%% Creación de la función que realiza el análisis 
# ind es el número del registro, fact el factor escalar

# En donde dice DEFINICION DEL MODELO, se deberá definir el modelo HASTA las cargas de gravedad
# logFile('log.out','-append','-noEcho')
def rundyn(fact,ind):
    #logFile('log.out','-noEcho')
    rec = str(ind+1) # linea que crea el indice para nombrar el registro
    factor = 9.81*fact
    # CARGAR MODELO DE GRAVEDAD
    # ================================================
    nombre = str(int(factor/9.81*100))
    
    # DEFINICION DEL MODELO
    # ============================================
    
    # execfile('parametricgenerationMVLEM_IDA.py') # Aquí debe estar el modelo que aplica las cargas de gravedad
    ops.wipe()
    #
    # wipe() mientras no funcione bien el script es mejor tener el wipe antes
    # creación del modelo
    
    diafragma = 1  # 1 para diafragma rígido, 0 para vigas
    seccionViga = [0.3, 0.5]  # [base, altura] de las vigas si diafragma=0
    
    an.CrearModelo(Arq1, diafragma=diafragma, seccionViga=seccionViga)
  
    #%% ANALISIS
    # ========================

    an2.gravedad()
    ops.loadConst('-time',0.0)
    # DEFINICIÓN DE RECORDERS
    # ================================================
    
    # para que grabe cada uno distinto se debe usar el indice que se creo en rec. Por ejemplo, para el nodo de techo:
    # recorder('Node','-file',nombre+'/roof'+rec+'.out','-time','-node',*nodes,'-dof',1,'disp') 
    
    # EJECUCION DEL ANÁLISIS DINÁMICO
    # ================================================
    
    factoran = SpectrumFactor[ind]*factor
    node_tags = ops.getNodeTags()
    node_c = np.max(node_tags)
    
    t,techo = an2.dinamicoIDA2(records[ind], DTs[ind], Nsteps[ind], 0.04, factoran, 0.025, int(node_c), 1,[0,2],1,1e-4)
    ops.wipe()
    return ind,fact,t,techo

#%%
 

num_cores = multiprocessing.cpu_count() # esta linea identifica el número de nucleos totales del PC.
# En equipos con SMT identifica los núcleos físicos y lógicos. La recomendación si se va a seguir usando el PC es dejar dos núcleos físicos libres

stime = time.time()

# resultados devuelve de momento cuatro cosas. La primera es el indice del terremoto, la segunda es el factor escalar, la tercera el tiempo del registro y la cuarta es el desplazamiento de techo
resultados = Parallel(n_jobs=num_cores)(delayed(rundyn)(ff,pp) for ff in SFactor for pp in GMcode) # loop paralelo

etime = time.time()

ttotal = etime - stime

print('tiempo de ejecucion: ',ttotal,'segundos')

#%%
hbuilding = Arq1.listaCoorY[0]
ind, Sa, tmax, techomax = [],[],[],[]
for res in resultados:
    ind.append(res[0])
    Sa.append(res[1]*Sa_d)
    tmax.append(np.max(res[2]))
    techomax.append(np.max(np.abs(res[3])))
    
dic = {'GM':ind, 'Sa':Sa, 'tmax':tmax, 'dertecho':techomax/hbuilding*100}
df = pd.DataFrame(dic)
df.to_pickle('Building.pkl')

    