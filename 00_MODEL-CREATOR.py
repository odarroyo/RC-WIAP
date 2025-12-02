# -*- coding: utf-8 -*-
"""

Created on Tue Feb 28 21:41:01 2023
@authores: Frank y JuanJo
====================================================================================
============= ENERADOR DE EDIFICIOS DE MUROS DE CON CONCRETO REFORZADO =============
====================================================================================

"""

#%% =============== IMPORTAR FUNCIONES, LIBRERIAS PARA LA EJECUCIÓN DEL MODELO ================
#==============================================================================================

#------ Librerias de python
import openseespy.opensees as ops                                               # Importa las librerias del software OpenSeesPy.
import opsvis as opsv                                                           # Importa librerias de. visualizador de OpenSeesPy opsvis para poder plotear la visualización del modelo.
import matplotlib.pyplot as plt                                                 # Importa librería estándar para plotear otras cosasy poder crear figuras.
import math
from enum import Enum
import pandas as pd                                                             # Importa librería estándar de análisis de datos.
import numpy as np                                                              # Importa librería estándar para operaciones matematicas.
import os                                                                       # Importa librería estándar para interactuar con el sistema operativo.
from pathlib import Path
import random
import pickle
import sys

#------ Librerias propias para los diferentes análisis
import Lib_materiales as mat                                                    # Importa librería creada para generar materiales para los modelos de estructurales.
import Lib_analisis as an                                                       # Importa librería creada para realizar analisis estructural de los modelos de estructurales.
from Lib_ClaseArquetipos import *                                               # Importar la documentacion de clases

#%% ================================ NOTAS IMPORTANTES ================================
"""
==================================================================================================================================
============ EL ARCHIVO 00 DEBE SER NOMBRADO CON LA TAXONOMIA DEL ARQUETIPO ===========
============ FORMATO DEL NOMBRE ==> 00_0000-MCR-XXX-XXP
============                           0000 ->ID del arquetipo en la base de datos, numero de 4 digitos
============                           MCR  ->Taxonomia del sistema estructural, Muros de Concreto Reforzado
============                           XXX  ->Abreviatura de la ciudad del arquetipo, ejemplo: Bogota --> "BGT"
============                           XXP ->Numero de pisos del arquetipo en formato de 2 digitos y P para indicar que son pisos.
==================================================================================================================================
"""

#%% ======================== INFORMACIÓN DE ENTRADA PARA LA GENERACION ========================
#==============================================================================================

#--------------------- Datos de entrada para la generacion de un modelo
carpetaInfoMuros = "Data"                           # Nombre de la carpeta en donde se encuentran los archvos de los muros en Excel
DireccionAnalisis = "Longitudinal"                          # Direccion del arquetipo

# Si el modelo tiene diafragma en cada piso (diafragma = 1) no es necesario que modifique las variables (bv, hv), 
# Ya que en este caso el generador no requiere estas variables.

diafragma = 1                                               # Colocar 1 si se desea diafragma en cada piso : 2, si se trata de modelo con vigas
# Si el modelo en lugar de diagragma en cada piso se modela con vigas (diafragma=0), 
# deberá definir las dimensiones de las vigas.
bv = 0.3                                                                       # Dimensión de la base de la viga
hv = 0.5                                                                       # Dimensión de la altura de la viga


#%% ================================== GENERACION DEL MODELO ==================================
#==============================================================================================
#---------------------------------NO CAMBIAR NADA DE ACA PARA ABAJO----------------------------

#--------------------- Validacion de la direccion del arquetipo
Dir = 'L'
if 't' in DireccionAnalisis.lower()[0]: Dir ="T"

#--------------------- Obtener la ruta del modelo y con base en la informacion de la carpeta generar los nombres del modelo
ruta = os.path.abspath(os.getcwd())                         # Obtener la ruta de la carpeta
folder_output = os.path.join(ruta, carpetaInfoMuros)        # Obtiene la ruta de la carpeta de la informacion de los muros

# Obtener el nombre del modelo: desde argumentos de línea de comando o desde el nombre del archivo
if len(sys.argv) > 1:
    # Si se proporciona un nombre de modelo como argumento
    NombreModelo = sys.argv[1]
else:
    # Obtener el nombre del modelo desde el nombre del archivo actual
    script_name = os.path.basename(__file__)                    # Obtener el nombre del archivo actual (ej: 00_MODEL-CREATOR.py)
    NombreModelo = script_name[3:-3]                            # Extraer el nombre del modelo (ej: MODEL-CREATOR)

ArchivoSerializado = NombreModelo + ".pkl"
ruta_Serializado = os.path.join(os.path.abspath(os.getcwd()), ArchivoSerializado)
DireccionArquetipo = eDireccion.Longitudinal

if NombreModelo[-1] != 'L':
    DireccionArquetipo = eDireccion.Transversal

#--------------------- Leer el numero de pisos desde Info.xlsx
ruta_Info = os.path.join(folder_output, "Info.xlsx")
Alturas_temp = pd.read_excel(ruta_Info, sheet_name="Hn")
npisos_arq = len(Alturas_temp)

#%% ================== GENERACION DE OBJETO CON LA INFORMACION DEL OBJETO ARQUETIPO, CASO CONTRARIO LO GENERA =================
#==============================================================================================================================

if os.path.isfile(ruta_Serializado):
    with open(ruta_Serializado, "rb") as f:
        serialized_person = f.read()
        Arq1 = pickle.loads(serialized_person) # Deserealizar la variable arquetipo con la informacion de la estructura
        print(ArchivoSerializado)
else:
    Arq1 = Arquetipo(nombre = NombreModelo, direccion = DireccionArquetipo, 
                     NumPisos = npisos_arq, w_ = 0,ws_ = 0)                     # Crear un objeto de clase Arquetipo
    Steel_WWM = cMatSteel(nombre = "Malla", iTag = 50, fy = 490)                # Crear el material "Malla electrosoldad"
    Steel_RB = cMatSteel(nombre = "Barra", iTag = 60, fy = 420)                 # Crear el material "Barras ductiles"
    
    Arq1.agregar_MatSteel(Steel_WWM)                                            # Agregar el material de Malla al objeto
    Arq1.agregar_MatSteel(Steel_RB)                                             # Agregar el material de Barras al objeto
    
    ruta_Info = os.path.join(folder_output, "Info.xlsx")            # Archivo de excel con la informacion general del edificio
    
    dw_ = pd.read_excel(ruta_Info, sheet_name="w")                  # Hoja de excel con la masa por piso
    dws_ = pd.read_excel(ruta_Info, sheet_name="ws")                # Hoja de excel con la masa sismica por piso
    fc_ = pd.read_excel(ruta_Info, sheet_name="fc")                 # Hoja de excel con la resistencia a la compresion f'c por piso
    Alturas_ = pd.read_excel(ruta_Info, sheet_name="Hn")            # Hoja de excel con las alturas de cada piso
        
    # Sort files to ensure correct order (00_, 01_, 02_, etc.)
    # Use numerical sorting on the prefix to handle files correctly
    archivos_temp = [f for f in os.listdir(folder_output) if os.path.splitext(f)[0].lower() != "info"]
    archivos_ordenados = sorted(archivos_temp, key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 999)
    
    for ind, archivo in enumerate(archivos_ordenados):
        if os.path.splitext(archivo)[0].lower() != "info":        
            ruta_Ar = os.path.join(folder_output, archivo)    
            
            Num_Macros = pd.read_excel(ruta_Ar, sheet_name="NumMacro")
            Anchos = pd.read_excel(ruta_Ar, sheet_name="Anchos")
            Espesores = pd.read_excel(ruta_Ar, sheet_name="Espesor")
            Concretos = pd.read_excel(ruta_Ar, sheet_name="Concreto")
            Aceros = pd.read_excel(ruta_Ar, sheet_name="Acero")
            Cuantias = pd.read_excel(ruta_Ar, sheet_name="Cuantia")
                
            Muro1 = cMuro(os.path.splitext(archivo)[0][3:], (100*(ind+1)), 3.5, 0.1, w_ = 0,ws_ = 0)
            
            for i in range(Num_Macros.Piso.count()):   
                #------------------- Generacion del objeto seccion simplificada de muro
                Muro_ = cSeccionMuro(id_ = i*1000, 
                                     nombre = os.path.splitext(archivo)[0][3:], 
                                     listaMacrofibras = Num_Macros.iloc[i, 1:].values, 
                                     listaAncho = Anchos.iloc[i, 1:].values, 
                                     listaEspesor = Espesores.iloc[i, 1:].values, 
                                     listaCuantia = Cuantias.iloc[i, 1:].values, 
                                     listaTipoConcreto = Concretos.iloc[i, 1:].values, 
                                     listaTipoAcero = Aceros.iloc[i, 1:].values)

                #------------------- Generacion del objeto seccion de muro con la informacion requerida por el MVLEM               
                Muro_Md =  cSeccionMuro(id_ = i*1000, 
                                     nombre = os.path.splitext(archivo)[0][3:]+str("_Md"), 
                                     listaMacrofibras = [], 
                                     listaAncho = [], 
                                     listaEspesor = [], 
                                     listaCuantia = [], 
                                     listaTipoConcreto = [], 
                                     listaTipoAcero = [])
               
                #------------------- Agregar las propiedades de los diferentes concretos que se tienen en el proyecto
                if len(Arq1.MatConcrete) == 0:
                    nombre_C = str(fc_.iloc[i, 1]) + "MPa"
                    val_Aleatorio = random.randint(1,20)
                    Mat_C = cMatConcrete(nombre_C, val_Aleatorio, int(val_Aleatorio + 20), fc_.iloc[i, 1])
                    
                    Arq1.agregar_MatConcrete(Mat_C)
                
                else:
                    listfc_ = list(set([getattr(cMatConcrete,'fc') for cMatConcrete in Arq1.MatConcrete]))
                    if fc_.iloc[i, 1] not in listfc_:
                        nombre_C = str(fc_.iloc[i, 1]) + "MPa"
                        lista_tag = set(range(1,20))
                        tag_dif = list(lista_tag.difference(list(set([getattr(cMatConcrete,'iTag') for cMatConcrete in Arq1.MatConcrete]))))
                        val_Aleatorio = random.choice(tag_dif)
                        Mat_C = cMatConcrete(nombre_C, val_Aleatorio, int(val_Aleatorio + 20), fc_.iloc[i, 1])
                        
                        Arq1.agregar_MatConcrete(Mat_C)
                
                for j, k in enumerate(Muro_.listaMacrofibras):
                    if not math.isnan(k):
                        for l in range(int(k)):
                            
                #------------------- Generacion del objeto seccion de muro con la informacion requerida por el MVLEM                                  
                            Muro_Md.listaAncho.append(round(Muro_.listaAncho[j]/k,3))
                            Muro_Md.listaEspesor.append(Muro_.listaEspesor[j])
                            Muro_Md.listaCuantia.append(Muro_.listaCuantia[j])
                            Muro_Md.listaTipoConcreto.append(Muro_.listaTipoConcreto[j])
                            Muro_Md.listaTipoAcero.append(Muro_.listaTipoAcero[j])
                            mat_an = next(filter(lambda p: p.fc == fc_.iloc[i, 1], Arq1.MatConcrete), None)
                            if Muro_.listaTipoConcreto[j].lower() == "unconf":
                                Muro_Md.listaTagConcret.append(mat_an.iTagUnc)
                            else:
                                Muro_Md.listaTagConcret.append(mat_an.iTag)
                                
                            if Muro_.listaTipoAcero[j].lower() == "wwm":
                                Muro_Md.listaTagSteel.append(Steel_WWM.iTag)
                            else:
                                Muro_Md.listaTagSteel.append(Steel_RB.iTag)
                
                #------------------- Agregar la informacion en cada piso del muro i                                
                P01 = cPiso((1000*(ind+1)+(Arq1.NumPisos-i)), "P"+str(int(Num_Macros.iloc[i, 0])), 
                            Muro_, Muro_Md, 2.5, fc = fc_.iloc[i, 1], CoorY = round(Alturas_.iloc[i, 1],2), 
                            w_ = dw_[Muro_.nombre][i], ws_= dws_[Muro_.nombre][i])
                
                Muro1.agregar_piso(P01)
                if P01.CoorY not in Arq1.listaCoorY:
                    Arq1.agregar_listaCoorY(P01.CoorY)
            
            Muro1.w_ = sum(np.array(list([getattr(cPiso,'w_') for cPiso in Muro1.pisos])))
            Muro1.ws_ = sum(np.array(list([getattr(cPiso,'ws_') for cPiso in Muro1.pisos])))
            Arq1.agregar_muros(Muro1)
    
    Arq1.w_ = sum(np.array(list([getattr(cMuro,'w_') for cMuro in Arq1.muros])))
    Arq1.ws_ = sum(np.array(list([getattr(cMuro,'ws_') for cMuro in Arq1.muros])))
            

#%% ======================== MODELACIÓN ARQUETIPO EN OPENSEES ========================
#=====================================================================================

#-------- Definición de la geometría de la viga de acople o conexión entre muros.
vigas = np.zeros(int(len(Arq1.muros)-1)).tolist()                              # Se deberá indicar 1 si entre un par de muros hay viga y 0 si se considera EqualDOF.

#-------- Vigas del modelo representan diafragma rigido entre muros.
Av = bv*hv                                                                     # Área de la viga
Iv = bv*hv**3/12                                                               # Inercia de la viga
Ev = 4700000*math.sqrt(21)                                                     # Modulo de elasticidad del concreto para la viga

#%% ======================== GENERADOR ARQUETIPO ========================
#========================================================================
ndf = 3                                                                        # Grados de libertad nodales
ops.wipe()                                                                     # Borra información de modelos existentes (Mientras no funcione bien el script es mejor tener el wipe).
ops.model('basic','-ndm',2,'-ndf',3)                                           # Crea modelo en 2D con 3 grados de libertad.

#=============== GENERADOR DE VECTORES Y MATRICES DE MASAS ===============


m_x = []
ms_x = []

for i in range(len(Arq1.muros)):
    lisp = []
    lisp_s = []
    lisp.append(0)
    lisp_s.append(0)
    for j in range(len(Arq1.muros[i].pisos)):
        lisp.append(Arq1.muros[i].pisos[j].w_)
        lisp_s.append(Arq1.muros[i].pisos[j].w_)
    m_x.append(lisp)                                                            #-------- MASA GRAVITACIONAL                            
    ms_x.append(lisp_s)                                                         #-------- MASA SISMICA
    
m_piso = np.sum(m_x,axis=0)
ms_piso = np.sum(ms_x,axis=0)
mat_Masa = np.diagflat(np.delete(m_piso, 0))
mat_Masas = np.diagflat(np.delete(ms_piso, 0))

#%% ======================== NODOS, RESTRICCIONES Y ASIGNACIÓN DE MASA ========================
# =============================================================================================

#-----Definición de matriz de rigidez
NodosLibres, nodes_control, node_record = [], [], []

ny = Arq1.NumPisos                                                                  # Número de nodos en dirección Y para un solo muro. 'Numero de pisos del edificio'
xloc = np.linspace(0,len(Arq1.muros)*5,len(Arq1.muros)).tolist()
nx = len(Arq1.muros)                                                                  # Número de nodos en dirección X por piso. 'Numero de muros del edificio'

# ------ Asignación de carga
ops.timeSeries('Linear', 1)                                                     # Definición de tipo de serie de carga para asignar al edificio.
ops.pattern('Plain', 1, 1)                                                      # Definición de patron de carga para asignar al edificio.
wallctrl = 0
for i, ele_i in enumerate(Arq1.muros):                                                             # Ciclo for para asignación de masa (toneladas) y fuerzas de gravedad (kN).
    nnode = int(ele_i.id_*10)
    ops.node(nnode, float(xloc[i]), 0.0)
    if i==0:
        nodes_control.append(nnode)
    for j in range(ny-1,-1,-1):
        piso_i = ele_i.pisos[j]
        nnode = int(piso_i.id_)
        ops.node(nnode, float(xloc[i]), float(piso_i.CoorY))
        # Convert seismic weight to mass: mass = weight / g (ensure float values)
        masa_sismica = float(piso_i.ws_) / 9.81
        ops.mass(nnode, masa_sismica, masa_sismica, 0.0)
        # Gravity load is -w_ (weight in kN, already includes gravity)
        ops.load(nnode, 0.0, float(-piso_i.w_), 0.0)
        if i==0:
            nodes_control.append(nnode)
            NodosLibres.append(nnode)
            
IDbaseNode = Arq1.muros[wallctrl].pisos[-1].id_                      # Definición de nodo en la base.
IDctrlNode = Arq1.muros[wallctrl].pisos[0].id_                       # Definición de nodo de control.    
IDctrlTecho = Arq1.muros[wallctrl].pisos[0].id_                      # Definición de nodo en el techo.

nodetags = ops.getNodeTags()                                                    # Definición de Tags asociados a los nodos del edificio.
N_Nodes = len(NodosLibres)                                                      # Definición del numero de modos de vibracion a analizar.

IDctrlDOF = 1
n_pisos = len(NodosLibres)                                                      # Definición del numero de pisos del arquetipo.

print('-------------------------------')
print('--------Nodos generados--------')
print('-------------------------------')
# apoyos
empotrado = [1,1,1]                                                             # Definición de grados de libertad empotrados.
grado2 = [1,1,0]                                                                # Definición de grados de libertad articulados.

# para colocarlos todos al nivel 0.0
ops.fixY(0.0,*empotrado)                                                        # Definición de restricción en la base del edificio.
print('-------------------------------')
print('----Restricciones asignadas----')
print('-------------------------------')

#%% ======================== ASIGNACIÓN DE DIAFRAGMAS ========================
# ============================================================================
if diafragma == 1:
    for j in range(ny):
        for i in range(1,nx):
            masternode = Arq1.muros[0].pisos[j].id_
            slavenode = Arq1.muros[i].pisos[j].id_
            ops.equalDOF(masternode, slavenode, 1)

print('-------------------------------')
print('------Diafragmas asignados-----')
print('-------------------------------')

#%% ======================== MATERIALES Y TRANSFORMACIONES ========================
# =================================================================================

# ---- Concreto
for i in range(len(Arq1.MatConcrete)):
    mat.Concreto('Unconf', int(Arq1.MatConcrete[i].iTagUnc), float(Arq1.MatConcrete[i].fc),'C01')                            #Definición del Concreto sin confinar a partir de la función Materials
    mat.Concreto('Conf', int(Arq1.MatConcrete[i].iTag), float(Arq1.MatConcrete[i].fc), 'C01')                            #Definición del Concreto sin confinar a partir de la función Materials

# ---- Acero de refuerzo
mat.Refuerzo('WWM', int(Arq1.MatSteel[0].iTag), float(Arq1.MatSteel[0].fy), 'HYS')      #Definición del acero de refuerzo con malla electrosoldada a partir de la función Materials
mat.Refuerzo('RB', int(Arq1.MatSteel[1].iTag), float(Arq1.MatSteel[1].fy), 'HYS')      #Definición del acero de refuerzo con malla electrosoldada a partir de la función Materials

#%% ======================== TRANSFORMACIÓN PARA ANÁLISIS =========================
# =================================================================================

lineal = 1
ops.geomTransf('Linear',lineal)                                                 # Definición de la transformación para analisis Lineal

pdelta = 2
ops.geomTransf('PDelta',pdelta)                                                 # Definición de la transformación para analisis P-Delta

cor = 3
ops.geomTransf('Corotational',cor)                                              # Definición de la transformación para analisis Corotational

#%% ======================== ELEMENTOS VERTICALES (MUROS) ========================
# ========================
#element('MVLEM', eleTag, Dens, *eleNodes, m, c, '-thick', *thick, '-width', *widths, '-rho', *rho, '-matConcrete', *matConcreteTags, '-matSteel', *matSteelTags, '-matShear', matShearTag)

electrl, ele_record = [], []
cMVLEM = 0.4

for i, ele_i in enumerate(Arq1.muros):                                                             # Definición del elemento (MVLEM) de análisis a partir de la sección creada para el edificio.
    for j in range(ny-1,-1,-1):
        if j == Arq1.NumPisos-1:
            nodeI = ele_i.id_*10
            nodeJ = ele_i.pisos[j].id_
        else:
            nodeI = ele_i.pisos[j+1].id_
            nodeJ = ele_i.pisos[j].id_
            
        piso_i = ele_i.pisos[j]
        eltag = piso_i.id_        
        nfib = len(piso_i.muro_Md.listaAncho)
        # ------- Definición del resorte de corte
        Ec = 4300000*math.sqrt(piso_i.fc)                                                      # Definicion del modulo de elasticidad del concreto de la sección del muro.
        G = Ec*0.4                                                                      # Definicion del modulo de rigidez a corte de la sección del muro.
        tagshear = 1500*(i+1)+j      
        ops.uniaxialMaterial('Elastic',tagshear,G*1.5)                                      # Definición del material asignado al resorte de corte ubicado a la altura ch del elemento.
        
        ops.element('MVLEM', eltag, 0.0, nodeI, nodeJ, nfib, cMVLEM, '-thick', *piso_i.muro_Md.listaEspesor, 
                    '-width', *piso_i.muro_Md.listaAncho, '-rho', *piso_i.muro_Md.listaCuantia, 
                    '-matConcrete', *piso_i.muro_Md.listaTagConcret, '-matSteel', *piso_i.muro_Md.listaTagSteel, 
                    '-matShear',tagshear)          

print('-------------------------------')
print('--------Muros generados--------')
print('-------------------------------')

ele_record =[]
for i in range(len(Arq1.muros)):    
    ele_record.append(Arq1.muros[i].pisos[-1].id_)


#%% ======================================= ELEMENTOS HORIZONTALES (VIGAS) ======================================
#================================================================================================================

if diafragma != 1:    
    tagvigas=[]
    for j in range(ny-1,-1,-1):
        for i in range(len(Arq1.muros)): 
            if i<(len(Arq1.muros)-1):
                nodeI = Arq1.muros[i].pisos[j].id_                              # nodo inicial de la viga
                nodeJ = Arq1.muros[i+1].pisos[j].id_                            # nodo final de la viga
                eltag = int(Arq1.muros[i].pisos[j].id_*10)                              # tag de la viga
                tagvigas.append(eltag)                                          # guarda los tags de las vigas
                ops.element('elasticBeamColumn',eltag,nodeI,nodeJ,Av,Ev,Iv,lineal)
    
    print('-------------------------------')
    print('--------Vigas generadas--------')
    print('-------------------------------')
        

plt.figure()
opsv.plot_model()
# Save the plot as an image file
plot_filename = f"{NombreModelo}_model.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"Model plot saved to: {plot_filename}")
plt.close()

#%% =============== GUARDAR LA VARIABLE ARQ1 (ARQUETIPO)  CON LA INFORMACION DE TODA LA ESTRUCTURA =============== 
#=================================================================================================================

ruta_Serializado = os.path.join(os.path.abspath(os.getcwd()), ArchivoSerializado)
if not os.path.isfile(ruta_Serializado):
    with open(ArchivoSerializado, "wb") as archivo:
        pickle.dump(Arq1, archivo)
    archivo.close()

