"""
Fecha: 01/08/2022
@authores: Frank y JuanJo
===================================================================
------------- Análisis PushOver estático no lineal ---------------
===================================================================

"""
#%% =============== IMPORTAR FUNCIONES, LIBRERIAS PARA LA EJECUCIÓN DEL MODELO ================
#==============================================================================================

#------Librerias Python
import os                                                                       # Importa librería estándar para interactuar con el sistema operativo.
import time                                                                     # Importa librerias estándar para manejar tareas relacionadas con el tiempo.
import shutil                                                                   # Importa librería estándar que proporciona la posibilidad de realizar la operación de archivos de alto nivel. Puede operar con el objeto de archivo y nos ofrece la posibilidad de copiar y eliminar los archivos.
import pickle
import sys                                                                      # Importa librería estándar para argumentos de línea de comandos.
import math                                                                     # Importa librería estándar para operaciones matemáticas.
import numpy as np                                                              # Importa librería estándar para operaciones matematicas.
import pandas as pd                                                             # Importa librería estándar de análisis de datos.
import multiprocessing                                                          # Importa librería que ofrece simultaneidad tanto local como remota, eludiendo efectivamente el bloqueo global del intérprete mediante el uso de subprocesos en lugar de subprocesos.
import matplotlib.pyplot as plt                                                 # Importa librería estándar para plotear otras cosasy poder crear figuras.
from joblib import Parallel, delayed                                            # Importa librería estándar que tiene un conjunto de herramientas para proporcionar una canalización ligera en Python
import subprocess

#------Librerias OpenSeesPy
import openseespy.opensees as ops                                               # Importa las librerias del software OpenSeesPy.
import opsvis as opsv                                                           # Importa librerias de. visualizador de OpenSeesPy opsvis para poder plotear la visualización del modelo
#------Librerias propias para los diferentes análisis
import Lib_materiales as mat                                                    # Importa librería creada para generar materiales para los modelos de estructurales.
import Lib_analisis as an                                                       # Importa librería creada para realizar analisis estructural de los modelos de estructurales.
from Lib_ClaseArquetipos import *

stime = time.time()
#%% RUTAS OUTPUT
#--------UNIDADES OUTPUTs:
# Deformaciones: milimetros (mm/mm)
# Derivas: porcentaje (%)
# Fuerzas: kiloNewton (kN)
# Momentos: kiloNewton-metro (kN-m)
# Deformaciones Unitarias: milimetros/milimetros (mm/mm)
# Aceleraciones: metros/segndocuadrado (m/s2)


#%% ======================== INFORMACIÓN DE ENTRADA PARA LA GENERACION ========================
#==============================================================================================

#--------------------- Datos de entrada para el analisis --------------------------------------
# Default values - can be overridden by command line arguments
#Definir si es "Longitudinal" o "Transversal"
DireccionAnalisis = "Longitudinal"                          # Direccion del arquetipo
pushlimit = 0.035                                           # Deriva Limite del Pushover
pushtype = 3                                                # Tipo de distribución de la carga para análisis PushOver. Donde: 1 para triangular, 2 para uniforme, 3 para proporcional al modo
modepush = 1                                                # Modo para las cargas del PushOver
wallctrl = 0                                                # Muro de control para el analisis PushOver: 0 para eje A, 1 para eje B.
Dincr = 0.001                                               # Incrementos de carga para el análisis PushOver

# Parse command line arguments if provided
# Usage: python 01_PUSHOVER.py model_name [pushlimit] [pushtype] [modepush] [wallctrl] [Dincr] [direction]
if len(sys.argv) > 2:
    try:
        pushlimit = float(sys.argv[2])
    except: pass
if len(sys.argv) > 3:
    try:
        pushtype = int(sys.argv[3])
    except: pass
if len(sys.argv) > 4:
    try:
        modepush = int(sys.argv[4])
    except: pass
if len(sys.argv) > 5:
    try:
        wallctrl = int(sys.argv[5])
    except: pass
if len(sys.argv) > 6:
    try:
        Dincr = float(sys.argv[6])
    except: pass
if len(sys.argv) > 7:
    try:
        DireccionAnalisis = sys.argv[7]
    except: pass

# Print the parameters being used
print('========== ANALYSIS PARAMETERS ==========')
print(f'Model Name: {sys.argv[1] if len(sys.argv) > 1 else "auto-detected"}')
print(f'pushlimit: {pushlimit}')
print(f'pushtype: {pushtype}')
print(f'modepush: {modepush}')
print(f'wallctrl: {wallctrl}')
print(f'Dincr: {Dincr}')
print(f'DireccionAnalisis: {DireccionAnalisis}')
print('=========================================')

record_PUSHOVER = "PO_"
#-----------------------------------------------------------------------------------------------


#%% ================================== INICIO DEL ANALISIS ====================================
#==============================================================================================

ruta = os.path.abspath(os.getcwd())                                                        # Obtiene el directorio actual
file_names_rec = os.listdir(ruta)

#--------- Validacion de la direccion del arquetipo
Dir = 'L'
if 't' in DireccionAnalisis.lower()[0]: Dir ="T"

# Check if model name was provided as command-line argument
if len(sys.argv) > 1:
    # Use the command-line argument as the model name
    # Extract just the basename if a path was provided
    NombreModelo = os.path.basename(sys.argv[1])
    # Find the corresponding 00_ file if it exists
    matching_files = [rec for rec in file_names_rec if np.logical_and(rec.startswith('00_'), rec.endswith(str(Dir) +'.py'))]
    modelname = matching_files[0] if matching_files else f"00_{NombreModelo}.py"
else:
    # Original behavior: search for 00_ file
    modelname = [rec for rec in file_names_rec if np.logical_and(rec.startswith('00_'), rec.endswith(str(Dir) +'.py'))][0]
    NombreModelo = modelname[3:-3]                                                  # Nombre del modelo a ejecutar

# Handle the case where the model file is in a subdirectory (e.g., models/)
ArchivoSerializado = NombreModelo + ".pkl"
# First try the provided path as-is
if len(sys.argv) > 1 and os.path.isfile(sys.argv[1] + ".pkl"):
    ruta_Serializado = sys.argv[1] + ".pkl"
else:
    # Otherwise look in current directory and subdirectories
    if os.path.isfile(ArchivoSerializado):
        ruta_Serializado = os.path.join(os.path.abspath(os.getcwd()), ArchivoSerializado)
    elif os.path.isfile(os.path.join("models", ArchivoSerializado)):
        ruta_Serializado = os.path.join(os.path.abspath(os.getcwd()), "models", ArchivoSerializado)
    else:
        ruta_Serializado = os.path.join(os.path.abspath(os.getcwd()), ArchivoSerializado)

if os.path.isfile(ruta_Serializado):
    with open(ruta_Serializado, "rb") as archivo:
        Arq1 = pickle.loads(archivo.read())
        
#--------- Disgregacion de info Taxonomia
n_arq = Arq1.nombre[:4]
tip_arq = Arq1.nombre[5:8]
ciudad_arq= Arq1.nombre[9:12]
npisos_arq = Arq1.nombre[13:15]
direccion_arq = Arq1.nombre[-1:]
name_arq = str(n_arq)+'_'+str(direccion_arq)

# Use simplified pushover/ folder structure
folder_PUSHOVER = os.path.join(ruta, "pushover")                               # Store all pushover results in pushover/ folder

# Create pushover folder if it doesn't exist
if not os.path.exists(folder_PUSHOVER):
    os.makedirs(folder_PUSHOVER)                                               # Create pushover folder

# Delete only the specific Excel file if it exists
excel_file = os.path.join(folder_PUSHOVER, str(NombreModelo)+'_pushover.xlsx')
if os.path.exists(excel_file):
    try:
        os.remove(excel_file)
    except:
        print(f"Warning: Could not delete {excel_file}. File will be overwritten.")

#%% ===================== GENERAR MODELO ARQUETIPO =====================
# ======================================================================
ops.wipe()

# Recrear el modelo usando la función CrearModelo del módulo Lib_analisis
# En lugar de ejecutar un archivo .py, usamos la función que crea el modelo desde el objeto Arquetipo
diafragma = 1  # 1 para diafragma rígido, 0 para vigas
seccionViga = [0.3, 0.5]  # [base, altura] de las vigas si diafragma=0

an.CrearModelo(Arq1, diafragma=diafragma, seccionViga=seccionViga)

# Obtener variables necesarias del modelo creado
NodosLibres = []
nodes_control = []
wallctrl_selected = wallctrl

for i, ele_i in enumerate(Arq1.muros):
    nnode = int(ele_i.id_*10)
    if i == 0:
        nodes_control.append(nnode)
    for j in range(Arq1.NumPisos-1,-1,-1):
        piso_i = ele_i.pisos[j]
        nnode = int(piso_i.id_)
        if i == 0:
            nodes_control.append(nnode)
            NodosLibres.append(nnode)

IDbaseNode = Arq1.muros[wallctrl_selected].pisos[-1].id_
IDctrlNode = Arq1.muros[wallctrl_selected].pisos[0].id_
IDctrlTecho = Arq1.muros[wallctrl_selected].pisos[0].id_
nodetags = ops.getNodeTags()
N_Nodes = len(NodosLibres)
IDctrlDOF = 1
n_pisos = len(NodosLibres)

#%% ========================= ANÁLISIS MODAL ===========================
# ======================================================================
eig = ops.eigen(1)                                                             # Determina el valor propio wn2 para el modo fundamental, este se requiere en el análisis pushover
T1 = 2*3.1416/np.sqrt(eig[0])                                                  # Determina el periodo para el modo fundamental

#%% ======================== ANALISIS DE GRAVEDAD ======================
# ======================================================================

an.gravedad_arquetipo()                                                                   # Función para análisis de gravedad del arquetipo del archivo analisis
ops.loadConst('-time',0.0)                                                          # Mantiene las cargas constantes al igual que los pasos luego del analisis de gravedad

#%% ========== FUERZAS EN LOS ELEMENTOS POR CARGA DE GRAVEDAD ==========
# ======================================================================

for i in range(len(Arq1.muros)):
    force = ops.eleForce(Arq1.muros[i].pisos[-1].id_)
    Reaccion = cReaccionesBase(force[1],force[0],force[2])
    Arq1.muros[i].agregar_Reacciones(Reaccion)

#%% ==================== PUSHOVER DEL ARQUETIPO ========================
# ======================================================================

#----------- Patron triangular proporcional a la altura
ylocation = np.array(Arq1.listaCoorY)
norm = np.sum(ylocation)                                                       # Suma las alturas de cada piso
forces = ylocation/norm                                                        # Normaliza cada coordenada Y por la suma total de las coordenadas

eigforce = np.zeros(Arq1.NumPisos)                                             # Predimensiona el vector para las "eigenfuerzas"

#----------- Patron modo escogido
for j in range(Arq1.NumPisos-1,-1,-1):
    eigforce[j] = ops.nodeEigenvector(Arq1.muros[0].pisos[j].id_, modepush, 1) # Obtiene la coordenada X del eigenvector deseado en el nodo de cada piso
norm2 = np.sum(eigforce)                                                       # Suma las "eigenfuerzas"
eigforces  = eigforce/norm2                                                    # Normaliza las "eigenfuerzas" dividiendo por su suma

#----------- Carga del pushover     
ops.timeSeries('Linear',2)                                                     # Crea serie de tiempo lineal 
ops.pattern('Plain',2,2)                                                       # Crea patron de carga Plain para el pushover

if pushtype == 1:
    for j in range(Arq1.NumPisos-1,-1,-1):
        ops.load(Arq1.muros[0].pisos[j].id_,forces[j],0.0,0.0)                 # Carga pushover triangular normalizado respecto a la altura
elif pushtype == 2:      
    for j in range(Arq1.NumPisos-1,-1,-1):
        ops.load(Arq1.muros[0].pisos[j].id_, 1/(Arq1.NumPisos),0.0,0.0)        # Carga pushover uniforme en altura para edificios altos
elif pushtype == 3:
    for j in range(Arq1.NumPisos-1,-1,-1):
        ops.load(Arq1.muros[0].pisos[j].id_, eigforces[j],0.0,0.0)             # Carga pushover respecto a los factores de participación modal en el modo fundamental

#================ ANALISIS PUSHOVER DEL MODELO ================
ht = Arq1.listaCoorY[0]
h1 = Arq1.listaCoorY[-1]
Dmax = pushlimit*Arq1.muros[0].pisos[0].CoorY
ARQ_R_PO = an.Pushover_Arquetipos_ADO(Dmax, Dincr, Arq1, NodosLibres, norm=[ht, Arq1.w_])

print('----------------------------------------------')
print('--------Análisis Push Over realizado----------')
print('----------------------------------------------')
#%% ==================== OUTPUT PUSHOVER ARQUETIPO ====================
# =====================================================================

#------- Resultados Analisis PushOver
Vbasal = ARQ_R_PO.ResultadosPushover.listaVbase
sDR_max = ARQ_R_PO.ResultadosPushover.listaSDR
SDR1 = ARQ_R_PO.ResultadosPushover.listaSDR1
RDR = ARQ_R_PO.ResultadosPushover.listaRDR

SDR1_PUSHOVER = pd.DataFrame({'SDR1': SDR1,'Vbasal': Vbasal }, dtype=float)
SDR_PUSHOVER = pd.DataFrame({'SDR':sDR_max,'Vbasal':Vbasal}, dtype=float)
RDR_PUSHOVER = pd.DataFrame({'RDR':RDR,'Vbasal':Vbasal}, dtype=float)

for i in ARQ_R_PO.muros:
    SDR1_PUSHOVER.insert(loc = len(SDR1_PUSHOVER.columns), column = i.nombre, value = -np.array(i.pisos[-1].ResultadosAltura.list_V))
    SDR_PUSHOVER.insert(loc = len(SDR_PUSHOVER.columns), column = i.nombre, value = -np.array(i.pisos[-1].ResultadosAltura.list_V))
    RDR_PUSHOVER.insert(loc = len(RDR_PUSHOVER.columns), column = i.nombre, value = -np.array(i.pisos[-1].ResultadosAltura.list_V))
    
Data_PUSHOVER = (SDR1_PUSHOVER, SDR_PUSHOVER, RDR_PUSHOVER)
excel_filename = os.path.join(folder_PUSHOVER, str(NombreModelo)+'_pushover.xlsx')
print(f'Saving Excel results to: {excel_filename}')
with pd.ExcelWriter(excel_filename) as writer:
    for i in range(len(Data_PUSHOVER)):
        Data_PUSHOVER[i].to_excel(writer, sheet_name=Data_PUSHOVER[i].columns[0])
print(f'Excel file saved successfully')

#%% ==== GUARDAR LA VARIABLE ARQ1 (ARQUETIPO) CON LA LOS RESULTADOS DEL PUSHOVER =====
# ====================================================================================

ArchivoResultadosPushover = NombreModelo + "-RPO.pkl"
ruta_ResultadosPushover = os.path.join(folder_PUSHOVER, ArchivoResultadosPushover)

# Always save/overwrite the RPO file
with open(ruta_ResultadosPushover, "wb") as archivo:
    pickle.dump(ARQ_R_PO, archivo)
archivo.close()

print(f'Results saved to: {ruta_ResultadosPushover}')


#%% ==================== CONTADOR TIEMPO DE EJECUCION =================
# =====================================================================
etime = time.time()
ttotal = (etime - stime)/60
print('---------------Tiempo Análisis----------------')
print('---------------'+str(round(ttotal,3))+' min ----------------')




