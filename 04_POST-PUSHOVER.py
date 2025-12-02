"""
Fecha: 01/08/2022
@authores: Frank y JuanJo
=============================================================================
---------- POST PROCESAMIENTO DEL ANÁLISIS PUSHOVER ARQUETIPO --------------
=============================================================================

"""
#%% ======================== IMPORTAR FUNCIONES, LIBRERIAS PARA LA EJECUCIÓN DEL MODELO ========================
#==============================================================================================

#------Librerias Python
import os                                                                       # Importa librería estándar para interactuar con el sistema operativo.
import time                                                                     # Importa librerias estándar para manejar tareas relacionadas con el tiempo.
import shutil                                                                   # Importa librería estándar que proporciona la posibilidad de realizar la operación de archivos de alto nivel. Puede operar con el objeto de archivo y nos ofrece la posibilidad de copiar y eliminar los archivos.
import pickle
import warnings
import numpy as np                                                              # Importa librería estándar para operaciones matematicas.
import pandas as pd                                                             # Importa librería estándar de análisis de datos.
import seaborn as sns                                                           # 
from matplotlib import style
from scipy import interpolate                                          # Importa librería estándar para plotear otras cosasy poder crear figuras.

#------Librerias OpenSeesPy
import openseespy.opensees as ops                                               # Importa las librerias del software OpenSeesPy.
import opsvis as opsv                                                           # Importa librerias de. visualizador de OpenSeesPy opsvis para poder plotear la visualización del modelo

#------Librerias propias para los diferentes análisis
import Lib_analisis as an                                                          # Importa librería creada para realizar analisis estructural de los modelos de estructurales.

# Configuración warnings
# ==============================================================================
warnings.filterwarnings('ignore')
pd.set_option("display.precision", 5)               #En resultados con precisión de n decimales

# Contador de tiempo
stime = time.time()

#%% ======================== INFORMACIÓN DE ENTRADA PARA LA GENERACION ========================
#==============================================================================================

#--------------------- Datos de entrada para el analisis---------------------------------
#Definir si es "Longitudinal" o "Transversal"
DireccionAnalisis = "Longitudinal"                     # Direccion del arquetipo
record_PUSHOVER = "PO_"
# %%NO TOCAR IMPORTACION DE DATOS PUSHOVER 
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ruta = os.path.abspath(os.getcwd())                                                        # Obtiene el directorio actual
file_names_rec = os.listdir(ruta)

#--------- Validacion de la direccion del arquetipo
Dir = 'L'
if 't' in DireccionAnalisis.lower()[0]: Dir ="T"
modelname = [rec for rec in file_names_rec if np.logical_and(rec.startswith('00_'), rec.endswith(str(Dir) +'.py'))][0]
NombreModelo = modelname[3:-3]                             # Nombre del modelo a ejecutar
ArchivoSerializado = NombreModelo + "-RPO.pkl"
ruta_Serializado = os.path.join(os.path.abspath(os.getcwd()), ArchivoSerializado)

if os.path.isfile(ruta_Serializado):
    with open(ruta_Serializado, "rb") as archivo:
        ARQ_R_PO = pickle.loads(archivo.read())
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# %% VARIABLES DE ANALISIS
#Definir muros a analizar, para obtener la lista de muros del modelo debe usar la siguiente linea con lista:
    # listamuros = [i.nombre for index, i in enumerate(ARQ_R_PO.muros)]
    # al ejecutar obtiene lo siguiente: 
    # listamuros = ['M03','M42','M59','M57','M86','M88','M90','M122','M120','M139','M177']
listamuros = [i.nombre for index, i in enumerate(ARQ_R_PO.muros)]
#--------------------- Datos de entrada para el analisis --------------------------------------
#Definir la deriva de analisis como:
    #SDR1_max: Deriva maxima del primer nivel
    #SDR_max: Deriva maxima del primer nivel
    #RDR_max: Deriva maxima del primer nivel
    
EDP_analisis = 'SDR_max'
edp_label = EDP_analisis[:-4].upper()
#Definir la metodologia y limites para evaluar el desempeño de los resultados del CNL
    #Metodologia con limites DETERMINISTICOS: 'DET'
    #Metodologia con limites PROBABILISTICOS: 'PROB'
metodo = 'DET'
# Los umbrales de daño de referencia que se encuentran en el archivo DS.xlxs, son los siguientes:
#-------------------------------------------------------------------------------
#   Propuesta Carrillo et al. 2022
    #Umbrales probabilistas:
        #RB (Barras Refuerzo Corrugado):
            #name_sheet_ds = 'CARRILLO_PROB_RB'
        #WWM (Mallas elesctrosoldadas): 
            #name_sheet_ds = 'CARRILLO_PROB_WWM'
    #Umbrales deterministas:
        #RB (Barras Refuerzo Corrugado):
            #name_sheet_ds = 'CARRILLO_DET_SDR_RB' o 'CARRILLO_DET_RDR_RB'
        #WWM (Mallas elesctrosoldadas):
            #name_sheet_ds = 'CARRILLO_DET_SDR_WWM' o 'CARRILLO_DET_RDR_WWM'
#-------------------------------------------------------------------------------
# Adicionalmente se permite evaluar DSi cuantos desee.
    #Metodo determinista:
        #name_sheet_ds = 'DS_DET'
    #Metodo probabilista:
        #name_sheet_ds = 'DS_PROB'
#-------------------------------------------------------------------------------
name_sheet_ds = 'DS_DET'

#Definir el material para el cual se evaluaran los limites de desempeño de los resultados del CNL
    #Barras corrugadas: 'RB'
    #Malla electrosoldada: 'WWM'
tipo_ref = 'WWM'

# Opciones Graficos
# Parámetros Locales
GraficoMomentosMuros = True
GraficoCortantesMuros = True
GraficoPerfilDeriva = True
PasodePerfilDeriva = 3                          # (1): Agrietamiento, (2): Primera Fluencia, (3): Punto pico del pushover, (4): Punto de capacidad ultima (Perdida del 20% de la capacidad)

#----------------------------------------------------------------------------------------
#%% ================================== INICIO DEL ANALISIS ====================================
#==============================================================================================

#--------- Disgregacion de info Taxonomia
n_arq = ARQ_R_PO.nombre[:4]
tip_arq = ARQ_R_PO.nombre[5:8]
ciudad_arq= ARQ_R_PO.nombre[9:12]
npisos_arq = ARQ_R_PO.nombre[13:15]
direccion_arq = ARQ_R_PO.nombre[-1:]
name_arq = str(n_arq)+'_'+str(direccion_arq)

folder_output = os.path.join(ruta, name_arq)                                                                     # Obtiene ruta dentro del directorio actual con nombre del modelo del arquetipo.
folder_PUSHOVER = os.path.join(folder_output, record_PUSHOVER+str(name_arq))                  # Obtiene ruta dentro del directorio actual en la carpeta del almacenamiento de los resultados del analisis pushover.

#%% ==================================  GENERAR MODELO ARQUETIPO ====================================
#====================================================================================================

execfile(modelname, locals())                                        # Ejecuta archivo donde se crea el modelo de análisis

#%% ANÁLISIS MODAL
# =============================================================================
eig = ops.eigen(1)                                                              # Determina el valor propio wn2 para el modo fundamental, este se requiere en el análisis pushover
T1 = 2*3.1416/np.sqrt(eig[0])                                                   # Determina el periodo para el modo fundamental

FormModals, Result_MA = an.Modal(mat_Masa, NodosLibres, N_Nodes, ndf)           # Función para análisis modal del arquetipo del archivo SDOF (ModoFund: imprime los parametros modales para el modo fundamental, 
                                                                                # y Result_MA: Imprime los resultados del análisis modal)
Modal = pd.DataFrame({'M*':Result_MA['M*'], 'Tn':Result_MA['Tn'],
                      'Ln':Result_MA['Ln'], 'Mn':Result_MA['Mn'],
                      'Gamman':Result_MA['Gamman']}, index=Result_MA['Modo'], dtype=float)
Modal.index.name = 'Modos'
Modal.to_excel(folder_PUSHOVER +'\\'+str(name_arq)+'_modal.xlsx')

#%% ==================================  ANÁLISIS DE GRAVEDAD ====================================
#================================================================================================

# ===============================================================================================
an.gravedad_arquetipo()                                                         # Función para análisis de gravedad del arquetipo del archivo analisis
ops.loadConst('-time',0.0)                                                      # Mantiene las cargas constantes al igual que los pasos luego del analisis de gravedad
print('----------------------------------------------')
print('--------Análisis de gravedad realizado--------')
print('----------------------------------------------')

#%% ==================================  PROCESAMIENTO PUNTOS DE CONTROL ====================================
#===========================================================================================================

# Puntos de control por muro en la base:
    #[0]: Agrietamiento en cada macrofibra.
    #[1]: Fluencia de las barras/malla en cada macrofibra.
    #[2]: Rotura de la barra/malla en cada macrofibra.
    #[3]: Falla por compresión del concreto en cada macrofibra.
    #[4]: Pandeo de la barra/malla por compresion.
    
paso_Cr = [] ; paso_RBy = [] ; paso_WWMy = [] ; paso_RB_Ult = [] ; paso_WWM_Ult = [] ; paso_RB_50Ult = []; paso_WWM_50Ult = []
Def_Cr = 0.00015 ; Def_RBy = 420/200000 ; Def_WWMy = 490/200000 ; Def_RB_Ult = 0.05 ; Def_WWM_Ult = 0.015
NumPaso = len(ARQ_R_PO.muros[0].pisos[-1].ResultadosAltura.list_Strain)

for ele_i in ARQ_R_PO.muros:
    minDiff_Cr = 99999 ;  minDiff_RBy = 99999 ; minDiff_WWMy = 99999 ; minDiff_RB_Ult = 99999 ; minDiff_WWM_Ult = 99999 ; minDiff_RB_50Ult = 99999 ; minDiff_WWM_50Ult = 99999
    i_Cr = 0 ; i_RBy = 0 ; i_WWMy = 0 ; i_RB_Ult = NumPaso-1 ; i_WWM_Ult = NumPaso-1 ; i_RB_50Ult = NumPaso-1 ; i_WWM_50Ult = NumPaso-1
    cont_RB_Ult = 0 ; cont_WWM_Ult = 0 ; cont_RB_50Ult = 0 ; cont_WWM_50Ult = 0 
    
    for i in range(NumPaso):        
        Diff_Cr = np.abs(Def_Cr - ele_i.pisos[-1].ResultadosAltura.list_Strain[i][0])
        Diff_RBy = np.abs(Def_RBy - ele_i.pisos[-1].ResultadosAltura.list_Strain[i][0])
        Diff_WWMy = np.abs(Def_WWMy - ele_i.pisos[-1].ResultadosAltura.list_Strain[i][0])
        Diff_RB_Ult = np.abs(Def_RB_Ult - ele_i.pisos[-1].ResultadosAltura.list_Strain[i][0])
        Diff_WWM_Ult = np.abs(Def_WWM_Ult - ele_i.pisos[-1].ResultadosAltura.list_Strain[i][0])
        Diff_RB_50Ult = np.abs(Def_RB_Ult - ele_i.pisos[-1].ResultadosAltura.list_Strain[i][int(ele_i.pisos[-1].nfib/2)])
        Diff_WWM_50Ult = np.abs(Def_WWM_Ult - ele_i.pisos[-1].ResultadosAltura.list_Strain[i][int(ele_i.pisos[-1].nfib/2)])
        
        if Diff_Cr < minDiff_Cr and ele_i.pisos[-1].ResultadosAltura.list_Strain[i][0] < 1.05*Def_Cr:
            minDiff_Cr = Diff_Cr
            i_Cr = i
        if Diff_RBy < minDiff_RBy and 'RB' in ele_i.pisos[-1].muro_Md.listaTipoAcero[:2] and ele_i.pisos[-1].ResultadosAltura.list_Strain[i][0] < 1.05*Def_RBy:
            minDiff_RBy = Diff_RBy
            i_RBy = i
        if Diff_WWMy < minDiff_WWMy and 'WWM' in ele_i.pisos[-1].muro_Md.listaTipoAcero[:2] and ele_i.pisos[-1].ResultadosAltura.list_Strain[i][0] < 1.05*Def_WWMy:
                
            minDiff_WWMy = Diff_WWMy
            i_WWMy = i
        if cont_RB_Ult != 1 and Diff_RB_Ult < minDiff_RB_Ult and 'RB' in ele_i.pisos[-1].muro_Md.listaTipoAcero[:2]:
            if (ele_i.pisos[-1].ResultadosAltura.list_Strain[i][0] > Def_RB_Ult):
                cont_RB_Ult = 1
                
            minDiff_RB_Ult = Diff_RB_Ult
            i_RB_Ult = i
        if cont_WWM_Ult != 1 and Diff_WWM_Ult < minDiff_WWM_Ult and 'WWM' in ele_i.pisos[-1].muro_Md.listaTipoAcero[:4]:
            for ind_fib, fib_ in enumerate(ele_i.pisos[-1].muro_Md.listaTipoAcero[:4]):
                if fib_ == 'WWM' and ele_i.pisos[-1].ResultadosAltura.list_Strain[i][ind_fib] > Def_WWM_Ult:                    
                    cont_WWM_Ult = 1
                        
                    minDiff_WWM_Ult = Diff_WWM_Ult
                    i_WWM_Ult = i
            
        if cont_RB_50Ult != 1 and Diff_RB_50Ult < minDiff_RB_50Ult and 'RB' in ele_i.pisos[-1].muro_Md.listaTipoAcero[int(ele_i.pisos[-1].nfib/2)]:
            if (ele_i.pisos[-1].ResultadosAltura.list_Strain[i][int(ele_i.pisos[-1].nfib/2)] > Def_RB_Ult):
                cont_RB_50Ult = 1
            elif (ele_i.pisos[-1].ResultadosAltura.list_Strain[i][int(ele_i.pisos[-1].nfib/2)] > 0.8*Def_RB_Ult) and i > i_RB_Ult:
                minDiff_RB_50Ult = Diff_RB_50Ult
                i_RB_50Ult = i
        if cont_WWM_50Ult != 1 and Diff_WWM_50Ult < minDiff_WWM_50Ult and 'WWM' in ele_i.pisos[-1].muro_Md.listaTipoAcero[int(ele_i.pisos[-1].nfib/2)]:
            if (ele_i.pisos[-1].ResultadosAltura.list_Strain[i][int(ele_i.pisos[-1].nfib/2)] > Def_WWM_Ult):
                cont_WWM_50Ult = 1
            elif (ele_i.pisos[-1].ResultadosAltura.list_Strain[i][int(ele_i.pisos[-1].nfib/2)] > 0.8*Def_WWM_Ult) and i > i_WWM_Ult:
                minDiff_WWM_50Ult = Diff_WWM_50Ult
                i_WWM_50Ult = i
            
    paso_Cr.append(i_Cr)
    paso_RBy.append(i_RBy)
    paso_WWMy.append(i_WWMy)
    paso_RB_Ult.append(i_RB_Ult)
    paso_WWM_Ult.append(i_WWM_Ult)
    paso_RB_50Ult.append(i_RB_50Ult)
    paso_WWM_50Ult.append(i_WWM_50Ult)

#%% ==================================  PUNTOS DE CONTROL (M1000) (primer muro del piso 1) ====================================
#==============================================================================================================================

Posicion_VMax = np.argmax(ARQ_R_PO.ResultadosPushover.listaVbase)                                              # Encuentra la posición del valor máximo pushover

minDiff = 99999
paso_80 = 0
for index, i in enumerate(ARQ_R_PO.ResultadosPushover.listaVbase[Posicion_VMax:]):
    Diff = np.abs(i-(0.8*max(ARQ_R_PO.ResultadosPushover.listaVbase)))
    if Diff < minDiff:
        minDiff = Diff
        paso_80 = index + Posicion_VMax
        

ptosctrl = pd.DataFrame(data={'SDR1':None, 'RDR':None, 'SDR':None, 'Vb':None, 'V/W':None}, 
                        index=['1er-Agrietamiento', '1ra-Fluencia', 'Cap-Maxima',
                               '1ra-Rotura', '80%-Capacidad', '50%-Rotura'], dtype= float)

# Deriva entre piso------------------------------------------------------------
pc_sdr1_1 = ARQ_R_PO.ResultadosPushover.listaSDR1[paso_Cr[0]]                               #Deriva entre piso en el agrietamiento en la primera macrofibra del primer nivel.
pc_sdr1_2 = ARQ_R_PO.ResultadosPushover.listaSDR1[paso_WWMy[0] if paso_WWMy[0] > 0 else paso_RBy[0]]                               #Deriva entre piso en la primera fluencia en la primera macrofibra del primer nivel.
pc_sdr1_3 = ARQ_R_PO.ResultadosPushover.listaSDR1[Posicion_VMax]                                             #Deriva entre piso en la cortante maxima en la base del arquetipo.
pc_sdr1_4 = ARQ_R_PO.ResultadosPushover.listaSDR1[min(paso_WWM_Ult) if min(paso_WWM_Ult) != (NumPaso-1) else min(paso_RB_Ult)]                               #Deriva entre piso en la primera rotura del acero en la primera macrofibra del primer nivel.
pc_sdr1_5 = ARQ_R_PO.ResultadosPushover.listaSDR1[paso_80]                                                                 #Deriva entre piso en la perdida del 20% del cortante maximo en la base.
pc_sdr1_6 = ARQ_R_PO.ResultadosPushover.listaSDR1[min(paso_WWM_50Ult) if min(paso_WWM_50Ult) != (NumPaso-1) else min(paso_RB_50Ult)]

#Deriva de techo---------------------------------------------------------------
pc_d1 = ARQ_R_PO.ResultadosPushover.listaRDR[paso_Cr[0]]                               #Desplazamiento de techo durante el agrietamiento en la primera macrofibra del primer nivel.
pc_d2 = ARQ_R_PO.ResultadosPushover.listaRDR[paso_WWMy[0] if paso_WWMy[0] > 0 else paso_RBy[0]]                               #Desplazamiento de techo durante la primera fluencia en la primera macrofibra del primer nivel.
pc_d3 = ARQ_R_PO.ResultadosPushover.listaRDR[Posicion_VMax]                                           #Desplazamiento de techo durante la cortante maxima en la base del arquetipo.
pc_d4 = ARQ_R_PO.ResultadosPushover.listaRDR[min(paso_WWM_Ult) if min(paso_WWM_Ult) != (NumPaso-1) else min(paso_RB_Ult)]                               #Desplazamiento de techo durante la primera rotura del acero en la primera macrofibra del primer nivel.
pc_d5 = ARQ_R_PO.ResultadosPushover.listaRDR[paso_80]                                                            #Desplazamiento de techo durante la perdida del 20% del cortante maximo en la base.
pc_d6 = ARQ_R_PO.ResultadosPushover.listaRDR[min(paso_WWM_50Ult) if min(paso_WWM_50Ult) != (NumPaso-1) else min(paso_RB_50Ult)]

#Deriva entre piso-------------------------------------------------------------
pc_sdr1 = ARQ_R_PO.ResultadosPushover.listaSDR[paso_Cr[0]]                               #Deriva entre piso en el agrietamiento en la primera macrofibra del primer nivel.
pc_sdr2 = ARQ_R_PO.ResultadosPushover.listaSDR[paso_WWMy[0] if paso_WWMy[0] > 0 else paso_RBy[0]]                               #Deriva entre piso en la primera fluencia en la primera macrofibra del primer nivel.
pc_sdr3 = ARQ_R_PO.ResultadosPushover.listaSDR[Posicion_VMax]                                             #Deriva entre piso en la cortante maxima en la base del arquetipo.
pc_sdr4 = ARQ_R_PO.ResultadosPushover.listaSDR[min(paso_WWM_Ult) if min(paso_WWM_Ult) != (NumPaso-1) else min(paso_RB_Ult)]                               #Deriva entre piso en la primera rotura del acero en la primera macrofibra del primer nivel.
pc_sdr5 = ARQ_R_PO.ResultadosPushover.listaSDR[paso_80]                                                                 #Deriva entre piso en la perdida del 20% del cortante maximo en la base.
pc_sdr6 = ARQ_R_PO.ResultadosPushover.listaSDR[min(paso_WWM_50Ult) if min(paso_WWM_50Ult) != (NumPaso-1) else min(paso_RB_50Ult)]

#Cortante----------------------------------------------------------------------
pc_v1 = ARQ_R_PO.ResultadosPushover.listaVbase[paso_Cr[0]]                               #Cortante en la base durante el agrietamiento en la primera macrofibra del primer nivel.
pc_v2 = ARQ_R_PO.ResultadosPushover.listaVbase[paso_WWMy[0] if paso_WWMy[0] > 0 else paso_RBy[0]]                               #Cortante en la base durante la primera fluencia en la primera macrofibra del primer nivel.
pc_v3 = ARQ_R_PO.ResultadosPushover.listaVbase[Posicion_VMax]                                             #Cortante maxima en la base del arquetipo.
pc_v4 = ARQ_R_PO.ResultadosPushover.listaVbase[min(paso_WWM_Ult) if min(paso_WWM_Ult) != (NumPaso-1) else min(paso_RB_Ult)]                               #Cortante en la base durante la primera rotura del acero en la primera macrofibra del primer nivel.
pc_v5 = ARQ_R_PO.ResultadosPushover.listaVbase[paso_80]                                                                     #20% del cortante maximo en la base.
pc_v6 = ARQ_R_PO.ResultadosPushover.listaVbase[min(paso_WWM_50Ult) if min(paso_WWM_50Ult) != (NumPaso-1) else min(paso_RB_50Ult)]
    
#----------Puntos de control---------------------------------------------------
ptosctrl['SDR1'] = np.array([pc_sdr1_1, pc_sdr1_2, pc_sdr1_3, pc_sdr1_4, pc_sdr1_5, pc_sdr1_6], dtype=float)
ptosctrl['RDR'] = np.array([pc_d1, pc_d2, pc_d3, pc_d4, pc_d5, pc_d6], dtype=float)
ptosctrl['SDR'] = np.array([pc_sdr1, pc_sdr2, pc_sdr3, pc_sdr4, pc_sdr5, pc_sdr6], dtype=float)
ptosctrl['Vb'] = np.array([pc_v1, pc_v2, pc_v3, pc_v4, pc_v5, pc_v6], dtype=float)
ptosctrl['V/W'] = ptosctrl['Vb']/(ARQ_R_PO.ws_*9.81)

ptosctrl = ptosctrl.dropna()
ptosctrl.to_excel(folder_PUSHOVER+'\\'+str(name_arq)+'_ptos_control.xlsx')

#%% ============================ ESTADO DAÑO (ARQUETIPO) ============================
# ===================================================================================

#----- lectura del archivo *.csv a dataframe
file_dsi_arq = open(ruta+'\\DS.xlsx', 'rb')
DSi = pd.read_excel(file_dsi_arq, sheet_name=name_sheet_ds)
file_dsi_arq.close()
DSi.set_index('DS', inplace=True)

#%% ================ PUSHOVER - RELACIÓN DE DERIVAS (ARQUETIPO) ===============================
#==============================================================================================
#%% ============ PUSHOVER - RELACIÓN DE DERIVAS (ARQUETIPO) ==========================
#==============================================================================================

if EDP_analisis == 'SDR_max':
    EDP_plot = ARQ_R_PO.ResultadosPushover.listaSDR
    EDP_legend = 'RDR'
    EDP_label_plot = 'SDR$_{max}$'+' (%)'
elif EDP_analisis == 'SDR1_max':
    EDP_plot = ARQ_R_PO.ResultadosPushover.listaSDR1
    EDP_legend = 'RDR'
    EDP_label_plot = 'SDR$_{1 max}$'+' (%)'
elif EDP_analisis == 'RDR_max':
    EDP_plot = ARQ_R_PO.ResultadosPushover.listaRDR
    EDP_legend = 'SDR'
    EDP_label_plot = 'RDR$_{max}$'+' (%)'
    
fig, axs = plt.subplots(figsize=(8.5, 7.5), dpi=150, nrows=1, ncols=1, sharex=False, sharey=False)
plt.rcParams["font.family"] = "San serif"
plt.rcParams['savefig.bbox'] = "tight"
style.use('default') or plt.style.use('default')
#------Relacion de deriva del techo
color = ['blue','orangered','gold','darkviolet',
         'dodgerblue','firebrick', 'olive', 'rosybrown']

for i, pc in enumerate(ptosctrl.index):    
    legh_i = [ptosctrl.index[i]+', '+EDP_legend+' ='+str(ptosctrl[EDP_legend][i].round(2))+'%']
    axs.scatter(ptosctrl[edp_label][i], ptosctrl['V/W'][i], marker = 'o', label=legh_i[0], c=color[i] )

axs.plot(EDP_plot, ARQ_R_PO.ResultadosPushover.listaVbase/(ARQ_R_PO.ws_*9.81), color='k', linestyle='--', label='Curva Capacidad')

axs.set_title('PUSHOVER - ' + str(NombreModelo), fontsize=24)
axs.set_xlabel(EDP_label_plot, fontsize=22)
axs.set_ylabel('V/W', fontsize=22)
axs.set_xlim([-0.05,3.0])
axs.set_xticks(np.arange(0,3.2,0.4))
axs.grid(ls='-.', lw=0.3)
axs.tick_params(labelsize=22)
axs.legend(loc='lower right', fontsize=16, framealpha=0.5)                   # place legend outside

plt.tight_layout()
plt.savefig(fname=folder_PUSHOVER+'\\'+str(name_arq)+'_PushOver_'+edp_label+'.pdf', format="pdf", dpi='figure')

#%% ====================== GRAFICO DE MOMENTOS EN LOS MUROS ====================================
#==============================================================================================

if (GraficoMomentosMuros == True):    
    fig, axs = plt.subplots(figsize=(8.5, 7.5), dpi=150, nrows=1, ncols=1, sharex=False, sharey=False)
    plt.rcParams["font.family"] = "San serif"
    plt.rcParams['savefig.bbox'] = "tight"
    style.use('default') or plt.style.use('default')
    
    
    for index, i in enumerate(ARQ_R_PO.muros): 
        if i.nombre in listamuros:
            axs.plot(i.pisos[-1].ResultadosAltura.list_SDR,
                     i.pisos[-1].ResultadosAltura.list_M,
                     label = i.nombre+': '+str(i.pisos[-1].id_),
                     lw = 1)  
    
    axs.set_title('Momentos en la base', fontsize=24)
    axs.set_xlabel('SDR$_{max}$'+' (%)', fontsize=22)
    axs.set_ylabel('Momento (kN.m)', fontsize=22)
    axs.grid(ls='-.', lw=0.3)
    axs.tick_params(labelsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()

#%% ====================== GRAFICO DE CORTANTES EN LOS MUROS ====================================
#==============================================================================================

if (GraficoCortantesMuros == True):    
    fig, axs = plt.subplots(figsize=(8.5, 7.5), dpi=150, nrows=1, ncols=1, sharex=False, sharey=False)
    plt.rcParams["font.family"] = "San serif"
    plt.rcParams['savefig.bbox'] = "tight"
    style.use('default') or plt.style.use('default')
    
    
    for index, i in enumerate(ARQ_R_PO.muros): 
        if i.nombre in listamuros:
            axs.plot(i.pisos[-1].ResultadosAltura.list_SDR,
                     i.pisos[-1].ResultadosAltura.list_V,
                     label = i.nombre+': '+str(i.pisos[-1].id_),
                     lw = 1)
    
    axs.set_title('Cortantes en la base', fontsize=24)
    axs.set_xlabel('SDR$_{max}$'+' (%)', fontsize=22)
    axs.set_ylabel('Cortante (kN)', fontsize=22)
    axs.grid(ls='-.', lw=0.3)
    axs.tick_params(labelsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()

#%% ====================== PERFIL DE DERIVA EN EL PUSHOVER ====================================
#==============================================================================================
Index = 0

if (GraficoPerfilDeriva == True):
        Index_Cr = paso_Cr[0]
        Index_y = paso_WWMy[0] if paso_WWMy[0] > 0 else paso_RBy[0]
        Index_max = np.where(ARQ_R_PO.ResultadosPushover.listaVbase == max(ARQ_R_PO.ResultadosPushover.listaVbase))[0][0]
        Index_80 = paso_80
        
        Index = (Index_Cr, Index_y, Index_max, Index_80)

sdr_pts_ctrl = pd.DataFrame()
fig, axs = plt.subplots(figsize=(8.5, 7.5), dpi=150, nrows=1, ncols=1, sharex=False, sharey=False)
plt.rcParams["font.family"] = "San serif"
plt.rcParams['savefig.bbox'] = "tight"
style.use('default') or plt.style.use('default')
color = ['blue','orangered','gold', 'dodgerblue']
legh_i = ['1$^{er}$-Agrietamiento', '1$^{ra}$-Fluencia', 'Cap-Máxima', '80% Capacidad']
columns_i = ['Agrietamiento', 'Fluencia', 'Cap-Máxima', '80% Capacidad']
for i, index_id in enumerate(Index):
    listSDRmax = []
    for piso in ARQ_R_PO.muros[0].pisos:
        listSDRmax.append(piso.ResultadosAltura.list_SDR[index_id])
    listSDRmax.append(0)
    hi_i = np.array(ARQ_R_PO.listaCoorY, dtype=float)/max(ARQ_R_PO.listaCoorY)
    hi_i = np.append(hi_i, 0)
    axs.plot(listSDRmax,
             hi_i,
             lw = 3,
             color=color[i],
             label = legh_i[i],
             marker = 'o')
    sdr_pts_ctrl[columns_i[i]] = listSDRmax
color_frag = ['forestgreen','gold','red']

# ----- Umbrales de daño
color_frag = ['forestgreen','gold','darkorange','red']
if metodo == 'PROB':
    for i_Ds, Ds_index in enumerate(DSi.index):
        axs.axvline(x=DSi['Theta'][i_Ds],
                    ls='-.', lw=2.5,
                    label=str(edp_label)+'$_{'+str(Ds_index)+'}$',
                    color=color_frag[i_Ds])
elif metodo == 'DET':
    for i_Ds, Ds_index in enumerate(DSi.index):
        axs.axvline(x=DSi.iloc[i_Ds].values[0],
                    ls='-.', lw=2.5,
                    label=str(edp_label)+'$_{'+str(Ds_index)+'}$',
                    color=color_frag[i_Ds])
sdr_pts_ctrl['hi/ht'] = hi_i
axs.set_xlabel('SDR$_{max}$'+' (%)', fontsize=22)
axs.set_ylabel('h$_{i}$/h$_{t}$', fontsize=22)
axs.grid(ls='-.', lw=0.3)
axs.tick_params(labelsize=22)
axs.legend(fontsize=16)
plt.tight_layout()

sdr_pts_ctrl.to_excel(ruta+'\\'+name_arq+'_perfil_SDRmax_PO.xlsx')
#%% =============================== CONTADOR TIEMPO DE EJECUCION ==============================
#==============================================================================================

etime = time.time()
ttotal = (etime - stime)/60
print('---------------Tiempo Análisis----------------')
print('---------------'+str(round(ttotal,3))+' min ----------------')

