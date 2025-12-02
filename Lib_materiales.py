# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:51:33 2022

@author: vidal
"""
import math
import numpy as np
import openseespy.opensees as ops
import random

# ------------ Definición de Materiales ----------------

# Tipo_R: Se refiere al tipo de refuerzo a utilizar (Acero dúctil-Barras o Malla de refuerzo)
# Tipo_R-> "Barras o Malla"
# Tipo_M-> "Hysteretic o SteelMPF"
# En los archivos en donde se definen los muros se especifica el tipo de material. WWM es para mallas electrosoldades (Welded Wire Mesh), RB es para barras dúctiles. Unconf es para el concreto sin confinar y conf es para el confinado. 

ops.wipe()

# ======================== SELECCION DEL REFUERZO =============================================
#==============================================================================================
# ---- "HYS" Para Hysteretic

def Refuerzo(Tipo_R, matTag, fyi, Tipo_M, p1 = [1, 1], p2 =[1, 1], p3 = [1, 1]):   
    
    Es = 210000000

    Tipo_Ref = Tipo_R.upper()
    Tipo_Mat = Tipo_M.upper()
    
    #-------- ------------BARRAS DE REFUERZO---------------------------------------------------
    if np.logical_and(Tipo_Ref.find('RB') >= 0, Tipo_Mat.find("HYS") >= 0):
        Mat_Refuerzo = 2500
        fy = fyi*1000
        eyR = fy/Es
        fu = 630000
        eult= 0.10
        fact_fy = 0.05
        fact_eu = 1.10
        ops.uniaxialMaterial('Hysteretic', Mat_Refuerzo, fy, eyR, fu, eult, fact_fy*fy, fact_eu*eult, -fy, -eyR, -fu, -eult, -fact_fy*fy, -fact_eu*eult, 1.0, 1.0, 0.0, 0.0)       
        return ops.uniaxialMaterial('MinMax', matTag, Mat_Refuerzo, '-min', -0.006, '-max', 0.05)


    #-------- ------------MALLAS ELECTROSOLODADAS--------------------------------------------
    #----- -----Parámetros histeresis Miranda et al. 2022 (Arteta)---------------------------
    elif np.logical_and(Tipo_Ref.find('WWM') >= 0, Tipo_Mat.find("HYS") >= 0): 
        Mat_Refuerzo = 6
        p1=509.10*1000
        p2=691.50*1000
        p3=734.44*1000
        e1=0.00248
        e2=0.005
        e3=0.01
        pinchX=0.34
        pinchY= 0.56
        damage1= 0.038
        damage2= 0.07
        beta= 0.086
        ops.uniaxialMaterial('Hysteretic', Mat_Refuerzo, p1, e1, p2, e2, p3, e3, -p1, -e1, -p2, -e2, -p3, -e3, pinchX, pinchY, damage1, damage2, beta)       
        return ops.uniaxialMaterial('MinMax', matTag, Mat_Refuerzo, '-min', -0.006, '-max', 0.0186)
    
    elif Tipo_Mat.find("MPF") >= 0:
        #Parámetros para SteelMPF
        by_flT = 0.00947
        by_flC = by_flT
        R0 = 20.0
        a1 = 0.92255
        a2 = 0.0015
        ops.uniaxialMaterial('SteelMPF', Mat_Refuerzo, fy, fy, Es, by_flT, by_flC, R0, a1, a2)
        return ops.uniaxialMaterial('MinMax', matTag, Mat_Refuerzo, '-min', -eult, '-max', eult)

def Refuerzo_SDOF(Tipo_R, matTag, Tipo_M, p1 = [1, 1], p2 =[1, 1], p3 = [1, 1]):
    Tipo_Mat = Tipo_M.upper()
    
    if Tipo_Mat.find("FOR") >= 0:
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        n1 = (-1)*p1
        n2 = (-1)*p2
        n3 = (-1)*p3
        pinchX = 1.0
        pinchY = 1.0
        damage1 = 0.00
        damage2 = 0.00
        beta = 0.0
        # ops.uniaxialMaterial('Hysteretic', matTag, *p1, *p2, *p3, *n1, *n2, *n3, pinchX, pinchY, damage1, damage2, beta)
        #uniaxialMaterial('Hysteretic',5,Fy,ey,fu,eult,0.2*Fy,0.0156,-Fy,-ey,-fu,-eult,-0.2*Fy,-0.0156,1.0,1.0,0.0,0.0)
        #R = uniaxialMaterial('MinMax', matTag, Mat_Refuerzo, '-min', -n3[1], '-max', p3[1])
        return ops.uniaxialMaterial('Hysteretic', matTag, *p1, *p2, *p3, *n1, *n2, *n3, pinchX, pinchY, damage1, damage2, beta)

# ======================== SELECCION DEL CONCRETO =============================================
#==============================================================================================
# ---- "C01" Para Concrete01
# ---- "C02" Para Concrete02

def Concreto(mat, matTag, fc, Tipo_C):
    if mat == 'Unconf':
        fpc = fc*1000
        Ec = 4300000*math.sqrt(fpc/1000)
        epsc = 2*fpc/Ec
        fpcu = 0.1*fpc
        epscu = 0.006
        
    elif mat == 'Conf':
        k=1.3
        fpc = fc*1000*k
        Ec = 4300000*math.sqrt(fpc/1000)
        epsc = 2*fpc/Ec
        fpcu = 0.2*fpc
        epscu = 0.02
    
    Tipo_Mat = Tipo_C.upper()
    
    if Tipo_Mat.find("C01") >= 0:
        return ops.uniaxialMaterial('Concrete01', matTag, -fpc, -epsc, -fpcu, -epscu)
    
    #FALTA POR REVISAR ESTOS MATERIALES
    elif Tipo_Mat.find("C02") >= 0:
        #Parámetros para Concrete02
        ft = 0.33*math.sqrt(fpc/1000)*1000
        lambda_C = 0.05
        Ets = ft/0.0020
        return ops.uniaxialMaterial('Concrete02', matTag, -fpc, -epsc, -fpcu, -epscu, lambda_C, ft, Ets)


