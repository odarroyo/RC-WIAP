# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:43:30 2023

@authores: Frank y JuanJo
======================== DEFINICION CLASE ARQUETIPO =========================
"""
#%% ==================== IMPORTAR FUNCIONES, LIBRERIAS PARA LA EJECUCIÓN ======================
#==============================================================================================
from enum import Enum

#%% ========================= CLASES PARA LA GENERACIÓN DE ARQUETIPOS =========================
#==============================================================================================

class eDireccion(Enum):
    Longitudinal = 0
    Transversal = 1

class eTipoRef(Enum):
    RB = 0
    WWM = 1

class eTipoConcrete(Enum):
    Unconf = 0
    Conf = 1

class cSeccionMuro:
    def __init__(self, id_, nombre, listaMacrofibras, 
                 listaAncho, listaEspesor, listaCuantia, listaTipoConcreto, 
                 listaTipoAcero):
        
        self.id_ = id_
        self.nombre = nombre
        self.listaMacrofibras = listaMacrofibras
        self.listaAncho = listaAncho
        self.listaEspesor = listaEspesor
        self.listaCuantia = listaCuantia
        self.listaTipoConcreto = listaTipoConcreto
        self.listaTipoAcero = listaTipoAcero
        self.listaTagConcret = []
        self.listaTagSteel = []
        
class cPiso:
    def __init__(self, id_, nombre, muro, muro_Md, hw, fc, CoorY, w_, ws_):
        
        self.id_ = id_
        self.nombre = nombre 
        self.muro = muro
        self.muro_Md = muro_Md
        self.fc = fc 
        self.CoorY = CoorY
        self.nfib = len(muro_Md.listaAncho)
        self.w_ = w_        
        self.ws_ = ws_
        self.DeforUnitaria = []
        self.ResultadosAltura = cResultadosAltura()
        
    def agregar_ResultadosAltura(self, Strain, StressConcrete, StressSteel, GlobalForce, Disp, SDR):
        self.ResultadosAltura.agregarResultados(Strain, StressConcrete, StressSteel, GlobalForce, Disp, SDR)

class cMuro:
    def __init__(self, nombre, id_, lw, tw, w_, ws_):
        self.nombre = nombre
        self.id_ = id_
        self.lw = lw
        self.tw = tw
        self.pisos = []
        self.w_ = w_
        self.ws_ = ws_
        self.Resultados = None
        self.Reacciones = None
        
    def agregar_piso(self, piso):
        self.pisos.append(piso)
        
    def eliminar_piso(self, piso):
        self.pisos.remove(piso)
        
    def agregar_Resultados(self, Resultado):
        self.Resultados = cResultadosMuros(Resultado)
        
    def agregar_Reacciones(self, Reaccion):
        self.Reacciones = Reaccion
        
class cMatConcrete:
    def __init__(self, nombre, iTag, iTagUnc, fc):
        self.nombre = nombre
        self.iTag = iTag
        self.iTagUnc = iTagUnc
        self.fc = fc
        
class cMatSteel:
    def __init__(self, nombre, iTag, fy):
        self.nombre = nombre
        self.iTag = iTag
        self.fy = fy
        
class Arquetipo:
    def __init__(self, nombre, direccion, NumPisos, w_, ws_):
        
        self.nombre = nombre
        self.direccion = direccion
        self.NumPisos = NumPisos
        self.w_ = w_
        self.ws_ = ws_
        self.listaCoorY = []
        self.muros = []
        self.MatConcrete = []
        self.MatSteel = []
        self.ResultadosPushover = None
        self.PuntosControl = []
    
    def agregar_muros(self, muro):
        self.muros.append(muro)
        
    def agregar_MatConcrete(self, concreto):
        self.MatConcrete.append(concreto)
        
    def agregar_MatSteel(self, Acero):
        self.MatSteel.append(Acero)
        
    def agregar_listaCoorY(self, CoorY):
        self.listaCoorY.append(CoorY)
        
    def agregar_ResultadosPushover(self, Resultado):
        self.ResultadosPushover = Resultado
        
    def __agregar_PtosControl(self, Pto):
        self.PuntosControl.append(Pto)
        
class cResultadosPushover:
    def __init__(self):
        self.listaSDisp1 = None
        self.listaRDisp = None
        self.listaVbase = None
        self.listaSDR1 = None
        self.listaRDR = None
        self.listaSDR = None

class cResultadosMuros:
    def __init__(self):
        self.Strain = None
        self.list_V = None
        self.list_P = None
        self.list_M = None
        
class cReaccionesBase:
    def __init__(self, P, V, M):
        self.P = P
        self.V = V
        self.M = M
        
class cPtoControl:
    def __init__(self, nombre, sDR, RDR, sDR1, Vbase):
        self.nombre = nombre
        self.sDR = sDR
        self.RDR = RDR
        self.sDR1 = sDR1
        self.Vbase = Vbase
        
class cResultadosAltura:
    def __init__(self):
        self.list_Strain = []
        self.list_StressConcrete = []
        self.list_StressSteel = []
        self.list_V = []
        self.list_P = []
        self.list_M = []
        self.list_Disp = []
        self.list_SDR = []
        
    def agregarResultados(self, Strain, StressConcrete, StressSteel, GlobalForce, Disp, SDR):
        self.list_Strain.append(Strain)
        self.list_StressConcrete.append(StressConcrete)
        self.list_StressSteel.append(StressSteel)
        self.list_V.append(GlobalForce[0])
        self.list_P.append(GlobalForce[1])
        self.list_M.append(GlobalForce[2])
        self.list_Disp.append(Disp)
        self.list_SDR.append(SDR)
