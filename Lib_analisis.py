"""
Fecha: 01/08/2022
@authores: Orlando, Dirsa, Frank y Juan Jose
===================================================================
------------------- Modelo Arquetipo ------------------------------
===================================================================

"""
#%% =========== IMPORTAR FUNCIONES, LIBRERIAS PARA LA EJECUCIÓN DEL MODELO =============
import openseespy.opensees as ops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
# Ajuste de distribuciones
# ==============================================================================
from scipy import stats, optimize
#---------- LIBRERIAS PROPIAS GENERADAS POR EL EQUIPO
import Lib_materiales as mat
from Lib_ClaseArquetipos import *

#%% ============================ 0. ANALISIS BASICOS ARQUETIPO ==========================
#========================================================================================

#------------------------------ 0.1 Análisis de gravedad del arquetipo.
def gravedad_arquetipo():

    ops.system('BandGeneral')                                                       # Create the system of equation, a sparse solver with partial pivoting
    ops.constraints('Plain')                                               # Create the constraint handler, the transformation method
    ops.numberer('RCM')                                                             # Create the DOF numberer, the reverse Cuthill-McKee algorithm
    ops.test('NormDispIncr', 1.0e-12, 10, 3)                                        # Create the convergence test, the norm of the residual with a tolerance of 1e-12 and a max number of iterations of 10
    ops.algorithm('Newton')                                                         # Create the solution algorithm, a Newton-Raphson algorithm
    ops.integrator('LoadControl', 0.1)                                              # Create the integration scheme, the LoadControl scheme using steps of 0.1
    ops.analysis('Static')                                                          # Create the analysis object
    ok = ops.analyze(10)   


#------------------------------ 0.2 Análisis modal del arquetipo.
def Modal(mat_Masa, NodosLibres, N_Nodes, ndf): 
    
    omega2 = ops.eigen('fullGenLapack', N_Nodes)                                    # Función para la obtención de los valores propios de la estructura
    #omega2 = eigen(N)
    #mat_Masa = np.flip(mat_Masa)                                           # Función para invertir el orden del vectos [1,2,3] -> [3,2,1]
    mat_Modos = []
    vec_Phi = []
    LnM = []
    MnM = []
    GammaM = []
    T_n = []
    w_n = []
    s_n = []
    vec_Me = []    
    for n in range(N_Nodes):   
        Phi_n = []    
        for nd in NodosLibres:
            nd=int(nd)
            ndMass = ops.nodeMass(nd)
            MNV = ops.nodeEigenvector(nd,n+1)                                       # Función para la obtención de los valores propios del nodo específico
            Phi_n.append(MNV[0])
            
        Tn = 2*np.pi/omega2[n]**0.5                                             # Periodo del modo n
        Phi_n = np.array(Phi_n)                                                 # Amplitudes del movimiento armónico para el modo n 
        Ln = np.trace(Phi_n*mat_Masa)
        Mn = np.trace(Phi_n**2*mat_Masa)
        Gamman = Ln/Mn                                                          # Factor de participación modal
        mat_Modos.append(n+1)
        vec_Phi.append(Phi_n)
        LnM.append(Ln)
        MnM.append(Mn)
        GammaM.append(Gamman)
        T_n.append(Tn)
        w_n.append(omega2[n]**0.5)
        sn = Gamman*np.dot(Phi_n,mat_Masa)                                      # Distribución espacial de los modos de vibrar
        s_n.append(sn)
        vec_Me.append(Ln*Gamman)
        
    # Generación del directorio con los resultados del análisis modal
    Result_MA = {'Modo': np.array(mat_Modos), 'Phi': np.array(vec_Phi), 'Ln': np.array(LnM), 'Mn': np.array(MnM), 'Gamman': np.array(GammaM), 'Tn': np.array(T_n),'wn': np.array(w_n), 'Sn': np.array(s_n), 'M*': np.array(vec_Me)}
    
    
    # Generación del dataframe con los resultados del del modo fundamental
    FormModals = ()
    for i in range(len(vec_Phi)):
        Data_ModoFund = {'phi1_norm': vec_Phi[i],'s1':s_n[i], 'Gamma_norm':np.ones(len(NodosLibres))*GammaM[i]}
        ModoFund = pd.DataFrame(Data_ModoFund, index =np.array(NodosLibres))
        FormModals = FormModals + (ModoFund,)
    return FormModals, Result_MA
#%% ============================ 1. ANALISIS PUSHOVER ARQUETIPO =========================
#========================================================================================

#------------------------------ 1.1 Análisis pushover del arquetipo.
def Pushover_Arquetipos(Dmax, Dincr, IDctrlNode, IDctrlDOF, NodosLibres, ele_record, wallctrl, nfib, norm=[-1,1], Tol=1e-5, plot = 'False', plotperiod = 'False', plotnorm = 'False'):
    
    # creación del recorder de techo y definición de la tolerancia
    #recorder('Node','-file','techo.out','-time','-node',IDctrlNode,'-dof',IDctrlDOF,'disp')
    maxNumIter = 10
          
    # configuración básica del análisis
    ops.wipeAnalysis()
    ops.constraints('Plain')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('EnergyIncr', Tol, maxNumIter)
    ops.algorithm('Newton')    
    ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)
    ops.analysis('Static')
    
    # Otras opciones de análisis    
    tests = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
    algoritmo = {1:'KrylovNewton', 2: 'SecantNewton' , 4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

    # rutina del análisis
    eig = ops.eigen(1)                                                              # Valor propio del primer modo
    TT = 2*3.1416/np.sqrt(eig[0])
    Nsteps =  int(Dmax/ Dincr)

    nels = len(ele_record)
    nnodes = len(NodosLibres)
    #-----Registros de las deformaciones unitarias en el primer nivel.
    Strains = np.zeros((nels, Nsteps+1, nfib))                                  # Graba las deformaciones unitarias de los muros en las nfib que tienen los elementos.
    # cStress = np.zeros((nels, Nsteps, nfib))                                  # Graba los esfuerzos del concreto de los muros en las nfib que tienen los elementos.
    # sStress = np.zeros((nels, Nsteps, nfib))                                  # Graba los esfuerzos del acero de los muros en las nfib que tienen los elementos.
    
    #-----Registros de las desplazamientos de piso
    Disp = np.zeros((nels, Nsteps+1, nnodes))                                  # Graba los desplazamiento en cada piso por muro.
    SDR = np.zeros((Nsteps+1, nnodes))
    #-----Registros de las fuerzas en la base del arquetipo.
    
    ForcesGlobal = np.zeros((nels, Nsteps+1, 3))                           # Graba las fuerzas globales (V, P, M) de cada muro en la base.
    # ShearForcesDef = np.zeros((nels, Nsteps, 2))
    #Curv = np.zeros((nels, Nsteps))
    for k in range(Nsteps):
        ok = ops.analyze(1)        
        # ok2 = ok;
        # En caso de no converger en un paso entra al condicional que sigue
        if ok != 0:
            print('configuración por defecto no converge en desplazamiento: ', ops.nodeDisp(IDctrlNode,IDctrlDOF))
            for j in algoritmo:
                if j < 4:
                    ops.algorithm(algoritmo[j], '-initial')  
                else:
                    ops.algorithm(algoritmo[j])                
                # el test se hace 50 veces más
                ops.test('EnergyIncr', Tol, maxNumIter*50)
                ok = ops.analyze(1)                
                if ok == 0:
                    # si converge vuelve a las opciones iniciales de análisi
                    ops.test('EnergyIncr', Tol, maxNumIter)
                    ops.algorithm('Newton')
                    break    
                
        if ok != 0:
            print('Pushover analisis fallido')
            print('Desplazamiento alcanzado: ', ops.nodeDisp(IDctrlNode,IDctrlDOF),'m')
            break

        
        for ne, ele in enumerate(ele_record):
            Strains[ne, k+1, :] = ops.eleResponse(ele, 'Fiber_Strain')                # Deformación unitaria en cada fibra de los muros en la base.
            # cStress[ne, k, :] = ops.eleResponse(ele, 'Fiber_Stress_Concrete')       # Esfuerzo del concreto en cada fibra de los muros en la base.
            # sStress[ne, k, :] = ops.eleResponse(ele, 'Fiber_Stress_Steel')          # Esfuerzo del acero en cada fibra de los muros en la base.
            
            ForcesGlobal[ne, k+1, :] = ops.eleResponse(ele, 'globalForce')[0:3]                  # Fuerzas globales en cada muro en la base.
            # ShearForcesDef[ne, k, :] = ops.eleResponse(ele, 'Shear_Force_Deformation')
            for nn, nnode in enumerate(NodosLibres):
                Disp[ne, k+1, nn] = ops.nodeDisp(nnode, IDctrlDOF)
                if ne == wallctrl:
                    SDR[k+1, nn] = ((ops.nodeDisp(nnode, IDctrlDOF) - ops.nodeDisp((nnode-1), IDctrlDOF))/
                                        (ops.nodeCoord(nnode, 2) - ops.nodeCoord((nnode-1),2)))*100
    #-------Desplazamientos en el arquetipo

    IDctrlTecho = NodosLibres.index(max(NodosLibres))
    
    dntecho = Disp[wallctrl,:,IDctrlTecho]
    dn1 = Disp[wallctrl,:,0]
    Vbase = (-1)*sum(ForcesGlobal[:,:,0])
    #-------Fuerzas en los elementos
    pushover = (dn1, dntecho, Vbase, Strains, SDR, ForcesGlobal)
    # ------Respuesta local del arquetipo en el pushover.
    # local_response = (Strains, cStress, sStress)
    # PER = np.array(periods)
    #-------Respuesta global del arquetipo en el pushover.
    # global_response = (ForcesGlobal, Disp, ShearForcesDef)
    return pushover

#------------------------------ 1.2 Análisis pushover del arquetipo adaptado.
def Pushover_Arquetipos_ADO(Dmax, Dincr, ARQ, NodosLibres, norm=[-1,1], Tol=1e-5, plot = 'False', plotperiod = 'False', plotnorm = 'False'):
    ARQ_R_PO = ARQ
    IDctrlNode = ARQ.muros[0].pisos[0].id_
    IDctrlDOF = 1

    maxNumIter = 10
          
    # =========== CONFIGURACION BASICA DEL ANALISIS ===========
    ops.wipeAnalysis()
    ops.constraints('Plain')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('EnergyIncr', Tol, maxNumIter)
    ops.algorithm('Newton')    
    ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)
    ops.analysis('Static')
    
    # ========== OTRAS OPCIONES DE ANALISIS ==========
    tests = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
    algoritmo = {1:'KrylovNewton', 2: 'SecantNewton' , 4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

    # rutina del análisis
    eig = ops.eigen(1)                                                              # Valor propio del primer modo
    TT = 2*3.1416/np.sqrt(eig[0])
    Nsteps =  int(Dmax/ Dincr)
    ListaVbasal = []

    for k in range(Nsteps):
        ok = ops.analyze(1)        
        # ok2 = ok;
        # En caso de no converger en un paso entra al condicional que sigue
        if ok != 0:
            print('configuración por defecto no converge en desplazamiento: ', ops.nodeDisp(IDctrlNode,IDctrlDOF))
            for j in algoritmo:
                if j < 4:
                    ops.algorithm(algoritmo[j], '-initial')  
                else:
                    ops.algorithm(algoritmo[j])                
                # el test se hace 50 veces más
                ops.test('EnergyIncr', Tol, maxNumIter*50)
                ok = ops.analyze(1)                
                if ok == 0:
                    # si converge vuelve a las opciones iniciales de análisi
                    ops.test('EnergyIncr', Tol, maxNumIter)
                    ops.algorithm('Newton')
                    break    
                
        if ok != 0:
            print('Pushover analisis fallido')
            print('Desplazamiento alcanzado: ', ops.nodeDisp(IDctrlNode,IDctrlDOF),'m')
            break

        if k != -1:
            for id_muro, muro_i in enumerate(ARQ_R_PO.muros):           # Ciclo para recorrer los muros que se tienen en el arquetipo
                for id_ele, ele_i in enumerate(muro_i.pisos):           # Ciclo para recorrer los pisos de cada muro
                    ele = ele_i.id_                                     # tag del muro del piso i
                    nnode = ele_i.id_                                   # tag del piso i
                    Strain_k = ops.eleResponse(nnode, 'Fiber_Strain')   # Obtener las deformaciones unitarias en el elemento del pios i
                    cStress_k = ops.eleResponse(nnode, 'Fiber_Stress_Concrete') # Obtiene los esfuerzos en el concreto en el elemento del piso i
                    sStress_k = ops.eleResponse(nnode, 'Fiber_Stress_Steel')    # Obtiene los esfuerzos en el acero en el elemento del piso i
                    GlobralForce_k = ops.eleResponse(ele, 'globalForce')[0:3]   # Obtiene las fuerzas en V, P, M del elemento del piso i
                    Disp_k = ops.nodeDisp(nnode, IDctrlDOF)                     # Obtiene los desplazamientos del piso i
                    SDR_k = (ops.nodeDisp(nnode, IDctrlDOF) - ops.nodeDisp((nnode-1), IDctrlDOF))/(ops.nodeCoord(nnode,     2) - ops.nodeCoord((nnode-1),2))*100    # Obtener la deriva de piso i
                    
                    #----- Agregar los resultados del PUSHOVER
                    ele_i.agregar_ResultadosAltura(Strain_k, cStress_k, sStress_k, GlobralForce_k, Disp_k, SDR_k)   # Instanciar a la funcion para guardar los resultados del elemento en el piso i
            
            ListaVbasal.append(ops.getTime())                           # Cortante del paso k
          
    # ====== GUARDAR LOS RESULTADOS GLOBALES DEL PUSHOVER ====== 
    ResultPush = cResultadosPushover()
    ResultPush.listaSDisp1 = np.array(ARQ_R_PO.muros[0].pisos[-1].ResultadosAltura.list_Disp)
    ResultPush.listaRDisp = np.array(ARQ_R_PO.muros[0].pisos[0].ResultadosAltura.list_Disp)
    ResultPush.listaSDR1 = ARQ_R_PO.muros[0].pisos[-1].ResultadosAltura.list_Disp/(ARQ_R_PO.listaCoorY[-1])*100
    ResultPush.listaRDR = ARQ_R_PO.muros[0].pisos[0].ResultadosAltura.list_Disp/(ARQ_R_PO.listaCoorY[0])*100
    ResultPush.listaSDR = np.array(ARQ_R_PO.muros[0].pisos[0].ResultadosAltura.list_SDR)
    ResultPush.listaVbase = np.array(ListaVbasal)
    ARQ_R_PO.agregar_ResultadosPushover(ResultPush)

    return ARQ_R_PO

#------------------------------ 1.3 Idealización del pushover del arquetipo.
def areaPushover(Drift_, Vbasal_):
    Area_Pushover = 0
    for i in range(len(Drift_)-1): 
        Area_Pushover +=(Vbasal_[i+1]+Vbasal_[i])/2*(Drift_[i+1]-Drift_[i])
    
    return Area_Pushover

#------------------------------ 1.2.1. Curva Cuatrilineal.
def CurvaCuatrilineal(RDR_, Vbasal_, ht, Area_Push_RDR, P_ctrl_RDR, Tol = 100):
#---Origen-----------------------------
    RDR_0 = 0.0
    Vb_0 = 0.0

#---Punto de Agrietamiento-------------
    Vb_1 = P_ctrl_RDR['Vb'][1]*0.5
    RDR_1 = RDR_[np.abs(Vbasal_- Vb_1).argmin()]
#---Punto Maximo-----------------------
    Vb_3 = np.max(Vbasal_)
    RDR_3 = np.max(RDR_)
#---Punto Fluencia---------------------
    paso = 20                                                                   #Valor a tomar para obtener la pendiente de la ultima parte de la curva  e intentar obtener una tangente
    Vb_cont = Vbasal_[len(Vbasal_) - paso]
    RDR_cont = RDR_[len(RDR_) - paso]
    #---Areas equivalentes RDR
    step = (RDR_3 - RDR_1)/100
    Darea = 100000
    for i in np.arange(RDR_1, RDR_3, step):
        vs = Vb_3 - ((Vb_3 - Vb_cont)*(RDR_3 - i) / (RDR_3 - RDR_cont))
        
        A1 = Vb_1 * RDR_1/2
        A2 = (vs + Vb_1) * (i - RDR_1) /2
        A3 = (Vb_3 + vs) * (RDR_3 - i) / 2
        
        ATri = A1 + A2 + A3
        if np.abs((Area_Push_RDR - ATri)) <= Darea:
            Darea = np.abs((Area_Push_RDR - ATri))
            Vb_2 = vs
            RDR_2 = i

#---Punto Falla---------------------
    RDR_4 = P_ctrl_RDR['RDR'][4]
    Vb_4 = P_ctrl_RDR['Vb'][4]
    
    Vbasal_cuatrilineal=np.array([Vb_0, Vb_1, Vb_2, Vb_3, Vb_4], dtype=(float))
    RDR_cuatrilineal=np.array([RDR_0, RDR_1, RDR_2, RDR_3, RDR_4], dtype=(float))
    RDisp_cuatrilineal = RDR_cuatrilineal*ht/100

    return  RDisp_cuatrilineal, RDR_cuatrilineal, Vbasal_cuatrilineal

#------------------------------ 1.2.2. Curva Cuatrilineal método secante.
def CurvaTrilineal_Sec(RDR_, Vbasal_, ht, P_ctrl_RDR):
#---Origen------------------------
    RDR_0 = 0.0
    Vb_0 = 0.0
#---Punto Secante-----------------
    Vb_sec = np.mean([P_ctrl_RDR['Vb'][0], P_ctrl_RDR['Vb'][1]])
    RDR_sec = np.mean([P_ctrl_RDR['RDR'][0], P_ctrl_RDR['RDR'][1]])
#---Punto Maximo------------------
    Vb_2 = np.max(Vbasal_)
    RDR_2 = np.max(RDR_)
#---Punto Fluencia----------------
    paso = 20 #Valor a tomar para obtener la pendiente de la ultima parte de la curva  e intentar obtener una tangente
    Vb_cont = Vbasal_[len(Vbasal_) - paso]
    RDR_cont = RDR_[len(RDR_) - paso]
    
    RDR_m1 = (Vb_sec-Vb_0)/(RDR_sec-RDR_0)
    RDR_m2 = (Vb_2-Vb_cont)/(RDR_2-RDR_cont)
    RDR_1 = (RDR_m1*RDR_sec-RDR_m2*RDR_2+Vb_2-Vb_sec)/(RDR_m1-RDR_m2)
    Vb_1 = RDR_m1*(RDR_1-RDR_sec)+Vb_sec

#---Punto Perdida de falla---------
    RDR_3 = P_ctrl_RDR['RDR'][4]
    Vb_3 = P_ctrl_RDR['Vb'][4]

    Vbasal_trilineal=np.array([Vb_0, Vb_1, Vb_2, Vb_3], dtype=(float))
    RDR_trilineal=np.array([RDR_0, RDR_1, RDR_2, RDR_3], dtype=(float))
    RDisp_trilineal = RDR_trilineal*ht/100


    return RDisp_trilineal, RDR_trilineal, Vbasal_trilineal

#%% ============================ 2. ANALISIS DINAMICOS ARQUETIPO ========================
#========================================================================================

#------------------------------ 2.1 Análisis Dinámico del arquetipo.
def dinamicoIDA4(recordName,dtrec,nPts,dtan,fact,damp,IDctrlNode,IDctrlDOF,ele_record,nodes_control, modes = [0,2],Kswitch = 1,Tol=1e-8):
  # dinamicoIDA4(recordName,dtrec,nPts,dtan,fact,damp,IDctrlNode,IDctrlDOF,elements,nodes_control,modes = [0,2],Kswitch = 1,Tol=1e-8)
    # PARA SER UTILIZADO PARA CORRER EN PARALELO LOS SISMOS Y EXTRAYENDO LAS FUERZAS DE LOS ELEMENTOS INDICADOS EN ELEMENTS
    
    #===== record es el nombre del registro, incluyendo extensión. P.ej. GM01.txt
    #===== dtrec es el dt del registro
    #===== nPts es el número de puntos del análisis
    #===== dtan es el dt del análisis
    #===== fact es el factor escalar del registro
    #===== damp es el porcentaje de amortiguamiento (EN DECIMAL. p.ej: 0.03 para 3%)
    #===== IDcrtlNode es el nodo de control para grabar desplazamientos
    #===== IDctrlDOF es el grado de libertad de control
    #===== elements son los elementos de los que se va a grabar información
    #===== nodes_control son los nodos donde se va a grabar las respuestas
    #===== Kswitch recibe: 1: matriz inicial, 2: matriz actual
    
    maxNumIter = 10
    
    #============== Creación del pattern ==============
    
    ops.timeSeries('Path',1000,'-filePath',recordName,'-dt',dtrec,'-factor',fact)
    ops.pattern('UniformExcitation',  1000,   1,  '-accel', 1000)
    
    #============== Damping ==============
    nmodes = max(modes)+1
    eigval = ops.eigen(nmodes)
    
    eig1 = eigval[modes[0]]
    eig2 = eigval[modes[1]]
    
    w1 = eig1**0.5
    w2 = eig2**0.5
    
    beta = 2.0*damp/(w1 + w2)
    alfa = 2.0*damp*w1*w2/(w1 + w2)
    
    if Kswitch == 1:
        ops.rayleigh(alfa, 0.0, beta, 0.0)
    else:
        ops.rayleigh(alfa, beta, 0.0, 0.0)
    
    #============== Configuración básica del análisis ==============
    ops.wipeAnalysis()
    ops.constraints('Plain')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('EnergyIncr', Tol, maxNumIter)
    ops.algorithm('Newton')    
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')
    
    #============== Otras opciones de análisis ==============
    tests = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
    algoritmo = {1:'KrylovNewton', 2: 'SecantNewton' , 4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

    #============== Rutina del análisis ==============
    
    Nsteps =  int(dtrec*nPts/dtan)                              # Numero de pasos para el analisis
    nels = len(ele_record)                                      # Numero de elementos seleccionados para guardar los resultados
    IDctrlTecho = nodes_control.index(max(nodes_control))           # Nodo de control para el analisis
    nnodos = len(nodes_control)                                   # Nummero de nodos libres
    
    #============== Desplazamientos en el arquetipo ==============
    tiempo = np.zeros(Nsteps)*np.nan

    #============== Registros de las deformaciones unitarias en el primer nivel ==============

    # Strains = np.zeros((nels, Nsteps, nfib))                                   # Graba las deformaciones unitarias de los muros en las nfib que tienen los elementos.
    # cStress = np.zeros((nels, Nsteps, nfib))                                 # Graba los esfuerzos del concreto de los muros en las nfib que tienen los elementos.
    # sStress = np.zeros((nels, Nsteps, nfib))                                 # Graba los esfuerzos del acero de los muros en las nfib que tienen los elementos.

    #============== Registros de las desplazamientos de piso ==============
    Disp = np.zeros((Nsteps, nnodos))                                    # Graba los desplazamiento en cada piso por muro.
    Accel = np.zeros((Nsteps, nnodos))                                   # Graba las aceleraciones en cada piso por muro.
    # Vel = np.zeros((Nsteps, nnodos))                                   # Graba las velocidades en cada piso por muro.
    SDR = np.zeros((Nsteps, nnodos))

    #============== Registros de las fuerzas en la base del arquetipo ==============
    # Curvature = np.zeros((nels, Nsteps,1))                                   # Graba la curvatura del elemento en cada piso por muro.
    ForcesGlobal = np.zeros((nels, Nsteps, 3))                                 # Eds: Graba las fuerzas globales (V, P, M) de cada muro en la base.

    for k in range(Nsteps):
        ok = ops.analyze(1,dtan)
        # ok2 = ok;
        # En caso de no converger en un paso entra al condicional que sigue
        if ok != 0:
            print('configuración por defecto no converge en tiempo: ',ops.getTime())
            # Print comentados buscando optimizar los tiempos de ejecucion
            for j in algoritmo:
                if j < 4:
                    ops.algorithm(algoritmo[j], '-initial')
    
                else:
                    ops.algorithm(algoritmo[j])
                
                # El test se hace 50 veces más
                ops.test('EnergyIncr', Tol, maxNumIter*50)
                ok = ops.analyze(1,dtan)
                if ok == 0:
                    # Si converge vuelve a las opciones iniciales de análisis
                    ops.test('EnergyIncr', Tol, maxNumIter)
                    ops.algorithm('Newton')
                    break
                    
        if ok != 0:
            print('Análisis dinámico fallido')
            print('Desplazamiento alcanzado: ',ops.nodeDisp(IDctrlNode,IDctrlDOF),'m')
            # Print comentados buscando optimizar los tiempos de ejecucion
            break
        
        if k != 0:
            for ne, ele in enumerate(ele_record):
                # Strains[ne, k, :] = ops.eleResponse(ele, 'Fiber_Strain')                # Deformación unitaria en cada fibra de los muros en la base.
                # cStress[ne, k, :] = ops.eleResponse(ele, 'Fiber_Stress_Concrete')       # Esfuerzo del concreto en cada fibra de los muros en la base.
                # sStress[ne, k, :] = ops.eleResponse(ele, 'Fiber_Stress_Steel')          # Esfuerzo del acero en cada fibra de los muros en la base.               
                # Curvature[ne, k, :] = ops.eleResponse(ele, 'Curvature')[0]              # Fuerzas globales en cada muro en la base.
                ForcesGlobal[ne, k, :] = ops.eleResponse(ele, 'globalForce')[3:]          # Fuerzas globales en cada muro en la base.

            for nn, nnode in enumerate(nodes_control):
                Disp[k, nn] = ops.nodeDisp(nnode, IDctrlDOF)                                                # Desplazamiento del nodo en el muro de control (Primero) en cada piso
                Accel[k, nn] = ops.nodeAccel(nnode, IDctrlDOF)                                              # Aceleracion del nodo en el muro de control (Primero) en cada piso
                SDR[k, nn] = ((ops.nodeDisp(nnode, IDctrlDOF) - ops.nodeDisp((nnode-1), IDctrlDOF))/        # Deriva de entrepiso del nodo en el muro de control (Primero) en cada piso
                                  (ops.nodeCoord(nnode, 2) - ops.nodeCoord((nnode-1),2)))*100
                
        tiempo[k] = ops.getTime()                                   # Tiempo de corrida 

    #------- Periodo elongado
    eig_i = ops.eigen('-fullGenLapack',1)                                                        # Determina el valor propio wn2 para el modo fundamental, este se requiere en el análisis pushover
    Telong = 2*np.pi/np.sqrt(eig_i[0])  
    
    #------- Deriva de Techo
    rDisp = Disp[:,nn]
    
    ops.wipe()
    # return tiempo,techo,ForcesGlobal,Strains,cStress,sStress,node_disp,node_vel,node_acel,drift
    return tiempo, rDisp, SDR, Telong, Disp, Accel, ForcesGlobal

#------------------------------ 2.2 Análisis Dinámico del arquetipo Modificando el calculo de la frecuencia por fuera de esta funcion.
def dinamicoIDA4W(recordName,dtrec,nPts,dtan,fact,damp,IDctrlNode,IDctrlDOF,ele_record,nodes_control, w, Kswitch = 1,Tol=1e-8):
    # PARA SER UTILIZADO PARA CORRER EN PARALELO LOS SISMOS Y EXTRAYENDO LAS FUERZAS DE LOS ELEMENTOS INDICADOS EN ELEMENTS
    
    #===== record es el nombre del registro, incluyendo extensión. P.ej. GM01.txt
    #===== dtrec es el dt del registro
    #===== nPts es el número de puntos del análisis
    #===== dtan es el dt del análisis
    #===== fact es el factor escalar del registro
    #===== damp es el porcentaje de amortiguamiento (EN DECIMAL. p.ej: 0.03 para 3%)
    #===== IDcrtlNode es el nodo de control para grabar desplazamientos
    #===== IDctrlDOF es el grado de libertad de control
    #===== elements son los elementos de los que se va a grabar información
    #===== nodes_control son los nodos donde se va a grabar las respuestas
    #===== Kswitch recibe: 1: matriz inicial, 2: matriz actual
    
    maxNumIter = 10
    
    #============== Creación del pattern ==============
    
    ops.timeSeries('Path',1000,'-filePath',recordName,'-dt',dtrec,'-factor',fact)
    ops.pattern('UniformExcitation',  1000,   1,  '-accel', 1000)
    
    #============== Damping ==============
    w1 = w[0]
    w2 = w[1]
    
    beta = 2.0*damp/(w1 + w2)
    alfa = 2.0*damp*w1*w2/(w1 + w2)
    
    if Kswitch == 1:
        ops.rayleigh(alfa, 0.0, beta, 0.0)
    else:
        ops.rayleigh(alfa, beta, 0.0, 0.0)
    
    #============== Configuración básica del análisis ==============
    ops.wipeAnalysis()
    ops.constraints('Plain')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('EnergyIncr', Tol, maxNumIter)
    ops.algorithm('Newton')    
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')
    
    #============== Otras opciones de análisis ==============
    tests = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
    algoritmo = {1:'KrylovNewton', 2: 'SecantNewton' , 4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

    #============== Rutina del análisis ==============
    
    Nsteps =  int(dtrec*nPts/dtan)                                                   # Numero de pasos para el analisis
    nels = len(ele_record)                                                           # Numero de elementos seleccionados para guardar los resultados
    IDctrlTecho = nodes_control.index(max(nodes_control))                            # Nodo de control para el analisis
    nnodos = len(nodes_control)                                                      # Nummero de nodos libres
    
    #============== Desplazamientos en el arquetipo ==============
    tiempo = np.zeros(Nsteps)*np.nan

    #============== Registros de las deformaciones unitarias en el primer nivel ==============

    # Strains = np.zeros((nels, Nsteps, nfib))                                                   # Graba las deformaciones unitarias de los muros en las nfib que tienen los elementos.
    # cStress = np.zeros((nels, Nsteps, nfib))                                                   # Graba los esfuerzos del concreto de los muros en las nfib que tienen los elementos.
    # sStress = np.zeros((nels, Nsteps, nfib))                                                   # Graba los esfuerzos del acero de los muros en las nfib que tienen los elementos.

    #============== Registros de las desplazamientos de piso ==============
    Disp = np.zeros((Nsteps, nnodos))                                                            # Graba los desplazamiento en cada piso por muro.
    Accel = np.zeros((Nsteps, nnodos))                                                           # Graba las aceleraciones en cada piso por muro.
    # Vel = np.zeros((Nsteps, nnodos))                                                           # Graba las velocidades en cada piso por muro.
    SDR = np.zeros((Nsteps, nnodos))

    #============== Registros de las fuerzas en la base del arquetipo ==============
    # Curvature = np.zeros((nels, Nsteps,1))                                                     # Graba la curvatura del elemento en cada piso por muro.
    ForcesGlobal = np.zeros((nels, Nsteps, 3))                                                   # Eds: Graba las fuerzas globales (V, P, M) de cada muro en la base.
    # Telong_i = []
    for k in range(Nsteps):
        ok = ops.analyze(1,dtan)
        # ok2 = ok;
        # En caso de no converger en un paso entra al condicional que sigue
        if ok != 0:
            # print('configuración por defecto no converge en tiempo: ',ops.getTime())
            # Print comentados buscando optimizar los tiempos de ejecucion
            for j in algoritmo:
                if j < 4:
                    ops.algorithm(algoritmo[j], '-initial')
    
                else:
                    ops.algorithm(algoritmo[j])
                
                # El test se hace 50 veces más
                ops.test('EnergyIncr', Tol, maxNumIter*5)
                ok = ops.analyze(1,dtan)
                if ok == 0:
                    # Si converge vuelve a las opciones iniciales de análisis
                    ops.test('EnergyIncr', Tol, maxNumIter)
                    ops.algorithm('Newton')
                    break
                    
        # if ok != 0:
            # print('Análisis dinámico fallido')
            # print('Desplazamiento alcanzado: ',ops.nodeDisp(IDctrlNode,IDctrlDOF),'m')
            # Print comentados buscando optimizar los tiempos de ejecucion
            # break
        
        if k != 0:
            for ne, ele in enumerate(ele_record):
                # Strains[ne, k, :] = ops.eleResponse(ele, 'Fiber_Strain')                                  # Deformación unitaria en cada fibra de los muros en la base.
                # cStress[ne, k, :] = ops.eleResponse(ele, 'Fiber_Stress_Concrete')                         # Esfuerzo del concreto en cada fibra de los muros en la base.
                # sStress[ne, k, :] = ops.eleResponse(ele, 'Fiber_Stress_Steel')                            # Esfuerzo del acero en cada fibra de los muros en la base.               
                # Curvature[ne, k, :] = ops.eleResponse(ele, 'Curvature')[0]                                # Fuerzas globales en cada muro en la base.
                ForcesGlobal[ne, k, :] = ops.eleResponse(ele, 'globalForce')[:3]                            # Fuerzas globales en cada muro en la base.

            for nn, nnode in enumerate(nodes_control):
                Disp[k, nn] = ops.nodeDisp(nnode, IDctrlDOF)                                                # Desplazamiento del nodo en el muro de control (Primero) en cada piso
                Accel[k, nn] = ops.nodeAccel(nnode, IDctrlDOF)                                              # Aceleracion del nodo en el muro de control (Primero) en cada piso
                SDR[k, nn] = ((ops.nodeDisp(nnode, IDctrlDOF) - ops.nodeDisp((nnode-1), IDctrlDOF))/        # Deriva de entrepiso del nodo en el muro de control (Primero) en cada piso
                                  (ops.nodeCoord(nnode, 2) - ops.nodeCoord((nnode-1),2)))*100
                
        tiempo[k] = ops.getTime()                                               # Tiempo de corrida 
        #------- Periodo elongado
        # eig_i = ops.eigen(1)                                                        # Determina el valor propio wn2 para el modo fundamental, este se requiere en el análisis pushover
        # Telong = 2*np.pi/np.sqrt(eig_i[0])
        # Telong_i.append(Telong)
    #------- Periodo elongado
    eig_i = ops.eigen('-fullGenLapack',1)                                                        # Determina el valor propio wn2 para el modo fundamental, este se requiere en el análisis pushover
    Telong = 2*np.pi/np.sqrt(eig_i[0])
    
    #------- Deriva de Techo
    rDisp = Disp[:,nn]
    
    ops.wipe()
    return tiempo, rDisp, SDR, Telong, Disp, Accel, ForcesGlobal

#------------------------------ 2.2 Análisis Dinámico del arquetipo Modificando el calculo de la frecuencia por fuera de esta funcion.
def dinamicoIDA4_Curvatura(recordName,dtrec,nPts,dtan,fact,damp,IDctrlNode,IDctrlDOF,ele_record,nodes_control, w, Kswitch = 1,Tol=1e-8):
    # PARA SER UTILIZADO PARA CORRER EN PARALELO LOS SISMOS Y EXTRAYENDO LAS FUERZAS DE LOS ELEMENTOS INDICADOS EN ELEMENTS
    
    #===== record es el nombre del registro, incluyendo extensión. P.ej. GM01.txt
    #===== dtrec es el dt del registro
    #===== nPts es el número de puntos del análisis
    #===== dtan es el dt del análisis
    #===== fact es el factor escalar del registro
    #===== damp es el porcentaje de amortiguamiento (EN DECIMAL. p.ej: 0.03 para 3%)
    #===== IDcrtlNode es el nodo de control para grabar desplazamientos
    #===== IDctrlDOF es el grado de libertad de control
    #===== elements son los elementos de los que se va a grabar información
    #===== nodes_control son los nodos donde se va a grabar las respuestas
    #===== Kswitch recibe: 1: matriz inicial, 2: matriz actual
    
    maxNumIter = 10
    
    #============== Creación del pattern ==============
    
    ops.timeSeries('Path',1000,'-filePath',recordName,'-dt',dtrec,'-factor',fact)
    ops.pattern('UniformExcitation',  1000,   1,  '-accel', 1000)
    
    #============== Damping ==============
    w1 = w[0]
    w2 = w[1]
    
    beta = 2.0*damp/(w1 + w2)
    alfa = 2.0*damp*w1*w2/(w1 + w2)
    
    if Kswitch == 1:
        ops.rayleigh(alfa, 0.0, beta, 0.0)
    else:
        ops.rayleigh(alfa, beta, 0.0, 0.0)
    
    #============== Configuración básica del análisis ==============
    ops.wipeAnalysis()
    ops.constraints('Plain')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('EnergyIncr', Tol, maxNumIter)
    ops.algorithm('Newton')    
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')
    
    #============== Otras opciones de análisis ==============
    tests = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
    algoritmo = {1:'KrylovNewton', 2: 'SecantNewton' , 4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

    #============== Rutina del análisis ==============
    
    Nsteps =  int(dtrec*nPts/dtan)                                                   # Numero de pasos para el analisis
    nels = len(ele_record)                                                           # Numero de elementos seleccionados para guardar los resultados
    IDctrlTecho = nodes_control.index(max(nodes_control))                            # Nodo de control para el analisis
    nnodos = len(nodes_control)                                                      # Nummero de nodos libres
    
    #============== Desplazamientos en el arquetipo ==============
    tiempo = np.zeros(Nsteps)*np.nan

    #============== Registros de las deformaciones unitarias en el primer nivel ==============

    # Strains = np.zeros((nels, Nsteps, nfib))                                                   # Graba las deformaciones unitarias de los muros en las nfib que tienen los elementos.
    # cStress = np.zeros((nels, Nsteps, nfib))                                                   # Graba los esfuerzos del concreto de los muros en las nfib que tienen los elementos.
    # sStress = np.zeros((nels, Nsteps, nfib))                                                   # Graba los esfuerzos del acero de los muros en las nfib que tienen los elementos.

    #============== Registros de las desplazamientos de piso ==============
    Disp = np.zeros((Nsteps, nnodos))                                                            # Graba los desplazamiento en cada piso por muro.
    # Accel = np.zeros((Nsteps, nnodos))                                                           # Graba las aceleraciones en cada piso por muro.
    # Vel = np.zeros((Nsteps, nnodos))                                                           # Graba las velocidades en cada piso por muro.
    # SDR = np.zeros((Nsteps, nnodos))

    #============== Registros de las fuerzas en la base del arquetipo ==============
    Curvature = np.zeros((nels, Nsteps, nnodos))                                                     # Graba la curvatura del elemento en cada piso por muro.
    # ForcesGlobal = np.zeros((nels, Nsteps, 3))                                                   # Eds: Graba las fuerzas globales (V, P, M) de cada muro en la base.
    # Telong_i = []
    for k in range(Nsteps):
        ok = ops.analyze(1,dtan)
        # ok2 = ok;
        # En caso de no converger en un paso entra al condicional que sigue
        if ok != 0:
            # print('configuración por defecto no converge en tiempo: ',ops.getTime())
            # Print comentados buscando optimizar los tiempos de ejecucion
            for j in algoritmo:
                if j < 4:
                    ops.algorithm(algoritmo[j], '-initial')
    
                else:
                    ops.algorithm(algoritmo[j])
                
                # El test se hace 50 veces más
                ops.test('EnergyIncr', Tol, maxNumIter*5)
                ok = ops.analyze(1,dtan)
                if ok == 0:
                    # Si converge vuelve a las opciones iniciales de análisis
                    ops.test('EnergyIncr', Tol, maxNumIter)
                    ops.algorithm('Newton')
                    break
                    
        # if ok != 0:
            # print('Análisis dinámico fallido')
            # print('Desplazamiento alcanzado: ',ops.nodeDisp(IDctrlNode,IDctrlDOF),'m')
            # Print comentados buscando optimizar los tiempos de ejecucion
            # break
        
        if k != 0:
            for ne, ele in enumerate(ele_record):
                # Strains[ne, k, :] = ops.eleResponse(ele, 'Fiber_Strain')                                  # Deformación unitaria en cada fibra de los muros en la base.
                # cStress[ne, k, :] = ops.eleResponse(ele, 'Fiber_Stress_Concrete')                         # Esfuerzo del concreto en cada fibra de los muros en la base.
                # sStress[ne, k, :] = ops.eleResponse(ele, 'Fiber_Stress_Steel')                            # Esfuerzo del acero en cada fibra de los muros en la base.               
                # Curvature[ne, k, :] = ops.eleResponse(ele, 'Curvature')[0]                                # Fuerzas globales en cada muro en la base.
                # ForcesGlobal[ne, k, :] = ops.eleResponse(ele, 'globalForce')[:3]                            # Fuerzas globales en cada muro en la base.

                for nn, nnode in enumerate(ele):
                    Disp[k, nn] = ops.nodeDisp(nnode, IDctrlDOF)                                                # Desplazamiento del nodo en el muro de control (Primero) en cada piso
                    # Accel[k, nn] = ops.nodeAccel(nnode, IDctrlDOF)                                              # Aceleracion del nodo en el muro de control (Primero) en cada piso
                    # SDR[k, nn] = ((ops.nodeDisp(nnode, IDctrlDOF) - ops.nodeDisp((nnode-1), IDctrlDOF))/        # Deriva de entrepiso del nodo en el muro de control (Primero) en cada piso
                                      # (ops.nodeCoord(nnode, 2) - ops.nodeCoord((nnode-1),2)))*100
                    Curvature[ne, k, nn] = ops.eleResponse(nnode, 'Curvature')[0]                                # Fuerzas globales en cada muro en la base.
        # tiempo[k] = ops.getTime()                                               # Tiempo de corrida 
        #------- Periodo elongado
        # eig_i = ops.eigen(1)                                                        # Determina el valor propio wn2 para el modo fundamental, este se requiere en el análisis pushover
        # Telong = 2*np.pi/np.sqrt(eig_i[0])
        # Telong_i.append(Telong)
    #------- Periodo elongado
    # eig_i = ops.eigen('-fullGenLapack',1)                                                        # Determina el valor propio wn2 para el modo fundamental, este se requiere en el análisis pushover
    # Telong = 2*np.pi/np.sqrt(eig_i[0])
    
    #------- Deriva de Techo
    # rDisp = Disp[:,nn]
    
    ops.wipe()
    return Curvature, Disp
#%% ============================ 3. ANÁLISIS DE UN GRADO DE LIBERTAD =========================
#========================================================================================

#------------------------------ 3.0 Análisis de gravedad del SEDOF.
def gravedad_SEDOF():

    # Realiza el analisis por gravedad
    ops.constraints('Plain'); # Create the constraint handler, the transformation method
    ops.numberer('Plain'); # Create the DOF numberer, the reverse Cuthill-McKee algorithm, renumber dof's to minimize band-width (optimization), if you want to
    ops.system('BandGeneral'); # Create the system of equation, a sparse solver with partial pivoting, how to store and solve the system of equations in the analysis
    ops.test('NormDispIncr', 1.0e-8, 10, 6); # Create the convergence test, the norm of the residual with a tolerance of 1e-12 and a max number of iterations of 10 determine if convergence has been achieved at the end of an iteration step
    ops.algorithm('Newton'); # Create the solution algorithm, use Newton's solution algorithm: updates tangent stiffness at every iteration
    ops.integrator('LoadControl', 0.1); # Create the integration scheme, the LoadControl scheme using steps of 0.1
    ops.analysis('Static');# Create the analysis object
    ok = ops.analyze(10)
        
    if ok != 0:
        print('----------------------------------------------')
        print('--------Análisis de gravedad fallido----------')
        print('----------------------------------------------')
        ops.sys.exit()
    else:
        print('----------------------------------------------')
        print('--------Análisis de gravedad realizado--------')
        print('----------------------------------------------')
   
#------------------------------ 3.1 Análisis modal de un sistema de un grado de libertad.        
def ModalAnalysis(mat_Masa, NodosLibres, N_Nodes, ndf):

    omega2 = ops.eigen('fullGenLapack', N_Nodes)
    mat_Modos = []
    vec_Phi = []
    LnM = []
    MnM = []
    GammaM = []
    T_n = []
    s_n = []
    vec_Me = []    
    for n in range(N_Nodes):
        Phi_n = []    
        for nd in NodosLibres:
            nd=int(nd)
            ndMass = ops.nodeMass(nd)
            MNV = ops.nodeEigenvector(nd,n+1)          
            Phi_n.append(MNV[0])
            
        Tn = 2*np.pi/omega2[n]**0.5
        Phi_n = np.array(Phi_n)     
        Ln = np.trace(Phi_n*mat_Masa)
        Mn = np.trace(Phi_n**2*mat_Masa)
        Gamman = Ln/Mn
        mat_Modos.append(n+1)
        vec_Phi.append(Phi_n)
        LnM.append(Ln)
        MnM.append(Mn)
        GammaM.append(Gamman)
        T_n.append(Tn) 
        sn = Gamman*np.dot(Phi_n,mat_Masa)
        s_n.append(sn)
        vec_Me.append(Ln*Gamman)
        
    Modos = np.array(mat_Modos)
    vec_Phi = np.array(vec_Phi)
    LnM = np.array(LnM)
    MnM = np.array(MnM)
    GammaM = np.array(GammaM)
    T_n = np.array(T_n)
    s_n = np.array(s_n)
    vec_Me = np.array(vec_Me)

    Result_MA = {'Modo': None, 'Phi': None, 'Ln': None, 'Mn': None, 'Gamman': None, 'Tn': None, 'Sn': None, 'M*': None}
    Result_MA['Modo'] = Modos
    Result_MA['Phi'] = vec_Phi
    Result_MA['Ln'] = LnM
    Result_MA['Mn'] = MnM
    Result_MA['Gamman'] = GammaM
    Result_MA['Tn'] = T_n
    Result_MA['Sn'] = s_n
    Result_MA['M*'] = vec_Me

    Data_ModoFund = {'phi1_norm':vec_Phi[0],'s1':s_n[0], 'Gamma_norm':np.ones(len(NodosLibres))*GammaM[0]}
    ModoFund = pd.DataFrame(Data_ModoFund, index =ops.getNodeTags()[1:N_Nodes+1])

    return ModoFund, Result_MA

#------------------------------ 3.2 Análisis inelastico de un sistema de un grado de libertad.
def inelastic_SDF(pushover, Result_MA, plot = False):
#---Convesión del pushover al sistema inelastico SDOF
    Fsn_Ln = pushover['Vbn']/Result_MA['M*'][0]                                                        
    Dn = pushover['Urn']/(Result_MA['Phi'][0][len(Result_MA['Phi'][0])-1]*Result_MA['Gamman'][0])

    #Punto de fluencia
    Fsny_Ln = Fsn_Ln[1]
    Dny = Dn[1]

    Ke = Fsny_Ln/Dny                                                            # Rigidez efectiva del sistema inelastico SDOF
    Me = Result_MA['M*'][0]                                                     # Masa modal efectiva
    Te = 2*np.pi*np.sqrt(Dny/Fsny_Ln)                                           # Periodo de vibración elastica
    wne = 2*np.pi/Te                                                            # Frecuencia angular del sistema de SDOF
    Data_P_SEDOF = {'DataSDOF':[Te,Me,Ke,wne],'Unit':['s','t','kN/m','rad/seg']}
    P_SEDOF = pd.DataFrame(Data_P_SEDOF, index =['T*','M*','K*','wn*'])
    if plot == True:
        plt.figure()
        plt.plot(Dn, Fsn_Ln, label='SEDOF', linewidth=1.5,linestyle = '--', marker = 'o',color='g' )
        plt.legend(loc='lower right')
        plt.xlabel('$D_1y$' + ' (m)')
        plt.ylabel('$Fs_n$'+'/'+'$L_n$'+ ' ($m/s^{2}$)')
    
    SEDOF = pd.concat([Dn, Fsn_Ln], axis=1)
    SEDOF = SEDOF.rename(columns = {'Urn':'Dn','Vbn':'Fsn_Ln'})
    return P_SEDOF, SEDOF

#------------------------------ 3.3 Modelo del SEDOF.
def Model_SDOF(P_SEDOF, SEDOF, Result_MA, idealtype):
    print('--------------------------------------------------')
    print('---------------Creando modelo SDOF----------------')
    print('--------------------------------------------------')
    
    # ------------------------ SET UP ----------------------------------------
    ops.model('basic', '-ndm', 2, '-ndf', 3)                                   # 2 dimensions, 3 dof per node
    # --------------------------------------------
    # Set geometry, nodes, boundary conditions
    # --------------------------------------------
    T = P_SEDOF['DataSDOF'][0]
    masa = P_SEDOF['DataSDOF'][1]
    w = P_SEDOF['DataSDOF'][3]
    K = 4*masa*(np.pi/T)**2
    
    # nodal coordinates:
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 0.0)

    # Set Control Node and DOF
    IDctrlNode = 2
    IDctrlDOF = 1

    eleNodes = [1,2]
    ops.equalDOF(1, 2, *[2, 3])
    # Mass asign
    ops.mass(2, masa, 0.0, 0.0)          # node#, Mx My Mz, Mass=Weight/g.

    # Single point constraints -- Boundary Conditions
    #   tag, DX, DY, RZ
    ops.fix(1, 1, 1, 1)

    # Define MATERIAL ---------------------------------------------------------
    #-----Steel 01
    # bilinear_mat_tag = 3
    # mat_type = "Steel01"
    # f_yield = 5.0
    # k_spring = K
    # r_post = 0.0
    # mat_props = [f_yield, k_spring, r_post]

    # ops.uniaxialMaterial(mat_type, bilinear_mat_tag, *mat_props)

    # Hysteretic Material
    if idealtype <= 1:
        Fpc = SEDOF['Fsn_Ln'][1]
        epc = SEDOF['Dn'][1]
        Fpy = SEDOF['Fsn_Ln'][2]
        epy = SEDOF['Dn'][2]
        Fpm = SEDOF['Fsn_Ln'][2]*1.000001
        epm = SEDOF['Dn'][2]*1.000001
    elif idealtype == 2:
        Fpc = SEDOF['Fsn_Ln'][1]
        epc = SEDOF['Dn'][1]
        Fpy = SEDOF['Fsn_Ln'][2]
        epy = SEDOF['Dn'][2]
        Fpm = SEDOF['Fsn_Ln'][3]
        epm = SEDOF['Dn'][3]
    elif idealtype == 3:
        Fpc = SEDOF['Fsn_Ln'][2]*Result_MA['Ln'][0]
        epc = SEDOF['Dn'][2]
        Fpy = SEDOF['Fsn_Ln'][3]*Result_MA['Ln'][0]
        epy = SEDOF['Dn'][3]
        Fpm = SEDOF['Fsn_Ln'][4]*Result_MA['Ln'][0]
        epm = SEDOF['Dn'][4]
    elif idealtype == 4:
        Fpc = SEDOF['Fsn_Ln'][1]*Result_MA['Ln'][0]
        epc = SEDOF['Dn'][1]
        Fpy = SEDOF['Fsn_Ln'][2]*Result_MA['Ln'][0]
        epy = SEDOF['Dn'][2]
        Fpm = SEDOF['Fsn_Ln'][3]*Result_MA['Ln'][0]
        epm = SEDOF['Dn'][3]
    matTag = 3000
    Tipo_M = 'FORCES'
    Tipo_R = 'WWM'
    R = mat.Refuerzo_SDOF(Tipo_R, matTag, Tipo_M, p1 = [Fpc, epc], p2 =[Fpy, epy], p3 = [Fpm, epm])    # Definición de una material que representará la histeresis del SDOF

    # Define ELEMENTS -------------------------------------------------------------
    #Tipo de sección
    rFlag=1
    eleTag = 1
    ops.element('zeroLength', eleTag, *eleNodes, '-mat', 3000, '-dir',1, '-doRayleigh', rFlag)
    
    print('--------------------------------------------------')
    print('---------------Modelo SDOF Creado-----------------')
    print('--------------------------------------------------')
    # return eleTag, IDctrlNode, IDctrlDOF

#------------------------------ 3.4 Análisis pushover del SEDOF
def Pushover_SDOF(Dmax, Dincr, IDctrlNode, IDctrlDOF, norm=[-1,1],Tol=1e-8, plot = 'False', plotperiod = 'False', plotnorm = 'False'):
    
    # creación del recorder de techo y definición de la tolerancia
    #recorder('Node','-file','techo.out','-time','-node',IDctrlNode,'-dof',IDctrlDOF,'disp')
    maxNumIter = 10
          
    # configuración básica del análisis
    ops.wipeAnalysis()
    ops.constraints('Plain')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('EnergyIncr', Tol, maxNumIter)
    ops.algorithm('Newton')    
    ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)
    ops.analysis('Static')
    
    # Otras opciones de análisis    
    tests = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
    algoritmo = {1:'KrylovNewton', 2: 'SecantNewton' , 4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

    Nsteps =  int(Dmax/ Dincr)
    dtecho = [ops.nodeDisp(IDctrlNode,IDctrlDOF)]                                                                            # Numero de pasos de análisis
    Vbasal = [ops.getTime()]

    for k in range(Nsteps):
        ok = ops.analyze(1)        
        # ok2 = ok;
        # En caso de no converger en un paso entra al condicional que sigue
        if ok != 0:
            print('configuración por defecto no converge en desplazamiento: ', ops.nodeDisp(IDctrlNode,IDctrlDOF))
            for j in algoritmo:
                if j < 4:
                    ops.algorithm(algoritmo[j], '-initial')  
                else:
                    ops.algorithm(algoritmo[j])                
                # el test se hace 50 veces más
                ops.test('EnergyIncr', Tol, maxNumIter*50)
                ok = ops.analyze(1)                
                if ok == 0:
                    # si converge vuelve a las opciones iniciales de análisi
                    ops.test('EnergyIncr', Tol, maxNumIter)
                    ops.algorithm('Newton')
                    break    
                
        if ok != 0:
            print('Pushover analisis fallido')
            print('Desplazamiento alcanzado: ', ops.nodeDisp(IDctrlNode,IDctrlDOF),'m')
            break       
        
        dtecho.append(ops.nodeDisp(IDctrlNode,IDctrlDOF))
        Vbasal.append(ops.getTime())
    dntecho = np.array(dtecho)     
    V = np.array(Vbasal)
    
    if plot == 'True':   
        plt.figure()
        plt.plot(dn1,V)
        plt.xlabel('desplazamiento de techo (m)')
        plt.ylabel('corte basal (kN)')

    if plotnorm == 'True':
        deriva = dn1/norm[0]*100
        VW = V/norm[1]
        plt.figure()
        plt.plot(deriva,VW)
        plt.xlabel('Deriva de techo (%)')
        plt.ylabel('V/W')  
        
    return dntecho, V

#------------------------------ 3.5 Análisis Dinámico Incremental (IDA) del SEDOF.
def dinamicoIDA_SDOF(accel,dtrec,nPts,dtan,fact,damp,IDctrlNode,IDctrlDOF,ele_record, wn,modes = [0,2],Kswitch = 1,Tol=1e-8):
    
    # PARA SER UTILIZADO PARA CORRER EN PARALELO LOS SISMOS Y EXTRAYENDO LAS FUERZAS DE LOS ELEMENTOS INDICADOS EN ELEMENTS
    
    # record es el nombre del registro, incluyendo extensión. P.ej. GM01.txt
    # dtrec es el dt del registro
    # nPts es el número de puntos del análisis
    # dtan es el dt del análisis
    # fact es el factor escalar del registro
    # damp es el porcentaje de amortiguamiento (EN DECIMAL. p.ej: 0.03 para 3%)
    # IDcrtlNode es el nodo de control para grabar desplazamientos
    # IDctrlDOF es el grado de libertad de control
    # elements son los elementos de los que se va a grabar información
    # nodes_control son los nodos donde se va a grabar las respuestas
    # Kswitch recibe: 1: matriz inicial, 2: matriz actual
    
    maxNumIter = 10
    
    # creación del pattern
    # creación del pattern
    ops.timeSeries('Path', 1000, '-dt', dtrec, '-values', *accel, '-factor', fact)     # timeSeries('Path', tag, '-dt', dt=0.0, '-values', *values, '-time', *time, '-filePath', filePath='', '-fileTime', fileTime='', '-factor', factor=1.0, '-startTime', startTime=0.0, '-useLast', '-prependZero')  
    
    # timeSeries('Path',1000,'-filePath',recordName,'-dt',dtrec,'-factor',fact)
    ops.pattern('UniformExcitation', 1000, 1, '-accel', 1000)

    # # damping
    # nmodes = max(modes)+1
    # eigval = eigen(nmodes)
    
    # eig1 = eigval[modes[0]]
    # eig2 = eigval[modes[1]]
    
    # w1 = eig1**0.5
    # w2 = eig2**0.5
    
    beta = 2.0*damp/(wn + 1.3*wn)
    alfa = 2.0*damp*wn*wn*1.3/(wn + 1.3*wn)

    if Kswitch == 1:
        ops.rayleigh(alfa, 0.0, beta, 0.0)
    else:
        ops.rayleigh(alfa, beta, 0.0, 0.0)
    
    # configuración básica del análisis
    ops.wipeAnalysis()
    ops.constraints('Plain')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('EnergyIncr', Tol, maxNumIter)
    ops.algorithm('Newton')    
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')
    
    # Otras opciones de análisis    
    tests = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
    algoritmo = {1:'KrylovNewton', 2: 'SecantNewton' , 4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

    # rutina del análisis
    Nsteps =  int(dtrec*nPts/dtan)
    nels = len(ele_record)
    nnodes = len(ele_record)
    tiempo = np.zeros(Nsteps)*np.nan
    #-----Registros de las deformaciones unitarias en el primer nivel.
    DeformationLocal = np.zeros(Nsteps)

    #-----Registros de las desplazamientos de piso
    Disp = np.zeros(Nsteps)                                     # Graba los desplazamiento en cada piso por muro.
    Accel = np.zeros(Nsteps)                                    # Graba las aceleraciones en cada piso por muro.
    Vel = np.zeros(Nsteps)                                      # Graba las velocidades en cada piso por muro.

    #-----Registros de las fuerzas en la base del arquetipo.
    ForcesGlobal = np.zeros((nels, Nsteps, 3))                                  # Graba las fuerzas globales (V, P, M) de cada muro en la base.

    tiempo[0] = ops.getTime()
    for k in range(Nsteps):
        ok = ops.analyze(1,dtan)
        # ok2 = ok;
        # En caso de no converger en un paso entra al condicional que sigue
        if ok != 0:
            print('configuración por defecto no converge en tiempo: ', ops.getTime())
            for j in algoritmo:
                if j < 4:
                    ops.algorithm(algoritmo[j], '-initial')
    
                else:
                    ops.algorithm(algoritmo[j])
                
                # el test se hace 50 veces más
                ops.test('EnergyIncr', Tol, maxNumIter*50)
                ok = ops.analyze(1,dtan)
                if ok == 0:
                    # si converge vuelve a las opciones iniciales de análisi
                    ops.test('EnergyIncr', Tol, maxNumIter)
                    ops.algorithm('Newton')
                    break
                    
        if ok != 0:
            print('Análisis dinámico fallido')
            print('Desplazamiento alcanzado: ', ops.nodeDisp(IDctrlNode,IDctrlDOF),'m')
            break

        if k != 0:
            for ne, ele in enumerate(ele_record):
                ForcesGlobal[ne, k, :] = ops.eleResponse(ele, 'force')[3:]                  # Fuerzas globales en cada muro en la base.
                DeformationLocal[k] = ops.eleResponse(ele, 'deformation')[0]
                for nn, nnode in enumerate([IDctrlNode]):
                    Disp[k] = ops.nodeDisp(nnode, IDctrlDOF)
                    Accel[k] = ops.nodeAccel(nnode, IDctrlDOF)
                    Vel[k] = ops.nodeVel(nnode, IDctrlDOF)
        tiempo[k] = ops.getTime()
        
    #-------Desplazamientos en el arquetipo
    dtecho = Disp
    
    #-------Fuerzas en los elementos
    CNL = (tiempo, dtecho)
    # ------Respuesta local del arquetipo en el analisis dinamico.
    local_response = (DeformationLocal)
    #-------Respuesta global del arquetipo en el analisis dinamico.
    global_response = (ForcesGlobal, Disp, Accel, Vel)
    ops.wipe()
    
    return CNL, local_response, global_response


#%%============================= 4. CREAR UN MODELO ============================
#========================================================================================

#------------------------------ 4.1 Generacion de un modelo de opensees con MVLEM a partir de una clase con la informacion del arquetipo
def CrearModelo(Arquetipo, diafragma=1, seccionViga = [0.3,0.5]):
    Arq1 = Arquetipo
    
    #-------- Definición de la geometría de la viga de acople o conexión entre muros.   

    # Vigas del modelo representan diafragma rigido entre muros.
    bv = seccionViga[0]                                                             # Base de Viga
    hv = seccionViga[1]                                                             # Alto de Viga
    
    Av = bv*hv                                                                      # Área de la viga
    Iv = bv*hv**3/12                                                                # Inercia de la viga
    Ev = 4700000*math.sqrt(21)                                                      # Modulo de elasticidad del concreto para la viga

    ops.wipe()                                                                      # Borra información de modelos existentes (Mientras no funcione bien el script es mejor tener el wipe).
    ops.model('basic', '-ndm', 2, '-ndf', 3)                                            # Crea modelo en 2D con 3 grados de libertad.

    #%% GENERADOR ARQUETIPO
    #%% NODOS, RESTRICCIONES Y ASIGNACIÓN DE MASA
    # =========================================

    ny = Arq1.NumPisos                                                                  # Número de nodos en dirección Y para un solo muro. 'Numero de pisos del edificio'
    xloc = np.linspace(0,len(Arq1.muros)*5,len(Arq1.muros)).tolist()
    nx = len(Arq1.muros)                                                                  # Número de nodos en dirección X por piso. 'Numero de muros del edificio'

    # ------ Asignación de carga
    ops.timeSeries('Linear', 1)                                                     # Definición de tipo de serie de carga para asignar al edificio.
    ops.pattern('Plain', 1, 1)                                                      # Definición de patron de carga para asignar al edificio.

    for i, ele_i in enumerate(Arq1.muros):                                                             # Ciclo for para asignación de masa (toneladas) y fuerzas de gravedad (kN).
        nnode = ele_i.id_*10
        ops.node(nnode, xloc[i], 0.0)
        for j in range(ny-1,-1,-1):
            piso_i = ele_i.pisos[j]
            nnode = piso_i.id_
            ops.node(int(nnode), float(xloc[i]), float(piso_i.CoorY))
            ops.mass(int(nnode), float(piso_i.ws_), float(piso_i.ws_), 0.0)
            ops.load(int(nnode), 0.0, float(-9.81*piso_i.w_), 0.0)
                   
    # apoyos
    empotrado = [1,1,1]                                                             # Definición de grados de libertad empotrados.

    # para colocarlos todos al nivel 0.0
    ops.fixY(0.0,*empotrado)                                                        # Definición de restricción en la base del edificio.

    #%% ASIGNACIÓN DE DIAFRAGMAS
    # =========================================
    if diafragma == 1:
        for j in range(ny):
            for i in range(1,nx):
                masternode = Arq1.muros[0].pisos[j].id_
                slavenode = Arq1.muros[i].pisos[j].id_
                ops.equalDOF(masternode, slavenode, 1)

    #%% MATERIALES Y TRANSFORMACIONES
    # =========================================
    # Arq1.MatConcrete[i].iTagUnc
    # ---- Concreto
    for i in range(len(Arq1.MatConcrete)):
        mat.Concreto('Unconf', int(Arq1.MatConcrete[i].iTagUnc), float(Arq1.MatConcrete[i].fc),'C01')                            #Definición del Concreto sin confinar a partir de la función Materials
        mat.Concreto('Conf', int(Arq1.MatConcrete[i].iTag), float(Arq1.MatConcrete[i].fc), 'C01')                            #Definición del Concreto sin confinar a partir de la función Materials

    # ---- Refuerzo
    mat.Refuerzo('WWM', int(Arq1.MatSteel[0].iTag), float(Arq1.MatSteel[0].fy), 'HYS')      #Definición del acero de refuerzo con malla electrosoldada a partir de la función Materials
    mat.Refuerzo('RB', int(Arq1.MatSteel[1].iTag), float(Arq1.MatSteel[1].fy), 'HYS')      #Definición del acero de refuerzo con malla electrosoldada a partir de la función Materials

    #%% TRANSFORMACIÓN PARA ANÁLISIS

    lineal = 1
    ops.geomTransf('Linear',lineal)                                                 # Definición de la transformación para analisis Lineal

    pdelta = 2
    ops.geomTransf('PDelta',pdelta)                                                 # Definición de la transformación para analisis P-Delta

    cor = 3
    ops.geomTransf('Corotational',cor)                                              # Definición de la transformación para analisis Corotational

    #%% ELEMENTOS VERTICALES (MUROS)
    # ========================
    #element('MVLEM', eleTag, Dens, *eleNodes, m, c, '-thick', *thick, '-width', *widths, '-rho', *rho, '-matConcrete', *matConcreteTags, '-matSteel', *matSteelTags, '-matShear', matShearTag)

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
            
            
    #%% ======================================= ELEMENTOS HORIZONTALES (VIGAS) ======================================
    #================================================================================================================

    if diafragma != 1:    
        for j in range(ny-1,-1,-1):
            for i in range(len(Arq1.muros)): 
                if i<(len(Arq1.muros)-1):
                    nodeI = Arq1.muros[i].pisos[j].id_                              # nodo inicial de la viga
                    nodeJ = Arq1.muros[i+1].pisos[j].id_                            # nodo final de la viga
                    eltag = int(Arq1.muros[i].pisos[j].id_*10)                              # tag de la viga
                    ops.element('elasticBeamColumn',eltag,nodeI,nodeJ,Av,Ev,Iv,lineal)
#