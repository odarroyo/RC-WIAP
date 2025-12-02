"""
Fecha: 01/08/2022
@authores: Frank y JuanJo
===================================================================
--------- PROCESAMIENTO DE LOS REGISTROS SÍSMICOS -----------------
===================================================================

"""
# %% IMPORTAR FUNCIONES, LIBRERIAS PARA LA EJECUCIÓN DEL MODELO
#------Librerias Python
import os, fnmatch                                                              # Importa librería estándar para interactuar con el sistema operativo.
import math                                                                     # Importa librería estándar para utilizar funciones integradoras matematicas.
import numpy as np                                                              # Importa librería estándar para operaciones matematicas.
import pandas as pd                                                             # Importa librería estándar para manejo de bases de datos.
import matplotlib.pyplot as plt                                                 # Importa librería estándar para plotear otras cosasy poder crear figuras.
from scipy import integrate                                                     # Importa librería estándar para analisis matematico y estadisco de datos.
import time as tm                                                                     # Importa librerias estándar para manejar tareas relacionadas con el tiempo.
from matplotlib import style
# %% VELOCIDAD Y DESPLAZAMIENTO DEBIDO A UNA ACELERACIÓN
def calculate_velocity_displacement(time, acc):
    dt = time[1]-time[0]
    velocity = dt * integrate.cumtrapz(acc, initial = 0.)
    displacement = dt * integrate.cumtrapz(velocity, initial = 0.)
    return velocity, displacement
# %% PLOTTING LINE
def Plot_Line(X, Y, Metodo, LabelX, LabelY):        
    plt.figure()
    plt.plot(X, Y, label = Metodo)
    plt.legend(loc = "upper right")
    plt.xlabel(LabelX)
    plt.ylabel(LabelY)
# %% LECTURA DE REGISTROS EN FORMATO .CSV
#Función para la lectura de los registros sísmicos que se tienen en la carpeta definida para el análisis, carga todos y los almacena en un directorio llamado "gmrs"
def Read_Acelerograms(folder, dtype='.csv'):
    time = []
    acc = []
    dt = []
    no_points = []
    name = []
    PGA = []
    x = []
    cont = 0
    for f in os.listdir(folder):
        l=f
        if f.endswith(dtype):
            itime, iacc = [], []
            with open(folder + '\\' + f) as f:
                for line in f.readlines():
                    line = line.split(',')
                    itime.append(float(line[0]))
                    #iacc.append(float(line[1])*9.81)
                    iacc.append(float(line[1]))      #Quitar comentario cuando se trabaje con funciones senoidales

                time.append(itime)
                acc.append(iacc)
                dt.append(itime[1] - itime[0])
                no_points.append(len(iacc))
                name.append(l[:-4])
            PGA.append(max(np.abs(acc[cont])))
            x.append(l)
            cont=cont+1
        
    time = np.array(time, dtype=(object))
    acc = np.array(acc, dtype=(object))	

    gmrs = {'time': time, 'acc': acc, 'dt': dt,'no_points': no_points, 'name': name}
   
    return gmrs

# %% GRAFICA LOS REGISTROS DE LA FUNCIÓN SDOF.Read_Acelerograms()
def Plot_GM(folder, folder_plot, gmrs, id_gmrs=38, Hzlvl=2,
            colors = ['salmon', 'lightsteelblue', 'moccasin', 'palegreen', 'plum', 'peru', 'bisque', 'goldenrod']):
    fig = plt.subplots(figsize=(10, 7.5), dpi=150)
    plt.rcParams["font.family"] = "Cambria"
    plt.rcParams['savefig.bbox'] = "tight"
    plt.plot(gmrs[Hzlvl]['time'][id_gmrs], gmrs[Hzlvl]['acc'][id_gmrs], c=colors[3], lw=0.4)
    plt.title('Nivel Amz 3 - GM='+str(gmrs[Hzlvl]['descripcion'][id_gmrs]), fontsize=22)
    plt.xlabel('Tiempo (s)', fontsize=22)
    plt.ylabel('Aceleración (g)', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.legend(loc='lower right', fontsize=18)                   # place legend outside
    plt.grid(True, which="both", color = 'gainsboro', ls='--', lw=0.5)
    plt.tight_layout()
    return
# %% LECTURA DE ACELEROGRAMAS EN FORMATO TIPO PEER
"""
@author: Daniel Hutabarat - UC Berkeley, 2017
"""

def processNGAfile(filepath, scalefactor=None):
    '''
    This function process acceleration history for NGA data file (.AT2 format)
    to a single column value and return the total number of data points and 
    time iterval of the recording.
    Parameters:
    ------------
    filepath : string (location and name of the file)
    scalefactor : float (Optional) - multiplier factor that is applied to each
                  component in acceleration array.
    
    Output:
    ------------
    desc: Description of the earthquake (e.g., name, year, etc)
    npts: total number of recorded points (acceleration data)
    dt: time interval of recorded points
    time: array (n x 1) - time array, same length with npts
    inp_acc: array (n x 1) - acceleration array, same length with time
             unit usually in (g) unless stated as other.
    
    Example: (plot time vs acceleration)
    filepath = os.path.join(os.getcwd(),'motion_1')
    desc, npts, dt, time, inp_acc = processNGAfile (filepath)
    plt.plot(time,inp_acc)
        
    '''    
    try:
        if not scalefactor:
            scalefactor = 1.0
        with open(filepath,'r') as f:
            content = f.readlines()
        counter = 0
        desc, row4Val, acc_data = "","",[]
        for x in content:
            if counter == 1:
                desc = x
            elif counter == 3:
                row4Val = x
                if row4Val[0][0] == 'N':
                    val = row4Val.split()
                    npts = float(val[(val.index('NPTS='))+1].rstrip(','))
                    dt = float(val[(val.index('DT='))+1])
                else:
                    val = row4Val.split()
                    npts = float(val[0])
                    dt = float(val[1])
            elif counter > 3:
                data = str(x).split()
                for value in data:
                    a = float(value) * scalefactor
                    acc_data.append(a)
                inp_acc = np.asarray(acc_data)
                time = []
                for i in range (0,len(acc_data)):
                    t = i * dt
                    time.append(t)
            counter = counter + 1
        return desc, npts, dt, time, inp_acc
    except IOError:
        print("processMotion FAILED!: File is not in the directory")

# %% LECTURA DE MULTIPLES REGISTROS EN FORMATO .AT2

def readaccePEER(folder, dtype='.AT2'):
    gmrs = ()
    list_dir = [f for f in os.scandir(folder) if f.is_dir()]                       # Extrae de un directorio unicamente las carpetas, ignora los archivos.
    for Hz in list_dir:
        print(Hz.name)
        forder_rec = os.path.join(folder, Hz.name, 'gmotions')
        folder_info = os.path.join(folder, Hz.name, 'listofgmotions.txt')
        list_GM = pd.read_table(folder_info, delimiter = '\s+', header=None)
        list_GM[0] = list_GM[0].replace({'a/':''}, regex=True)
        
        time_rec = []                   #Vector tiempo para cada registro de aceleración.
        acc = []                    #Vector aceleración en terminos de g para cada registro.
        dt_rec = []                     #Delta tiempo en cada registro de aceleración.
        no_points = []              #Numero de pasos de cada registro de aceleración.
        name = []                   #Nombre o código de cada registro.
        desc_rec = []               #Descripción del registro.
        path = []                   #Vector tiempo en el registro de aceleración
        SFactor = []                #Factor de escala de la señal.
        for rec in list_GM[0]:
            filepath = os.path.join(forder_rec, rec)
            scalefactor = list_GM[list_GM[0]==rec].values[0][1]
    
            #desc_rec, npts, dt_rec, t, inp_acc = processNGAfile(accfile)
            """
            @author: Daniel Hutabarat - UC Berkeley, 2017
            """
            try:
                if not scalefactor:
                    scalefactor = 1.0
                with open(filepath,'r') as f:
                    content = f.readlines()
                counter = 0
                desc, row4Val, acc_data = "","",[]
                for x in content:
                    if counter == 1:
                        desc = x
                    elif counter == 3:
                        row4Val = x
                        if row4Val[0][0] == 'N':
                            val = row4Val.split()
                            npts = float(val[(val.index('NPTS='))+1].rstrip(','))
                            dt = float(val[(val.index('DT='))+1])
                        else:
                            val = row4Val.split()
                            npts = float(val[0])
                            dt = float(val[1])
                    elif counter > 3:
                        data = str(x).split()
                        for value in data:
                            a = float(value) * scalefactor
                            acc_data.append(a)
                        inp_acc = np.asarray(acc_data)
                        time = []
                        for i in range (0,len(acc_data)):
                            t = i * dt
                            time.append(t)
                    counter = counter + 1
                if rec.endswith(dtype):
                    time_rec.append(time)
                    acc.append(inp_acc)
                    dt_rec.append(dt)
                    no_points.append(npts)
                    name.append(rec[:-4])
                    desc_rec.append(desc)
                    path.append(filepath)
                    SFactor.append(scalefactor)
            except IOError:
                print("processMotion FAILED!: File is not in the directory")
    
        Hzlv = {'time': np.array(time_rec, dtype=(object)), 'acc': np.array(acc, dtype=(object)),
                'dt': dt_rec,'no_points': no_points, 'name': name, 'descripcion': desc_rec,
                'ruta':path, 'Hazard':Hz.name, 'SFactor':SFactor}
        gmrs = gmrs + (Hzlv, )
    return gmrs

#%% ESPECTROS DE RESPUESTA SÍSMICA MÉTODO DE NIGAMJENNINGS
# def NigamJennings(gmrs, Ti, damp, Te):
def NigamJennings(gmrs, Ti, damp):
    no_gmrs = len(gmrs['time'])
    Nsteps = gmrs['no_points']
    DTs = gmrs['dt']
    nrecords = len(DTs)
    GMcode = np.arange(0,len(gmrs['acc']))
    Sa_T1, Sd_T1, Sv_T1 = [],[],[]
    # Sa_i = []
    # plt.figure(figsize = (15, 5))
    for igmr in range(no_gmrs):
        acc = gmrs['acc'][igmr]
        time = gmrs['time'][igmr]
        add_PGA = False
        if Ti[0] == 0:
            periods = np.delete(Ti, 0)
            add_PGA = True
        dt = time[1]-time[0]
        num_steps = len(acc)
        num_per = len(periods)
        #vel, disp = calculate_velocity_displacement(time, acc)
        omega = (2. * np.pi) / np.array(periods)
        omega2 = omega ** 2.
        omega3 = omega ** 3.
        omega_d = omega * math.sqrt(1.0 - (damp ** 2.))
        const = {'f1': (2.0 * damp) / (omega3 * dt), 'f2': 1.0 / omega2, 'f3': damp * omega, 'f4': 1.0 / omega_d}
        const['f5'] = const['f3'] * const['f4']
        const['f6'] = 2.0 * const['f3']
        const['e'] = np.exp(-const['f3'] * dt)
        const['s'] = np.sin(omega_d * dt)
        const['c'] = np.cos(omega_d * dt)
        const['g1'] = const['e'] * const['s']
        const['g2'] = const['e'] * const['c']
        const['h1'] = (omega_d * const['g2']) - (const['f3'] * const['g1'])
        const['h2'] = (omega_d * const['g1']) + (const['f3'] * const['g2'])

        x_d = np.zeros([num_steps - 1, num_per], dtype = float)
        x_v = np.zeros_like(x_d)
        x_a = np.zeros_like(x_d)

        for k in range(0, num_steps - 1):
            yval = k - 1
            dug = acc[k + 1] - acc[k]
            z_1 = const['f2'] * dug
            z_2 = const['f2'] * acc[k]
            z_3 = const['f1'] * dug
            z_4 = z_1 / dt
            if k == 0:
                b_val = z_2 - z_3
                a_val = (const['f5'] * b_val) + (const['f4'] * z_4)
            else:
                b_val = x_d[k - 1, :] + z_2 - z_3
                a_val = (const['f4'] * x_v[k - 1, :]) + (const['f5'] * b_val) + (const['f4'] * z_4)
                
            x_d[k, :] = (a_val * const['g1']) + (b_val * const['g2']) + z_3 - z_2 - z_1
            x_v[k, :] = (a_val * const['h1']) - (b_val * const['h2']) - z_4
            x_a[k, :] = (-const['f6'] * x_v[k, :]) - (omega2 * x_d[k, :])
        # spectrum = {'Sa': None, 'Sv': None, 'Sd': None, 'T': None,
        #             'Sa_T1': None,'Sd_T1': None, 'Sv_T1': None, 'Sai':None}
        spectrum = {'Sa': None, 'Sv': None, 'Sd': None, 'T': None,
                    'Sa_T1': None,'Sd_T1': None, 'Sv_T1': None}
        spectrum['Sa'] = np.max(np.fabs(x_a), axis = 0)
        spectrum['Sv'] = np.max(np.fabs(x_v), axis = 0)*9.81
        spectrum['Sd'] = np.max(np.fabs(x_d), axis = 0)*9.81
        
        # position_Sai = np.where(periods.round(2) == round(Te,2))
        # Sai = spectrum['Sa'][position_Sai[0].max()]
        Sa_T1.append(spectrum['Sa'])
        Sd_T1.append(spectrum['Sd'])
        Sv_T1.append(spectrum['Sv'])
        # Sa_i.append(Sai)
        
        # spectrum['Sai'] = Sa_i
        spectrum['Sa_T1'] = Sa_T1
        spectrum['Sd_T1'] = Sd_T1
        spectrum['Sv_T1'] = Sv_T1

        spectrum['PSv'] = spectrum['Sd']*omega
        spectrum['T'] = periods
        if add_PGA:
            # spectrum['Sa'] = np.append(np.max(np.fabs(acc)), spectrum['Sa'])
            spectrum['Sv'] = np.append(0, spectrum['Sv'])
            spectrum['Sd'] = np.append(0, spectrum['Sd'])
            spectrum['T']  = np.append(0, spectrum['T'])

    for i in range(nrecords):
        spectrum['Sa_T1'][i] = np.insert(arr=spectrum['Sa_T1'][i], obj=0, values=0)
        spectrum['Sd_T1'][i] = np.insert(arr=spectrum['Sd_T1'][i], obj=0, values=0)
        spectrum['Sv_T1'][i] = np.insert(arr=spectrum['Sv_T1'][i], obj=0, values=0)
    if len(periods)>1:
        return spectrum
    else:	
        return Sa_T1, Sd_T1, Sv_T1

#%% DEFINICIÓN DE LA FUNCIÓN PARA GRAFICO DE ESPECTRO SÍSMICO (NSR-10)
def spectrum_desing(arquetipo,tipo_suelo,Te,coef_imp='I'):
#---PARAMETROS INPUT (NSR-10)
    Sig_Ciudad = ['ARAUCA', 'ARM', 'BAR', 'BGT', 'BUC', 'CAL', 'CAR', 'CUC', 'FLO', 'IBA', 'LET',
                  'MAN', 'MED', 'MIT', 'MOC', 'MON','NEI','PAS','PER','POP','PCA','PIN', 'QUI',
                  'RCH','SAN','SJG','SMA','SIN','TUN','VAL','VIL','YOP','RIO']
    Ciudad = ['ARAUCA', 'ARMENIA', 'BARRANQUILLA', 'BOGOTA', 'BUCARAMANGA', 'CALI', 'CARTAGENA',
              'CUCUTA', 'FLORENCIA', 'IBAGUE', 'LETICIA', 'MANIZALES', 'MEDELLIN', 'MITU', 'MOCOA',
              'MONTERIA','NEIVA','PASTO','PEREIRA','POPAYAN','PUERTO CARRENO','PUERTO INIRIDA',
              'QUIBDO','RIOACHA','SAN ANDRES','SAN JOSE DEL GUAVIARE','SANTA MARTHA','SINCELEJO',
              'TUNJA','VALLEDUPAR','VILLAVICENCIO','YOPAL','RIONEGRO']
    Aa_ = np.array([0.15,0.25,0.10,0.15,0.25,0.25,0.10,0.35,0.20,0.20,0.05,0.25,0.15,0.05,0.30,
                   0.10,0.25,0.25,0.25,0.25,0.05,0.05,0.50,0.10,0.10,0.05,0.15,0.10,0.20,0.10,
                   0.25,0.30,0.15], dtype = float)
    Av_ = np.array([0.15,0.25,0.10,0.20,0.25,0.25,0.10,0.25,0.15,0.20,0.05,0.25,0.20,0.05,0.25,
                   0.20,0.25,0.25,0.25,0.20,0.05,0.05,0.35,0.15,0.10,0.05,0.10,0.15,0.20,0.1,
                   0.30,0.20,0.20], dtype = float)
    Suelo = ['A', 'B', 'C', 'D', 'E']
    Uso = ['IV','III','II','I']
    Importancia = np.array([1.5,1.25,1.1,1.0], dtype=float)
    
    Fa_ =[[0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
                   [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
                   [1.20, 1.20, 1.20, 1.15, 1.10, 1.05, 1.00, 1.00, 1.00],
                   [1.60, 1.50, 1.40, 1.30, 1.20, 1.15, 1.10, 1.05, 1.00],
                   [2.50, 2.10, 1.70, 1.45, 1.20, 1.05, 0.90, 0.90, 0.90]]
    
    Fv_ =[[0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
                   [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
                   [1.70, 1.65, 1.60, 1.55, 1.50, 1.45, 1.40, 1.35, 1.30],
                   [2.40, 2.20, 2.00, 1.90, 1.80, 1.70, 1.60, 1.55, 1.50],
                   [3.50, 3.35, 3.20, 3.00, 2.80, 2.60, 2.40, 2.40, 2.40]]
    
    Amenaza=pd.DataFrame(data=list(zip(Sig_Ciudad,Ciudad,Aa_,Av_)), columns=['Siglas','Ciudad','Aa','Av'])
    Grupo_uso=pd.DataFrame(data=list(zip(Uso,Importancia)),columns=['Uso','nivel'])
    Fai=pd.DataFrame(data=Fa_, columns=['Aa<=0.1','Aa=0.15','Aa=0.20','Aa=0.25','Aa=0.30','Aa=0.35','Aa=0.40','Aa=0.45','Aa>=0.50'],
                    index=Suelo, dtype=float)
    Fvj=pd.DataFrame(data=Fv_, columns=['Av<=0.1','Av=0.15','Av=0.20','Av=0.25','Av=0.30','Av=0.35','Av=0.40','Av=0.45','Av>=0.50'],
                    index=Suelo, dtype=float)
    #-------Encontrar ciudad:
    sig_ciudad = arquetipo[9:-4]

    parametros = (Amenaza,Grupo_uso,Fai,Fvj)
#---Calculos de parametros de amenaza
    I=Grupo_uso[Grupo_uso.Uso==coef_imp].values[0][1]                           #Importancia de la edificación
    Aa=Amenaza[Amenaza.Siglas==sig_ciudad].values[0][2]                          #Coeficiente que representa la aceleración horizontal pico efectiva, para diseño, dado en A.2.2. 
    Av=Amenaza[Amenaza.Siglas==sig_ciudad].values[0][3]                          #Coeficiente que representa la velocidad horizontal pico efectiva, para diseño, dado en A.2.2.
    if Aa<=0.1:
        i=0
    elif Aa==0.15:
        i=1
    elif Aa==0.20:
        i=2
    elif Aa==0.25:
        i=3
    elif Aa==0.30:
        i=4
    elif Aa==0.35:
        i=5
    elif Aa==0.4:    
        i=6
    elif Aa==0.45:    
        i=7
    elif Aa>=0.5:    
        i=8
    if Av<=0.1:
        j=0
    elif Av==0.15:
        j=1
    elif Av==0.20:
        j=2
    elif Av==0.25:
        j=3
    elif Av==0.30:
        j=4
    elif Av==0.35:
        j=5
    elif Av==0.4:    
        j=6
    elif Av==0.45:    
        j=7
    elif Av>=0.5:    
        j=8
        
    Fa=Fai[Fai.index==tipo_suelo].values[0][i]
    Fv=Fvj[Fvj.index==tipo_suelo].values[0][j]
    To=round(0.1*(Av*Fv)/(Aa*Fa),2)
    Sa_To =2.5*Aa*Fa*I*(0.4+0.6*(To/To))
    Sd_To =0.62*Aa*Fa*I*Te**2*(0.4+0.6*(To/To))
    Tc=round(0.48*(Av*Fv)/(Aa*Fa),2)
    Sa_Tc = 2.5*Aa*Fa*I
    Sd_Tc = 0.62*Aa*Fa*I*Tc**2
    Tl=round(2.4*Fv,2)
    Sa_Tl = 1.2*Av*Fv*Tl*I/(Tl**2)
    Sd_Tl = 0.3*Av*Fv*I*Tl
    T = np.linspace(0.0, round(Tl,0), 1000)
    Sa = np.zeros(len(T))
    Sd = np.zeros(len(T))
#-----Calculo de Sa y Sd
    Sa[0] = Aa*Fa*I                                                           # Ecuación para modos diferentes al fundamental
    #Sa[0] = 2.5*Aa*Fa*I
    Sd[0] = 0.0
    for z in range(1,len(T)):
        if T[z]<=To:
            Sa[z]=2.5*Aa*Fa*I
            #Sa[z]=2.5*Aa*Fa*I*(0.4+0.6*(T[z]/To))                             # Ecuación para modos diferentes al fundamental
            # Sd[z]=0.62*Aa*Fa*I*Te**2*(0.4+0.6*(T[z]/To))
            Sd[z]=0.62*Aa*Fa*I*T[z]**2
        elif To<=T[z]<=Tc:
            Sa[z]=2.5*Aa*Fa*I
            Sd[z]=0.62*Aa*Fa*I*T[z]**2
        elif T[z]>Tc:        
            Sa[z]=1.2*Av*Fv*I/T[z]      #cuando Te>0.85
            Sd[z]=0.3*Av*Fv*I*T[z]
        elif T[z]>=Tl:
            Sa[z]=1.2*Av*Fv*Tl*I/(T[z]**2)
            Sd[z]=0.3*Av*Fv*I*Tl
#----Calculo de Sa y Sd para el valor de Te
    if Te<=To:
        # Sa_Te=2.5*Aa*Fa*I*(0.4+0.6*(Te/To))                                  # Ecuación para modos diferentes al fundamental
        Sa_Te=2.5*Aa*Fa*I
        # Sd_Te=0.62*Aa*Fa*I*Te**2*(0.4+0.6*(Te/To))
        Sd_Te=0.62*Aa*Fa*I*Te**2
    elif To<=Te<=Tc:
        Sa_Te=2.5*Aa*Fa*I
        Sd_Te=0.62*Aa*Fa*I*Te**2
    elif Te>Tc:        
        Sa_Te=1.2*Av*Fv*I/Te      #cuando Te>0.85
        Sd_Te=0.3*Av*Fv*I*Te
    elif Te>=Tl:
        Sa_Te=1.2*Av*Fv*Tl*I/(Te**2)
        Sd_Te=0.3*Av*Fv*I*Tl
    Se=pd.DataFrame(data=[[Sa_Te, Sd_Te]], columns=['Sa(g)','Sd(m)'])
    spectrumdesign = pd.DataFrame(data=list(zip(T,Sa,Sd)), columns=['T(s)','Sa(g)','Sd(m)'])
    periods = np.array([To,Tc,Tl,Te], dtype=float)
    Sa_limits = np.array([Sa_To,Sa_Tc,Sa_Tl,Sa_Te], dtype=float)
    Sd_limits = np.array([Sd_To,Sd_Tc,Sd_Tl,Sd_Te], dtype=float)
    
    return Se, spectrumdesign, periods, Sa_limits, Sd_limits

#%% GRAFICA ESPECTROS DE DISEÑO SEGÚN NSR-10 (IDA)
def plot_spectrum_design_IDA(spectrumdesign, spectrum_response, SFactor, SpectrumFactor, gmrs, periods, Sa_limits):
    #-----Espectro Elástico de Aceleraciones de Diseño como fracción de (g)
    Nsteps = gmrs['no_points']
    DTs = gmrs['dt']
    nrecords = len(DTs)
    GMcode = np.arange(0,len(gmrs['acc']))
    limit_Sa = max(spectrumdesign['Sa(g)'])+0.20
    ylimit_Sa = max(np.arange(0,limit_Sa+0.20,0.10))
    plt.figure()
    #plt.style.use('grayscale')
    for j in range(len(SFactor)):
        for i in range(nrecords):
            #plt.plot(Tx, spectrum_response['Sa_T1'][i], color='gainsboro', lw=0.2)
            plt.plot(spectrum_response['T'], spectrum_response['Sa_T1'][i]*SpectrumFactor[i]*SFactor[j], lw=0.4)
    plt.plot(periods[3], Sa_limits[3], color='b', marker='d')
    plt.axvline(x=periods[0], ymin=0 , ymax=Sa_limits[0]/ylimit_Sa, color='gray', ls='--', lw=1, label='T0')
    plt.axvline(x=periods[1], ymin=0 , ymax=Sa_limits[1]/ylimit_Sa, color='gray', ls='--', lw=1, label='Tc')
    plt.axvline(x=periods[2], ymin=0 , ymax=Sa_limits[2]/ylimit_Sa, color='gray', ls='--', lw=1, label='Tl')
    plt.axvline(x=periods[3], ymin=0 , ymax=Sa_limits[3]/ylimit_Sa, color='b', ls='--', lw=1, label='Te')
    plt.plot(spectrumdesign['T(s)'], spectrumdesign['Sa(g)'], color='k', lw=1.5)
    plt.xscale('log')
    # plt.xlim(xmin=0.005, xmax=10)
    plt.yscale('log')
    # plt.ylim(ymin=0.07, ymax=0.7)
    # plt.ylim(0,ylimit_Sa)
    # plt.yticks(np.arange(0,limit_Sa+0.20,0.10))
    plt.title('Espectro Elástico de Aceleración de Diseño NSR-10', fontsize='x-large', fontweight="bold", fontname="Times New Roman")
    plt.xlabel('T (s)', fontsize='large', fontname="Times New Roman")
    plt.xticks(fontsize='x-large', fontname="Times New Roman")
    plt.ylabel('Sa (g)', fontsize='large', fontname="Times New Roman")
    plt.yticks(fontsize='large', fontname="Times New Roman")
    plt.tick_params(labelsize=14)
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='large')                   # place legend outside
    plt.grid(True, which="both", color = 'gainsboro', ls='--', lw=0.5)
    plt.tight_layout()
    #-----Espectro Elástico de Aceleraciones de Diseño como fracción de (g)
    limit_Sa = max(spectrumdesign['Sa(g)'])+0.20
    ylimit_Sa = max(np.arange(0,limit_Sa+0.20,0.10))
    Tx=np.delete(spectrum_response['T'],0)
    plt.figure()
    plt.plot(spectrumdesign['T(s)'], spectrumdesign['Sa(g)'], color='k')
    for j in range(len(SFactor)):
        for i in range(nrecords):
            plt.plot(spectrum_response['T'], spectrum_response['Sa_T1'][i], color='gainsboro', lw=0.2)
            #plt.plot(Tx, spectrum_response['Sa_T1'][i]*SpectrumFactor[i]*SFactor[j], color='gray', lw=0.2)
    plt.plot(periods[3], Sa_limits[3], color='b', marker='d')
    plt.axvline(x=periods[0], ymin=0 , ymax=Sa_limits[0]/ylimit_Sa, color='gray', ls='--', lw=1, label='T0')
    plt.axvline(x=periods[1], ymin=0 , ymax=Sa_limits[1]/ylimit_Sa, color='gray', ls='--', lw=1, label='Tc')
    plt.axvline(x=periods[2], ymin=0 , ymax=Sa_limits[2]/ylimit_Sa, color='gray', ls='--', lw=1, label='Tl')
    plt.axvline(x=periods[3], ymin=0 , ymax=Sa_limits[3]/ylimit_Sa, color='b', ls='--', lw=1, label='Te')
    # plt.xscale('log')
    # plt.xlim(xmin=0.005, xmax=10)
    # plt.yscale('log')
    # plt.ylim(ymin=0.07, ymax=0.7)
    plt.ylim(0,ylimit_Sa)
    plt.yticks(np.arange(0,limit_Sa+0.20,0.10))
    plt.title('Espectro Elástico de Aceleración de Diseño NSR-10', fontsize='x-large', fontweight="bold", fontname="Times New Roman")
    plt.xlabel('T (s)', fontsize='large', fontname="Times New Roman")
    plt.xticks(fontsize='x-large', fontname="Times New Roman")
    plt.ylabel('Sa (g)', fontsize='large', fontname="Times New Roman")
    plt.yticks(fontsize='large', fontname="Times New Roman")
    plt.tick_params(labelsize=14)
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='large')                   # place legend outside
    plt.tight_layout()
    #-----Espectro Elástico de Aceleraciones de Diseño como fracción de (g)
    ylimit_Sa = max(spectrumdesign['Sa(g)'])+0.05
    plt.figure()
    plt.plot(spectrumdesign['T(s)'], spectrumdesign['Sa(g)'], color='k')
    plt.axvline(x=periods[0], ymin=0 , ymax=Sa_limits[0]/ylimit_Sa, color='gainsboro', ls='--', lw=1, label='T0')
    plt.axvline(x=periods[1], ymin=0 , ymax=Sa_limits[1]/ylimit_Sa, color='lightgrey', ls='--', lw=1, label='Tc')
    plt.axvline(x=periods[2], ymin=0 , ymax=Sa_limits[2]/ylimit_Sa, color='lightgray', ls='--', lw=1, label='Tl')
    plt.axvline(x=periods[3], ymin=0 , ymax=Sa_limits[3]/ylimit_Sa, color='gray', ls='--', lw=1, label='Te')
    plt.ylim(0,ylimit_Sa)
    plt.yticks(np.arange(0,ylimit_Sa+0.05,0.05))
    plt.title('Espectro Elástico de Aceleración de Diseño NSR-10', fontsize='x-large', fontweight="bold", fontname="Times New Roman")
    plt.xlabel('T (s)', fontsize='large', fontname="Times New Roman")
    plt.xticks(fontsize='x-large', fontname="Times New Roman")
    plt.ylabel('Sa (g)', fontsize='large', fontname="Times New Roman")
    plt.yticks(fontsize='large', fontname="Times New Roman")
    plt.tick_params(labelsize=14)
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='large')                   # place legend outside
    plt.tight_layout()

    #-----Espectro Elástico de Desplazamientos (m) de Diseño
    # ylimit_Sd = max(spectrumdesign['Sd(m)'])+0.05
    # plt.figure()
    # plt.plot(spectrumdesign['T(s)'], spectrumdesign['Sd(m)'], color='k')
    # plt.axvline(x=periods[0], ymin=0 , ymax=Sd_limits[0]/ylimit_Sd, color='gainsboro', ls='--', lw=1, label='T0')
    # plt.axvline(x=periods[1], ymin=0 , ymax=Sd_limits[1]/ylimit_Sd, color='lightgrey', ls='--', lw=1, label='Tc')
    # plt.axvline(x=periods[2], ymin=0 , ymax=Sd_limits[2]/ylimit_Sd, color='lightgray', ls='--', lw=1, label='Tl')
    # plt.axvline(x=periods[3], ymin=0 , ymax=Sd_limits[3]/ylimit_Sd, color='gray', ls='--', lw=1, label='Te')
    # plt.ylim(0,ylimit_Sd)
    # plt.yticks(np.arange(0,ylimit_Sd+0.05,0.05))
    # plt.title('Espectro Elástico de Desplazamientos de Diseño NSR-10')
    # plt.xlabel('T(s)')
    # plt.ylabel('Sd(m)')
    # plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')                           # place legend outside
    # plt.tight_layout()
    return
#%% GRAFICA ESPECTROS DE RESPUESTA (CRONOLOGICO)
def plot_spectrum_design_CNL(spectrumdesign, spectrum_response, gmrs, ciudad, periodo, Hzlv_Tag = 'Hzlvs',):
    #-----Espectro Elástico de Aceleraciones de Diseño como fracción de (g)
    GMcode = np.arange(0,len(gmrs['acc']))
    nrecords = len(GMcode)
    fig, axs = plt.subplots(figsize=(24, 7.5), dpi=150, nrows=1, ncols=3, sharex=False, sharey=False)
    fig.suptitle('Espectros de respuesta de ' + str(ciudad) +' para '+ str(periodo), fontsize=18)
    plt.rcParams["font.family"] = "Cambria"
    colors = ['red', 'blue', 'yellow', 'green', 'indigo', 'rosybrown', 'orange', 'brown']
   
    #ACELERACION ESPECTRAL
    Sa = pd.DataFrame(dtype=float)
    Sd = pd.DataFrame(dtype=float)
    Sv = pd.DataFrame(dtype=float)
    for i in range(nrecords):
        # axs[0].loglog(spectrum_response['T'], spectrum_response['Sa_T1'][i], color='silver', lw=0.4)
        axs[0].plot(spectrum_response['T'], spectrum_response['Sa_T1'][i], color='silver', lw=0.4)
        Sa['GM_'+str(i+1)] = spectrum_response['Sa_T1'][i]
    Sa['mediana']=Sa.median(axis=1)
    # axs[0].loglog(spectrum_response['T'], Sa['mediana'], color='blue', lw=0.8, label='Sa mediana')
    # axs[0].loglog(spectrumdesign['T(s)'], spectrumdesign['Sa(g)'], color='k', lw=0.8, label='Sa diseño')
    axs[0].plot(spectrum_response['T'], Sa['mediana'], color='blue', lw=0.8, label='Sa mediana')
    axs[0].plot(spectrumdesign['T(s)'], spectrumdesign['Sa(g)'], color='k', lw=0.8, label='Sa diseño')
    axs[0].set_title('Espectro de Aceleración ' + str(Hzlv_Tag), fontsize=16)
    axs[0].set_xlabel('T (s)', fontsize=16)
    axs[0].set_ylabel('Sa (g)', fontsize=16)
    # axs[0].set_xlim((0.01,10))
    # axs[0].set_ylim((1e-5,3))
    axs[0].tick_params(labelsize=16, width=4)
    axs[0].grid(True, which="both", color = 'gainsboro', ls='--', lw=0.5)
    axs[0].legend(loc='lower right', fontsize=14)
    
    #DESPLAZAMIENTO ESPECTRAL
    for i in range(nrecords):
        # axs[1].loglog(spectrum_response['T'], spectrum_response['Sd_T1'][i], color='silver', lw=0.4)
        axs[1].plot(spectrum_response['T'], spectrum_response['Sd_T1'][i], color='silver', lw=0.4)
        Sd['GM_'+str(i+1)] = spectrum_response['Sd_T1'][i]
    Sd['mediana']=Sd.median(axis=1)
    # axs[1].loglog(spectrum_response['T'], Sd['mediana'], color='green', lw=0.8, label='Sd mediana')
    # axs[1].loglog(spectrumdesign['T(s)'], spectrumdesign['Sd(m)'], color='k', lw=0.8, label='Sd diseño')
    axs[1].plot(spectrum_response['T'], Sd['mediana'], color='green', lw=0.8, label='Sd mediana')
    axs[1].plot(spectrumdesign['T(s)'], spectrumdesign['Sd(m)'], color='k', lw=0.8, label='Sd diseño')
    axs[1].set_title('Espectro de Desplazamiento ' + str(Hzlv_Tag), fontsize=18)
    axs[1].set_xlabel('T (s)', fontsize=16)
    axs[1].set_ylabel('Sd (m)', fontsize=16)
    axs[1].set_xlim((0,1))
    axs[1].set_ylim((0,0.2))
    axs[1].tick_params(labelsize=16, width=4)
    axs[1].grid(True, which="both", color = 'gainsboro', ls='--', lw=0.5)
    axs[1].legend(loc='lower right', fontsize=14)
    #DESPLAZAMIENTO ESPECTRAL
    for i in range(nrecords):
        # axs[2].loglog(spectrum_response['T'], spectrum_response['Sv_T1'][i], color='silver', lw=0.4)
        axs[2].plot(spectrum_response['T'], spectrum_response['Sv_T1'][i], color='silver', lw=0.4)
        Sv['GM_'+str(i+1)] = spectrum_response['Sv_T1'][i]
    Sv['mediana']=Sv.median(axis=1)
    # axs[2].loglog(spectrum_response['T'], Sv['mediana'], color='red', lw=0.8, label='Sv mediana')
    axs[2].plot(spectrum_response['T'], Sv['mediana'], color='red', lw=0.8, label='Sv mediana')
    axs[2].set_title('Espectro de Velocidades ' + str(Hzlv_Tag), fontsize=18)
    axs[2].set_xlabel('T (s)', fontsize=16)
    axs[2].set_ylabel('Sv (m/s)', fontsize=16)
    # axs[2].set_xlim((0,10))
    # axs[2].set_ylim((0,0.1))
    axs[2].tick_params(labelsize=16, width=4)
    axs[2].grid(True, which="both", color = 'gainsboro', ls='--', lw=0.5)
    axs[2].legend(loc='lower right', fontsize=14)
    plt.tight_layout()
    
    return fig, Sa






def plot_spectrum_design_CNLs(spectrumdesign, spectrum_response, gmrs,
                              colors = ['salmon', 'lightsteelblue', 'moccasin', 'palegreen', 'plum', 'peru', 'bisque', 'goldenrod']):
    #-----Espectro Elástico de Aceleraciones de Diseño como fracción de (g)
    
    Hzlv = np.arange(len(gmrs))
    fig = plt.subplots(figsize=(10, 7.5), dpi=150)
    plt.rcParams["font.family"] = "Cambria"
    plt.rcParams['savefig.bbox'] = "tight"
    # style.use('ggplot') or plt.style.use('ggplot')
    for hzlv in range(len(Hzlv)):
        Hzlv_Tag = 'Nivel Amz '+ str(hzlv + 1)
        # GMcode = np.arange(0,len(gmrs[hzlv]['acc']))
        nrecords = len(gmrs[hzlv]['acc'])
        plt.plot(np.nan, np.nan, c=colors[hzlv], label= Hzlv_Tag, lw=2.0)
        for i in range(nrecords):
            plt.plot(spectrum_response[hzlv]['T'], spectrum_response[hzlv]['Sa_T1'][i], c=colors[hzlv], lw=0.4)
            
    # plt.axvline(x=T1, ymin=0.0 , ymax=1.0, color='gray', ls='--', lw=1, label='T1')
    plt.plot(spectrumdesign['T(s)'], spectrumdesign['Sa(g)'], color='k', lw=1.5, label='Espectro Diseño')
    plt.xscale('log')
    plt.yscale('log')
    # plt.title('Espectro Elástico de Aceleración')
    plt.xlabel('T (s)', fontsize=26)
    plt.ylabel('Sa (g)', fontsize=26)
    plt.tick_params(labelsize=26)
    plt.legend(loc='lower left', fontsize=18)                   # place legend outside
    plt.grid(True, which="both", color = 'gainsboro', ls='--', lw=0.5)
    plt.tight_layout()
    return






















