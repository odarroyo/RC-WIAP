import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import pickle
import joblib as jb
import matplotlib.pyplot as plt
from matplotlib import style
import time
import multiprocessing
from joblib import Parallel, delayed
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Import OpenSees libraries
import openseespy.opensees as ops
import Lib_analisis as an
import opseestools.analisis as an2
import opseestools.Lib_frag as lf
from scipy import stats

# Set page configuration
st.set_page_config(page_title="RC-WIAP IDA Analysis", layout="wide")

# Initialize session state
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'ida_analysis_complete' not in st.session_state:
    st.session_state.ida_analysis_complete = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'page' not in st.session_state:
    st.session_state.page = "IDA Analysis"
if 'selected_results_file' not in st.session_state:
    st.session_state.selected_results_file = None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["IDA Analysis", "Results Visualization", "Fragility Functions"])

# Title
st.title("RC-WIAP - Incremental Dynamic Analysis (IDA)")
st.markdown("---")

# ==================== PAGE 1: IDA ANALYSIS ====================
if page == "IDA Analysis":
    st.header("Incremental Dynamic Analysis Configuration")

    # Step 1: Model Selection
    st.subheader("Step 1: Select Model")

    # Get all .pkl files from the models folder
    models_folder = "models"
    if not os.path.exists(models_folder):
        st.error(f"❌ Models folder '{models_folder}' not found!")
        st.stop()

    model_files = glob.glob(os.path.join(models_folder, "*.pkl"))

    if len(model_files) == 0:
        st.warning(f"⚠️ No model files found in '{models_folder}' folder.")
        st.stop()
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_file = st.selectbox(
                "Available Models:",
                options=model_files,
                format_func=lambda x: os.path.basename(x),
                index=model_files.index(st.session_state.selected_model) if st.session_state.selected_model in model_files else 0,
                help="Select the model file for IDA analysis"
            )
            
            # Display model information
            if selected_file:
                st.session_state.selected_model = selected_file
                model_name = os.path.basename(selected_file).replace('.pkl', '')
                
                st.info(f"📦 Selected Model: **{model_name}**")
                
                # Load model info
                try:
                    Arq1 = jb.load(selected_file)
                    
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("Number of Walls", Arq1.NumMuros)
                    with col_info2:
                        st.metric("Number of Stories", Arq1.NumPisos)
                    with col_info3:
                        st.metric("Building Height (m)", f"{Arq1.listaCoorY[0]:.2f}")
                except Exception as e:
                    st.warning(f"Could not load model details: {str(e)}")
        
        with col2:
            # Show model image if available
            model_image = selected_file.replace('.pkl', '_model.png')
            if os.path.exists(model_image):
                st.image(model_image, caption="Model Visualization", use_container_width=True)

    st.markdown("---")

    # Step 2: Check Ground Motions
    st.subheader("Step 2: Ground Motion Records")

    gms_folder = "GMs"
    if not os.path.exists(gms_folder):
        st.error(f"❌ Ground motions folder '{gms_folder}' not found!")
        st.stop()

    gm_files = sorted([f for f in os.listdir(gms_folder) if f.startswith('GM') and f.endswith('.txt')])

    if len(gm_files) == 0:
        st.error(f"❌ No GM files found in '{gms_folder}' folder!")
        st.stop()
    else:
        col_gm1, col_gm2 = st.columns(2)
        
        with col_gm1:
            st.success(f"✅ Found {len(gm_files)} ground motion records")
        
        with col_gm2:
            # Check for spectral data files
            sa_fema_exists = os.path.exists(os.path.join(gms_folder, 'Sa_FEMA.pkl'))
            t_exists = os.path.exists(os.path.join(gms_folder, 'T.pkl'))
            
            if sa_fema_exists and t_exists:
                st.success("✅ Spectral data files found (Sa_FEMA.pkl, T.pkl)")
            else:
                st.error("❌ Missing spectral data files in GMs folder!")
                if not sa_fema_exists:
                    st.warning("Missing: Sa_FEMA.pkl")
                if not t_exists:
                    st.warning("Missing: T.pkl")
        
        with st.expander(f"View Ground Motion Records ({len(gm_files)} files)"):
            # Display in columns
            num_cols = 5
            cols = st.columns(num_cols)
            for i, gm_file in enumerate(gm_files):
                with cols[i % num_cols]:
                    st.text(f"• {gm_file}")

    st.markdown("---")

    # Step 3: IDA Parameters
    st.subheader("Step 3: IDA Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Scale Factors (SFactor)**")
        st.write("Define the intensity levels for the IDA analysis")
        
        # Default scale factors
        default_sfactors = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        
        # Option to use default or custom
        use_default = st.checkbox("Use default scale factors", value=True)
        
        if use_default:
            sfactor_input = ", ".join([str(x) for x in default_sfactors])
            st.info(f"Scale Factors: {sfactor_input}")
        else:
            sfactor_input = st.text_input(
                "Enter scale factors (comma-separated):",
                value=", ".join([str(x) for x in default_sfactors]),
                help="Example: 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0"
            )
        
        # Parse scale factors
        try:
            SFactor = [float(x.strip()) for x in sfactor_input.split(',')]
            st.success(f"✅ {len(SFactor)} scale factors defined")
        except:
            st.error("❌ Invalid scale factors format. Use comma-separated numbers.")
            SFactor = default_sfactors

    with col2:
        st.markdown("**Spectral Acceleration (Sa_d)**")
        st.write("Design spectral acceleration value")
        
        Sa_d = st.number_input(
            "Sa_d [g]:",
            min_value=0.01,
            max_value=5.0,
            value=0.5625,
            step=0.01,
            format="%.4f",
            help="Design spectral acceleration in g's"
        )
        
        st.markdown("**Scaling Period (T_scaling)**")
        st.write("Period for spectral scaling")
        
        T_scaling = st.number_input(
            "T_scaling [s]:",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            format="%.4f",
            help="Fundamental period for scaling ground motions"
        )
        
        st.markdown("**Computational Settings**")
        max_cores = multiprocessing.cpu_count()
        
        num_cores = st.number_input(
            "Number of CPU Cores:",
            min_value=1,
            max_value=max_cores,
            value=max_cores,
            step=1,
            help=f"Number of parallel processes (max: {max_cores} available)"
        )
        
        st.info(f"💻 Using {num_cores} of {max_cores} available cores")

    st.markdown("---")

    # Step 4: Analysis Summary
    st.subheader("Step 4: Analysis Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Analyses", len(SFactor) * len(gm_files))
    with col2:
        st.metric("Ground Motions", len(gm_files))
    with col3:
        st.metric("Intensity Levels", len(SFactor))

    st.info(f"⚠️ This analysis will run {len(SFactor) * len(gm_files)} nonlinear time-history analyses. This may take considerable time.")

    st.markdown("---")

    # Step 5: Run Analysis
    st.subheader("Step 5: Run IDA Analysis")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Run IDA Analysis", type="primary", use_container_width=True):
            if st.session_state.selected_model is None:
                st.error("Please select a model first!")
            else:
                try:
                    # Load the model
                    Arq1 = jb.load(st.session_state.selected_model)
                    model_name = os.path.basename(st.session_state.selected_model).replace('.pkl', '')
                    
                    st.info("⏳ Running IDA analysis... Please wait, this may take several minutes.")
                    
                    # Create IDAs folder if it doesn't exist
                    idas_folder = "IDAs"
                    if not os.path.exists(idas_folder):
                        os.makedirs(idas_folder)
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Loading ground motion records...")
                    
                    # Build full paths for GM files
                    records = [os.path.join(gms_folder, gm) for gm in gm_files]
                    
                    # Load spectral data from pickle files
                    try:
                        # Try loading with joblib first (more compatible)
                        try:
                            Sa_FEMA = jb.load(os.path.join(gms_folder, 'Sa_FEMA.pkl'))
                            T_periods = jb.load(os.path.join(gms_folder, 'T.pkl'))
                        except:
                            # Fallback to pickle with different encoding
                            with open(os.path.join(gms_folder, 'Sa_FEMA.pkl'), 'rb') as f:
                                Sa_FEMA = pickle.load(f, encoding='latin1')
                            with open(os.path.join(gms_folder, 'T.pkl'), 'rb') as f:
                                T_periods = pickle.load(f, encoding='latin1')
                        
                        # Calculate SpectrumFactor based on T_scaling
                        # Find Sa values at T_scaling for each ground motion
                        SpectrumFactor = []
                        for i in range(len(gm_files)):
                            # Interpolate Sa at T_scaling
                            Sa_at_T = np.interp(T_scaling, T_periods, Sa_FEMA[i])
                            # Calculate spectrum factor as Sa_d / Sa_at_T
                            factor = Sa_d / Sa_at_T if Sa_at_T > 0 else 1.0
                            SpectrumFactor.append(factor)
                        
                        status_text.text(f"✅ Calculated SpectrumFactor for T = {T_scaling} s")
                        
                    except Exception as e:
                        st.error(f"Error loading spectral data files: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.stop()
                    
                    # Ground motion parameters (from original script)
                    Nsteps = [3000, 3000, 2000, 2000, 5590, 5590, 4535, 4535, 9995, 9995, 7810, 7810, 4100, 4100, 4100, 4100, 5440, 5440, 6000, 6000, 2200, 2200, 11190, 11190, 7995, 7995, 7990, 7990, 2680, 2300, 8000, 8000, 2230, 2230, 1800, 1800, 18000, 18000, 18000, 18000, 2800, 2800, 7270, 7270]
                    DTs = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.005,0.005,0.01,0.01,0.01,0.01,0.005,0.005,0.05,0.05,0.02,0.02,0.0025,0.0025,0.005,0.005,0.005,0.005,0.02,0.02,0.005,0.005,0.01,0.01,0.02,0.02,0.005,0.005,0.005,0.005,0.01,0.01,0.005,0.005]
                    GMcode = list(range(len(gm_files)))
                    GMtime = [Nsteps[i]*DTs[i] for i in range(len(gm_files))]
                    status_text.text("Setting up parallel analysis...")
                    progress_bar.progress(5)
                    
                    # Define the analysis function
                    def rundyn(fact, ind):
                        rec = str(ind+1)
                        factor = 9.81*fact
                        
                        nombre = str(int(factor/9.81*100))
                        
                        # DEFINICION DEL MODELO
                        ops.wipe()
                        
                        diafragma = 1  # 1 para diafragma rígido, 0 para vigas
                        seccionViga = [0.3, 0.5]  # [base, altura] de las vigas si diafragma=0
                        
                        an.CrearModelo(Arq1, diafragma=diafragma, seccionViga=seccionViga)
                    
                        # ANALISIS
                        an2.gravedad()
                        ops.loadConst('-time',0.0)
                        
                        # EJECUCION DEL ANÁLISIS DINÁMICO
                        factoran = SpectrumFactor[ind]*factor
                        node_tags = ops.getNodeTags()
                        node_c = np.max(node_tags)
                        
                        t, techo = an2.dinamicoIDA2(records[ind], DTs[ind], Nsteps[ind], 0.04, factoran, 0.025, int(node_c), 1, [0,2], 1, 1e-4)
                        ops.wipe()
                        return ind, fact, t, techo
                    
                    # Run parallel analysis
                    status_text.text(f"Running {len(SFactor) * len(GMcode)} analyses in parallel...")
                    progress_bar.progress(10)
                    
                    stime = time.time()
                    
                    resultados = Parallel(n_jobs=num_cores)(
                        delayed(rundyn)(ff, pp) for ff in SFactor for pp in GMcode
                    )
                    
                    etime = time.time()
                    ttotal = etime - stime
                    
                    progress_bar.progress(90)
                    status_text.text("Processing results...")
                    
                    # Process results
                    hbuilding = Arq1.listaCoorY[0]
                    ind, Sa, tmax, techomax = [], [], [], []
                    
                    for res in resultados:
                        ind.append(res[0])
                        Sa.append(res[1]*Sa_d)
                        tmax.append(np.max(res[2]))
                        
                        if np.max(res[2]) < 0.97*GMtime[res[0]]:
                            techomax.append(0.1*hbuilding)
                        else:
                            techomax.append(np.max(np.abs(res[3])))
                    
                    dic = {'GM': ind, 'Sa': Sa, 'tmax': tmax, 'dertecho': techomax/hbuilding*100}
                    df = pd.DataFrame(dic)
                    
                    # Save to IDAs folder with model name
                    output_filename = os.path.join(idas_folder, f'{model_name}_IDA.pkl')
                    df.to_pickle(output_filename)
                    
                    # Also save as Excel for easy viewing
                    excel_filename = os.path.join(idas_folder, f'{model_name}_IDA.xlsx')
                    df.to_excel(excel_filename, index=False)
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    st.success(f"✅ IDA analysis completed successfully in {ttotal:.2f} seconds!")
                    st.balloons()
                    
                    st.session_state.ida_analysis_complete = True
                    st.session_state.results_df = df
                    
                    # Display results location
                    st.info(f"📁 Results saved to:")
                    st.code(f"  • {output_filename}\n  • {excel_filename}")
                    
                    # Display basic statistics
                    st.markdown("---")
                    st.subheader("Analysis Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Analyses", len(df))
                    with col2:
                        st.metric("Execution Time", f"{ttotal:.1f} s")
                    with col3:
                        st.metric("Max Drift", f"{df['dertecho'].max():.3f}%")
                    with col4:
                        st.metric("Max Sa", f"{df['Sa'].max():.3f} g")
                    
                    st.info("📊 Go to 'Results Visualization' page to view interactive plots")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during analysis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# ==================== PAGE 2: RESULTS VISUALIZATION ====================
elif page == "Results Visualization":
    st.header("IDA Results Visualization")
    
    # Step 1: Load Results
    st.subheader("Step 1: Load IDA Results")
    
    # Find available IDA result files
    idas_folder = "IDAs"
    if not os.path.exists(idas_folder):
        st.warning("⚠️ IDAs folder not found. Please run an IDA analysis first.")
        st.stop()
    
    # Get all .pkl files from IDAs folder
    ida_files = glob.glob(os.path.join(idas_folder, "*_IDA.pkl"))
    
    if len(ida_files) == 0:
        st.warning("⚠️ No IDA result files found. Please run an IDA analysis first.")
        st.stop()
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_results = st.selectbox(
                "Available IDA Results:",
                options=ida_files,
                format_func=lambda x: os.path.basename(x).replace('_IDA.pkl', ''),
                index=ida_files.index(st.session_state.selected_results_file) if st.session_state.selected_results_file in ida_files else 0,
                help="Select IDA results to visualize"
            )
            
            if selected_results:
                st.session_state.selected_results_file = selected_results
                results_name = os.path.basename(selected_results).replace('_IDA.pkl', '')
                st.info(f"📊 Selected Results: **{results_name}**")
        
        with col2:
            # Check for corresponding model image
            model_name = os.path.basename(selected_results).replace('_IDA.pkl', '')
            model_image = os.path.join("models", f"{model_name}_model.png")
            if os.path.exists(model_image):
                st.image(model_image, caption="Model", use_container_width=True)
        
        # Load the results
        try:
            df = pd.read_pickle(selected_results)
            
            st.markdown("---")
            st.subheader("Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyses", len(df))
            with col2:
                st.metric("Ground Motions", df['GM'].nunique())
            with col3:
                st.metric("Intensity Levels", len(df) // df['GM'].nunique())
            with col4:
                st.metric("Max Drift", f"{df['dertecho'].max():.2f}%")
            
            # Data preview
            with st.expander("📋 View Data Table"):
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"{results_name}_IDA.csv",
                    mime="text/csv"
                )
            
            st.markdown("---")
            
            # ==================== PLOTS ====================
            st.subheader("Interactive Plots")
            
            # Add small positive value to avoid log(0) issues
            df_plot = df.copy()
            min_drift = df_plot[df_plot['dertecho'] > 0]['dertecho'].min()
            df_plot['dertecho_plot'] = df_plot['dertecho'].replace(0, min_drift * 0.1)
            
            # Create tabs for different plots
            tab1, tab2, tab3 = st.tabs(["📈 Sa vs Roof Drift", "📊 IDA Curves", "📦 Boxplots by Sa Level"])
            
            # ========== TAB 1: Sa vs Roof Drift ==========
            with tab1:
                st.markdown("### Spectral Acceleration vs Roof Drift")
                
                # Plot controls
                col_ctrl1, col_ctrl2 = st.columns([1, 2])
                
                with col_ctrl1:
                    st.markdown("**Scale Options**")
                    log_x_tab1 = st.checkbox("Log scale X-axis (Sa)", value=False, key="log_x_tab1")
                    log_y_tab1 = st.checkbox("Log scale Y-axis (Drift)", value=False, key="log_y_tab1")
                
                with col_ctrl2:
                    st.markdown("**Axis Limits**")
                    col_lim1, col_lim2 = st.columns(2)
                    with col_lim1:
                        sa_min_tab1 = st.number_input("Sa min [g]", value=0.0, step=0.1, format="%.2f", key="sa_min_tab1")
                        sa_max_tab1 = st.number_input("Sa max [g]", value=float(df['Sa'].max()*1.1), step=0.1, format="%.2f", key="sa_max_tab1")
                    with col_lim2:
                        drift_min_tab1 = st.number_input("Drift min [%]", value=float(max(0.01, df['dertecho'].min()*0.9)), step=0.01, format="%.3f", key="drift_min_tab1")
                        drift_max_tab1 = st.number_input("Drift max [%]", value=float(df['dertecho'].max()*1.1), step=0.1, format="%.2f", key="drift_max_tab1")
                
                st.markdown("---")
                
                # Create figure
                fig = go.Figure()
                
                # Add scatter plot for all data (swapped axes: Sa on X, Drift on Y)
                fig.add_trace(go.Scatter(
                    x=df_plot['Sa'],
                    y=df_plot['dertecho_plot'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=df_plot['GM'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=dict(text="GM ID", font=dict(size=14))),
                        opacity=0.6,
                        line=dict(width=0.5, color='white')
                    ),
                    text=[f"GM: {gm}<br>Sa: {sa:.3f} g<br>Drift: {drift:.3f}%" 
                          for gm, sa, drift in zip(df['GM'], df['Sa'], df['dertecho'])],
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    name='IDA Points'
                ))
                
                # Update layout with user controls
                fig.update_layout(
                    title=dict(
                        text=f'Sa vs Roof Drift - {results_name}',
                        font=dict(size=20, color='#2c3e50', family='Arial Black')
                    ),
                    xaxis=dict(
                        title='Sa(T₁) [g]',
                        type='log' if log_x_tab1 else 'linear',
                        range=[sa_min_tab1, sa_max_tab1] if not log_x_tab1 else [np.log10(max(sa_min_tab1, 0.001)), np.log10(sa_max_tab1)],
                        gridcolor='lightgray',
                        showgrid=True,
                        zeroline=False,
                        title_font=dict(size=16, color='black'),
                        tickfont=dict(size=14, color='black')
                    ),
                    yaxis=dict(
                        title='Roof Drift [%]',
                        type='log' if log_y_tab1 else 'linear',
                        range=[drift_min_tab1, drift_max_tab1] if not log_y_tab1 else [np.log10(max(drift_min_tab1, 0.001)), np.log10(drift_max_tab1)],
                        gridcolor='lightgray',
                        showgrid=True,
                        zeroline=False,
                        title_font=dict(size=16, color='black'),
                        tickfont=dict(size=14, color='black')
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hovermode='closest',
                    height=600,
                    font=dict(family='Arial', size=14)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Drift Statistics**")
                    st.write(f"Min: {df['dertecho'].min():.4f}%")
                    st.write(f"Max: {df['dertecho'].max():.4f}%")
                    st.write(f"Mean: {df['dertecho'].mean():.4f}%")
                    st.write(f"Median: {df['dertecho'].median():.4f}%")
                with col2:
                    st.markdown("**Sa Statistics**")
                    st.write(f"Min: {df['Sa'].min():.4f} g")
                    st.write(f"Max: {df['Sa'].max():.4f} g")
                    st.write(f"Mean: {df['Sa'].mean():.4f} g")
                    st.write(f"Median: {df['Sa'].median():.4f} g")
            
            # ========== TAB 2: IDA Curves ==========
            with tab2:
                st.markdown("### IDA Curves")
                
                # Plot controls
                col_ctrl1, col_ctrl2 = st.columns([1, 2])
                
                with col_ctrl1:
                    st.markdown("**Scale Options**")
                    log_x_tab2 = st.checkbox("Log scale X-axis (Drift)", value=False, key="log_x_tab2")
                    log_y_tab2 = st.checkbox("Log scale Y-axis (Sa)", value=False, key="log_y_tab2")
                
                with col_ctrl2:
                    st.markdown("**Axis Limits**")
                    col_lim1, col_lim2 = st.columns(2)
                    with col_lim1:
                        drift_min_tab2 = st.number_input("Drift min [%]", value=float(max(0.01, df['dertecho'].min()*0.9)), step=0.01, format="%.3f", key="drift_min_tab2")
                        drift_max_tab2 = st.number_input("Drift max [%]", value=float(df['dertecho'].max()*1.1), step=0.1, format="%.2f", key="drift_max_tab2")
                    with col_lim2:
                        sa_min_tab2 = st.number_input("Sa min [g]", value=0.0, step=0.1, format="%.2f", key="sa_min_tab2")
                        sa_max_tab2 = st.number_input("Sa max [g]", value=float(df['Sa'].max()*1.1), step=0.1, format="%.2f", key="sa_max_tab2")
                
                st.markdown("---")
                
                # Create figure
                fig = go.Figure()
                
                # Color palette
                colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set3
                
                # Plot each ground motion as a separate curve
                for i, gm_id in enumerate(sorted(df['GM'].unique())):
                    df_gm = df_plot[df_plot['GM'] == gm_id].sort_values('Sa')
                    
                    fig.add_trace(go.Scatter(
                        x=df_gm['dertecho_plot'],
                        y=df_gm['Sa'],
                        mode='lines+markers',
                        name=f'GM{gm_id:02d}',
                        line=dict(width=2),
                        marker=dict(size=6, line=dict(width=0.5, color='white')),
                        hovertemplate=f'<b>GM{gm_id:02d}</b><br>' +
                                     'Drift: %{x:.3f}%<br>' +
                                     'Sa: %{y:.3f} g<extra></extra>',
                        legendgroup=f'gm{gm_id}',
                        showlegend=True
                    ))
                
                # Update layout with user controls
                fig.update_layout(
                    title=dict(
                        text=f'IDA Curves - {results_name}',
                        font=dict(size=20, color='#2c3e50', family='Arial Black')
                    ),
                    xaxis=dict(
                        title='Roof Drift [%]',
                        type='log' if log_x_tab2 else 'linear',
                        range=[drift_min_tab2, drift_max_tab2] if not log_x_tab2 else [np.log10(max(drift_min_tab2, 0.001)), np.log10(drift_max_tab2)],
                        gridcolor='lightgray',
                        showgrid=True,
                        zeroline=False,
                        title_font=dict(size=16, color='black'),
                        tickfont=dict(size=14, color='black')
                    ),
                    yaxis=dict(
                        title='Sa(T₁) [g]',
                        type='log' if log_y_tab2 else 'linear',
                        range=[sa_min_tab2, sa_max_tab2] if not log_y_tab2 else [np.log10(max(sa_min_tab2, 0.001)), np.log10(sa_max_tab2)],
                        gridcolor='lightgray',
                        showgrid=True,
                        zeroline=False,
                        title_font=dict(size=16, color='black'),
                        tickfont=dict(size=14, color='black')
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hovermode='closest',
                    height=600,
                    font=dict(family='Arial', size=14),
                    legend=dict(
                        title=dict(text="Ground Motions", font=dict(size=14, color='#2c3e50')),
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02,
                        font=dict(size=12),
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='lightgray',
                        borderwidth=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to show/hide specific GMs
                with st.expander("⚙️ Filter Ground Motions"):
                    all_gms = sorted(df['GM'].unique())
                    selected_gms = st.multiselect(
                        "Select GMs to display (leave empty for all):",
                        options=all_gms,
                        default=[],
                        format_func=lambda x: f"GM{x:02d}"
                    )
                    
                    if len(selected_gms) > 0:
                        # Plot controls for filtered plot
                        st.markdown("---")
                        col_ctrl_f1, col_ctrl_f2 = st.columns([1, 2])
                        
                        with col_ctrl_f1:
                            st.markdown("**Scale Options**")
                            log_x_filtered = st.checkbox("Log scale X-axis (Drift)", value=False, key="log_x_filtered")
                            log_y_filtered = st.checkbox("Log scale Y-axis (Sa)", value=False, key="log_y_filtered")
                        
                        with col_ctrl_f2:
                            st.markdown("**Axis Limits**")
                            col_lim_f1, col_lim_f2 = st.columns(2)
                            with col_lim_f1:
                                drift_min_filtered = st.number_input("Drift min [%]", value=float(max(0.01, df['dertecho'].min()*0.9)), step=0.01, format="%.3f", key="drift_min_filtered")
                                drift_max_filtered = st.number_input("Drift max [%]", value=float(df['dertecho'].max()*1.1), step=0.1, format="%.2f", key="drift_max_filtered")
                            with col_lim_f2:
                                sa_min_filtered = st.number_input("Sa min [g]", value=0.0, step=0.1, format="%.2f", key="sa_min_filtered")
                                sa_max_filtered = st.number_input("Sa max [g]", value=float(df['Sa'].max()*1.1), step=0.1, format="%.2f", key="sa_max_filtered")
                        
                        st.markdown("---")
                        # Create filtered plot
                        fig_filtered = go.Figure()
                        
                        for gm_id in selected_gms:
                            df_gm = df_plot[df_plot['GM'] == gm_id].sort_values('Sa')
                            
                            fig_filtered.add_trace(go.Scatter(
                                x=df_gm['dertecho_plot'],
                                y=df_gm['Sa'],
                                mode='lines+markers',
                                name=f'GM{gm_id:02d}',
                                line=dict(width=3),
                                marker=dict(size=8, line=dict(width=0.5, color='white')),
                                hovertemplate=f'<b>GM{gm_id:02d}</b><br>' +
                                             'Drift: %{x:.3f}%<br>' +
                                             'Sa: %{y:.3f} g<extra></extra>'
                            ))
                        
                        fig_filtered.update_layout(
                            title=dict(text=f'IDA Curves - Selected GMs', font=dict(size=20)),
                            xaxis=dict(
                                title='Roof Drift [%]', 
                                type='log' if log_x_filtered else 'linear',
                                range=[np.log10(drift_min_filtered), np.log10(drift_max_filtered)] if log_x_filtered else [drift_min_filtered, drift_max_filtered], 
                                gridcolor='lightgray', 
                                showgrid=True,
                                title_font=dict(size=16, color='black'),
                                tickfont=dict(size=14, color='black')
                            ),
                            yaxis=dict(
                                title='Sa(T₁) [g]',
                                type='log' if log_y_filtered else 'linear',
                                range=[np.log10(max(0.01, sa_min_filtered)), np.log10(sa_max_filtered)] if log_y_filtered else [sa_min_filtered, sa_max_filtered], 
                                gridcolor='lightgray', 
                                showgrid=True,
                                title_font=dict(size=16, color='black'),
                                tickfont=dict(size=14, color='black')
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=600,
                            font=dict(family='Arial', size=14)
                        )
                        
                        st.plotly_chart(fig_filtered, use_container_width=True)
            
            # ========== TAB 3: Boxplots by Sa Level ==========
            with tab3:
                st.markdown("### Roof Drift Distribution by Sa Level")
                
                # Plot controls
                col_ctrl1, col_ctrl2 = st.columns([1, 2])
                
                with col_ctrl1:
                    st.markdown("**Scale Options**")
                    log_y_tab3 = st.checkbox("Log scale Y-axis (Drift)", value=False, key="log_y_tab3")
                
                with col_ctrl2:
                    st.markdown("**Y-Axis Limits**")
                    drift_min_tab3 = st.number_input("Drift min [%]", value=0.0, step=0.1, format="%.2f", key="drift_min_tab3")
                    drift_max_tab3 = st.number_input("Drift max [%]", value=float(df['dertecho'].max()*1.2), step=0.1, format="%.2f", key="drift_max_tab3")
                
                st.markdown("---")
                
                # Create figure with boxplots
                fig = go.Figure()
                
                # Get unique Sa levels and sort them
                sa_levels = sorted(df['Sa'].unique())
                
                # Create boxplot for each Sa level
                for sa in sa_levels:
                    df_sa = df[df['Sa'] == sa]
                    
                    fig.add_trace(go.Box(
                        y=df_sa['dertecho'],
                        name=f"{sa:.3f}",
                        boxmean='sd',  # Show mean and standard deviation
                        marker=dict(
                            color='lightblue',
                            line=dict(color='darkblue', width=1.5)
                        ),
                        line=dict(color='darkblue', width=2),
                        fillcolor='rgba(135, 206, 250, 0.5)',
                        hovertemplate='<b>Sa: ' + f'{sa:.3f} g</b><br>' +
                                     'Drift: %{y:.3f}%<extra></extra>'
                    ))
                
                # Update layout
                fig.update_layout(
                    title=dict(
                        text=f'Roof Drift Distribution by Sa Level - {results_name}',
                        font=dict(size=20, color='#2c3e50', family='Arial Black')
                    ),
                    xaxis=dict(
                        title='Sa(T₁) [g]',
                        gridcolor='lightgray',
                        showgrid=True,
                        title_font=dict(size=16, color='black'),
                        tickfont=dict(size=14, color='black')
                    ),
                    yaxis=dict(
                        title='Roof Drift [%]',
                        type='log' if log_y_tab3 else 'linear',
                        range=[drift_min_tab3, drift_max_tab3] if not log_y_tab3 else [np.log10(max(drift_min_tab3, 0.001)), np.log10(drift_max_tab3)],
                        gridcolor='lightgray',
                        showgrid=True,
                        zeroline=False,
                        title_font=dict(size=16, color='black'),
                        tickfont=dict(size=14, color='black')
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=False,
                    height=600,
                    font=dict(family='Arial', size=14),
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics table
                st.markdown("---")
                st.markdown("### Statistical Summary by Sa Level")
                
                # Calculate statistics for each Sa level
                stats_data = []
                for sa in sa_levels:
                    df_sa = df[df['Sa'] == sa]
                    stats_data.append({
                        'Sa [g]': f"{sa:.3f}",
                        'Count': len(df_sa),
                        'Mean [%]': f"{df_sa['dertecho'].mean():.3f}",
                        'Std [%]': f"{df_sa['dertecho'].std():.3f}",
                        'Min [%]': f"{df_sa['dertecho'].min():.3f}",
                        'Median [%]': f"{df_sa['dertecho'].median():.3f}",
                        'Max [%]': f"{df_sa['dertecho'].max():.3f}"
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
                # Additional violin plot option
                with st.expander("📊 Show Violin Plot"):
                    fig_violin = go.Figure()
                    
                    for sa in sa_levels:
                        df_sa = df[df['Sa'] == sa]
                        
                        fig_violin.add_trace(go.Violin(
                            y=df_sa['dertecho'],
                            name=f"{sa:.3f}",
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor='lightseagreen',
                            opacity=0.6,
                            line_color='darkslategray',
                            hovertemplate='<b>Sa: ' + f'{sa:.3f} g</b><br>' +
                                         'Drift: %{y:.3f}%<extra></extra>'
                        ))
                    
                    fig_violin.update_layout(
                        title=dict(text='Roof Drift Distribution (Violin Plot)', font=dict(size=20)),
                        xaxis=dict(
                            title='Sa(T₁) [g]', 
                            gridcolor='lightgray', 
                            showgrid=True,
                            title_font=dict(size=16, color='black'),
                            tickfont=dict(size=14, color='black')
                        ),
                        yaxis=dict(
                            title='Roof Drift [%]', 
                            gridcolor='lightgray', 
                            showgrid=True,
                            title_font=dict(size=16, color='black'),
                            tickfont=dict(size=14, color='black')
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        showlegend=False,
                        height=600,
                        font=dict(family='Arial', size=14)
                    )
                    
                    st.plotly_chart(fig_violin, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error loading results: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ==================== PAGE 3: FRAGILITY FUNCTIONS ====================
elif page == "Fragility Functions":
    st.header("Fragility Function Calculation")
    
    # Step 1: Load IDA Results
    st.subheader("Step 1: Load IDA Results")
    
    # Find available IDA result files
    idas_folder = "IDAs"
    if not os.path.exists(idas_folder):
        st.warning("⚠️ IDAs folder not found. Please run an IDA analysis first.")
        st.stop()
    
    # Get all .pkl files from IDAs folder
    ida_files = glob.glob(os.path.join(idas_folder, "*_IDA.pkl"))
    
    if len(ida_files) == 0:
        st.warning("⚠️ No IDA result files found. Please run an IDA analysis first.")
        st.stop()
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_results = st.selectbox(
                "Available IDA Results:",
                options=ida_files,
                format_func=lambda x: os.path.basename(x).replace('_IDA.pkl', ''),
                index=ida_files.index(st.session_state.selected_results_file) if st.session_state.selected_results_file in ida_files else 0,
                help="Select IDA results for fragility calculation"
            )
            
            if selected_results:
                st.session_state.selected_results_file = selected_results
                results_name = os.path.basename(selected_results).replace('_IDA.pkl', '')
                st.info(f"📊 Selected Results: **{results_name}**")
        
        with col2:
            # Check for corresponding model image
            model_name = os.path.basename(selected_results).replace('_IDA.pkl', '')
            model_image = os.path.join("models", f"{model_name}_model.png")
            if os.path.exists(model_image):
                st.image(model_image, caption="Model", use_container_width=True)
        
        # Load the results
        try:
            df = pd.read_pickle(selected_results)
            
            # Load model to get building height
            model_file = os.path.join("models", f"{model_name}.pkl")
            if os.path.exists(model_file):
                Arq1 = jb.load(model_file)
                h_building = Arq1.listaCoorY[0]
            else:
                st.warning("⚠️ Model file not found. Using default building height of 15m.")
                h_building = 15.0
            
            st.markdown("---")
            
            # Step 2: Define Damage States
            st.subheader("Step 2: Define Damage States")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Number of Damage States**")
                num_ds = st.number_input(
                    "Number of Damage States:",
                    min_value=1,
                    max_value=10,
                    value=4,
                    step=1,
                    help="Number of damage state thresholds to define"
                )
            
            with col2:
                st.info("📊 Using **dertecho** (roof drift %) as EDP and **Sa** as IM")
                st.info(f"🏢 Building Height: {h_building:.2f} m")
            
            st.markdown("---")
            
            # Define damage state limits
            st.subheader("Damage State Limits")
            st.write("Define the threshold values for each damage state:")
            
            # Create columns for limit inputs
            limit_names = []
            limits = []
            
            cols = st.columns(min(4, num_ds))
            for i in range(num_ds):
                with cols[i % 4]:
                    st.markdown(f"**Damage State {i+1}**")
                    
                    # Default name
                    ds_name = st.text_input(
                        f"DS{i+1} Name:",
                        value=f"ds{i+1}",
                        key=f"ds_name_{i}"
                    )
                    limit_names.append(ds_name)
                    
                    # Default limits based on typical drift percentages
                    default_limits = [0.4, 0.8, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
                    default_value = default_limits[i] if i < len(default_limits) else 1.0 * (i + 1)
                    
                    # Limit value as drift percentage (directly compatible with dertecho)
                    limit_val = st.number_input(
                        f"Drift Limit [%]:",
                        min_value=0.01,
                        max_value=20.0,
                        value=default_value,
                        step=0.1,
                        format="%.2f",
                        key=f"limit_{i}",
                        help="Drift percentage threshold (e.g., 0.4 = 0.4%)"
                    )
                    limits.append(limit_val)
            
            limits = np.array(limits)
            
            st.markdown("---")
            
            # Step 3: Calculate Fragility Functions
            st.subheader("Step 3: Calculate Fragility Functions")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("📊 Calculate Fragility Functions", type="primary", use_container_width=True):
                    try:
                        # Calculate fragility functions using dertecho (drift %)
                        with st.spinner("Calculating fragility functions..."):
                            thetas, betas = lf.calculate_fragility(
                                df, 
                                limit_names, 
                                limits, 
                                'Sa', 
                                'dertecho',
                                plot=False
                            )
                        
                        st.success("✅ Fragility functions calculated successfully!")
                        
                        # Store in session state
                        st.session_state.fragility_thetas = thetas
                        st.session_state.fragility_betas = betas
                        st.session_state.fragility_limit_names = limit_names
                        st.session_state.fragility_limits = limits
                        
                    except Exception as e:
                        st.error(f"❌ Error calculating fragility functions: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Display results if calculated
            if 'fragility_thetas' in st.session_state:
                st.markdown("---")
                st.subheader("Fragility Function Parameters")
                
                # Create summary table
                summary_data = []
                for i, name in enumerate(st.session_state.fragility_limit_names):
                    summary_data.append({
                        'Damage State': name,
                        'Threshold': f"{limits[i]:.3f}",
                        'Theta (θ)': f"{st.session_state.fragility_thetas[i]:.4f}",
                        'Beta (β)': f"{st.session_state.fragility_betas[i]:.4f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Fragility Curves")
                
                # Plot controls
                col_ctrl1, col_ctrl2 = st.columns([1, 2])
                
                with col_ctrl1:
                    st.markdown("**Scale Options**")
                    log_x_frag = st.checkbox("Log scale X-axis", value=False, key="log_x_frag")
                    log_y_frag = st.checkbox("Log scale Y-axis", value=False, key="log_y_frag")
                
                with col_ctrl2:
                    st.markdown("**Axis Limits**")
                    col_lim1, col_lim2 = st.columns(2)
                    with col_lim1:
                        sa_min_frag = st.number_input("Sa min [g]", value=0.0, step=0.1, format="%.2f", key="sa_min_frag")
                        sa_max_frag = st.number_input("Sa max [g]", value=4.0, step=0.1, format="%.2f", key="sa_max_frag")
                    with col_lim2:
                        prob_min_frag = st.number_input("Probability min", value=0.0, step=0.1, format="%.2f", key="prob_min_frag")
                        prob_max_frag = st.number_input("Probability max", value=1.0, step=0.1, format="%.2f", key="prob_max_frag")
                
                st.markdown("---")
                
                # Create fragility curves plot
                x = np.linspace(max(0.01, sa_min_frag), sa_max_frag, 200)
                
                fig = go.Figure()
                
                # Color scheme
                colors = px.colors.qualitative.Set2
                
                for i, name in enumerate(st.session_state.fragility_limit_names):
                    theta = st.session_state.fragility_thetas[i]
                    beta = st.session_state.fragility_betas[i]
                    
                    # Calculate lognormal CDF
                    y = stats.lognorm.cdf(x, s=beta, scale=theta)
                    
                    # Add trace
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        name=f'{name} (θ={theta:.3f}, β={beta:.3f})',
                        line=dict(width=3, color=colors[i % len(colors)]),
                        hovertemplate='<b>' + name + '</b><br>' +
                                     'Sa: %{x:.3f} g<br>' +
                                     'P[DS≥' + name + ']: %{y:.3f}<extra></extra>'
                    ))
                
                # Update layout
                fig.update_layout(
                    title=dict(
                        text=f'Fragility Curves - {results_name}',
                        font=dict(size=20, color='#2c3e50', family='Arial Black')
                    ),
                    xaxis=dict(
                        title='Sa(T₁) [g]',
                        type='log' if log_x_frag else 'linear',
                        range=[sa_min_frag, sa_max_frag] if not log_x_frag else [np.log10(max(sa_min_frag, 0.001)), np.log10(sa_max_frag)],
                        gridcolor='lightgray',
                        showgrid=True,
                        zeroline=False,
                        title_font=dict(size=16, color='black'),
                        tickfont=dict(size=14, color='black')
                    ),
                    yaxis=dict(
                        title='Probability of Exceedance',
                        type='log' if log_y_frag else 'linear',
                        range=[prob_min_frag, prob_max_frag] if not log_y_frag else [np.log10(max(prob_min_frag, 0.001)), np.log10(prob_max_frag)],
                        gridcolor='lightgray',
                        showgrid=True,
                        zeroline=False,
                        title_font=dict(size=16, color='black'),
                        tickfont=dict(size=14, color='black')
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hovermode='closest',
                    height=600,
                    font=dict(family='Arial', size=14),
                    legend=dict(
                        title=dict(text="Damage States", font=dict(size=14, color='#2c3e50')),
                        orientation="v",
                        yanchor="top",
                        y=0.3,
                        xanchor="right",
                        x=0.98,
                        font=dict(size=12),
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='lightgray',
                        borderwidth=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    # Create CSV for download
                    download_data = {
                        'Sa [g]': x,
                    }
                    for i, name in enumerate(st.session_state.fragility_limit_names):
                        theta = st.session_state.fragility_thetas[i]
                        beta = st.session_state.fragility_betas[i]
                        y = stats.lognorm.cdf(x, s=beta, scale=theta)
                        download_data[f'P[{name}]'] = y
                    
                    download_df = pd.DataFrame(download_data)
                    csv = download_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Fragility Curves Data",
                        data=csv,
                        file_name=f"{results_name}_fragility.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error loading results: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")
st.caption("RC-WIAP - Incremental Dynamic Analysis Tool")
