import streamlit as st
import pandas as pd
import os
import subprocess
import sys
from pathlib import Path
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="RC-WIAP Analysis", layout="wide")

# Initialize session state
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'page' not in st.session_state:
    st.session_state.page = "Analysis"

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Analysis", "Results Visualization"])

# Title
st.title("RC-WIAP Pushover Analysis")
st.markdown("---")

# ==================== PAGE 1: ANALYSIS ====================
if page == "Analysis":
    # Step 1: Model Selection
    st.header("Step 1: Select Model")
    st.write("Select a model file (.pkl) to analyze")

    # Get all .pkl files in the models directory (excluding RPO files)
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    pkl_files = [f for f in glob.glob(f"{models_dir}/*.pkl") if not f.endswith("-RPO.pkl")]
    # Extract just the filename for display
    pkl_filenames = [os.path.basename(f) for f in pkl_files]

    if len(pkl_files) == 0:
        st.warning("⚠️ No model files found in 'models/' folder. Please create a model using the Model Generator app first.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_filename = st.selectbox(
                "Available Models:",
                options=pkl_filenames,
                index=pkl_filenames.index(st.session_state.selected_model) if st.session_state.selected_model in pkl_filenames else 0,
                help="Select the model file you want to analyze"
            )
            
            # Get full path
            selected_file = os.path.join(models_dir, selected_filename)
            
            # Save to session state
            st.session_state.selected_file = selected_file
            st.session_state.selected_model = selected_filename
            
            # Display model information
            if selected_filename:
                model_name = selected_filename.replace('.pkl', '')
                
                st.info(f"📦 Selected Model: **{model_name}**")
                
                # Try to display basic model info
                file_size = os.path.getsize(selected_file) / 1024  # KB
                st.text(f"File size: {file_size:.2f} KB")
        
        with col2:
            # Show model image if available
            model_image = selected_file.replace('.pkl', '_model.png')
            if os.path.exists(model_image):
                st.image(model_image, caption="Model Visualization", use_container_width=True)

    st.markdown("---")

    # Step 2: Analysis Parameters
    st.header("Step 2: Analysis Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pushover Settings")
        
        pushlimit = st.number_input(
            "Drift Limit (pushlimit):",
            min_value=0.001,
            max_value=0.100,
            value=0.035,
            step=0.005,
            format="%.3f",
            help="Maximum drift ratio for pushover analysis (default: 0.035 = 3.5%)"
        )
        
        pushtype = st.selectbox(
            "Load Distribution Pattern:",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Triangular", 2: "Uniform", 3: "Modal (Recommended)"}[x],
            index=2,
            help="1: Triangular, 2: Uniform, 3: Proportional to mode shape"
        )
        
        modepush = st.number_input(
            "Mode for Load Pattern:",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Mode number for modal load pattern (usually 1 for fundamental mode)"
        )

    with col2:
        st.subheader("Control Settings")
        
        wallctrl = st.number_input(
            "Control Wall Index:",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Wall index for control (0 for first wall, 1 for second, etc.)"
        )
        
        Dincr = st.number_input(
            "Load Increment (Dincr):",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f",
            help="Load increment for pushover analysis"
        )
        
        direction = st.selectbox(
            "Analysis Direction:",
            options=["Longitudinal", "Transversal"],
            index=0,
            help="Direction of analysis"
        )

    st.markdown("---")

    # Step 3: Run Analysis
    st.header("Step 3: Run Analysis")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Run Pushover Analysis", type="primary", use_container_width=True):
            if st.session_state.selected_model is None:
                st.error("Please select a model first!")
            else:
                # Run the analysis directly with command line arguments
                try:
                    # Get the full path from session state
                    model_full_path = st.session_state.selected_file
                    model_name = st.session_state.selected_model.replace('.pkl', '')
                    
                    # Build command with parameters
                    # Usage: python 01_PUSHOVER.py model_name [pushlimit] [pushtype] [modepush] [wallctrl] [Dincr] [direction]
                    cmd = [
                        sys.executable, 
                        "01_PUSHOVER.py", 
                        model_full_path.replace('.pkl', ''),  # Pass full path without .pkl
                        str(pushlimit),
                        str(pushtype),
                        str(modepush),
                        str(wallctrl),
                        str(Dincr),
                        direction
                    ]
                    
                    # Run the analysis
                    st.info("⏳ Running pushover analysis... This may take a few minutes.")
                    st.info(f"Parameters: pushlimit={pushlimit}, pushtype={pushtype}, modepush={modepush}, wallctrl={wallctrl}, Dincr={Dincr}, direction={direction}")
                    
                    progress_bar = st.progress(0)
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True
                    )
                    
                    progress_bar.progress(100)
                    
                    if result.returncode == 0:
                        st.success("✅ Pushover analysis completed successfully!")
                        st.balloons()
                        
                        st.session_state.analysis_complete = True
                        
                        # Show output
                        with st.expander("View Analysis Output"):
                            st.code(result.stdout)
                        
                        # Look for results Excel file in pushover folder
                        excel_file = os.path.join("pushover", f"{model_name}_pushover.xlsx")
                        
                        if os.path.exists(excel_file):
                            results_file = excel_file
                            st.info(f"📊 Results saved to: {results_file}")
                            
                            # Display results preview
                            st.markdown("---")
                            st.subheader("Results Preview")
                            
                            try:
                                # Read the Excel file
                                xls = pd.ExcelFile(results_file)
                                
                                tabs = st.tabs(xls.sheet_names)
                                
                                for i, sheet_name in enumerate(xls.sheet_names):
                                    with tabs[i]:
                                        df = pd.read_excel(results_file, sheet_name=sheet_name)
                                        
                                        # Hide data table in expander
                                        with st.expander("View Data Table"):
                                            st.markdown(df.to_html(), unsafe_allow_html=True)
                                        
                                        # Plot if it has Vbasal column
                                        if 'Vbasal' in df.columns:
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            
                                            x_col = df.columns[1]  # Second column (first is index)
                                            
                                            # Add (0,0) starting point
                                            x_data = np.concatenate([[0], df[x_col].values])
                                            y_vbasal = np.concatenate([[0], df['Vbasal'].values])
                                            
                                            ax.plot(x_data, y_vbasal, 'b-', linewidth=2, label='Total Base Shear')
                                            
                                            # Plot individual walls if available
                                            for col in df.columns[3:]:  # Skip index, drift, and Vbasal
                                                y_wall = np.concatenate([[0], df[col].values])
                                                ax.plot(x_data, y_wall, '--', linewidth=1, alpha=0.7, label=col)
                                            
                                            ax.set_xlabel(sheet_name, fontsize=16, fontweight='bold')
                                            ax.set_ylabel('Base Shear (kN)', fontsize=16, fontweight='bold')
                                            ax.set_title(f'Pushover Curve - {sheet_name}', fontsize=18, fontweight='bold')
                                            ax.tick_params(axis='both', which='major', labelsize=14)
                                            ax.grid(True, alpha=0.3)
                                            ax.legend(loc='best', fontsize=12)
                                            
                                            st.pyplot(fig)
                                            plt.close()
                            
                            except Exception as e:
                                st.warning(f"Could not preview results: {str(e)}")
                        
                        # Check for RPO pickle file in pushover folder
                        rpo_file = os.path.join("pushover", f"{model_name}-RPO.pkl")
                        if os.path.exists(rpo_file):
                            st.info(f"📦 Analysis object saved to: {rpo_file}")
                    
                    else:
                        st.error("❌ Error during analysis!")
                        st.error("Error output:")
                        st.code(result.stderr)
                        if result.stdout:
                            st.code(result.stdout)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# ==================== PAGE 2: RESULTS VISUALIZATION ====================
elif page == "Results Visualization":
    st.header("Pushover Results Visualization")
    st.write("Load and visualize pushover analysis results")
    
    # Define pushover directory
    pushover_dir = "pushover"
    
    # Create pushover directory if it doesn't exist
    if not os.path.exists(pushover_dir):
        os.makedirs(pushover_dir)
    
    # Get all RPO.pkl files from pushover folder
    rpo_files = glob.glob(os.path.join(pushover_dir, "*-RPO.pkl"))
    
    # Extract just the filenames for display
    rpo_basenames = [os.path.basename(f) for f in rpo_files]
    
    if len(rpo_files) == 0:
        st.warning("⚠️ No results found. Please run a pushover analysis first.")
    else:
        # Select results file
        selected_rpo_basename = st.selectbox(
            "Select Results File:",
            options=rpo_basenames,
            help="Select the pushover results file to visualize"
        )
        
        # Get full path
        selected_rpo = os.path.join(pushover_dir, selected_rpo_basename) if selected_rpo_basename else None
        
        if selected_rpo and st.button("📊 Load and Visualize Results"):
            try:
                # Load the RPO file
                with open(selected_rpo, "rb") as archivo:
                    ARQ_R_PO = pickle.loads(archivo.read())
                
                st.success(f"✅ Loaded results from: {selected_rpo}")
                
                # Extract model information
                model_name = selected_rpo_basename.replace('-RPO.pkl', '')
                st.info(f"**Model:** {ARQ_R_PO.nombre}")
                
                # Calculate control points
                st.markdown("---")
                st.subheader("🎯 Control Points Analysis")
                
                with st.spinner("Calculating control points..."):
                    # Initialize lists for control points
                    paso_Cr = []
                    paso_RBy = []
                    paso_WWMy = []
                    paso_RB_Ult = []
                    paso_WWM_Ult = []
                    paso_RB_50Ult = []
                    paso_WWM_50Ult = []
                    
                    # Strain thresholds
                    Def_Cr = 0.00015
                    Def_RBy = 420/200000
                    Def_WWMy = 490/200000
                    Def_RB_Ult = 0.05
                    Def_WWM_Ult = 0.015
                    
                    NumPaso = len(ARQ_R_PO.muros[0].pisos[-1].ResultadosAltura.list_Strain)
                    
                    # Find control points for each wall
                    for ele_i in ARQ_R_PO.muros:
                        minDiff_Cr = 99999
                        minDiff_RBy = 99999
                        minDiff_WWMy = 99999
                        minDiff_RB_Ult = 99999
                        minDiff_WWM_Ult = 99999
                        minDiff_RB_50Ult = 99999
                        minDiff_WWM_50Ult = 99999
                        
                        i_Cr = 0
                        i_RBy = 0
                        i_WWMy = 0
                        i_RB_Ult = NumPaso-1
                        i_WWM_Ult = NumPaso-1
                        i_RB_50Ult = NumPaso-1
                        i_WWM_50Ult = NumPaso-1
                        
                        cont_RB_Ult = 0
                        cont_WWM_Ult = 0
                        cont_RB_50Ult = 0
                        cont_WWM_50Ult = 0
                        
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
                        
                        paso_Cr.append(i_Cr)
                        paso_RBy.append(i_RBy)
                        paso_WWMy.append(i_WWMy)
                        paso_RB_Ult.append(i_RB_Ult)
                        paso_WWM_Ult.append(i_WWM_Ult)
                        paso_RB_50Ult.append(i_RB_50Ult)
                        paso_WWM_50Ult.append(i_WWM_50Ult)
                    
                    # Find position of maximum base shear
                    Posicion_VMax = np.argmax(ARQ_R_PO.ResultadosPushover.listaVbase)
                    
                    # Find 80% capacity point
                    minDiff = 99999
                    paso_80 = 0
                    for index, i in enumerate(ARQ_R_PO.ResultadosPushover.listaVbase[Posicion_VMax:]):
                        Diff = np.abs(i-(0.8*max(ARQ_R_PO.ResultadosPushover.listaVbase)))
                        if Diff < minDiff:
                            minDiff = Diff
                            paso_80 = index + Posicion_VMax
                    
                    # Create control points dataframe
                    ptosctrl = pd.DataFrame(data={'SDR1':None, 'RDR':None, 'SDR':None, 'Vb':None, 'V/W':None}, 
                                            index=['1er-Agrietamiento', '1ra-Fluencia', 'Cap-Maxima',
                                                   '1ra-Rotura', '80%-Capacidad', '50%-Rotura'], dtype=float)
                    
                    # Populate control points
                    pc_sdr1_1 = ARQ_R_PO.ResultadosPushover.listaSDR1[paso_Cr[0]]
                    pc_sdr1_2 = ARQ_R_PO.ResultadosPushover.listaSDR1[paso_WWMy[0] if paso_WWMy[0] > 0 else paso_RBy[0]]
                    pc_sdr1_3 = ARQ_R_PO.ResultadosPushover.listaSDR1[Posicion_VMax]
                    pc_sdr1_4 = ARQ_R_PO.ResultadosPushover.listaSDR1[min(paso_WWM_Ult) if min(paso_WWM_Ult) != (NumPaso-1) else min(paso_RB_Ult)]
                    pc_sdr1_5 = ARQ_R_PO.ResultadosPushover.listaSDR1[paso_80]
                    pc_sdr1_6 = ARQ_R_PO.ResultadosPushover.listaSDR1[min(paso_WWM_50Ult) if min(paso_WWM_50Ult) != (NumPaso-1) else min(paso_RB_50Ult)]
                    
                    pc_d1 = ARQ_R_PO.ResultadosPushover.listaRDR[paso_Cr[0]]
                    pc_d2 = ARQ_R_PO.ResultadosPushover.listaRDR[paso_WWMy[0] if paso_WWMy[0] > 0 else paso_RBy[0]]
                    pc_d3 = ARQ_R_PO.ResultadosPushover.listaRDR[Posicion_VMax]
                    pc_d4 = ARQ_R_PO.ResultadosPushover.listaRDR[min(paso_WWM_Ult) if min(paso_WWM_Ult) != (NumPaso-1) else min(paso_RB_Ult)]
                    pc_d5 = ARQ_R_PO.ResultadosPushover.listaRDR[paso_80]
                    pc_d6 = ARQ_R_PO.ResultadosPushover.listaRDR[min(paso_WWM_50Ult) if min(paso_WWM_50Ult) != (NumPaso-1) else min(paso_RB_50Ult)]
                    
                    pc_sdr1 = ARQ_R_PO.ResultadosPushover.listaSDR[paso_Cr[0]]
                    pc_sdr2 = ARQ_R_PO.ResultadosPushover.listaSDR[paso_WWMy[0] if paso_WWMy[0] > 0 else paso_RBy[0]]
                    pc_sdr3 = ARQ_R_PO.ResultadosPushover.listaSDR[Posicion_VMax]
                    pc_sdr4 = ARQ_R_PO.ResultadosPushover.listaSDR[min(paso_WWM_Ult) if min(paso_WWM_Ult) != (NumPaso-1) else min(paso_RB_Ult)]
                    pc_sdr5 = ARQ_R_PO.ResultadosPushover.listaSDR[paso_80]
                    pc_sdr6 = ARQ_R_PO.ResultadosPushover.listaSDR[min(paso_WWM_50Ult) if min(paso_WWM_50Ult) != (NumPaso-1) else min(paso_RB_50Ult)]
                    
                    pc_v1 = ARQ_R_PO.ResultadosPushover.listaVbase[paso_Cr[0]]
                    pc_v2 = ARQ_R_PO.ResultadosPushover.listaVbase[paso_WWMy[0] if paso_WWMy[0] > 0 else paso_RBy[0]]
                    pc_v3 = ARQ_R_PO.ResultadosPushover.listaVbase[Posicion_VMax]
                    pc_v4 = ARQ_R_PO.ResultadosPushover.listaVbase[min(paso_WWM_Ult) if min(paso_WWM_Ult) != (NumPaso-1) else min(paso_RB_Ult)]
                    pc_v5 = ARQ_R_PO.ResultadosPushover.listaVbase[paso_80]
                    pc_v6 = ARQ_R_PO.ResultadosPushover.listaVbase[min(paso_WWM_50Ult) if min(paso_WWM_50Ult) != (NumPaso-1) else min(paso_RB_50Ult)]
                    
                    ptosctrl['SDR1'] = np.array([pc_sdr1_1, pc_sdr1_2, pc_sdr1_3, pc_sdr1_4, pc_sdr1_5, pc_sdr1_6], dtype=float)
                    ptosctrl['RDR'] = np.array([pc_d1, pc_d2, pc_d3, pc_d4, pc_d5, pc_d6], dtype=float)
                    ptosctrl['SDR'] = np.array([pc_sdr1, pc_sdr2, pc_sdr3, pc_sdr4, pc_sdr5, pc_sdr6], dtype=float)
                    ptosctrl['Vb'] = np.array([pc_v1, pc_v2, pc_v3, pc_v4, pc_v5, pc_v6], dtype=float)
                    ptosctrl['V/W'] = ptosctrl['Vb']/(ARQ_R_PO.ws_*9.81)
                    
                    ptosctrl = ptosctrl.dropna()
                
                # Display control points table (optional)
                with st.expander("View Control Points Table"):
                    st.markdown(ptosctrl.to_html(), unsafe_allow_html=True)
                
                # Visualization with tabs
                st.markdown("---")
                st.subheader("📊 Pushover Results Visualization")
                
                # Create tabs for different EDPs - RDR first (default)
                edp_tabs = st.tabs(["RDR (Roof Drift)", "SDR (Story Drift)", "SDR1 (1st Story Drift)"])
                
                # Get list of walls for filtering
                listamuros = [i.nombre for i in ARQ_R_PO.muros]
                
                # Common control point indices
                Index_Cr = paso_Cr[0]
                Index_y = paso_WWMy[0] if paso_WWMy[0] > 0 else paso_RBy[0]
                Index_max = Posicion_VMax
                Index_80 = paso_80
                
                # Process each EDP
                edp_configs = [
                    ('RDR_max', ARQ_R_PO.ResultadosPushover.listaRDR, 'SDR', 'RDR (%)', 'RDR'),
                    ('SDR_max', ARQ_R_PO.ResultadosPushover.listaSDR, 'RDR', 'SDR<sub>max</sub> (%)', 'SDR'),
                    ('SDR1_max', ARQ_R_PO.ResultadosPushover.listaSDR1, 'RDR', 'SDR<sub>1</sub> (%)', 'SDR1')
                ]
                
                for tab_idx, (edp_name, edp_data, edp_legend, edp_label, edp_col) in enumerate(edp_configs):
                    with edp_tabs[tab_idx]:
                        
                        # Pushover Curve with Control Points
                        st.markdown("#### Pushover Curve with Control Points")
                        
                        fig_push = go.Figure()
                        
                        # Add (0,0) starting point and plot capacity curve
                        edp_data_with_zero = np.concatenate([[0], edp_data])
                        vbase_data_with_zero = np.concatenate([[0], ARQ_R_PO.ResultadosPushover.listaVbase/(ARQ_R_PO.ws_*9.81)])
                        
                        fig_push.add_trace(go.Scatter(
                            x=edp_data_with_zero,
                            y=vbase_data_with_zero,
                            mode='lines',
                            name='Capacity Curve',
                            line=dict(color='black', width=3)
                        ))
                        
                        # Plot control points
                        colors = ['blue','orangered','gold','darkviolet',
                                 'dodgerblue','firebrick', 'olive', 'rosybrown']
                        
                        for i, pc in enumerate(ptosctrl.index):
                            fig_push.add_trace(go.Scatter(
                                x=[ptosctrl[edp_col][i]],
                                y=[ptosctrl['V/W'][i]],
                                mode='markers',
                                name=f"{ptosctrl.index[i]}, {edp_legend}={ptosctrl[edp_legend][i]:.2f}%",
                                marker=dict(size=12, color=colors[i])
                            ))
                        
                        fig_push.update_layout(
                            title=dict(text=f'PUSHOVER - {ARQ_R_PO.nombre}', font=dict(size=20, family='Arial Black')),
                            xaxis_title=edp_label,
                            yaxis_title='V/W',
                            xaxis=dict(
                                rangemode='tozero',
                                title_font=dict(size=18, family='Arial', color='black'),
                                tickfont=dict(size=16, color='#000000'),
                                tickcolor='#000000',
                                tickwidth=2
                            ),
                            yaxis=dict(
                                rangemode='tozero',
                                title_font=dict(size=18, family='Arial', color='black'),
                                tickfont=dict(size=16, color='#000000'),
                                tickcolor='#000000',
                                tickwidth=2
                            ),
                            hovermode='closest',
                            template='plotly_white',
                            height=600,
                            showlegend=True,
                            legend=dict(x=1.02, y=1, xanchor='left', yanchor='top', font=dict(size=14))
                        )
                        
                        st.plotly_chart(fig_push, use_container_width=True, key=f"push_{tab_idx}")
                        
                        # Additional plots tabs
                        st.markdown("---")
                        sub_tabs = st.tabs(["Moments", "Shear Forces", "Drift Profile"])
                        
                        # Moments plot
                        with sub_tabs[0]:
                            st.markdown("#### Moments at Base")
                            fig_moment = go.Figure()
                            
                            for muro in ARQ_R_PO.muros:
                                # Add (0,0) starting point - use RDR from global results
                                x_data = np.concatenate([[0], ARQ_R_PO.ResultadosPushover.listaRDR])
                                y_data = np.concatenate([[0], muro.pisos[-1].ResultadosAltura.list_M])
                                
                                fig_moment.add_trace(go.Scatter(
                                    x=x_data,
                                    y=y_data,
                                    mode='lines',
                                    name=f"{muro.nombre}: {muro.pisos[-1].id_}",
                                    line=dict(width=2)
                                ))
                            
                            fig_moment.update_layout(
                                xaxis_title='RDR (%)',
                                yaxis_title='Moment (kN·m)',
                                xaxis=dict(
                                    rangemode='tozero',
                                    title_font=dict(size=18, family='Arial', color='black'),
                                    tickfont=dict(size=16, color='#000000'),
                                    tickcolor='#000000',
                                    tickwidth=2
                                ),
                                yaxis=dict(
                                    rangemode='tozero',
                                    title_font=dict(size=18, family='Arial', color='black'),
                                    tickfont=dict(size=16, color='#000000'),
                                    tickcolor='#000000',
                                    tickwidth=2
                                ),
                                hovermode='x unified',
                                template='plotly_white',
                                height=500,
                                legend=dict(font=dict(size=14))
                            )
                            
                            st.plotly_chart(fig_moment, use_container_width=True, key=f"moment_{tab_idx}")
                        
                        # Shear plot
                        with sub_tabs[1]:
                            st.markdown("#### Shear Forces at Base")
                            fig_shear = go.Figure()
                            
                            for muro in ARQ_R_PO.muros:
                                # Add (0,0) starting point - use RDR from global results
                                x_data = np.concatenate([[0], ARQ_R_PO.ResultadosPushover.listaRDR])
                                y_data = np.concatenate([[0], muro.pisos[-1].ResultadosAltura.list_V])
                                
                                fig_shear.add_trace(go.Scatter(
                                    x=x_data,
                                    y=y_data,
                                    mode='lines',
                                    name=f"{muro.nombre}: {muro.pisos[-1].id_}",
                                    line=dict(width=2)
                                ))
                            
                            fig_shear.update_layout(
                                xaxis_title='RDR (%)',
                                yaxis_title='Shear Force (kN)',
                                xaxis=dict(
                                    rangemode='tozero',
                                    title_font=dict(size=18, family='Arial', color='black'),
                                    tickfont=dict(size=16, color='#000000'),
                                    tickcolor='#000000',
                                    tickwidth=2
                                ),
                                yaxis=dict(
                                    rangemode='tozero',
                                    title_font=dict(size=18, family='Arial', color='black'),
                                    tickfont=dict(size=16, color='#000000'),
                                    tickcolor='#000000',
                                    tickwidth=2
                                ),
                                hovermode='x unified',
                                template='plotly_white',
                                height=500,
                                legend=dict(font=dict(size=14))
                            )
                            
                            st.plotly_chart(fig_shear, use_container_width=True, key=f"shear_{tab_idx}")
                        
                        # Drift profile
                        with sub_tabs[2]:
                            st.markdown("#### Drift Profile Along Height")
                            fig_profile = go.Figure()
                            
                            Index = (Index_Cr, Index_y, Index_max, Index_80)
                            color_profile = ['blue','orangered','gold', 'dodgerblue']
                            legh_i = ['1st Cracking', '1st Yielding', 'Max Capacity', '80% Capacity']
                            
                            for i, index_id in enumerate(Index):
                                listSDRmax = []
                                for piso in ARQ_R_PO.muros[0].pisos:
                                    listSDRmax.append(piso.ResultadosAltura.list_SDR[index_id])
                                listSDRmax.append(0)
                                
                                hi_i = np.array(ARQ_R_PO.listaCoorY, dtype=float)/max(ARQ_R_PO.listaCoorY)
                                hi_i = np.append(hi_i, 0)
                                
                                fig_profile.add_trace(go.Scatter(
                                    x=listSDRmax,
                                    y=hi_i,
                                    mode='lines+markers',
                                    name=legh_i[i],
                                    line=dict(width=3, color=color_profile[i]),
                                    marker=dict(size=8)
                                ))
                            
                            fig_profile.update_layout(
                                xaxis_title='SDR<sub>max</sub> (%)',
                                yaxis_title='h<sub>i</sub>/h<sub>t</sub>',
                                xaxis=dict(
                                    rangemode='tozero',
                                    title_font=dict(size=18, family='Arial', color='black'),
                                    tickfont=dict(size=16, color='#000000'),
                                    tickcolor='#000000',
                                    tickwidth=2
                                ),
                                yaxis=dict(
                                    rangemode='tozero',
                                    title_font=dict(size=18, family='Arial', color='black'),
                                    tickfont=dict(size=16, color='#000000'),
                                    tickcolor='#000000',
                                    tickwidth=2
                                ),
                                hovermode='x unified',
                                template='plotly_white',
                                height=600,
                                legend=dict(font=dict(size=14))
                            )
                            
                            st.plotly_chart(fig_profile, use_container_width=True, key=f"profile_{tab_idx}")
                
            except Exception as e:
                st.error(f"Error loading or processing results: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")
st.caption("RC-WIAP Pushover Analysis Interface v2.0")
