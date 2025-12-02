import streamlit as st
import pandas as pd
import openpyxl
from openpyxl import Workbook
import os
import pickle
import glob

# Set page configuration
st.set_page_config(page_title="RC-WIAP Model Creator", layout="wide")

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state.mode = None  # 'create' or 'edit'
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'num_walls' not in st.session_state:
    st.session_state.num_walls = 0
if 'num_stories' not in st.session_state:
    st.session_state.num_stories = 0
if 'wall_names' not in st.session_state:
    st.session_state.wall_names = []
if 'info_data' not in st.session_state:
    st.session_state.info_data = {}
if 'model_name' not in st.session_state:
    st.session_state.model_name = ""
if 'walls_saved' not in st.session_state:
    st.session_state.walls_saved = set()
if 'loaded_model' not in st.session_state:
    st.session_state.loaded_model = None
if 'loaded_model_name' not in st.session_state:
    st.session_state.loaded_model_name = ""

# Title
st.title("RC-WIAP Model Creator")
st.markdown("Create new models or edit existing ones")
st.markdown("---")

# Mode Selection (only show if not in a workflow)
if st.session_state.mode is None:
    st.header("Select Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🆕 Create New Model")
        st.write("Start from scratch to create a new building model")
        if st.button("Create New Model", type="primary", use_container_width=True):
            st.session_state.mode = 'create'
            st.session_state.step = 1
            # Reset all data
            st.session_state.num_walls = 0
            st.session_state.num_stories = 0
            st.session_state.wall_names = []
            st.session_state.info_data = {}
            st.session_state.model_name = ""
            st.session_state.walls_saved = set()
            st.session_state.loaded_model = None
            st.session_state.loaded_model_name = ""
            st.rerun()
    
    with col2:
        st.subheader("✏️ Edit Existing Model")
        st.write("Load and modify an existing model from the models folder")
        
        # Get list of .pkl files in models directory
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        pkl_files = glob.glob(f"{models_dir}/*.pkl")
        pkl_filenames = [os.path.basename(f) for f in pkl_files]
        
        if len(pkl_files) == 0:
            st.warning("⚠️ No model files found in 'models/' folder")
        else:
            selected_model = st.selectbox(
                "Select model to edit:",
                options=pkl_filenames,
                help="Choose a .pkl file to load and edit"
            )
            
            if st.button("Load Model", type="primary", use_container_width=True):
                try:
                    # Load the pickle file
                    model_path = os.path.join(models_dir, selected_model)
                    with open(model_path, "rb") as f:
                        loaded_arq = pickle.load(f)
                    
                    st.session_state.loaded_model = loaded_arq
                    st.session_state.loaded_model_name = selected_model.replace('.pkl', '')
                    st.session_state.mode = 'edit'
                    st.session_state.step = 1
                    
                    # Sort walls by id_ to ensure correct order
                    sorted_muros = sorted(loaded_arq.muros, key=lambda m: m.id_)
                    
                    # Extract model information
                    st.session_state.num_walls = len(sorted_muros)
                    st.session_state.num_stories = loaded_arq.NumPisos
                    st.session_state.wall_names = [muro.nombre for muro in sorted_muros]
                    
                    # Extract Info.xlsx equivalent data
                    w_data = {'Piso': list(range(loaded_arq.NumPisos, 0, -1))}
                    ws_data = {'Piso': list(range(loaded_arq.NumPisos, 0, -1))}
                    fc_data = {'Piso': list(range(loaded_arq.NumPisos, 0, -1))}
                    
                    for muro in sorted_muros:
                        w_values = [piso.w_ for piso in muro.pisos]
                        ws_values = [piso.ws_ for piso in muro.pisos]
                        w_data[muro.nombre] = w_values
                        ws_data[muro.nombre] = ws_values
                    
                    # Extract fc from first wall (use sorted)
                    fc_values = [piso.fc for piso in sorted_muros[0].pisos]
                    fc_data['fc'] = fc_values
                    
                    # Extract heights
                    hn_data = {'Piso': list(range(loaded_arq.NumPisos, 0, -1))}
                    hn_data['Hi'] = sorted(loaded_arq.listaCoorY, reverse=True)
                    
                    st.session_state.info_data = {
                        'w': w_data,
                        'ws': ws_data,
                        'fc': fc_data,
                        'Hn': hn_data
                    }
                    
                    # Extract wall data for each wall (use sorted)
                    st.session_state.wall_files_data = {}
                    for muro in sorted_muros:
                        # Get number of macrofibras from first story (use muro, not muro_Md)
                        num_macrofibras = len(muro.pisos[0].muro.listaMacrofibras)
                        
                        # Initialize data structures
                        nummacro_data = {'Piso': list(range(loaded_arq.NumPisos, 0, -1))}
                        anchos_data = {'Piso': list(range(loaded_arq.NumPisos, 0, -1))}
                        espesor_data = {'Piso': list(range(loaded_arq.NumPisos, 0, -1))}
                        concreto_data = {'Piso': list(range(loaded_arq.NumPisos, 0, -1))}
                        acero_data = {'Piso': list(range(loaded_arq.NumPisos, 0, -1))}
                        cuantia_data = {'Piso': list(range(loaded_arq.NumPisos, 0, -1))}
                        
                        # Extract data from the original macrofibra structure (muro, not muro_Md)
                        for i in range(num_macrofibras):
                            col_name = f"F{i+1}"
                            
                            nummacro_data[col_name] = []
                            anchos_data[col_name] = []
                            espesor_data[col_name] = []
                            concreto_data[col_name] = []
                            acero_data[col_name] = []
                            cuantia_data[col_name] = []
                            
                            # For each story, get values for this macrofibra
                            for piso in muro.pisos:
                                # Get data from muro (original input with macrofibras)
                                nummacro_data[col_name].append(int(piso.muro.listaMacrofibras[i]))
                                anchos_data[col_name].append(piso.muro.listaAncho[i])
                                espesor_data[col_name].append(piso.muro.listaEspesor[i])
                                cuantia_data[col_name].append(piso.muro.listaCuantia[i])
                                concreto_data[col_name].append(piso.muro.listaTipoConcreto[i])
                                acero_data[col_name].append(piso.muro.listaTipoAcero[i])
                        
                        st.session_state.wall_files_data[muro.nombre] = {
                            'num_columns': num_macrofibras,
                            'NumMacro': nummacro_data,
                            'Anchos': anchos_data,
                            'Espesor': espesor_data,
                            'Concreto': concreto_data,
                            'Acero': acero_data,
                            'Cuantia': cuantia_data
                        }
                    
                    # Mark all walls as saved
                    st.session_state.walls_saved = set(range(st.session_state.num_walls))
                    
                    st.success(f"✅ Model '{st.session_state.loaded_model_name}' loaded successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

else:
    # Show current mode and allow going back
    mode_text = "Creating New Model" if st.session_state.mode == 'create' else f"Editing Model: {st.session_state.loaded_model_name}"
    st.sidebar.info(f"**Mode:** {mode_text}")
    
    if st.sidebar.button("← Back to Mode Selection"):
        st.session_state.mode = None
        st.session_state.step = 1
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Current Step:** {st.session_state.step} of 3")

# Step 1: Configure building information and create Info.xlsx
if st.session_state.mode is not None and st.session_state.step == 1:
    st.header("Step 1: Building Configuration")
    st.write("Enter the basic information about the building to create the Info.xlsx file.")
    
    # Input for number of walls and stories
    col1, col2 = st.columns(2)
    
    with col1:
        num_walls = st.number_input(
            "Number of walls:", 
            min_value=1, 
            max_value=50, 
            value=st.session_state.num_walls if st.session_state.num_walls > 0 else 4,
            step=1
        )
    
    with col2:
        num_stories = st.number_input(
            "Number of stories:", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.num_stories if st.session_state.num_stories > 0 else 5,
            step=1
        )
    
    st.markdown("---")
    st.subheader("Wall Names")
    st.write("Enter the name for each wall:")
    
    # Create input fields for wall names
    wall_names = []
    cols = st.columns(min(4, num_walls))
    
    for i in range(num_walls):
        col_idx = i % 4
        with cols[col_idx]:
            default_name = st.session_state.wall_names[i] if i < len(st.session_state.wall_names) else f"Wall-{i+1}"
            wall_name = st.text_input(
                f"Wall {i+1}:",
                value=default_name,
                key=f"wall_name_{i}"
            )
            wall_names.append(wall_name)
    
    st.markdown("---")
    st.subheader("Info.xlsx Data")
    st.write("Enter the data for each sheet in Info.xlsx:")
    
    # Tab for different sheets
    tab1, tab2, tab3, tab4 = st.tabs(["w (Gravity weight)", "ws (Seismic weight)", "fc (Concrete Strength)", "Hn (Height)"])
    
    with tab1:
        st.write("Enter the gravity load weight (w) for each wall at each story:")
        
        w_data = {}
        w_data['Piso'] = list(range(num_stories, 0, -1))
        
        for wall_idx, wall_name in enumerate(wall_names):
            col = st.columns(1)[0]
            st.write(f"**{wall_name}**")
            
            # Quick fill option per wall
            with st.expander(f"⚡ Quick Fill for {wall_name}"):
                quick_w_value = st.number_input("Value to apply to all stories:", value=10.0, format="%.6f", key=f"quick_w_{wall_idx}")
                if st.button("Apply to all stories", key=f"apply_w_{wall_idx}"):
                    if 'w_quick_fill' not in st.session_state:
                        st.session_state.w_quick_fill = {}
                    st.session_state.w_quick_fill[wall_name] = [quick_w_value] * num_stories
                    st.success(f"Applied {quick_w_value} to all stories of {wall_name}!")
                    st.rerun()
            wall_values = []
            for story in range(num_stories, 0, -1):
                key = f"w_w{wall_idx}_{wall_name}_{story}"
                # Check quick fill first, then existing data
                if 'w_quick_fill' in st.session_state and wall_name in st.session_state.w_quick_fill:
                    default_val = st.session_state.w_quick_fill[wall_name][num_stories - story]
                else:
                    default_val = st.session_state.info_data.get('w', {}).get(wall_name, [10.0]*num_stories)[num_stories - story] if num_stories - story < len(st.session_state.info_data.get('w', {}).get(wall_name, [])) else 10.0
                val = st.number_input(
                    f"Story {story}:",
                    value=float(default_val),
                    format="%.6f",
                    key=key
                )
                wall_values.append(val)
            w_data[wall_name] = wall_values
    
    with tab2:
        st.write("Enter the seismic weight (ws) for each wall at each story:")
        
        ws_data = {}
        ws_data['Piso'] = list(range(num_stories, 0, -1))
        
        for wall_idx, wall_name in enumerate(wall_names):
            col = st.columns(1)[0]
            st.write(f"**{wall_name}**")
            
            # Quick fill option per wall
            with st.expander(f"⚡ Quick Fill for {wall_name}"):
                quick_ws_value = st.number_input("Value to apply to all stories:", value=35.0, format="%.6f", key=f"quick_ws_{wall_idx}")
                if st.button("Apply to all stories", key=f"apply_ws_{wall_idx}"):
                    if 'ws_quick_fill' not in st.session_state:
                        st.session_state.ws_quick_fill = {}
                    st.session_state.ws_quick_fill[wall_name] = [quick_ws_value] * num_stories
                    st.success(f"Applied {quick_ws_value} to all stories of {wall_name}!")
                    st.rerun()
            wall_values = []
            for story in range(num_stories, 0, -1):
                key = f"ws_w{wall_idx}_{wall_name}_{story}"
                # Check quick fill first, then existing data
                if 'ws_quick_fill' in st.session_state and wall_name in st.session_state.ws_quick_fill:
                    default_val = st.session_state.ws_quick_fill[wall_name][num_stories - story]
                else:
                    default_val = st.session_state.info_data.get('ws', {}).get(wall_name, [35.0]*num_stories)[num_stories - story] if num_stories - story < len(st.session_state.info_data.get('ws', {}).get(wall_name, [])) else 35.0
                val = st.number_input(
                    f"Story {story}:",
                    value=float(default_val),
                    format="%.6f",
                    key=key
                )
                wall_values.append(val)
            ws_data[wall_name] = wall_values
    
    with tab3:
        st.write("Enter the concrete strength (fc) for each story:")
        
        # Quick fill option
        with st.expander("⚡ Quick Fill - Apply same value to all stories (applies to entire building)"):
            quick_fc_value = st.number_input("Value to apply:", value=21.0, format="%.2f", key="quick_fc")
            if st.button("Apply to all stories", key="apply_fc"):
                st.session_state.fc_quick_fill = [quick_fc_value] * num_stories
                st.success(f"Applied {quick_fc_value} MPa to all stories!")
                st.rerun()
        
        fc_data = {}
        fc_data['Piso'] = list(range(num_stories, 0, -1))
        fc_values = []
        
        for story in range(num_stories, 0, -1):
            key = f"fc_{story}"
            # Check quick fill first, then existing data
            if 'fc_quick_fill' in st.session_state:
                default_val = st.session_state.fc_quick_fill[num_stories - story]
            else:
                default_val = st.session_state.info_data.get('fc', {}).get('fc', [21.0]*num_stories)[num_stories - story] if num_stories - story < len(st.session_state.info_data.get('fc', {}).get('fc', [])) else 21.0
            val = st.number_input(
                f"Story {story}:",
                value=float(default_val),
                format="%.2f",
                key=key
            )
            fc_values.append(val)
        fc_data['fc'] = fc_values
    
    with tab4:
        st.write("Enter the height (Hi) for each story:")
        hn_data = {}
        hn_data['Piso'] = list(range(num_stories, 0, -1))
        hi_values = []
        
        # Default heights: [15, 12, 9, 6, 3] for 5 stories (from top to bottom)
        default_heights = [15.0, 12.0, 9.0, 6.0, 3.0]
        
        for story in range(num_stories, 0, -1):
            key = f"hi_{story}"
            # Use default heights if available, otherwise 0.0
            story_index = num_stories - story
            if story_index < len(default_heights):
                default_height = default_heights[story_index]
            else:
                default_height = 0.0
            default_val = st.session_state.info_data.get('Hn', {}).get('Hi', [default_height]*num_stories)[num_stories - story] if num_stories - story < len(st.session_state.info_data.get('Hn', {}).get('Hi', [])) else default_height
            val = st.number_input(
                f"Story {story}:",
                value=float(default_val),
                format="%.2f",
                key=key
            )
            hi_values.append(val)
        hn_data['Hi'] = hi_values
    
    st.markdown("---")
    
    # Button to generate/update Info.xlsx
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        button_text = "Update Info.xlsx & Continue →" if st.session_state.mode == 'edit' else "Generate Info.xlsx & Continue →"
        if st.button(button_text, type="primary", use_container_width=True):
            # Save to session state
            st.session_state.num_walls = num_walls
            st.session_state.num_stories = num_stories
            st.session_state.wall_names = wall_names
            st.session_state.info_data = {
                'w': w_data,
                'ws': ws_data,
                'fc': fc_data,
                'Hn': hn_data
            }
            
            # Reset wall tracking when regenerating Info.xlsx (only in create mode)
            if st.session_state.mode == 'create':
                st.session_state.current_wall_index = 0
                st.session_state.walls_saved = set()
                st.session_state.wall_files_data = {}
            else:
                # In edit mode, keep existing wall data but adjust if needed
                if 'current_wall_index' not in st.session_state:
                    st.session_state.current_wall_index = 0
            
            # Create Data folder if it doesn't exist
            if not os.path.exists('Data'):
                os.makedirs('Data')
            
            # In create mode, delete old wall Excel files
            if st.session_state.mode == 'create':
                old_wall_files = glob.glob('Data/[0-9][0-9]_*.xlsx')
                for old_file in old_wall_files:
                    try:
                        os.remove(old_file)
                    except:
                        pass
            
            # Create Info.xlsx
            try:
                with pd.ExcelWriter('Data/Info.xlsx', engine='openpyxl') as writer:
                    # Write w sheet
                    df_w = pd.DataFrame(w_data)
                    df_w.to_excel(writer, sheet_name='w', index=False)
                    
                    # Write ws sheet
                    df_ws = pd.DataFrame(ws_data)
                    df_ws.to_excel(writer, sheet_name='ws', index=False)
                    
                    # Write fc sheet
                    df_fc = pd.DataFrame(fc_data)
                    df_fc.to_excel(writer, sheet_name='fc', index=False)
                    
                    # Write Hn sheet
                    df_hn = pd.DataFrame(hn_data)
                    df_hn.to_excel(writer, sheet_name='Hn', index=False)
                
                success_msg = "✅ Info.xlsx has been successfully updated!" if st.session_state.mode == 'edit' else "✅ Info.xlsx has been successfully created in the Data folder!"
                st.success(success_msg)
                st.balloons()
                
                # Move to next step
                st.session_state.step = 2
                st.rerun()
                
            except Exception as e:
                st.error(f"Error creating Info.xlsx: {str(e)}")

# Step 2: Create/Edit wall files (00_, 01_, etc.)
elif st.session_state.mode is not None and st.session_state.step == 2:
    st.header("Step 2: Wall Information Files")
    st.write(f"Create/edit individual wall files for {st.session_state.num_walls} walls")
    
    # Initialize wall data in session state
    if 'current_wall_index' not in st.session_state:
        st.session_state.current_wall_index = 0
    if 'wall_files_data' not in st.session_state:
        st.session_state.wall_files_data = {}
    
    current_idx = st.session_state.current_wall_index
    wall_name = st.session_state.wall_names[current_idx]
    
    st.subheader(f"Wall {current_idx + 1} of {st.session_state.num_walls}: {wall_name}")
    st.write(f"File: {current_idx:02d}_{wall_name}.xlsx")
    
    st.markdown("---")
    
    # First, ask for number of columns (F1, F2, F3, etc.)
    st.write("**Configuration:**")
    num_columns = st.number_input(
        "Number of sections/columns (F1, F2, F3, etc.):",
        min_value=1,
        max_value=20,
        value=st.session_state.wall_files_data.get(wall_name, {}).get('num_columns', 1),
        step=1,
        key=f"num_cols_{current_idx}"
    )
    
    column_names = [f"F{i+1}" for i in range(num_columns)]
    
    st.markdown("---")
    
    # Create tabs for each sheet
    tabs = st.tabs(["NumMacro", "Anchos (Width)", "Espesor (Thickness)", "Concreto", "Acero", "Cuantia"])
    
    # NumMacro sheet
    with tabs[0]:
        st.write("Enter the number of macro elements for each section at each story:")
        st.info("NumMacro: Number of macro elements for each fiber section")
        
        nummacro_data = {}
        nummacro_data['Piso'] = list(range(st.session_state.num_stories, 0, -1))
        
        cols = st.columns(min(4, num_columns))
        for idx, col_name in enumerate(column_names):
            col_idx = idx % 4
            with cols[col_idx]:
                st.write(f"**{col_name}**")
                
                # Quick fill option per section
                with st.expander(f"⚡ Quick Fill for {col_name}"):
                    quick_nummacro_value = st.number_input("Value to apply to all stories:", value=10, min_value=1, step=1, key=f"quick_nummacro_{wall_name}_{col_name}")
                    if st.button("Apply to all stories", key=f"apply_nummacro_{wall_name}_{col_name}"):
                        if 'nummacro_quick_fill' not in st.session_state:
                            st.session_state.nummacro_quick_fill = {}
                        if wall_name not in st.session_state.nummacro_quick_fill:
                            st.session_state.nummacro_quick_fill[wall_name] = {}
                        st.session_state.nummacro_quick_fill[wall_name][col_name] = [quick_nummacro_value] * st.session_state.num_stories
                        st.success(f"Applied {quick_nummacro_value} to all stories of {col_name}!")
                        st.rerun()
                
                col_values = []
                for story in range(st.session_state.num_stories, 0, -1):
                    key = f"nummacro_{wall_name}_{col_name}_{story}"
                    story_index = st.session_state.num_stories - story
                    # Check quick fill first, then existing data
                    if 'nummacro_quick_fill' in st.session_state and wall_name in st.session_state.nummacro_quick_fill and col_name in st.session_state.nummacro_quick_fill[wall_name]:
                        default_val = st.session_state.nummacro_quick_fill[wall_name][col_name][story_index]
                    else:
                        default_list = st.session_state.wall_files_data.get(wall_name, {}).get('NumMacro', {}).get(col_name, [10]*st.session_state.num_stories)
                        default_val = default_list[story_index] if story_index < len(default_list) else 10
                    val = st.number_input(
                        f"Story {story}:",
                        value=int(default_val),
                        min_value=1,
                        step=1,
                        key=key
                    )
                    col_values.append(val)
                nummacro_data[col_name] = col_values
    
    # Anchos sheet
    with tabs[1]:
        st.write("Enter the width (Anchos) for each section at each story:")
        st.info("Anchos: Width of each fiber section")
        
        anchos_data = {}
        anchos_data['Piso'] = list(range(st.session_state.num_stories, 0, -1))
        
        cols = st.columns(min(4, num_columns))
        for idx, col_name in enumerate(column_names):
            col_idx = idx % 4
            with cols[col_idx]:
                st.write(f"**{col_name}**")
                
                # Quick fill option per section
                with st.expander(f"⚡ Quick Fill for {col_name}"):
                    quick_anchos_value = st.number_input("Value to apply to all stories:", value=9.0, format="%.2f", key=f"quick_anchos_{wall_name}_{col_name}")
                    if st.button("Apply to all stories", key=f"apply_anchos_{wall_name}_{col_name}"):
                        if 'anchos_quick_fill' not in st.session_state:
                            st.session_state.anchos_quick_fill = {}
                        if wall_name not in st.session_state.anchos_quick_fill:
                            st.session_state.anchos_quick_fill[wall_name] = {}
                        st.session_state.anchos_quick_fill[wall_name][col_name] = [quick_anchos_value] * st.session_state.num_stories
                        st.success(f"Applied {quick_anchos_value} to all stories of {col_name}!")
                        st.rerun()
                
                col_values = []
                for story in range(st.session_state.num_stories, 0, -1):
                    key = f"anchos_{wall_name}_{col_name}_{story}"
                    story_index = st.session_state.num_stories - story
                    # Check quick fill first, then existing data
                    if 'anchos_quick_fill' in st.session_state and wall_name in st.session_state.anchos_quick_fill and col_name in st.session_state.anchos_quick_fill[wall_name]:
                        default_val = st.session_state.anchos_quick_fill[wall_name][col_name][story_index]
                    else:
                        default_list = st.session_state.wall_files_data.get(wall_name, {}).get('Anchos', {}).get(col_name, [9.0]*st.session_state.num_stories)
                        default_val = default_list[story_index] if story_index < len(default_list) else 9.0
                    val = st.number_input(
                        f"Story {story}:",
                        value=float(default_val),
                        format="%.2f",
                        key=key
                    )
                    col_values.append(val)
                anchos_data[col_name] = col_values
    
    # Espesor sheet
    with tabs[2]:
        st.write("Enter the thickness (Espesor) for each section at each story:")
        st.info("Espesor: Thickness of each fiber section")
        
        espesor_data = {}
        espesor_data['Piso'] = list(range(st.session_state.num_stories, 0, -1))
        
        cols = st.columns(min(4, num_columns))
        for idx, col_name in enumerate(column_names):
            col_idx = idx % 4
            with cols[col_idx]:
                st.write(f"**{col_name}**")
                
                # Quick fill option per section
                with st.expander(f"⚡ Quick Fill for {col_name}"):
                    quick_espesor_value = st.number_input("Value to apply to all stories:", value=0.1, format="%.2f", key=f"quick_espesor_{wall_name}_{col_name}")
                    if st.button("Apply to all stories", key=f"apply_espesor_{wall_name}_{col_name}"):
                        if 'espesor_quick_fill' not in st.session_state:
                            st.session_state.espesor_quick_fill = {}
                        if wall_name not in st.session_state.espesor_quick_fill:
                            st.session_state.espesor_quick_fill[wall_name] = {}
                        st.session_state.espesor_quick_fill[wall_name][col_name] = [quick_espesor_value] * st.session_state.num_stories
                        st.success(f"Applied {quick_espesor_value} to all stories of {col_name}!")
                        st.rerun()
                
                col_values = []
                for story in range(st.session_state.num_stories, 0, -1):
                    key = f"espesor_{wall_name}_{col_name}_{story}"
                    story_index = st.session_state.num_stories - story
                    # Check quick fill first, then existing data
                    if 'espesor_quick_fill' in st.session_state and wall_name in st.session_state.espesor_quick_fill and col_name in st.session_state.espesor_quick_fill[wall_name]:
                        default_val = st.session_state.espesor_quick_fill[wall_name][col_name][story_index]
                    else:
                        default_list = st.session_state.wall_files_data.get(wall_name, {}).get('Espesor', {}).get(col_name, [0.1]*st.session_state.num_stories)
                        default_val = default_list[story_index] if story_index < len(default_list) else 0.1
                    val = st.number_input(
                        f"Story {story}:",
                        value=float(default_val),
                        format="%.2f",
                        key=key
                    )
                    col_values.append(val)
                espesor_data[col_name] = col_values
    
    # Concreto sheet
    with tabs[3]:
        st.write("Enter the concrete type for each section at each story:")
        st.info("Concreto: Concrete type (e.g., Unconf for unconfined, Conf for confined)")
        
        concreto_data = {}
        concreto_data['Piso'] = list(range(st.session_state.num_stories, 0, -1))
        
        cols = st.columns(min(4, num_columns))
        for idx, col_name in enumerate(column_names):
            col_idx = idx % 4
            with cols[col_idx]:
                st.write(f"**{col_name}**")
                
                # Quick fill option per section
                with st.expander(f"⚡ Quick Fill for {col_name}"):
                    quick_concreto_value = st.text_input("Value to apply to all stories:", value="Unconf", key=f"quick_concreto_{wall_name}_{col_name}")
                    if st.button("Apply to all stories", key=f"apply_concreto_{wall_name}_{col_name}"):
                        if 'concreto_quick_fill' not in st.session_state:
                            st.session_state.concreto_quick_fill = {}
                        if wall_name not in st.session_state.concreto_quick_fill:
                            st.session_state.concreto_quick_fill[wall_name] = {}
                        st.session_state.concreto_quick_fill[wall_name][col_name] = [quick_concreto_value] * st.session_state.num_stories
                        st.success(f"Applied '{quick_concreto_value}' to all stories of {col_name}!")
                        st.rerun()
                
                col_values = []
                for story in range(st.session_state.num_stories, 0, -1):
                    key = f"concreto_{wall_name}_{col_name}_{story}"
                    story_index = st.session_state.num_stories - story
                    # Check quick fill first, then existing data
                    if 'concreto_quick_fill' in st.session_state and wall_name in st.session_state.concreto_quick_fill and col_name in st.session_state.concreto_quick_fill[wall_name]:
                        default_val = st.session_state.concreto_quick_fill[wall_name][col_name][story_index]
                    else:
                        default_list = st.session_state.wall_files_data.get(wall_name, {}).get('Concreto', {}).get(col_name, ['Unconf']*st.session_state.num_stories)
                        default_val = default_list[story_index] if story_index < len(default_list) else 'Unconf'
                    val = st.text_input(
                        f"Story {story}:",
                        value=str(default_val),
                        key=key
                    )
                    col_values.append(val)
                concreto_data[col_name] = col_values
    
    # Acero sheet
    with tabs[4]:
        st.write("Enter the steel type for each section at each story:")
        st.info("Acero: Steel type (e.g., WWM for welded wire mesh)")
        
        acero_data = {}
        acero_data['Piso'] = list(range(st.session_state.num_stories, 0, -1))
        
        cols = st.columns(min(4, num_columns))
        for idx, col_name in enumerate(column_names):
            col_idx = idx % 4
            with cols[col_idx]:
                st.write(f"**{col_name}**")
                
                # Quick fill option per section
                with st.expander(f"⚡ Quick Fill for {col_name}"):
                    quick_acero_value = st.text_input("Value to apply to all stories:", value="WWM", key=f"quick_acero_{wall_name}_{col_name}")
                    if st.button("Apply to all stories", key=f"apply_acero_{wall_name}_{col_name}"):
                        if 'acero_quick_fill' not in st.session_state:
                            st.session_state.acero_quick_fill = {}
                        if wall_name not in st.session_state.acero_quick_fill:
                            st.session_state.acero_quick_fill[wall_name] = {}
                        st.session_state.acero_quick_fill[wall_name][col_name] = [quick_acero_value] * st.session_state.num_stories
                        st.success(f"Applied '{quick_acero_value}' to all stories of {col_name}!")
                        st.rerun()
                
                col_values = []
                for story in range(st.session_state.num_stories, 0, -1):
                    key = f"acero_{wall_name}_{col_name}_{story}"
                    story_index = st.session_state.num_stories - story
                    # Check quick fill first, then existing data
                    if 'acero_quick_fill' in st.session_state and wall_name in st.session_state.acero_quick_fill and col_name in st.session_state.acero_quick_fill[wall_name]:
                        default_val = st.session_state.acero_quick_fill[wall_name][col_name][story_index]
                    else:
                        default_list = st.session_state.wall_files_data.get(wall_name, {}).get('Acero', {}).get(col_name, ['WWM']*st.session_state.num_stories)
                        default_val = default_list[story_index] if story_index < len(default_list) else 'WWM'
                    val = st.text_input(
                        f"Story {story}:",
                        value=str(default_val),
                        key=key
                    )
                    col_values.append(val)
                acero_data[col_name] = col_values
    
    # Cuantia sheet
    with tabs[5]:
        st.write("Enter the reinforcement ratio (Cuantia) for each section at each story:")
        st.info("Cuantia: Reinforcement ratio")
        
        cuantia_data = {}
        cuantia_data['Piso'] = list(range(st.session_state.num_stories, 0, -1))
        
        cols = st.columns(min(4, num_columns))
        for idx, col_name in enumerate(column_names):
            col_idx = idx % 4
            with cols[col_idx]:
                st.write(f"**{col_name}**")
                
                # Quick fill option per section
                with st.expander(f"⚡ Quick Fill for {col_name}"):
                    quick_cuantia_value = st.number_input("Value to apply to all stories:", value=0.0012, format="%.6f", key=f"quick_cuantia_{wall_name}_{col_name}")
                    if st.button("Apply to all stories", key=f"apply_cuantia_{wall_name}_{col_name}"):
                        if 'cuantia_quick_fill' not in st.session_state:
                            st.session_state.cuantia_quick_fill = {}
                        if wall_name not in st.session_state.cuantia_quick_fill:
                            st.session_state.cuantia_quick_fill[wall_name] = {}
                        st.session_state.cuantia_quick_fill[wall_name][col_name] = [quick_cuantia_value] * st.session_state.num_stories
                        st.success(f"Applied {quick_cuantia_value} to all stories of {col_name}!")
                        st.rerun()
                
                col_values = []
                for story in range(st.session_state.num_stories, 0, -1):
                    key = f"cuantia_{wall_name}_{col_name}_{story}"
                    story_index = st.session_state.num_stories - story
                    # Check quick fill first, then existing data
                    if 'cuantia_quick_fill' in st.session_state and wall_name in st.session_state.cuantia_quick_fill and col_name in st.session_state.cuantia_quick_fill[wall_name]:
                        default_val = st.session_state.cuantia_quick_fill[wall_name][col_name][story_index]
                    else:
                        default_list = st.session_state.wall_files_data.get(wall_name, {}).get('Cuantia', {}).get(col_name, [0.0012]*st.session_state.num_stories)
                        default_val = default_list[story_index] if story_index < len(default_list) else 0.0012
                    val = st.number_input(
                        f"Story {story}:",
                        value=float(default_val),
                        format="%.6f",
                        key=key
                    )
                    col_values.append(val)
                cuantia_data[col_name] = col_values
    
    st.markdown("---")
    
    # Navigation buttons
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    
    with col1:
        if st.button("← Back to Step 1"):
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if current_idx > 0:
            if st.button("← Previous Wall", use_container_width=True):
                st.session_state.current_wall_index -= 1
                st.rerun()
    
    with col3:
        # Save current wall button
        button_text = "Update & Continue →" if st.session_state.mode == 'edit' else "Save & Continue →"
        if st.button(button_text, type="primary", use_container_width=True):
            # Save data to session state
            st.session_state.wall_files_data[wall_name] = {
                'num_columns': num_columns,
                'NumMacro': nummacro_data,
                'Anchos': anchos_data,
                'Espesor': espesor_data,
                'Concreto': concreto_data,
                'Acero': acero_data,
                'Cuantia': cuantia_data
            }
            
            # Create the Excel file for this wall
            try:
                # Create Data folder if it doesn't exist
                if not os.path.exists('Data'):
                    os.makedirs('Data')
                
                filename = f"{current_idx:02d}_{wall_name}.xlsx"
                filepath = f"Data/{filename}"
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    # Write each sheet
                    df_nummacro = pd.DataFrame(nummacro_data)
                    df_nummacro.to_excel(writer, sheet_name='NumMacro', index=False)
                    
                    df_anchos = pd.DataFrame(anchos_data)
                    df_anchos.to_excel(writer, sheet_name='Anchos', index=False)
                    
                    df_espesor = pd.DataFrame(espesor_data)
                    df_espesor.to_excel(writer, sheet_name='Espesor', index=False)
                    
                    df_concreto = pd.DataFrame(concreto_data)
                    df_concreto.to_excel(writer, sheet_name='Concreto', index=False)
                    
                    df_acero = pd.DataFrame(acero_data)
                    df_acero.to_excel(writer, sheet_name='Acero', index=False)
                    
                    df_cuantia = pd.DataFrame(cuantia_data)
                    df_cuantia.to_excel(writer, sheet_name='Cuantia', index=False)
                
                success_msg = f"✅ {filename} has been successfully updated!" if st.session_state.mode == 'edit' else f"✅ {filename} has been successfully created in the Data folder!"
                st.success(success_msg)
                
                # Mark this wall as saved
                st.session_state.walls_saved.add(current_idx)
                
                # Move to next wall or finish
                if current_idx < st.session_state.num_walls - 1:
                    st.session_state.current_wall_index += 1
                    st.rerun()
                else:
                    st.balloons()
                    st.success("🎉 All wall files have been created/updated successfully!")
                
            except Exception as e:
                st.error(f"Error creating {filename}: {str(e)}")
    
    with col4:
        pass  # Empty column for spacing
    
    # Show Continue button ONLY if all walls have been saved
    all_walls_saved = len(st.session_state.walls_saved) == st.session_state.num_walls
    
    if all_walls_saved:
        st.markdown("---")
        st.success("✅ All wall files have been created/updated successfully!")
        st.info("You can now continue to Model Creation/Update.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            continue_text = "Continue to Model Update →" if st.session_state.mode == 'edit' else "Continue to Model Creation →"
            if st.button(continue_text, type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()

# Step 3: Model Creation/Update
elif st.session_state.mode is not None and st.session_state.step == 3:
    # Verify all walls have been saved before allowing model creation
    all_walls_saved = len(st.session_state.walls_saved) == st.session_state.num_walls
    
    if not all_walls_saved:
        st.error("❌ Cannot create/update model: Not all wall files have been created!")
        st.warning(f"Saved walls: {len(st.session_state.walls_saved)} / {st.session_state.num_walls}")
        st.info("Please go back to Step 2 and complete all wall definitions.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("← Back to Step 2", type="primary", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        st.stop()
    
    header_text = "Step 3: Model Update" if st.session_state.mode == 'edit' else "Step 3: Model Creation"
    st.header(header_text)
    
    if st.session_state.mode == 'edit':
        st.write("Update the existing OpenSees model with the new parameters.")
    else:
        st.write("Create the OpenSees model and generate the Python script.")
    
    st.markdown("---")
    
    # Model name input
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_model_name = st.session_state.loaded_model_name if st.session_state.mode == 'edit' else (st.session_state.model_name if st.session_state.model_name else "0041-MCR-BGT-05P")
        model_name = st.text_input(
            "Model Name:",
            value=default_model_name,
            help="Enter a name for your model (e.g., 0041-MCR-BGT-05P)"
        )
    
    with col2:
        # Extract direction from loaded model name if in edit mode
        default_direction = 0
        if st.session_state.mode == 'edit' and st.session_state.loaded_model_name:
            if st.session_state.loaded_model_name.endswith('-T'):
                default_direction = 1
        
        direction = st.selectbox(
            "Analysis Direction:",
            options=["Longitudinal (L)", "Transversal (T)"],
            index=default_direction
        )
        direction_code = "L" if "Longitudinal" in direction else "T"
    
    st.markdown("---")
    
    # Display model information summary
    st.subheader("Model Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Walls", st.session_state.num_walls)
    with col2:
        st.metric("Number of Stories", st.session_state.num_stories)
    with col3:
        st.metric("Model File", f"{model_name}-{direction_code}.pkl")
    
    st.markdown("---")
    
    # Navigation and action buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Step 2"):
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        button_text = "Update Model" if st.session_state.mode == 'edit' else "Generate Model"
        if st.button(button_text, type="primary", use_container_width=True):
            # Save model name to session state
            st.session_state.model_name = model_name
            
            # Create the full model name with direction
            full_model_name = f"{model_name}-{direction_code}"
            
            # Check if master script exists
            master_script = "00_MODEL-CREATOR.py"
            
            if not os.path.exists(master_script):
                st.error(f"Master script '{master_script}' not found!")
            else:
                try:
                    # Create models folder if it doesn't exist
                    if not os.path.exists('models'):
                        os.makedirs('models')
                    
                    # Run the master script with model name as argument
                    action_text = "Updating" if st.session_state.mode == 'edit' else "Generating"
                    st.info(f"{action_text} the model...")
                    
                    import subprocess
                    import sys
                    
                    result = subprocess.run(
                        [sys.executable, master_script, full_model_name],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        # Move generated files to models folder
                        import shutil
                        pkl_file = f"{full_model_name}.pkl"
                        plot_file = f"{full_model_name}_model.png"
                        
                        if os.path.exists(pkl_file):
                            shutil.move(pkl_file, f"models/{pkl_file}")
                        if os.path.exists(plot_file):
                            shutil.move(plot_file, f"models/{plot_file}")
                        
                        success_text = "✅ Model updated successfully!" if st.session_state.mode == 'edit' else "✅ Model generated successfully!"
                        st.success(success_text)
                        st.balloons()
                        
                        # Show output in an expander
                        with st.expander("View Model Generation Output"):
                            st.code(result.stdout)
                        
                        # Show generated files
                        if os.path.exists(f"models/{pkl_file}"):
                            st.info(f"📦 Model object saved to: models/{pkl_file}")
                        
                        # Display the model plot
                        if os.path.exists(f"models/{plot_file}"):
                            st.markdown("---")
                            st.subheader("Model Visualization")
                            st.image(f"models/{plot_file}", caption=f"3D Model: {full_model_name}", use_container_width=True)
                        
                    else:
                        st.error("❌ Error generating/updating model!")
                        st.error("Error output:")
                        st.code(result.stderr)
                        if result.stdout:
                            st.code(result.stdout)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col3:
        pass  # Empty column for spacing

st.markdown("---")
st.caption("RC-WIAP Model Creator v2.0")
