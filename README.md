# RC-WIAP: Reinforced Concrete Wall Inelastic Analysis Program

## Overview

RC-WIAP is a comprehensive structural analysis platform for reinforced concrete buildings with shear walls. The program provides an integrated workflow from model generation through nonlinear static and dynamic analyses to fragility function calculation, all through intuitive Streamlit web interfaces.

## Features

- **Model Generation** (`app_modelcreator.py`): Create parametric building models with multiple stories and shear walls
- **Pushover Analysis** (`app_analysis.py`): Perform nonlinear static pushover analysis using OpenSeesPy
- **IDA & Dynamic Analysis** (`app_dynamic.py`): Incremental Dynamic Analysis with ground motion scaling
- **Fragility Functions** (`app_dynamic.py`): Calculate and visualize seismic fragility curves
- **Interactive Visualization**: Plotly-based charts with zoom, pan, and export capabilities
- **Material Models**: Support for both reinforcing bars (RB) and welded wire mesh (WWM)
- **Multiple EDPs**: Analyze using Story Drift Ratio (SDR), First Story Drift (SDR1), or Roof Drift Ratio (RDR)
- **Parallel Processing**: Multi-core support for IDA analyses

## System Requirements

### Software Dependencies

- Python 3.11.5 or higher (Anaconda recommended)
- OpenSeesPy
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- SciPy
- openpyxl
- joblib
- pickle

### Installation

1. Install Anaconda Python 3.11.5
2. Install required packages:
```bash
pip install openseespy streamlit pandas numpy matplotlib plotly scipy openpyxl joblib opseestools
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

## Project Structure

```
RC-WIAP/
├── app_modelcreator.py         # Model generation interface
├── app_analysis.py             # Pushover analysis and visualization
├── app_dynamic.py              # IDA analysis and fragility functions
├── 00_MODEL-CREATOR.py         # Master model template
├── 01_PUSHOVER.py              # Pushover analysis script
├── 02_IDA.py                   # IDA analysis script
├── 02_CNL.py                   # Nonlinear cyclic analysis
├── 04_POST-PUSHOVER.py         # Post-processing script
├── 05_POST_CNL_FULL.py         # CNL post-processing
├── Lib_ClaseArquetipos.py      # Archetype class definitions
├── Lib_materiales.py           # Material models library
├── Lib_analisis.py             # Analysis functions library
├── Lib_GM.py                   # Ground motion processing
├── Lib_frag.py                 # Fragility function library
├── requirements.txt            # Python dependencies
├── Data/                       # Input data folder
│   ├── Info.xlsx              # Building configuration
│   └── 00_Wall-*.xlsx         # Wall property files
├── models/                     # Generated models
│   ├── *.pkl                  # Serialized model objects
│   └── *_model.png            # Model visualizations
├── pushover/                   # Pushover results
│   ├── *-RPO.pkl              # Results objects
│   └── *_pushover.xlsx        # Results data
├── IDAs/                       # IDA results
│   ├── *_IDA.pkl              # IDA results objects
│   └── *_IDA.xlsx             # IDA results data
├── GMs/                        # Ground motion records
│   ├── GM*.txt                # Time history files
│   ├── Sa_FEMA.pkl            # Spectral accelerations
│   └── T.pkl                  # Periods
└── README.md                   # This file
```

## Workflow

### 1. Model Generation (`app_modelcreator.py`)

The model generation interface allows you to create building models through three steps:

#### Step 1: Building Configuration

Define the basic building parameters:

- **Number of Stories**: Total number of building levels (1-20)
- **Story Heights (Hi)**: Heights for each story in meters, comma-separated
  - Example: `15, 12, 9, 6, 3` for a 5-story building
- **Building Weight (w)**: Seismic weight per floor in kN (typical: 10-50)
- **Seismic Weight (ws)**: Mass weight per floor in kN (typical: 10-100)
- **Concrete Strength (fc)**: Concrete compressive strength in MPa (typical: 21-35)

**Output**: `Data/Info.xlsx` containing building configuration

#### Step 2: Wall Properties

Define properties for each shear wall:

- **Wall Name**: Unique identifier (e.g., M01, M02)
- **Wall Thickness**: In meters (typical: 0.10-0.30)
- **Wall Length**: In meters (typical: 1.0-6.0)
- **X-Coordinate**: Position along X-axis in meters
- **Reinforcement Type**: RB (Reinforcing Bars) or WWM (Welded Wire Mesh)
- **Number of Fibers**: Discretization (typical: 10-30)
- **Boundary Elements**: Yes/No for confined concrete regions

**Important**: Save each wall before proceeding to the next. The interface tracks saved walls and prevents moving to Step 3 until all walls are saved.

**Output**: `Data/00_Wall-*.xlsx` for each wall

#### Step 3: Model Creation

Generate the structural model:

- **Model Name**: Custom name for your model (e.g., "Building-5S-L")
- **Direction**: Longitudinal (L) or Transversal (T)

**Process**:
1. Click "Create Model" to generate the OpenSeesPy model
2. The system runs `00_MODEL-CREATOR.py` with your parameters
3. Creates `.pkl` file containing the serialized Arquetipo object
4. Generates visualization plot showing the model geometry

**Output**: 
- `<model_name>-L.pkl`: Serialized model object
- `<model_name>-L_model.png`: Model visualization

### 2. Pushover Analysis (`app_analysis.py` - Analysis Page)

Perform nonlinear static pushover analysis on generated models.

#### Step 1: Model Selection

- Select from available `.pkl` files in the workspace
- View model information (file size, visualization)

#### Step 2: Analysis Parameters

**Pushover Settings**:
- **Drift Limit (pushlimit)**: Maximum drift ratio for analysis (0.001-0.100)
  - Default: 0.035 (3.5%)
  - Recommended: 0.025-0.050 depending on structure
- **Load Distribution Pattern (pushtype)**:
  - `1`: Triangular (increases linearly with height)
  - `2`: Uniform (constant at all levels)
  - `3`: Modal (proportional to mode shape) - **Recommended**
- **Mode for Load Pattern (modepush)**: Mode number (usually 1 for fundamental mode)

**Control Settings**:
- **Control Wall Index (wallctrl)**: Wall used for displacement control (0-based index)
- **Load Increment (Dincr)**: Step size for load increments (0.0001-0.01)
  - Default: 0.001
  - Smaller values = more accurate but slower
- **Analysis Direction**: Longitudinal or Transversal

#### Step 3: Run Analysis

Click "Run Pushover Analysis" to execute:

1. Command-line parameters are passed to `01_PUSHOVER.py`
2. Model is reconstructed from `.pkl` file using `CrearModelo()`
3. Gravity analysis is performed
4. Pushover analysis is executed with specified parameters
5. Results are saved to Excel and `.pkl` files

**Outputs**:
- `<model_name>-RPO.pkl`: Results object with pushover data
- `<folder>/<model_name>_pushover.xlsx`: Excel file with three sheets:
  - **SDR1**: First story drift vs base shear
  - **SDR**: Maximum story drift vs base shear
  - **RDR**: Roof drift vs base shear

**Analysis Time**: Typically 0.003-0.010 minutes depending on model size and drift limit

### 3. Results Visualization (`app_analysis.py` - Visualization Page)

Interactive visualization of pushover analysis results using Plotly.

#### Loading Results

- Select from available `*-RPO.pkl` files
- Click "Load and Visualize Results"

#### Control Points Analysis

The system automatically identifies key performance points:

1. **1st Cracking (Agrietamiento)**: Concrete strain reaches ~0.00015
2. **1st Yielding (Primera Fluencia)**: Steel reaches yield strain
   - RB: 0.0021 (420 MPa / 200 GPa)
   - WWM: 0.00245 (490 MPa / 200 GPa)
3. **Maximum Capacity (Capacidad Máxima)**: Peak base shear
4. **1st Rupture (Primera Rotura)**: Steel reaches ultimate strain
   - RB: 0.05
   - WWM: 0.015
5. **80% Capacity**: Base shear drops to 80% of maximum
6. **50% Rupture**: Mid-fiber reaches ultimate strain

**Control Points Table**: Displays SDR1, RDR, SDR, Vb (base shear), and V/W for each point

#### Visualization Tabs

Three main tabs provide different Engineering Demand Parameter (EDP) views:

##### Tab 1: RDR (Roof Drift Ratio) - **Default**

- X-axis: Roof drift ratio as percentage of total height
- Useful for overall building deformation assessment

##### Tab 2: SDR (Story Drift Ratio)

- X-axis: Maximum story drift ratio among all stories
- Critical for seismic design compliance
- Most codes limit SDR to 0.7-2.5%

##### Tab 3: SDR1 (First Story Drift Ratio)

- X-axis: Drift ratio of first story only
- Important for soft-story assessment

#### Sub-Tabs (Available in each EDP tab)

**Pushover Curve with Control Points**:
- Interactive Plotly chart
- Black solid line: Capacity curve (V/W vs drift)
- Colored markers: Control points with hover information
- Click legend items to show/hide traces
- Zoom, pan, and export capabilities

**Moments**:
- Moment vs SDR_max for each wall
- Shows moment development at wall base
- Each wall is a separate trace (toggleable)

**Shear Forces**:
- Shear force vs SDR_max for each wall
- Displays shear demand evolution
- Interactive legend for wall selection

**Drift Profile**:
- Normalized height (hi/ht) vs SDR_max
- Shows drift distribution along building height
- Four curves for different performance points:
  - Blue: 1st Cracking
  - Orange-red: 1st Yielding
  - Gold: Maximum Capacity
  - Dodger Blue: 80% Capacity

#### Interactive Features

All Plotly charts support:
- **Hover**: Display exact values
- **Zoom**: Box zoom, zoom in/out buttons
- **Pan**: Click and drag to move view
- **Legend Toggle**: Click legend items to show/hide traces
- **Export**: Download as PNG using camera icon
- **Reset**: Double-click to reset view

### 4. Incremental Dynamic Analysis (`app_dynamic.py` - IDA Analysis Page)

Perform nonlinear time-history analyses at multiple intensity levels using ground motion records.

#### Step 1: Model Selection

- Select from available `.pkl` files in the `models/` folder
- View model information:
  - Number of walls
  - Number of stories
  - Building height
  - Model visualization (if available)

#### Step 2: Ground Motion Records

The system automatically detects ground motion files in the `GMs/` folder:
- **GM Files**: Individual time-history records (GM01.txt - GM44.txt)
- **Spectral Data**: Required pickle files for scaling
  - `Sa_FEMA.pkl`: Spectral accelerations for each ground motion
  - `T.pkl`: Period array for spectral interpolation

**Status Indicators**:
- ✅ Green: All files found
- ❌ Red: Missing files or spectral data

#### Step 3: IDA Parameters

**Scale Factors (SFactor)**:
- Define intensity levels for analysis
- Default: `[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]`
- Custom: Enter comma-separated values
- Each factor scales the ground motions

**Spectral Acceleration (Sa_d)**:
- Design spectral acceleration in g's
- Default: 0.5625 g
- Range: 0.01 - 5.0 g
- Used as target for ground motion scaling

**Scaling Period (T_scaling)**:
- Fundamental period for spectral scaling
- Default: 1.0 seconds
- Range: 0.01 - 10.0 seconds
- Ground motions are scaled to match Sa_d at this period

**Computational Settings**:
- **CPU Cores**: Number of parallel processes
- Default: All available cores
- Each ground motion + scale factor combination runs in parallel
- More cores = faster analysis

#### Step 4: Analysis Summary

Displays total number of analyses:
- **Total Analyses** = (Scale Factors) × (Ground Motions)
- Example: 7 factors × 44 GMs = 308 analyses

#### Step 5: Run IDA Analysis

Click "🚀 Run IDA Analysis" to execute:

**Process**:
1. **Load Model**: Reconstruct from `.pkl` file
2. **Load Spectral Data**: Read `Sa_FEMA.pkl` and `T.pkl`
3. **Calculate Scale Factors**: 
   - Interpolate Sa at T_scaling for each GM
   - Factor = Sa_d / Sa_at_T
4. **Run Analyses**: Parallel execution using joblib
5. **Process Results**: Collect drift and Sa data
6. **Save Results**: 
   - `IDAs/<model>_IDA.pkl`: Pickle format
   - `IDAs/<model>_IDA.xlsx`: Excel format

**Output Data**:
- `GM`: Ground motion index (0-43)
- `Sa`: Spectral acceleration [g]
- `tmax`: Maximum time [s]
- `dertecho`: Roof drift [%]

**Analysis Time**: Depends on:
- Number of analyses
- Number of CPU cores
- Model complexity
- Ground motion duration

### 5. Results Visualization (`app_dynamic.py` - Results Visualization Page)

Interactive visualization of IDA results with three different plot types.

#### Loading Results

- Select from available `*_IDA.pkl` files in the `IDAs/` folder
- View dataset overview:
  - Total analyses
  - Number of ground motions
  - Intensity levels
  - Maximum drift
- Data table preview with download option

#### Visualization Tabs

##### Tab 1: Sa vs Roof Drift

Scatter plot showing relationship between spectral acceleration and roof drift:
- **X-axis**: Sa(T₁) [g]
- **Y-axis**: Roof Drift [%]
- **Color**: Ground motion ID
- **Markers**: Individual analysis points

**Controls**:
- Log/normal scale toggles for both axes
- Adjustable axis limits (Sa and Drift)
- Color scale shows GM distribution

**Statistics Panel**:
- Drift: Min, Max, Mean, Median
- Sa: Min, Max, Mean, Median

##### Tab 2: IDA Curves

Individual curves for each ground motion showing capacity:
- **X-axis**: Roof Drift [%]
- **Y-axis**: Sa(T₁) [g]
- **Lines**: One per ground motion (GM01-GM44)
- **Interactive Legend**: Toggle individual curves

**Main Plot Controls**:
- Log/normal scale for both axes
- Adjustable axis limits
- Hover shows GM ID, drift, and Sa

**Filter Ground Motions Expander**:
- Select specific GMs to display
- Separate plot with larger markers
- Independent scale and limit controls
- Useful for detailed comparison

##### Tab 3: Boxplots by Sa Level

Distribution of roof drift at each intensity level:
- **X-axis**: Sa(T₁) [g] - categorical by level
- **Y-axis**: Roof Drift [%]
- **Boxplots**: Show distribution statistics
  - Box: Interquartile range (IQR)
  - Line in box: Median
  - Whiskers: 1.5 × IQR
  - Diamond: Mean
  - Crosses: Outliers

**Controls**:
- Log/normal scale for Y-axis
- Adjustable Y-axis limits

**Statistical Summary Table**:
- Sa level [g]
- Count of analyses
- Mean, Std, Min, Median, Max drift [%]

**Violin Plot Option**:
- Alternative visualization showing distribution density
- Toggle in expander
- Same statistical information as boxplots

#### Interactive Features

All plots include:
- **Zoom**: Box zoom, scroll zoom
- **Pan**: Click and drag
- **Hover**: Detailed information
- **Legend**: Click to show/hide traces
- **Export**: PNG download
- **Reset**: Double-click to restore view

### 6. Fragility Functions (`app_dynamic.py` - Fragility Functions Page)

Calculate and visualize seismic fragility curves based on IDA results.

#### Step 1: Load IDA Results

- Select IDA results file (same as Results Visualization)
- Automatically loads building height from model file
- Shows model visualization if available

#### Step 2: Define Damage States

**Configuration**:
- **Number of Damage States**: 1-10 (default: 4)
- **EDP**: Uses `dertecho` (roof drift %)
- **IM**: Uses `Sa` (spectral acceleration)

**Damage State Inputs** (for each state):
- **Name**: Identifier (e.g., ds1, ds2, ds3, ds4)
- **Drift Limit [%]**: Threshold value
  - Example: 0.4, 0.8, 1.5, 2.0 for typical performance levels
  - Values directly compared with `dertecho` column

**Default Limits**:
- DS1: 0.4% (Immediate Occupancy)
- DS2: 0.8% (Damage Control)
- DS3: 1.5% (Life Safety)
- DS4: 2.0% (Collapse Prevention)

#### Step 3: Calculate Fragility Functions

Click "📊 Calculate Fragility Functions" to:

**Process**:
1. **Maximum Likelihood Estimation**: Fit lognormal distribution
2. **Parameter Calculation**: Determine θ (theta) and β (beta)
3. **Optimization**: Nelder-Mead simplex method
4. **Store Results**: Save in session state

**Theory**:
- Lognormal CDF: P[DS≥ds|Sa] = Φ[(ln(Sa) - ln(θ))/β]
- θ: Median spectral acceleration causing damage state
- β: Logarithmic standard deviation (dispersion)

#### Fragility Function Parameters

**Summary Table**:
- Damage State name
- Threshold value [%]
- θ (theta) [g]
- β (beta) [dimensionless]

#### Fragility Curves Plot

Interactive Plotly visualization:
- **X-axis**: Sa(T₁) [g]
- **Y-axis**: Probability of Exceedance
- **Curves**: One per damage state
- **Color-coded**: Different colors for each DS
- **Legend**: Shows θ and β values

**Plot Controls**:
- Log/normal scale toggles
- Adjustable axis limits for Sa and Probability
- Hover information showing exact values

**Interpretation**:
- Curves typically don't cross (higher DS at higher Sa)
- Steeper curves = less uncertainty (smaller β)
- Median (P=0.5) occurs at θ
- β controls curve spread

#### Data Export

**Download Button**:
- CSV file with fragility curve data
- Columns: Sa [g], P[ds1], P[ds2], P[ds3], P[ds4]
- 200 Sa values from min to max
- Suitable for further analysis or reporting

## Running Multiple Apps Simultaneously

The three apps can run concurrently without conflicts:

### Method 1: Automatic Port Assignment

Open three terminal windows and run:

```bash
# Terminal 1 - Model Creation
streamlit run app.py

# Terminal 2 - Pushover Analysis  
streamlit run app_analysis.py

# Terminal 3 - IDA & Fragility
streamlit run app_dynamic.py
```

Streamlit automatically assigns ports: 8501, 8502, 8503

### Method 2: Manual Port Assignment

Specify ports explicitly:

```bash
streamlit run app.py --server.port 8501
streamlit run app_analysis.py --server.port 8502
streamlit run app_dynamic.py --server.port 8503
```

### Workflow Integration

**Typical Analysis Sequence**:
1. **Create Model** (`app.py`) → `models/<name>.pkl`
2. **Run Pushover** (`app_analysis.py`) → `pushover/<name>-RPO.pkl`
3. **Run IDA** (`app_dynamic.py`) → `IDAs/<name>_IDA.pkl`
4. **Calculate Fragility** (`app_dynamic.py`) → Curves and CSV

**Benefits**:
- Independent sessions
- No resource conflicts
- Parallel workflows possible
- File-based communication

## File Formats

### Input Files

#### `Data/Info.xlsx`
Excel file with building configuration:
- **Sheet**: "Hoja1"
- **Columns**:
  - `Propiedad`: Parameter name
  - `Valor`: Parameter value
- **Rows**:
  - `Alturas`: Story heights (comma-separated)
  - `w`: Building weight
  - `ws`: Seismic weight
  - `fc`: Concrete strength

#### `Data/00_Wall-<name>.xlsx`
Excel file per wall with multiple sheets:
- **Concreto**: Concrete properties
- **Acero**: Steel reinforcement properties
- **Geometria**: Wall geometry
- **ElementosBorde**: Boundary elements (if applicable)

### Output Files

#### Model Files (`.pkl`)
Python pickle files containing serialized objects:
- **`<model>-L.pkl`**: Arquetipo object with model definition
- **`<model>-RPO.pkl`**: Arquetipo object with pushover results

Structure:
```python
Arquetipo:
    .nombre: str                    # Model name
    .muros: List[Muro]             # List of wall objects
    .ws_: float                     # Seismic weight
    .ResultadosPushover:           # Results container
        .listaVbase: List[float]   # Base shear history
        .listaSDR: List[float]     # SDR history
        .listaSDR1: List[float]    # SDR1 history
        .listaRDR: List[float]     # RDR history
```

#### Excel Results (`*_pushover.xlsx`)
Three sheets with pushover curve data:
- **SDR1**: First story drift results
- **SDR**: Maximum story drift results
- **RDR**: Roof drift results

Each sheet contains:
- Column 1: Drift ratio (%)
- Column 2: Vbasal - Total base shear (kN)
- Columns 3+: Individual wall shear forces (kN)

#### IDA Results (`*_IDA.pkl` and `*_IDA.xlsx`)

Pandas DataFrame with IDA analysis results:

**Columns**:
- `GM`: Ground motion index (0-based)
- `Sa`: Spectral acceleration [g]
- `tmax`: Maximum analysis time [s]
- `dertecho`: Roof drift percentage [%]

**Format**: One row per analysis (GM × Scale Factor combination)

**Excel Structure**:
- Single sheet with all data
- Easily imported to other tools
- Compatible with fragility calculation

#### Ground Motion Files

**Time History Files** (`GMs/GM*.txt`):
- ASCII text format
- Single column of acceleration values
- Units: Depends on ground motion source
- Scaled by SpectrumFactor during analysis

**Spectral Data Files** (`GMs/*.pkl`):
- `Sa_FEMA.pkl`: List of spectral acceleration arrays (one per GM)
- `T.pkl`: Period array corresponding to Sa values
- Used for scaling ground motions to target spectrum

## Analysis Theory

### MVLEM Element

The Multiple Vertical Line Element Model (MVLEM) is used for wall modeling:
- **Fibers**: Vertical uniaxial elements representing wall segments
- **Shear Spring**: Horizontal spring for shear behavior
- **DOFs**: 6 DOF per node (3 translations + 3 rotations)

### Material Models

**Concrete**:
- Uniaxial material with compression/tension behavior
- Mander model for confined concrete (boundary elements)
- Unconfined concrete for web regions

**Steel**:
- Bilinear or Giuffré-Menegotto-Pinto model
- Different properties for RB vs WWM
- Strain hardening included

### Pushover Analysis Procedure

1. **Gravity Analysis**: Apply dead loads
2. **Lateral Load Pattern**: Apply according to pushtype
3. **Displacement Control**: Push to target drift
4. **Convergence**: Newton-Raphson algorithm
5. **Recording**: Capture response at each step

### Control Point Detection Algorithm

For each wall, the algorithm searches through analysis steps to find:
- **Cracking**: First step where concrete strain > 0.00015
- **Yielding**: First step where steel strain > yield strain
- **Maximum**: Step with maximum base shear
- **Ultimate**: First step where steel strain > ultimate strain
- **80% Capacity**: First step after peak where V < 0.8·V_max

## Common Issues and Solutions

### Issue 1: Python Not Found
**Error**: `no se encontró Python`
**Solution**: Use full Python path: `C:/Users/HOME/anaconda3/python.exe`

### Issue 2: PyArrow DLL Error
**Error**: `ImportError: DLL load failed while importing lib`
**Solution**: The app now uses HTML tables instead of `st.dataframe()` to avoid this issue

### Issue 3: Model Creation Fails
**Error**: Invalid mass or load values
**Solution**: 
- Ensure all numeric inputs are positive
- Check that wall properties are physically reasonable
- Verify Excel files are not corrupted

### Issue 4: Pushover Analysis Doesn't Converge
**Error**: Analysis terminates early
**Solution**:
- Reduce `Dincr` for smaller load steps
- Reduce `pushlimit` to stop before instability
- Check model for geometric issues

### Issue 5: Duplicate Chart IDs
**Error**: Multiple plotly_chart elements with same ID
**Solution**: Fixed by adding unique `key` parameters to all charts

### Issue 6: RPO File Not Updating
**Error**: Old results displayed after new analysis
**Solution**: Script now always overwrites RPO files (no existence check)

## Advanced Usage

### Command-Line Execution

#### Model Generation
```bash
python 00_MODEL-CREATOR.py <model_name>
```

#### Pushover Analysis
```bash
python 01_PUSHOVER.py <model_name> [pushlimit] [pushtype] [modepush] [wallctrl] [Dincr] [direction]
```

Example:
```bash
python 01_PUSHOVER.py building-5S-L 0.035 3 1 0 0.001 Longitudinal
```

### Custom Material Models

Edit `Lib_materiales.py` to add custom material definitions:
- Concrete models: Modify `Concreto_Conf()` or `Concreto_NoConf()`
- Steel models: Modify `Acero_RB()` or `Acero_WWM()`

### Extending Analysis Capabilities

The modular structure allows easy extension:
- **New Analysis Types**: Add to `Lib_analisis.py`
- **Custom EDPs**: Modify result extraction in `01_PUSHOVER.py`
- **Additional Visualizations**: Extend `app_analysis.py`

## Best Practices

### Model Generation

1. **Start Simple**: Begin with 2-3 stories and few walls
2. **Consistent Units**: Use kN, m, MPa throughout
3. **Reasonable Parameters**: 
   - Wall thickness: 0.10-0.30 m
   - Story heights: 2.5-4.0 m
   - fc: 21-35 MPa
4. **Save Frequently**: Save each wall immediately after defining

### Analysis Setup

1. **Modal Pattern**: Use pushtype=3 for realistic load distribution
2. **Appropriate Drift**: 
   - Low-rise (1-3 stories): 0.025-0.035
   - Mid-rise (4-8 stories): 0.030-0.040
   - High-rise (9+ stories): 0.035-0.050
3. **Increment Size**: Balance accuracy vs speed
   - Quick analysis: Dincr = 0.002
   - Normal: Dincr = 0.001
   - Detailed: Dincr = 0.0005

### Results Interpretation

1. **Check Control Points**: Ensure they occur in logical sequence
2. **Review Curves**: Look for smooth, monotonic behavior
3. **Compare EDPs**: SDR typically governs for regular structures
4. **Validate**: Compare V/W with hand calculations (V/W ≈ C_s)

## Validation Examples

### Example 1: 5-Story Building

**Input**:
- Stories: 5
- Heights: 15, 12, 9, 6, 3 m (bottom to top)
- w = 10 kN, ws = 35 kN
- fc = 21 MPa
- 2 walls: 0.15m × 3.0m each
- Direction: Longitudinal

**Expected Results**:
- V/W at yielding: ~0.15-0.25
- Maximum V/W: ~0.30-0.45
- SDR at max: 0.5-1.5%

### Example 2: 3-Story Building

**Input**:
- Stories: 3
- Heights: 12, 9, 6 m
- w = 15 kN, ws = 40 kN
- fc = 28 MPa
- 3 walls: 0.20m × 4.0m each

**Expected Results**:
- V/W at yielding: ~0.20-0.30
- Maximum V/W: ~0.40-0.60
- More ductile behavior (higher drift capacity)

## Troubleshooting

### Debug Mode

Enable debug output in `01_PUSHOVER.py` by checking the printed parameters:
```
========== ANALYSIS PARAMETERS ==========
Model Name: building-5S-L
pushlimit: 0.035
pushtype: 3
...
=========================================
```

### Log Files

Check terminal output for:
- Convergence issues
- Material failures
- Geometric problems
- File I/O errors

### Verification Steps

1. **Model Creation**:
   - Verify `.pkl` file created
   - Check plot shows correct geometry
   - Confirm Excel files in Data/

2. **Analysis Execution**:
   - Check for "Análisis Push Over realizado" message
   - Verify Excel results file created
   - Confirm RPO.pkl file updated

3. **Visualization**:
   - Ensure control points are reasonable
   - Check curve monotonicity
   - Verify all walls appear in plots

## References

### Technical Background

- MVLEM Element: Lu et al. (2015)
- Pushover Analysis: FEMA 356, ATC-40
- Material Models: Mander et al. (1988), Menegotto-Pinto (1973)
- Damage States: Carrillo et al. (2022)

### Software Documentation

- OpenSeesPy: https://openseespydoc.readthedocs.io
- Streamlit: https://docs.streamlit.io
- Plotly: https://plotly.com/python

## Version History

### v3.0 (Current)
- **NEW**: IDA Analysis module (`app_dynamic.py`)
  - Incremental Dynamic Analysis with parallel processing
  - Multi-core support for faster execution
  - Ground motion scaling based on target spectrum
  - Dynamic SpectrumFactor calculation from spectral data
- **NEW**: Fragility Function calculation
  - Maximum likelihood estimation of parameters
  - Interactive Plotly visualization of fragility curves
  - Multiple damage state support (1-10)
  - CSV export of fragility data
- **NEW**: Three comprehensive visualization tabs for IDA results
  - Sa vs Roof Drift scatter plots
  - IDA curves with GM filtering
  - Boxplots and violin plots by intensity level
- Interactive plot controls: log/normal scale toggles, axis limits
- Cross-platform path compatibility fixes (macOS/Windows)
- Enhanced session state management for multi-page apps

### v2.0
- Added Plotly interactive visualizations for pushover results
- Implemented tab-based EDP selection (RDR, SDR, SDR1)
- Removed wall filtering (use legend instead)
- Fixed pyarrow compatibility issues
- Added unique keys for all widgets
- Improved command-line parameter passing

### v1.0
- Initial release
- Basic model generation
- Matplotlib-based plotting
- Single EDP selection dropdown

## Support

For issues, questions, or contributions:
- Check this documentation first
- Review common issues section
- Verify input file formats
- Test with simple example models

## License

RC-WIAP is developed for academic and research purposes.

## Authors

- Orlando
- Frank
- JuanJo

## Acknowledgments

Special thanks to the structural engineering research community for developing the underlying analysis methods and OpenSeesPy framework.

---

**Last Updated**: November 9, 2025
**Version**: 3.0

