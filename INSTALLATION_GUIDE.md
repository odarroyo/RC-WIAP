# RC-WIAP Installation Guide

This guide provides step-by-step instructions for setting up RC-WIAP (Reinforced Concrete Wall Interaction Analysis Program) from scratch on your local machine.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installing Python](#installing-python)
3. [Setting Up a Virtual Environment](#setting-up-a-virtual-environment)
4. [Installing Required Libraries](#installing-required-libraries)
5. [Installing OpenSeesPy](#installing-openseespy)
6. [Installing Custom Libraries](#installing-custom-libraries)
7. [Verifying the Installation](#verifying-the-installation)
8. [Running RC-WIAP Applications](#running-rc-wiap-applications)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

RC-WIAP can run on the following operating systems:

- **Windows** 10 or later
- **macOS** 10.14 (Mojave) or later
- **Linux** (Ubuntu 18.04 or equivalent)

**Hardware recommendations:**
- Minimum 8 GB RAM (16 GB recommended for large dynamic analyses)
- Multi-core processor (4+ cores recommended for parallel processing)
- At least 2 GB of free disk space

---

## Installing Python

RC-WIAP requires Python 3.8 or later (Python 3.9 or 3.10 recommended).

### Windows

1. Download Python from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Run the installer
3. **Important:** Check the box "Add Python to PATH" during installation
4. Click "Install Now"
5. Verify installation by opening Command Prompt and typing:
   ```cmd
   python --version
   ```

### macOS

Python 3 may already be installed. Check by opening Terminal and typing:
```bash
python3 --version
```

If not installed, you can install Python using Homebrew:
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10
```

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3-pip

# Verify installation
python3 --version
```

---

## Setting Up a Virtual Environment

A virtual environment isolates RC-WIAP's dependencies from other Python projects on your system. This is highly recommended.

### Creating a Virtual Environment

Navigate to the directory where you want to install RC-WIAP, then create a virtual environment:

**Windows:**
```cmd
cd C:\path\to\RC-WIAP
python -m venv rcwiap_env
```

**macOS/Linux:**
```bash
cd /path/to/RC-WIAP
python3 -m venv rcwiap_env
```

### Activating the Virtual Environment

You need to activate the virtual environment each time you want to use RC-WIAP.

**Windows:**
```cmd
rcwiap_env\Scripts\activate
```

**macOS/Linux:**
```bash
source rcwiap_env/bin/activate
```

After activation, your command prompt should show `(rcwiap_env)` at the beginning, indicating the virtual environment is active.

### Deactivating the Virtual Environment

When you're done working with RC-WIAP:
```bash
deactivate
```

---

## Installing Required Libraries

With your virtual environment activated, install the core dependencies.

### Installing from requirements.txt

If you have the `requirements.txt` file in the RC-WIAP directory:

```bash
pip install -r requirements.txt
```

### Manual Installation

If you need to install packages individually:

```bash
# Core web framework
pip install streamlit>=1.28.0

# Data manipulation and analysis
pip install pandas>=2.0.0
pip install numpy>=1.24.0

# Excel file handling
pip install openpyxl>=3.1.0

# Plotting and visualization
pip install matplotlib>=3.7.0
pip install plotly>=5.14.0

# Scientific computing
pip install scipy>=1.10.0

# Parallel processing
pip install joblib>=1.3.0
```

---

## Installing OpenSeesPy

OpenSeesPy is the core analysis engine for RC-WIAP. Install it using pip:

```bash
pip install openseespy
```

**Note:** OpenSeesPy installation may take a few minutes as it includes compiled libraries.

### Verifying OpenSeesPy Installation

Test that OpenSeesPy is properly installed:

```bash
python -c "import openseespy.opensees as ops; print('OpenSeesPy version:', ops.version())"
```

If successful, you should see the OpenSeesPy version number printed.

---

## Installing Custom Libraries

RC-WIAP uses custom libraries for structural analysis and visualization that are not available on PyPI.

### Installing opsvis

`opsvis` is used for visualizing OpenSees models. Install it via pip:

```bash
pip install opsvis
```

### Installing opseestools

`opseestools` is a custom library that contains analysis functions and fragility calculation modules used by RC-WIAP.

**Option 1: If opseestools is available as a package**
```bash
pip install opseestools
```

**Option 2: If opseestools is provided as source files**

If `opseestools` is included as a folder in the RC-WIAP directory (containing `analisis.py`, `Lib_frag.py`, etc.), ensure the folder structure is:

```
RC-WIAP/
├── opseestools/
│   ├── __init__.py
│   ├── analisis.py
│   ├── Lib_frag.py
│   └── ...
├── app.py
├── app_analysis.py
├── app_dynamic.py
└── ...
```

Python will automatically find this local package when running the applications.

---

## Verifying the Installation

After installing all dependencies, verify that everything is working correctly.

### Check All Required Packages

Create a test script `check_installation.py`:

```python
import sys

packages = {
    'streamlit': 'Streamlit',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'openpyxl': 'OpenPyXL',
    'matplotlib': 'Matplotlib',
    'plotly': 'Plotly',
    'scipy': 'SciPy',
    'joblib': 'Joblib',
    'openseespy.opensees': 'OpenSeesPy',
    'opsvis': 'OpsVis',
}

print("Checking installed packages...\n")
all_installed = True

for module, name in packages.items():
    try:
        __import__(module)
        print(f"✓ {name} is installed")
    except ImportError:
        print(f"✗ {name} is NOT installed")
        all_installed = False

# Check custom libraries
try:
    import opseestools.analisis as an2
    import opseestools.Lib_frag as lf
    print(f"✓ opseestools is installed")
except ImportError:
    print(f"✗ opseestools is NOT installed or not in Python path")
    all_installed = False

print("\n" + "="*50)
if all_installed:
    print("SUCCESS: All packages are installed correctly!")
else:
    print("ERROR: Some packages are missing. Please install them.")
print("="*50)
```

Run the script:
```bash
python check_installation.py
```

---

## Running RC-WIAP Applications

RC-WIAP consists of three main applications:

### 1. Model Generation Application (`app.py`)

This application is used to create parametric building models.

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### 2. Pushover Analysis Application (`app_analysis.py`)

This application performs nonlinear static pushover analyses.

```bash
streamlit run app_analysis.py --server.port 8502
```

The application will open at `http://localhost:8502`

### 3. Dynamic Analysis Application (`app_dynamic.py`)

This application performs incremental dynamic analysis (IDA) and fragility function calculations.

```bash
streamlit run app_dynamic.py --server.port 8503
```

The application will open at `http://localhost:8503`

### Running Multiple Applications Simultaneously

You can run all three applications at the same time on different ports. Open three separate terminal windows (or tabs) and run each command with the virtual environment activated.

**Terminal 1:**
```bash
streamlit run app.py --server.port 8501
```

**Terminal 2:**
```bash
streamlit run app_analysis.py --server.port 8502
```

**Terminal 3:**
```bash
streamlit run app_dynamic.py --server.port 8503
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "python: command not found" or "python3: command not found"

**Solution:** Python is not in your system PATH. 
- On Windows, reinstall Python and check "Add Python to PATH"
- On macOS/Linux, use the full path to Python or create an alias

#### Issue: "pip: command not found"

**Solution:** pip is not installed or not in PATH.
```bash
# Windows
python -m ensurepip --upgrade

# macOS/Linux
python3 -m ensurepip --upgrade
```

#### Issue: OpenSeesPy installation fails

**Solution:** 
- Ensure you have a 64-bit Python installation (OpenSeesPy requires 64-bit)
- On Linux, you may need to install additional system libraries:
  ```bash
  sudo apt-get install gcc g++ gfortran libopenblas-dev liblapack-dev
  ```
- Try installing a specific version:
  ```bash
  pip install openseespy==3.4.0.1
  ```

#### Issue: "Module not found" errors when running applications

**Solution:**
- Ensure your virtual environment is activated
- Verify all packages are installed: `pip list`
- Check that custom libraries (Lib_analisis.py, Lib_materiales.py, etc.) are in the same directory as the application files

#### Issue: Streamlit applications won't start

**Solution:**
- Check if the port is already in use
- Try a different port: `streamlit run app.py --server.port 8505`
- Check firewall settings that might block local connections

#### Issue: Applications run but show import errors for custom modules

**Solution:**
- Ensure the following files are in the RC-WIAP directory:
  - `Lib_analisis.py`
  - `Lib_materiales.py`
  - `Lib_ClaseArquetipos.py`
  - `Lib_GM.py`
  - `opseestools/` directory (if applicable)
- Verify file permissions (read access required)

#### Issue: Parallel processing not working in dynamic analysis

**Solution:**
- Check the number of CPU cores available: 
  ```python
  import multiprocessing
  print(multiprocessing.cpu_count())
  ```
- On some systems, you may need to set the number of workers manually in the application
- On macOS, ensure you're not using the default system Python (use Homebrew Python instead)

#### Issue: Memory errors during large analyses

**Solution:**
- Reduce the number of ground motions analyzed simultaneously
- Close other applications to free up RAM
- Consider analyzing in batches for very large IDA studies
- Increase system virtual memory/swap space

#### Issue: Excel files not saving or loading properly

**Solution:**
- Ensure openpyxl is installed: `pip install openpyxl`
- Check file permissions in the output directories
- Close any Excel files that may be open and locked
- On macOS, ensure you're not syncing the folder with iCloud (may cause locking issues)

---

## Updating RC-WIAP

To update RC-WIAP dependencies to the latest versions:

```bash
# Activate virtual environment first
pip install --upgrade streamlit pandas numpy openseespy plotly scipy joblib
```

To update individual packages:
```bash
pip install --upgrade package_name
```

---

## Uninstalling RC-WIAP

To completely remove RC-WIAP and its virtual environment:

1. Deactivate the virtual environment if active:
   ```bash
   deactivate
   ```

2. Delete the virtual environment folder:
   **Windows:**
   ```cmd
   rmdir /s rcwiap_env
   ```
   
   **macOS/Linux:**
   ```bash
   rm -rf rcwiap_env
   ```

3. Optionally, delete the RC-WIAP application files

---

## Additional Resources

- **OpenSeesPy Documentation:** [https://openseespydoc.readthedocs.io/](https://openseespydoc.readthedocs.io/)
- **Streamlit Documentation:** [https://docs.streamlit.io/](https://docs.streamlit.io/)
- **Python Virtual Environments:** [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html)
- **RC-WIAP User Manual:** See `README.md` in the RC-WIAP directory

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check the `README.md` file for additional documentation
2. Review error messages carefully - they often indicate the specific problem
3. Search for the error message online (Stack Overflow, GitHub issues)
4. Ensure you're using compatible versions of Python (3.8-3.10) and all dependencies

---

**Installation Date:** November 9, 2025  
**Document Version:** 1.0  
**Compatible with:** RC-WIAP v3.0
