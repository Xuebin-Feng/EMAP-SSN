import unicodedata  # Pre-load to prevent Windows DLL search path conflicts with Qt/OpenGL
# Import Libraries
import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false"

# --- Placeholder Parameters ---
SEQUENCE_SET = ""
ALIGNMENT_REFERENCE = ""
ALIGNMENT_OFFSET = 0
SIMILARITY_THRESHOLD = None
TOP_EDGE_PERCENT = None    
ALIGNMENT_SCORE = None
NORM_MODE = None
UMAP_MODE = None
UMAP_NEIGHBORS = None
UMAP_MIN_DIST = None
TARGET_CACHE_FILE = os.environ.get("SSN_TARGET_CACHE", None)

# --- Directory & File Paths ---
FASTA_DIR = os.path.join("Input_Files", "Sequence_Sets")
MSA_DIR = os.path.join("Input_Files", "Multiple_Alignments")
HDF5_DIR = os.path.join("Input_Files", "Networks_EValues")
SAVED_LAYOUT_DIR = os.path.join("Cache_Files", "Saved_Layouts")
METADATA_DIR = os.path.join("Cache_Files", "Meta_Data")
PRINT_SAVE_DIR = os.path.join("Results", "Saved_Images")
FASTA_SPLIT_DIR = os.path.join("Cache_Files", "FASTA_Split")
CLUSTER_LABEL_DIR = os.path.join("Results", "Cluster_Label")
HEADER_LIST_DIR = os.path.join("Cache_Files", "Header_Lists")
LOGO_DIR = os.path.join("Results", "Sequence_Logos")
STRUCTURES_DIR = os.path.join("Cache_Files", "Structures")

# --- Explicit Input File Paths ---
# You can manually replace these string paths to decouple file logic:
NODE_FASTA_FILE = ""

# Default values if scanning directories fails:
MSA_FILE = ""
INPUT_HDF5 = ""

# Sequences File points to NODE_FASTA_FILE for backward compatibility
SEQUENCES_FILE = ""

# --- Command Settings ---
GAP_CHARS = ['-', '.']
FILTER_MIN_OCCUPANCY = 10.0

# --- Visual Defaults ---
NODE_SIZE = 10
EDGE_WIDTH = 1.0
EDGE_ALPHA = 0.1     
NODE_BOUNDARY_WIDTH = 0.5
TEXT_SIZE = 8
TEXT_COLOR = 'grey'
INITIAL_NODE_COLOR = '#4488ff'
HOVER_COLOR = '#ffaa00'
CONNECTED_NODE_COLOR = '#ff0000'
EDGE_COLOR = '#000000'
LOW_RESOURCE_MODE = False
NODE_BOUNDARY_COLOR = '#000000'

# --- Grid Packing Settings ---
PACKING_GRID_SIZE = 20.0  # The base size of one grid square
PACKING_PADDING = 10.0     # Extra padding applied to the bounding box of each cluster

# --- Simulation & Physics Settings ---
PHYSICS_ENGINE = "Molecular Dynamics (Style)"
SPRING_K = 5.0             
COULOMB_K = 10.0            
COULOMB_CUTOFF = 30.0      
DAMPING = 0.9              
MAX_FORCE_LIMIT = 20.0      

DT = 0.005
BOX_SCALE = 2.0
MAX_STEPS = 10000           
RMSD_THRESHOLD = 0.005 
PERCENTAGE_DROP_THRESHOLD = 0.1    
RMSD_WINDOW = 50
ENABLE_PROGRESSIVE_SIMULATION = False
PACKING_GEOMETRY = "Square"

# --- Monte Carlo / SGLD Settings ---
SGLD_MIN_K = 20
SGLD_K_PERCENT = 0.01
SGLD_START_TEMP = 1.5
SGLD_NOISE_SCALE = 1.0

# --- JSON Settings Override ---
import json
import ast

SETTINGS_FILE = os.path.join("Input_Files", "viewer_settings.json")
if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r") as f:
            viewer_settings = json.load(f)
            # Map legacy settings keys to new keys
            LEGACY_KEYS_MAPPING = {
                "NEIGHBOR_COLOR": "INITIAL_NODE_COLOR"
            }
            for k, v in viewer_settings.items():
                if k in LEGACY_KEYS_MAPPING:
                    k = LEGACY_KEYS_MAPPING[k]
                if k in globals() and v is not None and (str(v).strip() != "" or k in ["MSA_FILE", "ALIGNMENT_REFERENCE"]):
                    orig = globals()[k]
                    if isinstance(orig, int) and not isinstance(orig, bool):
                        try: v = int(v)
                        except: pass
                    elif isinstance(orig, float):
                        try: v = float(v)
                        except: pass
                    elif isinstance(orig, bool):                                  # <--- ADD THIS
                        v = str(v).lower() in ['true', '1', 't', 'y', 'yes']      # <--- ADD THIS
                    elif isinstance(orig, list):
                        try: v = ast.literal_eval(v) if isinstance(v, str) else v
                        except: pass
                    elif orig is None:
                        if v == "None": v = None
                        elif str(v).replace('.', '', 1).isdigit():
                            v = float(v) if '.' in str(v) else int(v)
                    globals()[k] = v
                        
                # ---> NEW: SYNC LEGACY VARIABLES <---
                if "NODE_FASTA_FILE" in globals() and globals()["NODE_FASTA_FILE"]:
                    globals()["SEQUENCES_FILE"] = globals()["NODE_FASTA_FILE"]
                    globals()["SEQUENCE_SET"] = os.path.splitext(os.path.basename(globals()["NODE_FASTA_FILE"]))[0]
                    # Also update default fallback paths if they still contain "None" or are empty after SEQUENCE_SET is resolved
                    if globals().get("MSA_FILE") and "none_[e1_ra]_alignment.fasta" in globals()["MSA_FILE"].lower():
                        globals()["MSA_FILE"] = os.path.join(globals()["MSA_DIR"], f"{globals()['SEQUENCE_SET']}_[E1_RA]_alignment.fasta")
                    if not globals().get("INPUT_HDF5") or "none_[e1_ra]_network.h5" in globals()["INPUT_HDF5"].lower():
                        globals()["INPUT_HDF5"] = os.path.join(globals()["HDF5_DIR"], f"{globals()['SEQUENCE_SET']}_[E1_RA]_network.h5")
                    
    except Exception as e:
        print(f"Failed to load viewer settings: {e}")

# =============================================================================
# GUI APPLICATION
# =============================================================================
if __name__ == "__main__":
    import sys
    import subprocess

    os.environ["QT_API"] = "pyqt6"
    os.environ["QT_MAC_WANTS_LIGHT_THEME"] = "1"
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QTabWidget, QFormLayout, QLineEdit, 
                                 QComboBox, QPushButton, QMessageBox, QTextEdit,
                                 QLabel, QSplitter, QSlider, QSpinBox, QDoubleSpinBox,
                                 QStyle, QStyleOptionSlider, QFileDialog, QColorDialog)
    from PyQt6.QtCore import Qt, QUrl
    from PyQt6.QtGui import QDesktopServices, QIcon
    
    # --- Custom Widget Classes ---
    class NoScrollComboBox(QComboBox):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            
        def wheelEvent(self, event):
            if not self.hasFocus():
                event.ignore()
            else:
                super().wheelEvent(event)
                
    class DynamicComboBox(NoScrollComboBox):
        def __init__(self, refresh_callback, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.refresh_callback = refresh_callback
            
        def showPopup(self):
            if self.refresh_callback:
                self.refresh_callback()
            super().showPopup()
            
    class NoScrollSpinBox(QSpinBox):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            
        def wheelEvent(self, event):
            if not self.hasFocus():
                event.ignore()
            else:
                super().wheelEvent(event)
                
    class NoScrollDoubleSpinBox(QDoubleSpinBox):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            
        def wheelEvent(self, event):
            if not self.hasFocus():
                event.ignore()
            else:
                super().wheelEvent(event)
                
    class NoScrollSlider(QSlider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            
        def wheelEvent(self, event):
            if not self.hasFocus():
                event.ignore()
            else:
                super().wheelEvent(event)
                
        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                opt = QStyleOptionSlider()
                self.initStyleOption(opt)
                sr = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self)
                if not sr.contains(event.position().toPoint()):
                    val = self.style().sliderValueFromPosition(self.minimum(), self.maximum(), int(event.position().x()), self.width())
                    self.setValue(val)
                    event.accept()
                    return
            super().mousePressEvent(event)

    class ConfigGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("SSN Configuration Editor")
            
            # Set Window Icon
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.ico")
            if not os.path.exists(icon_path):
                icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                
            self.resize(1000, 650)
            
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            
            # --- SPLIT LAYOUT ---
            self.main_layout = QHBoxLayout(self.central_widget)
            
            # Retain native Fusion theme graphics but increase the invisible grab area width
            self.main_split = QSplitter(Qt.Orientation.Horizontal)
            self.main_split.setHandleWidth(12) 
            self.main_layout.addWidget(self.main_split)
            
            # Left Panel Splitter (Vertical)
            self.left_split = QSplitter(Qt.Orientation.Vertical)
            self.left_split.setHandleWidth(12)
            self.main_split.addWidget(self.left_split)
            
            # Left Top: Tabs Only
            self.left_top_widget = QWidget()
            self.left_top_layout = QVBoxLayout(self.left_top_widget)
            self.left_top_layout.setContentsMargins(0, 0, 0, 0)
            self.tabs = QTabWidget()
            self.left_top_layout.addWidget(self.tabs)
            
            # Left Bottom: Tool Tip Box + Action Buttons
            self.left_bottom_widget = QWidget()
            self.left_bottom_layout = QVBoxLayout(self.left_bottom_widget)
            self.left_bottom_layout.setContentsMargins(0, 0, 0, 0)
            
            self.tip_panel = QLabel("Click or tab to an input or its label to see helpful tips here.")
            self.tip_panel.setWordWrap(True)
            self.tip_panel.setStyleSheet("color: #444; font-style: italic; background-color: #e8eaed; padding: 10px; border-radius: 5px;")
            self.left_bottom_layout.addWidget(self.tip_panel)
            
            btn_layout = QHBoxLayout()
            self.btn_check = QPushButton("Consistency Check")
            self.btn_check.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; padding: 5px;")
            self.btn_check.clicked.connect(self.run_consistency_check)
            
            btn_save_run = QPushButton("Save && Run")
            btn_save_run.clicked.connect(self.save_and_run)
            btn_save_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
            
            btn_save = QPushButton("Save")
            btn_save.clicked.connect(self.save_only)
            
            btn_exit = QPushButton("Exit")
            btn_exit.clicked.connect(self.close)
            
            btn_layout.addWidget(btn_save_run)
            btn_layout.addWidget(self.btn_check)
            btn_layout.addWidget(btn_save)
            btn_layout.addWidget(btn_exit)
            self.left_bottom_layout.addLayout(btn_layout)
            
            self.left_split.addWidget(self.left_top_widget)
            self.left_split.addWidget(self.left_bottom_widget)
            
            # Explicitly force the initial pixel heights (tabs get 550px, bottom gets 100px)
            self.left_split.setSizes([450, 200])
            
            # Ensure that if the user resizes the window, extra space goes to the tabs, not the bottom
            self.left_split.setStretchFactor(0, 1)
            self.left_split.setStretchFactor(1, 0)
            
            # Right Panel: Statistics
            self.right_panel = QWidget()
            self.right_layout = QVBoxLayout(self.right_panel)
            self.right_layout.setContentsMargins(0, 0, 0, 0)
            self.stat_label = QLabel("Network Statistics Report")
            self.stat_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.stat_display = QTextEdit()
            self.stat_display.setReadOnly(True)
            self.stat_display.setPlaceholderText("Select Fasta subset and HDF5 Network file, then click compute.")
            self.stat_display.setStyleSheet("font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; background-color: #f5f5f5;")
            self.right_layout.addWidget(self.stat_label)
            self.right_layout.addWidget(self.stat_display)
            
            self.main_split.addWidget(self.right_panel)
            self.main_split.setStretchFactor(0, 6) 
            self.main_split.setStretchFactor(1, 4) 
            
            # Data Containers
            self.inputs = {}
            self.labels = {} 
            self.color_swatches = {} 
            
            self.create_inputs_tab()
            self.create_visuals_tab()
            self.create_physics_tab()
            self.create_directories_tab()
            
            self.cb_fasta.currentTextChanged.connect(self.update_live_validators)
            self.cb_hdf5.currentTextChanged.connect(self.update_live_validators)
            self.cb_score_mode.currentTextChanged.connect(self.update_live_validators)
            self.cb_norm_mode.currentTextChanged.connect(self.update_live_validators)
            self.line_ref.textChanged.connect(self.update_live_validators)
            self.line_thresh.textChanged.connect(self.update_live_validators)
            self.line_top.textChanged.connect(self.update_live_validators)
            self.cb_msa.currentTextChanged.connect(self.update_live_validators)
            
            self.update_live_validators()
            self.setup_tips()
            
        def run_consistency_check(self):
            import h5py
            from Bio import SeqIO
            import os
            
            # 1. Define the file names by grabbing them from the UI dropdowns
            fasta_file = self.cb_fasta.currentText()
            hdf5_file = self.cb_hdf5.currentText()
            msa_file = self.cb_msa.currentText()
            
            # 2. Halt if the user hasn't selected the required files
            if not fasta_file or not hdf5_file:
                return
            
            # 3. Build the paths safely
            fasta_path = os.path.join(self.inputs["FASTA_DIR"].text(), fasta_file)
            hdf5_path = os.path.join(self.inputs["HDF5_DIR"].text(), hdf5_file)
            msa_path = os.path.join(self.inputs["MSA_DIR"].text(), msa_file) if msa_file else None
                
            self.tip_panel.setText("Running Consistency Check...")
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            
            try:
                fasta_ids = set()
                fasta_headers = set()
                for rec in SeqIO.parse(fasta_path, "fasta"):
                    fasta_ids.add(rec.id)
                    fasta_headers.add(rec.description)
                
                with h5py.File(hdf5_path, "r") as hf:
                    raw_headers = hf['headers'][:]
                    headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
                    
                net_headers_set = set(headers)
                net_id_set = {h.split()[0] for h in headers}
                
                missing_nodes = [hid for hid in fasta_ids if hid not in net_id_set and hid not in net_headers_set]
                
                num_matched = len(fasta_ids) - len(missing_nodes)
                num_missing = len(missing_nodes)
                
                msg = f"FASTA vs HDF5:\nMatched: {num_matched} of {len(net_headers_set)} | Missing: {num_missing}"
                
                if missing_nodes:
                    msg = f"ERROR: FASTA is NOT a subset of HDF5.\n{msg}\nMissing examples: {', '.join(missing_nodes[:5])}"
                else:
                    msg = f"SUCCESS: FASTA is a strict subset of HDF5.\n{msg}"
                
                if msa_path and os.path.exists(msa_path):
                    msa_ids = set()
                    
                    if msa_path.endswith('.h5'):
                        import h5py
                        with h5py.File(msa_path, "r") as hf:
                            raw_headers = hf['headers'][:]
                            msa_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
                            msa_ids = {h.split()[0] for h in msa_headers}
                    else:
                        for rec in SeqIO.parse(msa_path, "fasta"):
                            msa_ids.add(rec.id)
                            
                    msa_missing = [hid for hid in fasta_ids if hid not in msa_ids]
                    msa_matched = len(fasta_ids) - len(msa_missing)
                    
                    msa_msg = f"FASTA vs MSA:\nMatched: {msa_matched} of {len(msa_ids)} | Missing: {len(msa_missing)}"
                    
                    if msa_missing:
                        msg += f"\n\nERROR: FASTA is NOT a subset of MSA.\n{msa_msg}\nMissing examples: {', '.join(msa_missing[:5])}"
                    else:
                        msg += f"\n\nSUCCESS: FASTA is a strict subset of MSA.\n{msa_msg}"
                
                # Check Reference ID if provided
                ref_id = self.line_ref.text().strip()
                if ref_id:
                    # Proceed with normal matching only (Case-Insensitive)
                    ref_id_lower = ref_id.lower()
                    matched_refs = [h for h in fasta_headers if ref_id_lower in h.lower()]
                    if matched_refs:
                        msg += f"\n\nSUCCESS: Reference ID '{ref_id}' matched {len(matched_refs)} header(s) in FASTA."
                        for h in matched_refs[:5]:
                            msg += f"\n  - {h}"
                    else:
                        msg += f"\n\nWARNING: Reference ID '{ref_id}' NOT found in FASTA headers."
                
                self.tip_panel.setText(msg)
            
            except Exception as e:
                self.tip_panel.setText(f"Error during consistency check: {e}")
            
        def eventFilter(self, obj, event):
            from PyQt6.QtCore import QEvent
            if event.type() in (QEvent.Type.FocusIn, QEvent.Type.MouseButtonPress):
                if hasattr(self, 'tip_db'):
                    tip = self.tip_db.get(obj, None)
                    if tip:
                        self.tip_panel.setText(tip)
            return super().eventFilter(obj, event)

        def setup_tips(self):
            self.tip_db_keys = {
                "NODE_FASTA_FILE": "The FASTA file representing the primary sequence set or subset to be visualized as nodes in the SSN.\nEnsure this file resides in the designated input directory and aligns with the selected network edges and alignments.",
                "MSA_FILE": "The multiple sequence alignment file (.fasta, .h5, or _sparse.pkl) containing alignments for the sequence set.\nThis is required to calculate positional conservation, gaps, and occupancy thresholds during structural clustering.",
                "INPUT_HDF5": "The core network or similarity matrix file (.h5) containing pair-wise sequence similarity metrics and edge list coordinates.\nMust contain similarity scores or alignment metrics for at least all sequences present in the active FASTA sequence set.",
                "ALIGNMENT_SCORE": "(For embedding-based SSNs only) Specifies whether to use global alignment similarity scores or local alignment similarity scores.\nLocal alignment is recommended for multi-domain proteins, whereas global alignment is best for full-length comparisons.",
                "NORM_MODE": "(For embedding-based SSNs only) Normalization strategy for pairwise sequence alignment scores.\nOptions include normalizing by alignment length, shorter sequence, longer sequence, or average sequence length to reduce length bias.",
                "ALIGNMENT_REFERENCE": "Substring from a sequence header to identify the reference sequence (e.g. wildtype or specific construct) in the alignment.\nThis sequence is used to calculate absolute relative positions and mapping offsets across the entire network.",
                "ALIGNMENT_OFFSET": "Integer added to reference-anchored alignment positions. For example, an offset of 10 changes reference position 1 to 11.\nThe offset is applied only when the Alignment Reference ID resolves successfully.",
                "SIMILARITY_THRESHOLD": "Minimum similarity score threshold (e.g. identity fraction, normalized score, or Log10 E-Value) required to retain an edge.\nEdges with scores below this value are filtered out and will not be rendered or computed in the physics simulation.",
                "TOP_EDGE_PERCENT": "Alternative edge filtering method that automatically calculates a threshold to retain only the top N% of all possible edges.\nUseful for maintaining network connectivity and density without manually tuning raw similarity score thresholds (Overrides Similarity Threshold).",
                "FILTER_MIN_OCCUPANCY": "Minimum percentage of non-gap characters required at an alignment column to retain it in the clustering calculations.\nColumns with occupancy below this percentage are treated as noise and are excluded to improve signal-to-noise ratio.",
                "NODE_SIZE": "Visual parameter controlling the average render size of each sequence node in the network visualization window.\nAdjust this parameter to optimize visual density; smaller nodes are recommended for very large networks.",
                "EDGE_WIDTH": "Visual parameter determining the thickness of the connection lines drawn between related sequence nodes in the network.\nThinner lines are recommended for dense networks to avoid visual cluttering, while thicker lines highlight strong relationships.",
                "EDGE_ALPHA": "Visual parameter controlling the transparency of the network edge lines, ranging from 0.0 (fully transparent) to 1.0 (opaque).\nLower opacity helps reveal the underlying node distribution and cluster density in highly connected graphs.",
                "TEXT_SIZE": "Font size used for rendering text labels on clusters or individual nodes in the visualizer window.\nAdjust this to make labels legible against the background without overlapping or obstructing structural features of the network.",
                "TEXT_COLOR": "Color of standard text labels in the viewer. Can be specified as a standard web color name (e.g. 'grey', 'black') or hex code.\nChoose a color that contrasts well with your background to ensure readability of cluster annotations.",
                "INITIAL_NODE_COLOR": "The default fill color applied to all nodes when the network is first loaded. This serves as the baseline node color before any custom coloring is applied.",
                "HOVER_COLOR": "Highlight color applied to a node and its adjacent connections when hovering over it with the cursor, or when selected.\nThis color should be highly vibrant to give immediate interactive visual feedback to the user.",
                "CONNECTED_NODE_COLOR": "Highlight color applied to the outer contour/border of nodes that are directly connected to the currently selected node.\nAllows easy visual identification of the local neighborhood network topology surrounding any selected node.",
                "EDGE_COLOR": "Color of the standard connection lines (edges) drawn between similar nodes in the network plot.\nCan be specified as a standard color name or hex code; lighter colors are often preferred to reduce visual dominance of edges.",
                "NODE_BOUNDARY_COLOR": "Color of the outer border ring outline drawn around each sequence node in the network visualization plot.\nTypically set to dark grey or black to cleanly separate adjacent nodes and enhance the depth of the visualization.",
                "NODE_BOUNDARY_WIDTH": "Visual rendering parameter controlling the thickness of the outer border ring outline drawn around each sequence node.\nSetting this to a small non-zero value helps distinguish overlapping nodes in dense cluster regions.",
                "PHYSICS_ENGINE": "Selects the simulation engine used to compute node coordinates: Molecular Dynamics or Monte Carlo (SGLD).\nMolecular Dynamics uses deterministic force integration, while Monte Carlo uses stochastic Langevin dynamics for escape from local minima.",
                "SPRING_K": "Attractive spring constant controlling the magnitude of hookian tension pulling connected node pairs closer together.\nLarger values pull highly similar sequences into tighter, more compact clusters, which increases local network density.",
                "COULOMB_K": "Repulsive constant controlling the electrostatic-like force pushing all nodes away from each other.\nLarger values push unrelated nodes and clusters apart, increasing separation distance between distinct sequence families.",
                "COULOMB_CUTOFF": "Maximum distance threshold past which the repulsive force between unrelated nodes drops off completely to zero.\nLowering this cutoff speeds up calculation and prevents distant clusters from exerting unnecessary forces on each other.",
                "DAMPING": "Frictional resistance coefficient applied to node velocities in the physics simulation to dissipate kinetic energy.\nValues near 1.0 allow smooth movement; values below 0.8 quickly freeze nodes, preventing oscillations and stabilizing the layout.",
                "DT": "Timestep size for each numerical integration step of the physics simulation.\nSmaller timesteps increase layout calculation precision and stability, whereas larger timesteps speed up convergence but may cause erratic jitter.",
                "MAX_STEPS": "The maximum number of physics iterations the simulation engine will run before forcing termination.\nEnsure this value is large enough to allow the network layout to settle and converge to a stable configuration.",
                "RMSD_THRESHOLD": "Root Mean Square Deviation stopping threshold for early termination of the physics layout calculation.\nIf the average displacement of nodes between consecutive steps falls below this value, the simulation halts as converged.",
                "PERCENTAGE_DROP_THRESHOLD": "Early termination criteria based on the percentage change of the moving average RMSD over the window size.\nIf the rate of layout change drops below this percentage (representing a plateau), the simulation terminates. Set to 0 to disable.",
                "RMSD_WINDOW": "The number of simulation steps over which the moving average RMSD is calculated for plateau and convergence detection.\nLarger windows smooth out transient spikes in node velocities, ensuring that early termination is only triggered on true convergence.",
                "ENABLE_PROGRESSIVE_SIMULATION": "Gradually lowers the similarity threshold in stages for massive connected components to prevent massive grid-lock.\nHelps resolve fine-grained sub-clusters in large, dense components. Disable if layout fails to converge.",
                "PACKING_GEOMETRY": "Sets the boundary geometry for cluster layout packing. Options include Square and Circle.",
                "PACKING_GRID_SIZE": "The base size of one grid square in the macro-grid layout packing.\nControls spacing and size scaling of packed independent components in the final network visualization.",
                "SGLD_MIN_K": "Minimum number of nearest neighbors (K) to retain for each node during Monte Carlo / SGLD physics simulation.\nPrevents nodes in small or disconnected clusters from collapsing onto each other by maintaining a baseline neighborhood.",
                "SGLD_K_PERCENT": "Fraction of total nodes used to compute the dynamic neighborhood size (K) for each node in Monte Carlo mode.\nSpecifically, K is set to max(SGLD_MIN_K, Fraction * total_nodes). Higher fractions preserve global structure but increase memory usage.",
                "SGLD_START_TEMP": "Starting temperature for the Simulated Annealing schedule in Monte Carlo / SGLD mode.\nControls the initial stochastic thermal noise; higher temperatures allow nodes to escape local energy minima and resolve gridlocks.",
                "SGLD_NOISE_SCALE": "Scaling factor for the stochastic Brownian noise term added to the node velocities in the SGLD simulation.\nLarger noise scales introduce more thermal random fluctuations, which can be adjusted to prevent premature layout freezing.",
                "UMAP_MODE": "Enables the Uniform Manifold Approximation and Projection (UMAP) dimensionality reduction algorithm instead of a physics engine.\nUMAP is highly optimized for projecting complex high-dimensional sequence embeddings down to 2D coordinates.",
                "UMAP_NEIGHBORS": "Size of the local neighborhood (k) used by the UMAP algorithm to learn the manifold structure of the sequence data.\nSmaller values preserve local sub-clusters, whereas larger values capture the global topological relationships between clusters.",
                "UMAP_MIN_DIST": "Controls how tightly UMAP packs points together in the low-dimensional projection space (ranging from 0.0 to 1.0).\nLower values result in extremely tight, dense point clouds; larger values distribute points more evenly.",
                "FASTA_DIR": "Directory containing the input FASTA files for sequence sets and subsets.\nFiles in this directory populate the Sequence Set dropdown in the Inputs tab.",
                "MSA_DIR": "Directory containing multiple sequence alignment files (Fasta format, Sparse alignment pickle, or HDF5 alignment matrices).\nFiles in this directory populate the MSA dropdown in the Inputs tab.",
                "HDF5_DIR": "Directory containing HDF5 files of pairwise sequence similarity scores and network edge coordinates.\nFiles in this directory populate the Network Edges dropdown in the Inputs tab.",
                "SAVED_LAYOUT_DIR": "Directory where calculated 2D layout coordinate files and network metadata (.h5 format) are saved and loaded from.\nThis serves as the layout cache to avoid recalculating coordinate layouts when re-opening a network.",
                "METADATA_DIR": "Directory where uploaded node metadata spreadsheet and CSV files are stored and retrieved from.",
                "PRINT_SAVE_DIR": "Directory where high-resolution image snapshots (PDF, PNG, SVG) of the sequence similarity networks are exported.\nEnsure this path is writable and has sufficient disk space for vector graphic output.",
                "FASTA_SPLIT_DIR": "Directory where dynamically split or extracted sequence subset FASTA files are temporarily stored.\nUsed by clustering tools when analyzing specific sub-clusters or regions of the network.",
                "CLUSTER_LABEL_DIR": "Directory where exported cluster metadata, sequence IDs, and automatically generated label descriptions are saved.\nUseful for downstream annotation pipelines or manual network inspection.",
                "HEADER_LIST_DIR": "Directory containing text files with lists of sequence headers matching specific network filtering or query criteria.\nUsed to keep track of interesting sequence cohorts identified in the visualizer.",
                "LOGO_DIR": "Directory where exported sequence logos (representing positional consensus sequence conservation) are saved.\nTypically saved in PNG or vector format for research publication and presentation.",
                "STRUCTURES_DIR": "Directory where ESMFold predicted 3D structures and PDB files are stored.",
                "TARGET_CACHE_FILE": "Selects a specific saved network coordinate cache file from the target directory to load into the visualizer.\nAllows restoring previous layout configurations or comparing different layout iterations directly.",
                "NEW_CACHE_NAME": "Specifies a custom filename when saving a new layout configuration iteration.\nOnly editable when the 'Selected Cache File' dropdown is set to '(New Layout Cache)'."
            }
            
            self.tip_db = {}
            for key, tip in self.tip_db_keys.items():
                if key in self.labels:
                    lbl = self.labels[key]
                    self.tip_db[lbl] = tip
                    lbl.installEventFilter(self)
                
                if key in self.inputs:
                    widget = self.inputs[key]
                    self.tip_db[widget] = tip
                    widget.installEventFilter(self)
                    
            for key, tip in self.tip_db_keys.items():
                if key in self.inputs:
                    widget = self.inputs[key]
                    parent = widget.parentWidget()
                    if parent and parent.objectName() == "wrapper":
                        for child in parent.children():
                            if child.isWidgetType() and child not in self.tip_db:
                                self.tip_db[child] = tip
                                child.installEventFilter(self)
        
        def _toggle_new_cache_input(self, text):
            if text == "(New Layout Cache)":
                self.line_new_cache.setEnabled(True)
            else:
                self.line_new_cache.setEnabled(False)

        def refresh_combo(self, combo, dir_key, ext_list):
            import os  # Moved here to ensure it's loaded before use
            if dir_key not in self.inputs: 
                return
            combo.blockSignals(True)
            current = combo.currentText()
            combo.clear()
            combo.addItem("")
            dir_path = self.inputs[dir_key].text()
            
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if any(f.endswith(ext) for ext in ext_list)]
                combo.addItems(files)
                if current in files:
                    combo.setCurrentText(current)
            combo.blockSignals(False)
            self.update_live_validators()

        def create_inputs_tab(self):
            tab = QWidget()
            layout = QFormLayout(tab)
            layout.setHorizontalSpacing(30)
            layout.setVerticalSpacing(10)
            
            def add_row(key, label_text, widget):
                lbl = QLabel(label_text)
                layout.addRow(lbl, widget)
                self.labels[key] = lbl
                self.inputs[key] = widget

            # Helper for adding a combobox + dynamic folder button
            def add_row_with_dynamic_btn(key, label_text, combo, dir_key, default_dir):
                container = QWidget()
                container.setObjectName("wrapper")
                h_lay = QHBoxLayout(container)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                btn = QPushButton("📂")
                btn.setFixedWidth(30)
                btn.setToolTip("Open Folder")
                
                def open_folder(checked):
                    import os
                    # Read from the directory input if it exists, otherwise use the default global
                    path = self.inputs[dir_key].text() if dir_key in self.inputs else globals().get(dir_key, default_dir)
                    abs_path = os.path.abspath(path)
                    os.makedirs(abs_path, exist_ok=True)
                    from PyQt6.QtGui import QDesktopServices
                    from PyQt6.QtCore import QUrl
                    QDesktopServices.openUrl(QUrl.fromLocalFile(abs_path))
                    
                btn.clicked.connect(open_folder)
                
                h_lay.addWidget(combo)
                h_lay.addWidget(btn)
                
                lbl = QLabel(label_text)
                layout.addRow(lbl, container)
                self.labels[key] = lbl
                self.inputs[key] = combo 
            
            # --- Sequence Set Input ---
            # Use DynamicComboBox and bind the refresh function directly to the click event!
            self.cb_fasta = DynamicComboBox(lambda: self.refresh_combo(self.cb_fasta, "FASTA_DIR", ['.fasta']))
            seq_dir = globals().get("FASTA_DIR", os.path.join("Input_Files", "Sequence_Sets"))
            fasta_files = [f for f in os.listdir(seq_dir) if f.endswith('.fasta')] if os.path.exists(seq_dir) else []
            self.cb_fasta.addItems([""] + fasta_files)
            fasta_val = globals().get("NODE_FASTA_FILE") or ""
            if os.path.basename(fasta_val) in fasta_files:
                self.cb_fasta.setCurrentText(os.path.basename(fasta_val))
            add_row_with_dynamic_btn("NODE_FASTA_FILE", "Sequence Set / Subset (.fasta):", self.cb_fasta, "FASTA_DIR", seq_dir)
            
            # --- MSA Input ---
            self.cb_msa = DynamicComboBox(lambda: self.refresh_combo(self.cb_msa, "MSA_DIR", ['.fasta', '.h5']))
            msa_dir_path = globals().get("MSA_DIR", os.path.join("Input_Files", "Multiple_Alignments"))
            msa_files = [f for f in os.listdir(msa_dir_path) if f.endswith('.fasta') or f.endswith('.h5')] if os.path.exists(msa_dir_path) else []
            self.cb_msa.addItems([""] + msa_files)
            msa_val = globals().get("MSA_FILE") or ""
            if os.path.basename(msa_val) in msa_files:
                self.cb_msa.setCurrentText(os.path.basename(msa_val))
            add_row_with_dynamic_btn("MSA_FILE", "MSA Input (.fasta / _sparse.h5):", self.cb_msa, "MSA_DIR", msa_dir_path)

            # --- HDF5 Input ---
            self.cb_hdf5 = DynamicComboBox(lambda: self.refresh_combo(self.cb_hdf5, "HDF5_DIR", ['.h5']))
            hdf5_dir = globals().get("HDF5_DIR", os.path.join("Input_Files", "Networks_EValues"))
            hdf5_files = [f for f in os.listdir(hdf5_dir) if f.endswith('.h5')] if os.path.exists(hdf5_dir) else []
            self.cb_hdf5.addItems([""] + hdf5_files)
            hdf5_val = globals().get("INPUT_HDF5") or ""
            if os.path.basename(hdf5_val) in hdf5_files:
                self.cb_hdf5.setCurrentText(os.path.basename(hdf5_val))
            add_row_with_dynamic_btn("INPUT_HDF5", "Network Edges Input (.h5):", self.cb_hdf5, "HDF5_DIR", hdf5_dir)
            
            # --- Rest of Inputs ---
            # Use NoScrollComboBox here to prevent accidental scroll wheel changes
            self.cb_score_mode = NoScrollComboBox()
            self.cb_score_mode.addItems(["global", "local"])
            self.cb_score_mode.setCurrentText(str(globals().get("ALIGNMENT_SCORE", "global")))
            add_row("ALIGNMENT_SCORE", "Alignment Score Mode:", self.cb_score_mode)
            
            self.cb_norm_mode = NoScrollComboBox()
            add_row("NORM_MODE", "Normalization Mode:", self.cb_norm_mode)
            
            self.cb_score_mode.currentTextChanged.connect(self.update_norm_mode_options)
            self.update_norm_mode_options()
            
            initial_norm = str(globals().get("NORM_MODE", "alignment_length"))
            if self.cb_score_mode.currentText() == "local" and initial_norm == "alignment_length":
                initial_norm = "longer_sequence"
            self.cb_norm_mode.setCurrentText(initial_norm)
            
            ref_val = globals().get("ALIGNMENT_REFERENCE", "")
            self.line_ref = QLineEdit("" if ref_val in [None, "None"] else str(ref_val))

            ref_container = QWidget()
            ref_container.setObjectName("alignment_reference_wrapper")
            ref_layout = QHBoxLayout(ref_container)
            ref_layout.setContentsMargins(0, 0, 0, 0)
            ref_layout.addWidget(self.line_ref, 1)

            self.lbl_alignment_offset = QLabel("Alignment Offset:")
            self.spin_alignment_offset = QSpinBox()
            self.spin_alignment_offset.setRange(-1000000, 1000000)
            try:
                offset_value = int(globals().get("ALIGNMENT_OFFSET", 0) or 0)
            except (TypeError, ValueError):
                offset_value = 0
            self.spin_alignment_offset.setValue(offset_value)
            self.spin_alignment_offset.setAccelerated(True)
            self.spin_alignment_offset.setFixedWidth(100)
            self.spin_alignment_offset.setStyleSheet(
                "QSpinBox:disabled { background-color: #f0f0f0; color: #888; }"
            )
            ref_layout.addWidget(self.lbl_alignment_offset)
            ref_layout.addWidget(self.spin_alignment_offset)

            ref_label = QLabel("Alignment Reference ID:")
            layout.addRow(ref_label, ref_container)
            self.labels["ALIGNMENT_REFERENCE"] = ref_label
            self.inputs["ALIGNMENT_REFERENCE"] = self.line_ref
            self.labels["ALIGNMENT_OFFSET"] = self.lbl_alignment_offset
            self.inputs["ALIGNMENT_OFFSET"] = self.spin_alignment_offset
            
            # --- UMAP Controls ---
            umap_container = QWidget()
            umap_container.setObjectName("wrapper")
            umap_layout = QHBoxLayout(umap_container)
            umap_layout.setContentsMargins(0, 0, 0, 0)
            
            self.check_umap = QPushButton()
            self.check_umap.setCheckable(True)
            self.check_umap.setFixedSize(60, 28)
            def switch_umap_style(checked, btn=self.check_umap):
                if checked:
                    btn.setText("ON")
                    btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 14px; font-weight: bold; border: 1px solid #388E3C; }")
                else:
                    btn.setText("OFF")
                    btn.setStyleSheet("QPushButton { background-color: #e0e0e0; color: #333; border-radius: 14px; font-weight: bold; border: 1px solid #bdbdbd; }")
            self.check_umap.toggled.connect(switch_umap_style)
            
            umap_mode_val = globals().get("UMAP_MODE", False)
            if isinstance(umap_mode_val, str):
                umap_mode_val = umap_mode_val.lower() in ['true', '1', 't', 'y', 'yes']
            self.check_umap.setChecked(bool(umap_mode_val))
            switch_umap_style(bool(umap_mode_val))
            
            from PyQt6.QtWidgets import QSizePolicy
            lbl_k = QLabel("   UMAP Nearest Neighbors (k):")
            lbl_md = QLabel("   UMAP Minimum Distance:")
            
            self.spin_umap_k = NoScrollSpinBox()
            self.spin_umap_k.setRange(2, 500)
            self.spin_umap_k.setValue(int(globals().get("UMAP_NEIGHBORS") or 15))
            self.spin_umap_k.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            
            self.spin_umap_md = NoScrollDoubleSpinBox()
            self.spin_umap_md.setRange(0.0, 1.0)
            self.spin_umap_md.setSingleStep(0.1)
            self.spin_umap_md.setDecimals(2)
            self.spin_umap_md.setValue(float(globals().get("UMAP_MIN_DIST") or 0.1))
            self.spin_umap_md.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            # Match the taller spinboxes used beside sliders on the Visuals/Physics tabs.
            input_spinbox_height = max(
                self.spin_umap_k.sizeHint().height(),
                self.spin_umap_md.sizeHint().height(),
            )
            for spinbox in (
                self.spin_alignment_offset,
                self.spin_umap_k,
                self.spin_umap_md,
            ):
                spinbox.setFixedHeight(input_spinbox_height)
            
            # Apply styling for disabled states (grayed out)
            disabled_spinbox_style = "QSpinBox:disabled, QDoubleSpinBox:disabled { background-color: #f0f0f0; color: #888; }"
            disabled_label_style = "QLabel:disabled { color: #888; }"
            self.spin_umap_k.setStyleSheet(disabled_spinbox_style)
            self.spin_umap_md.setStyleSheet(disabled_spinbox_style)
            lbl_k.setStyleSheet(disabled_label_style)
            lbl_md.setStyleSheet(disabled_label_style)
            
            # Set initial enabled/disabled states based on current toggle value
            umap_enabled = self.check_umap.isChecked()
            self.spin_umap_k.setEnabled(umap_enabled)
            self.spin_umap_md.setEnabled(umap_enabled)
            lbl_k.setEnabled(umap_enabled)
            lbl_md.setEnabled(umap_enabled)
            
            def toggle_umap(state):
                self.spin_umap_k.setEnabled(state)
                self.spin_umap_md.setEnabled(state)
                lbl_k.setEnabled(state)
                lbl_md.setEnabled(state)
                self.update_live_validators()
                
            self.check_umap.toggled.connect(toggle_umap)
            self.spin_umap_k.valueChanged.connect(self.update_live_validators)
            
            umap_layout.addWidget(self.check_umap)
            umap_layout.addWidget(lbl_k)
            umap_layout.addWidget(self.spin_umap_k)
            umap_layout.addWidget(lbl_md)
            umap_layout.addWidget(self.spin_umap_md)
            
            layout.addRow("Enable UMAP Layout:", umap_container)
            self.labels["UMAP_MODE"] = layout.labelForField(umap_container)
            self.inputs["UMAP_MODE"] = self.check_umap
            self.inputs["UMAP_NEIGHBORS"] = self.spin_umap_k
            self.inputs["UMAP_MIN_DIST"] = self.spin_umap_md
            self.labels["UMAP_NEIGHBORS"] = lbl_k
            self.labels["UMAP_MIN_DIST"] = lbl_md
            
            thresh_val = globals().get("SIMILARITY_THRESHOLD", "")
            self.line_thresh = QLineEdit("" if thresh_val in [None, "None"] else str(thresh_val))
            
            val_top = globals().get("TOP_EDGE_PERCENT", "")
            self.line_top = QLineEdit("" if val_top in [None, "None"] else str(val_top))
            self.line_top.setPlaceholderText("Overrides Similarity Threshold (e.g. 1.0)")
            self.line_top.textChanged.connect(self.update_live_validators)
            
            thresh_container = QWidget()
            thresh_container.setObjectName("wrapper")
            thresh_layout = QHBoxLayout(thresh_container)
            thresh_layout.setContentsMargins(0, 0, 0, 0)
            
            lbl_top = QLabel("   Top Edge % (Optional):")
            
            thresh_layout.addWidget(self.line_thresh)
            thresh_layout.addWidget(lbl_top)
            thresh_layout.addWidget(self.line_top)
            
            lbl_thresh = QLabel("Similarity Threshold:")
            layout.addRow(lbl_thresh, thresh_container)
            
            self.labels["SIMILARITY_THRESHOLD"] = lbl_thresh
            self.inputs["SIMILARITY_THRESHOLD"] = self.line_thresh
            
            self.labels["TOP_EDGE_PERCENT"] = lbl_top
            self.inputs["TOP_EDGE_PERCENT"] = self.line_top
            
            min_occ_val = globals().get("FILTER_MIN_OCCUPANCY", "")
            self.line_min_occ = QLineEdit("" if min_occ_val in [None, "None"] else str(min_occ_val))
            add_row("FILTER_MIN_OCCUPANCY", "Filter Min Occupancy %:", self.line_min_occ)
            
            # Horizontal layout for both statistics and histogram buttons
            btn_container = QWidget()
            btn_lay = QHBoxLayout(btn_container)
            btn_lay.setContentsMargins(0, 0, 0, 0)
            
            self.btn_stats = QPushButton("Compute Network Statistics")
            self.btn_stats.setStyleSheet("background-color: #2196F3; color: white;")
            self.btn_stats.clicked.connect(self.run_statistics)
            
            self.btn_hist = QPushButton("Histogram")
            self.btn_hist.setStyleSheet("background-color: #9C27B0; color: white;")
            self.btn_hist.clicked.connect(self.run_histogram)
            
            btn_lay.addWidget(self.btn_stats, 3)
            btn_lay.addWidget(self.btn_hist, 1)
            layout.addRow("", btn_container)
            
            # --- Target Cache Tracker & Folder Button ---
            cache_container = QWidget()
            cache_lay = QHBoxLayout(cache_container)
            cache_lay.setContentsMargins(0, 0, 0, 0)
            
            self.lbl_cache_tracker = QLabel("Target Folder: None")
            self.lbl_cache_tracker.setStyleSheet("color: gray;")
            self.lbl_cache_tracker.setWordWrap(True)
            
            btn_open_cache = QPushButton("📂")
            btn_open_cache.setFixedWidth(30)
            btn_open_cache.setToolTip("Open Target Cache Folder")
            
            def open_cache_folder(checked):
                import os
                from PyQt6.QtGui import QDesktopServices
                from PyQt6.QtCore import QUrl
                
                # Exclusively open the parent directory
                dir_input = self.inputs.get("SAVED_LAYOUT_DIR")
                path = dir_input.text() if dir_input else globals().get("SAVED_LAYOUT_DIR", os.path.join("Cache_Files", "Saved_Layouts"))
                
                abs_path = os.path.abspath(path)
                os.makedirs(abs_path, exist_ok=True)
                QDesktopServices.openUrl(QUrl.fromLocalFile(abs_path))
                
            btn_open_cache.clicked.connect(open_cache_folder)
            
            cache_lay.addWidget(self.lbl_cache_tracker)
            cache_lay.addWidget(btn_open_cache)
            layout.addRow("", cache_container)
            
            # ---> NEW: Cache File Dropdown & Specific Folder Button <---
            target_container = QWidget()
            target_container.setObjectName("wrapper")
            target_lay = QHBoxLayout(target_container)
            target_lay.setContentsMargins(0, 0, 0, 0)

            self.cb_cache_file = NoScrollComboBox()
            self.cb_cache_file.setEnabled(False)
            
            self.btn_open_target_folder = QPushButton("📂")
            self.btn_open_target_folder.setFixedWidth(30)
            self.btn_open_target_folder.setToolTip("Open Specific Target Folder")
            self.btn_open_target_folder.setEnabled(False)  # Greyed out by default
            
            def open_target_folder(checked):
                import os
                from PyQt6.QtGui import QDesktopServices
                from PyQt6.QtCore import QUrl
                
                path = getattr(self, 'current_cache_folder', None)
                if path and os.path.exists(path):
                    abs_path = os.path.abspath(path)
                    QDesktopServices.openUrl(QUrl.fromLocalFile(abs_path))
                    
            self.btn_open_target_folder.clicked.connect(open_target_folder)
            
            target_lay.addWidget(self.cb_cache_file)
            target_lay.addWidget(self.btn_open_target_folder)
            
            layout.addRow("Selected Cache File:", target_container)
            self.labels["TARGET_CACHE_FILE"] = layout.labelForField(target_container)
            self.inputs["TARGET_CACHE_FILE"] = self.cb_cache_file
            
            # ---> NEW: Custom Cache Name Box <---
            self.line_new_cache = QLineEdit()
            self.line_new_cache.setEnabled(False)
            self.line_new_cache.setStyleSheet("QLineEdit:disabled { background-color: #f0f0f0; color: #888; }")
            lbl_new_cache = QLabel("New Cache Name:")
            layout.addRow(lbl_new_cache, self.line_new_cache)
            self.labels["NEW_CACHE_NAME"] = lbl_new_cache
            self.inputs["NEW_CACHE_NAME"] = self.line_new_cache
            
            # Hook up the toggle switch
            self.cb_cache_file.currentTextChanged.connect(self._toggle_new_cache_input)

            self.tabs.addTab(tab, "Inputs && Outputs")
            
        def update_norm_mode_options(self):
            if not hasattr(self, 'cb_score_mode') or not hasattr(self, 'cb_norm_mode'):
                return
            current_norm = self.cb_norm_mode.currentText()
            self.cb_norm_mode.blockSignals(True)
            self.cb_norm_mode.clear()
            
            is_local = self.cb_score_mode.currentText() == "local"
            if is_local:
                self.cb_norm_mode.addItems(["shorter_sequence", "longer_sequence", "average_sequence"])
                if current_norm == "alignment_length":
                    current_norm = "longer_sequence"
            else:
                self.cb_norm_mode.addItems(["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"])
                
            self.cb_norm_mode.setCurrentText(current_norm)
            self.cb_norm_mode.blockSignals(False)

        def update_live_validators(self):
            has_fasta = bool(self.cb_fasta.currentText().strip())
            has_hdf5 = bool(self.cb_hdf5.currentText().strip())
            has_reference = bool(self.line_ref.text().strip())
            if hasattr(self, 'spin_alignment_offset'):
                self.spin_alignment_offset.setEnabled(has_reference)
            if hasattr(self, 'lbl_alignment_offset'):
                self.lbl_alignment_offset.setEnabled(has_reference)
            self.btn_stats.setEnabled(has_fasta and has_hdf5)
            if hasattr(self, 'btn_hist'):
                self.btn_hist.setEnabled(has_fasta and has_hdf5)
            
            is_umap = hasattr(self, 'check_umap') and self.check_umap.isChecked()
            
            if hasattr(self, 'line_thresh') and hasattr(self, 'line_top'):
                has_top_edge = bool(self.line_top.text().strip())
                self.line_thresh.setEnabled(not is_umap and not has_top_edge)
                
                if not self.line_thresh.isEnabled():
                    self.line_thresh.setStyleSheet("QLineEdit:disabled { background-color: #f0f0f0; color: #888; }")
                else:
                    self.line_thresh.setStyleSheet("")
                    
                self.line_top.setEnabled(not is_umap)
                
            if hasattr(self, 'tabs') and self.tabs.count() > 2:
                self.tabs.setTabEnabled(2, not is_umap)
            
            if hasattr(self, 'btn_check'):
                self.btn_check.setEnabled(has_fasta and has_hdf5)
            
            # --- STABILIZE LAYOUT (SINGLE LINE MODE) ---
            # Turning off WordWrap forces PyQt to keep it strictly on one line.
            # It will gracefully truncate at the edge of the window instead of jumping.
            self.lbl_cache_tracker.setWordWrap(False) 
            self.lbl_cache_tracker.setMinimumHeight(0) # Clear any previous manual heights
            self.lbl_cache_tracker.setMaximumHeight(30) # Prevent vertical expansion
            
            if not has_hdf5:
                self.lbl_cache_tracker.setText("Target Cache: Missing HDF5")
                self.lbl_cache_tracker.setStyleSheet("color: gray;")
                return
                
            from SSN_Utils import simplify_node_label
            import re
            
            # --- 1. FATAL OS CHARACTER CHECK ---
            raw_ref = self.line_ref.text().strip()
            prohibited_pattern = r'[\\/:*?"<>|]'
            
            if re.search(prohibited_pattern, raw_ref):
                # Condensed error message to fit beautifully on one line
                self.lbl_cache_tracker.setText("Error: Invalid OS characters in Alignment Reference")
                self.lbl_cache_tracker.setStyleSheet("color: #d32f2f; font-weight: bold;")
                
                # Sync Dropdown
                self.cb_cache_file.blockSignals(True)
                self.cb_cache_file.clear()
                self.cb_cache_file.setEnabled(False)
                self.cb_cache_file.addItem("Folder does not exist")
                self.cb_cache_file.blockSignals(False)
                
                self.btn_open_target_folder.setEnabled(False)
                return
            # ------------------------------------

            fasta_base = self.cb_fasta.currentText()
            if fasta_base: fasta_base = os.path.splitext(fasta_base)[0]
            else: fasta_base = "Network"
                
            hdf5_base = self.cb_hdf5.currentText()
            
            # Resolve Model String
            match = re.search(r'(\[.*?\])', hdf5_base)
            if match:
                model_str = f"_{match.group(1)}"
            else:
                hdf5_no_ext = hdf5_base[:-3] if hdf5_base.endswith(".h5") else os.path.splitext(hdf5_base)[0]
                stripped = re.sub(r'_(network|evalue)$', '', hdf5_no_ext, flags=re.IGNORECASE)
                old_match = re.search(r'_(e[0-9]+_.*|blast.*)$', stripped, flags=re.IGNORECASE)
                model_str = f"_{old_match.group(1)}" if old_match else ""
                
            net_prefix = f"{fasta_base}{model_str}"
            is_blast = "blast" in hdf5_base.lower()
            
            # Update Score/Norm Toggles
            self.cb_score_mode.setEnabled(not is_blast)
            self.cb_norm_mode.setEnabled(not is_blast)
            
            # ---> NEW: Clear selections for blast, restore defaults for others <---
            if is_blast:
                self.cb_score_mode.setCurrentIndex(-1)
                self.cb_norm_mode.setCurrentIndex(-1)
            else:
                # Restore sensible defaults if switching back from a blast network
                if self.cb_score_mode.currentIndex() == -1:
                    self.cb_score_mode.setCurrentText("global")
                if self.cb_norm_mode.currentIndex() == -1:
                    self.cb_norm_mode.setCurrentText("alignment_length")
                    
            suffix = ""
            if not is_blast:
                norm_m = self.cb_norm_mode.currentText()
                if norm_m: suffix += f"_{norm_m}"
                score_m = self.cb_score_mode.currentText()
                if score_m: suffix += f"_{score_m}"
            
            if is_umap:
                suffix += f"_UMAP_k{self.spin_umap_k.value()}"
            else:
                top_val = self.line_top.text().strip()
                if top_val and top_val != "None":
                    try: suffix += f"_Top{float(top_val)}Pct"
                    except: pass
                else:
                    try: suffix += f"_Score{float(self.line_thresh.text().strip())}"
                    except: pass
                
            # --- 2. FOLDER & HDF5 CHECKING ---
            cache_file = f"{net_prefix}{suffix}.h5"
            folder_name = os.path.splitext(cache_file)[0]
            cache_folder = os.path.join("Cache_Files", "Saved_Layouts", folder_name)
            self.current_cache_folder = cache_folder
            
            self.cb_cache_file.blockSignals(True)
            self.cb_cache_file.clear()
            
            # Calculate default next cache name
            max_ver = -1
            if os.path.exists(cache_folder):
                 for f in os.listdir(cache_folder):
                     if f.startswith(f"{folder_name}_ver.") and f.endswith(".h5"):
                         match = re.search(r'_ver\.(\d+)\.h5$', f)
                         if match:
                             max_ver = max(max_ver, int(match.group(1)))
                             
            next_ver = max_ver + 1
            default_new_name = f"{folder_name}_ver.{next_ver:02d}.h5"
            self.line_new_cache.setPlaceholderText(default_new_name)
            self.line_new_cache.setText("") # Clear previous user input
            
            self.cb_cache_file.setEnabled(True) # Always enable the dropdown now
            
            if os.path.exists(cache_folder):
                 self.lbl_cache_tracker.setText(f"Target Folder: {folder_name} [✅ Exists]")
                 self.lbl_cache_tracker.setStyleSheet("color: green; font-weight: bold;")
                 self.btn_open_target_folder.setEnabled(True)
                 
                 h5_files = [f for f in os.listdir(cache_folder) if f.endswith(".h5")]
                 if h5_files:
                     h5_files.sort(key=lambda x: os.path.getmtime(os.path.join(cache_folder, x)), reverse=True)
                     self.cb_cache_file.addItems(h5_files)
            else:
                 self.lbl_cache_tracker.setText(f"Target Folder: {folder_name} [❌ Needs Computing]")
                 self.lbl_cache_tracker.setStyleSheet("color: #d32f2f;")
                 self.btn_open_target_folder.setEnabled(False)
                 
            # Always append the 'New Layout' option at the very bottom
            self.cb_cache_file.addItem("(New Layout Cache)")
                 
            # Auto-select newest cache (Index 0), or 'New Layout Cache' if the folder is empty
            self.cb_cache_file.setCurrentIndex(0)
            self.cb_cache_file.blockSignals(False)
            
            # Force the UI to evaluate the correct grey-out state on refresh
            self._toggle_new_cache_input(self.cb_cache_file.currentText())

        def run_statistics(self):
            import h5py
            import numpy as np
            import math
            import sys
            
            fasta_path = os.path.join(self.inputs["FASTA_DIR"].text(), self.cb_fasta.currentText())
            hdf5_path = os.path.join(self.inputs["HDF5_DIR"].text(), self.cb_hdf5.currentText())
            
            self.stat_display.setText("Computing... This may take a moment for large HDF5 networks.")
            QApplication.processEvents()
            
            try:
                is_blast = "EValue" in os.path.basename(hdf5_path) or "Evalue" in os.path.basename(hdf5_path)
                score_mode = self.cb_score_mode.currentText()
                norm_mode = self.cb_norm_mode.currentText()
                
                with h5py.File(hdf5_path, "r") as hf:
                    from Bio import SeqIO
                    kept_mask = None
                    if fasta_path and os.path.exists(fasta_path):
                        fasta_ids = set()
                        fasta_headers = set()
                        for rec in SeqIO.parse(fasta_path, "fasta"):
                            fasta_ids.add(rec.id)
                            fasta_headers.add(rec.description)
                            
                        raw_headers = hf['headers'][:]
                        headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
                        
                        net_headers_set = set(headers)
                        net_id_set = {h.split()[0] for h in headers}
                        missing_nodes = [hid for hid in fasta_ids if hid not in net_id_set and hid not in net_headers_set]
                        if missing_nodes:
                            raise ValueError(f"FASTA file is NOT a strict subset of the network file. {len(missing_nodes)} sequences are missing from the network.")
                        
                        valid_indices = []
                        for i, h in enumerate(headers):
                            rec_id = h.split()[0]
                            if h in fasta_headers or rec_id in fasta_ids:
                                valid_indices.append(i)
                                
                        if len(valid_indices) < len(headers):
                            kept_mask = np.zeros(len(headers), dtype=bool)
                            kept_mask[valid_indices] = True
                    
                    if is_blast:
                        raw_scores = hf['score'][:]
                        sources = hf['i'][:]
                        targets = hf['j'][:]
                    else:
                        sources = hf['i'][:].astype(np.int64)
                        targets = hf['j'][:].astype(np.int64)
                        if score_mode == "local":
                            raw_scores = hf['l_score'][:].astype(np.float32)
                            align_lens = hf['l_len'][:].astype(np.float32)
                        else:
                            raw_scores = hf['g_score'][:].astype(np.float32)
                            align_lens = hf['g_len'][:].astype(np.float32)
                            
                        if 'seq_lens' in hf:
                            seq_lens = hf['seq_lens'][:]
                        else:
                            seq_lens = np.ones(np.max([np.max(sources), np.max(targets)]) + 1)
                        
                    if kept_mask is not None:
                        valid_edges_mask = kept_mask[sources] & kept_mask[targets]
                        raw_scores = raw_scores[valid_edges_mask]
                        sources = sources[valid_edges_mask]
                        targets = targets[valid_edges_mask]
                        if not is_blast:
                            align_lens = align_lens[valid_edges_mask]
                            
                    if is_blast:
                        scores = raw_scores.astype(np.float32)
                    else:
                        epsilon = 1e-6
                        if norm_mode == "alignment_length":
                            denom = align_lens
                        else:
                            len_src = seq_lens[sources].astype(np.float32)
                            len_dst = seq_lens[targets].astype(np.float32)
                            if norm_mode == "shorter_sequence": denom = np.minimum(len_src, len_dst)
                            elif norm_mode == "longer_sequence": denom = np.maximum(len_src, len_dst)
                            elif norm_mode == "average_sequence": denom = (len_src + len_dst) / 2.0
                            else: denom = align_lens
                            
                        denom = np.maximum(denom, epsilon)
                        scores = (raw_scores / denom).astype(np.float32)
                
                if len(scores) == 0:
                    self.stat_display.setText("Warning: No valid edges found in the selected Fasta subset.")
                    return
                    
                max_score = np.max(scores)
                min_score = np.min(scores)
                avg_score = np.mean(scores)
                
                sorted_scores = np.sort(scores)
                stored_edges = len(sorted_scores)
                
                total_nodes = np.sum(kept_mask) if kept_mask is not None else len(headers)
                theoretical_max_edges = (total_nodes * (total_nodes - 1)) / 2.0
                
                if is_blast:
                    start_val = int(math.floor(min_score))
                    limit_step_1 = min(100, int(math.ceil(max_score)))
                    thresh_low = np.arange(start_val, limit_step_1 + 1, 1)
                    if max_score > 100:
                        thresh_high = np.arange(105, int(math.ceil(max_score)) + 5, 5)
                        thresholds = np.concatenate([thresh_low, thresh_high])
                    else:
                        thresholds = thresh_low
                else:
                    start_val = math.floor(min_score * 10) / 10.0
                    thresholds = np.arange(start_val, max_score + 0.1, 0.1)
                    
                indices = np.searchsorted(sorted_scores, thresholds, side='left')
                counts = stored_edges - indices
                
                import re
                hdf5_base_stat = os.path.basename(hdf5_path)
                match_stat = re.search(r'(\[.*?\])', hdf5_base_stat)
                if match_stat:
                    model_name = match_stat.group(1)
                else:
                    hdf5_no_ext_stat = hdf5_base_stat[:-3] if hdf5_base_stat.endswith(".h5") else os.path.splitext(hdf5_base_stat)[0]
                    stripped_stat = re.sub(r'_(network|evalue)$', '', hdf5_no_ext_stat, flags=re.IGNORECASE)
                    old_match = re.search(r'_(e[0-9]+_.*|blast.*)$', stripped_stat, flags=re.IGNORECASE)
                    model_name = old_match.group(1) if old_match else hdf5_base_stat
                
                lines = []
                lines.append(f"====== Network Statistics ======")
                lines.append(f"Network Model: {model_name}")
                lines.append(f"Fasta Node Subset: {os.path.basename(fasta_path)}")
                lines.append(f"Total Nodes Processed: {total_nodes}")
                display_norm = norm_mode.replace('_', ' ').title()
                lines.append(f"Metric: {'Log10(E-Value)' if is_blast else f'{score_mode.title()} Alignment Score with {display_norm} Normalization'}")
                lines.append(f"Stored Edges: {stored_edges} (Max possible: {int(theoretical_max_edges)})")
                lines.append(f"Max: {max_score:.4f} | Min: {min_score:.4f} | Avg: {avg_score:.4f}")
                lines.append("-" * 40)
                lines.append(f"{'Threshold':<10} | {'Count':<10} | {'Percentage':<10}")
                lines.append("-" * 40)
                
                for thresh, count in zip(thresholds, counts):
                    pct = (count / theoretical_max_edges) * 100.0 if theoretical_max_edges > 0 else 0
                    if is_blast:
                         lines.append(f"{int(thresh):<10} | {count:<10} | {pct:<9.2f}%")
                    else:
                         lines.append(f"{thresh:<10.1f} | {count:<10} | {pct:<9.2f}%")
                         
                self.stat_display.setText("\n".join(lines))
                
            except Exception as e:
                self.stat_display.setText(f"Error during computation:\n{e}")

        def run_histogram(self):
            import h5py
            import numpy as np
            import math
            
            fasta_path = os.path.join(self.inputs["FASTA_DIR"].text(), self.cb_fasta.currentText())
            hdf5_path = os.path.join(self.inputs["HDF5_DIR"].text(), self.cb_hdf5.currentText())
            
            self.stat_display.setText("Computing score distribution... This may take a moment.")
            QApplication.processEvents()
            
            try:
                is_blast = "EValue" in os.path.basename(hdf5_path) or "Evalue" in os.path.basename(hdf5_path)
                score_mode = self.cb_score_mode.currentText()
                norm_mode = self.cb_norm_mode.currentText()
                
                with h5py.File(hdf5_path, "r") as hf:
                    from Bio import SeqIO
                    kept_mask = None
                    if fasta_path and os.path.exists(fasta_path):
                        fasta_ids = set()
                        fasta_headers = set()
                        for rec in SeqIO.parse(fasta_path, "fasta"):
                            fasta_ids.add(rec.id)
                            fasta_headers.add(rec.description)
                            
                        raw_headers = hf['headers'][:]
                        headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
                        
                        net_headers_set = set(headers)
                        net_id_set = {h.split()[0] for h in headers}
                        missing_nodes = [hid for hid in fasta_ids if hid not in net_id_set and hid not in net_headers_set]
                        if missing_nodes:
                            raise ValueError(f"FASTA file is NOT a strict subset of the network file. {len(missing_nodes)} sequences are missing from the network.")
                        
                        valid_indices = []
                        for i, h in enumerate(headers):
                            rec_id = h.split()[0]
                            if h in fasta_headers or rec_id in fasta_ids:
                                valid_indices.append(i)
                                
                        if len(valid_indices) < len(headers):
                            kept_mask = np.zeros(len(headers), dtype=bool)
                            kept_mask[valid_indices] = True
                    
                    if is_blast:
                        raw_scores = hf['score'][:]
                        sources = hf['i'][:]
                        targets = hf['j'][:]
                    else:
                        sources = hf['i'][:].astype(np.int64)
                        targets = hf['j'][:].astype(np.int64)
                        if score_mode == "local":
                            raw_scores = hf['l_score'][:].astype(np.float32)
                            align_lens = hf['l_len'][:].astype(np.float32)
                        else:
                            raw_scores = hf['g_score'][:].astype(np.float32)
                            align_lens = hf['g_len'][:].astype(np.float32)
                            
                        if 'seq_lens' in hf:
                            seq_lens = hf['seq_lens'][:]
                        else:
                            seq_lens = np.ones(np.max([np.max(sources), np.max(targets)]) + 1)
                        
                    if kept_mask is not None:
                        valid_edges_mask = kept_mask[sources] & kept_mask[targets]
                        raw_scores = raw_scores[valid_edges_mask]
                        sources = sources[valid_edges_mask]
                        targets = targets[valid_edges_mask]
                        if not is_blast:
                            align_lens = align_lens[valid_edges_mask]
                            
                    if is_blast:
                        scores = raw_scores.astype(np.float32)
                    else:
                        epsilon = 1e-6
                        if norm_mode == "alignment_length":
                            denom = align_lens
                        else:
                            len_src = seq_lens[sources].astype(np.float32)
                            len_dst = seq_lens[targets].astype(np.float32)
                            if norm_mode == "shorter_sequence": denom = np.minimum(len_src, len_dst)
                            elif norm_mode == "longer_sequence": denom = np.maximum(len_src, len_dst)
                            elif norm_mode == "average_sequence": denom = (len_src + len_dst) / 2.0
                            else: denom = align_lens
                            
                        denom = np.maximum(denom, epsilon)
                        scores = (raw_scores / denom).astype(np.float32)
                
                if len(scores) == 0:
                    self.stat_display.setText("Warning: No valid edges found in the selected Fasta subset.")
                    return
                
                # Determine threshold based on top edge % override
                is_umap = hasattr(self, 'check_umap') and self.check_umap.isChecked()
                top_percent_str = self.line_top.text().strip()
                try:
                    top_percent = float(top_percent_str) if top_percent_str else None
                except ValueError:
                    top_percent = None
                    
                if top_percent is not None and not is_umap:
                    total_active_nodes = np.sum(kept_mask) if kept_mask is not None else len(headers)
                    theoretical_max_edges = (total_active_nodes * (total_active_nodes - 1)) / 2.0
                    k = int(theoretical_max_edges * (top_percent / 100.0))
                    if len(scores) == 0:
                        threshold = 0.0
                    else:
                        k = max(1, min(k, len(scores)))
                        sorted_all = np.sort(scores)[::-1]
                        threshold = sorted_all[k - 1]
                else:
                    thresh_str = self.line_thresh.text().strip()
                    try:
                        threshold = float(thresh_str) if thresh_str else 0.0
                    except ValueError:
                        threshold = 0.0
                
                self.stat_display.setText("Displaying score histogram...")
                QApplication.processEvents()
                
                # Call the utility function to show the histogram
                from SSN_Utils import plot_score_histogram
                
                # Make sure to mock/set up the cfg configuration just in case it is read inside the function
                import SSN_Config as cfg_mod
                cfg_mod.INPUT_IS_EVALUE = is_blast
                cfg_mod.NORM_MODE = norm_mode
                
                plot_score_histogram(scores, threshold)
                
                self.stat_display.setText("Histogram displayed successfully.")
                
            except Exception as e:
                self.stat_display.setText(f"Error during histogram generation:\n{e}")

        def create_visuals_tab(self):
            tab = QWidget()
            main_layout = QVBoxLayout(tab)
            form_layout = QFormLayout()
            form_layout.setHorizontalSpacing(30)
            form_layout.setVerticalSpacing(10)
            
            # 1. Sliders Setup
            slider_settings = [
                {"key": "NODE_SIZE", "type": "int", "min": 1, "max": 20, "default": 10},
                {"key": "EDGE_WIDTH", "type": "float", "min": 0.1, "max": 3.0, "scale": 10.0, "decimals": 1, "default": 1.0},
                {"key": "NODE_BOUNDARY_WIDTH", "type": "float", "min": 0.0, "max": 2.0, "scale": 10.0, "decimals": 1, "default": 0.5},
                {"key": "EDGE_ALPHA", "type": "float", "min": 0.0, "max": 1.0, "scale": 100.0, "decimals": 2, "default": 0.1},
                {"key": "TEXT_SIZE", "type": "int", "min": 1, "max": 24, "default": 8}
            ]
            
            for s in slider_settings:
                key = s["key"]
                display_name = key.replace('_', ' ').title()
                if key == "EDGE_ALPHA": display_name = "Edge Opacity"
                
                val_raw = globals().get(key, s["default"])
                
                ui_element = QWidget()
                ui_element.setObjectName("wrapper")
                h_lay = QHBoxLayout(ui_element)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                sl = NoScrollSlider(Qt.Orientation.Horizontal)
                
                if s["type"] == "int":
                    try: val = int(val_raw)
                    except: val = s["default"]
                    
                    sl.setMinimum(s["min"])
                    sl.setMaximum(s["max"])
                    
                    box = NoScrollSpinBox()
                    box.setRange(-999999, 999999)
                    box.setFixedWidth(60)
                    
                    sl.setValue(val)
                    box.setValue(val)
                    
                    sl.valueChanged.connect(box.setValue)
                    box.valueChanged.connect(sl.setValue)
                    
                else: 
                    try: val = float(val_raw)
                    except: val = s["default"]
                    
                    sc = s["scale"]
                    sl.setMinimum(int(s["min"] * sc))
                    sl.setMaximum(int(s["max"] * sc))
                    
                    box = NoScrollDoubleSpinBox()
                    box.setRange(-999999.0, 999999.0)
                    box.setDecimals(s["decimals"])
                    box.setFixedWidth(70)
                    
                    sl.setValue(int(val * sc))
                    box.setValue(val)
                    
                    sl.valueChanged.connect(lambda v, b=box, scale=sc: b.setValue(v / scale))
                    box.valueChanged.connect(lambda v, s=sl, scale=sc: s.setValue(int(v * scale)))
                    
                h_lay.addWidget(sl)
                h_lay.addWidget(box)
                
                lbl = QLabel(f"{display_name}:")
                form_layout.addRow(lbl, ui_element)
                self.labels[key] = lbl
                self.inputs[key] = box
            
            # 2. Colors Setup
            color_keys = ["TEXT_COLOR", "INITIAL_NODE_COLOR", "HOVER_COLOR", "CONNECTED_NODE_COLOR", "EDGE_COLOR", "NODE_BOUNDARY_COLOR"]
            self.visual_defaults = {
                "NODE_SIZE": 10, "EDGE_WIDTH": 1.0, "NODE_BOUNDARY_WIDTH": 0.5, "EDGE_ALPHA": 0.1, "TEXT_SIZE": 8,
                "TEXT_COLOR": "grey", "INITIAL_NODE_COLOR": "#4488ff", "HOVER_COLOR": "#ffaa00", "CONNECTED_NODE_COLOR": "#ff0000",
                "EDGE_COLOR": "#000000", "NODE_BOUNDARY_COLOR": "#000000", "LOW_RESOURCE_MODE": False
            }
            
            for key in color_keys:
                if key == "HOVER_COLOR": display_name = "Hover and Selected Node Color"
                else: display_name = key.replace('_', ' ').title()
                
                color_container = QWidget()
                color_container.setObjectName("wrapper")
                h_layout = QHBoxLayout(color_container)
                h_layout.setContentsMargins(0, 0, 0, 0)
                
                val = globals().get(key, self.visual_defaults[key])
                
                swatch = QLabel()
                swatch.setFixedSize(20, 20)
                swatch.setStyleSheet(f"background-color: {val}; border: 1px solid gray; border-radius: 3px;")
                
                le = QLineEdit("" if val in [None, "None"] else str(val))
                
                btn = QPushButton("Pick")
                btn.setFixedWidth(50)
                
                def pick_color(checked, line_edit=le, color_swatch=swatch):
                    initial = line_edit.text()
                    from PyQt6.QtGui import QColor
                    color = QColorDialog.getColor(QColor(initial) if initial else QColor("white"), self, "Select Color")
                    if color.isValid():
                        hex_val = color.name()
                        line_edit.setText(hex_val)
                        color_swatch.setStyleSheet(f"background-color: {hex_val}; border: 1px solid gray; border-radius: 3px;")
                
                btn.clicked.connect(pick_color)
                
                h_layout.addWidget(swatch)
                h_layout.addWidget(le)
                h_layout.addWidget(btn)
                
                lbl = QLabel(f"{display_name}:")
                form_layout.addRow(lbl, color_container)
                self.labels[key] = lbl
                self.inputs[key] = le
                self.color_swatches[key] = swatch
            # --- Low Resource Mode Toggle ---
            lbl_low_res = QLabel("Low Resource Mode:")
            lbl_low_res.setFixedWidth(180)
            cb_low_res = QPushButton()
            cb_low_res.setCheckable(True)
            cb_low_res.setFixedSize(60, 28)
            
            def switch_toggle_style_low_res(checked, btn=cb_low_res):
                if checked:
                    btn.setText("ON")
                    btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 14px; font-weight: bold; border: 1px solid #388E3C; }")
                else:
                    btn.setText("OFF")
                    btn.setStyleSheet("QPushButton { background-color: #e0e0e0; color: #333; border-radius: 14px; font-weight: bold; border: 1px solid #bdbdbd; }")
            
            cb_low_res.toggled.connect(switch_toggle_style_low_res)
            initial_low_res = bool(globals().get("LOW_RESOURCE_MODE", False))
            cb_low_res.setChecked(initial_low_res)
            switch_toggle_style_low_res(initial_low_res)
            
            form_layout.addRow(lbl_low_res, cb_low_res)
            self.labels["LOW_RESOURCE_MODE"] = lbl_low_res
            self.inputs["LOW_RESOURCE_MODE"] = cb_low_res
            
            main_layout.addLayout(form_layout)
            main_layout.addStretch()
            
            btn_reset = QPushButton("Reset to Default")
            btn_reset.setMinimumSize(140, 35)
            btn_reset.setStyleSheet("background-color: #e0e0e0; color: #333; font-weight: bold;")
            
            def reset_visuals():
                for k, v in self.visual_defaults.items():
                    widget = self.inputs.get(k)
                    if widget:
                        if hasattr(widget, 'setValue'):
                            widget.setValue(v)
                        elif hasattr(widget, 'setChecked'):
                            widget.setChecked(v)
                        else:
                            widget.setText(str(v))
                            if k in self.color_swatches:
                                self.color_swatches[k].setStyleSheet(f"background-color: {v}; border: 1px solid gray; border-radius: 3px;")
                                
            btn_reset.clicked.connect(reset_visuals)
            
            bottom_lay = QHBoxLayout()
            bottom_lay.addStretch()
            bottom_lay.addWidget(btn_reset)
            main_layout.addLayout(bottom_lay)
            
            self.tabs.addTab(tab, "Visual Effects")
            
        def create_physics_tab(self):
            tab = QWidget()
            main_layout = QVBoxLayout(tab)
            form_layout = QFormLayout()
            form_layout.setHorizontalSpacing(30)
            form_layout.setVerticalSpacing(10)
            
            self.physics_defaults = {
                "PHYSICS_ENGINE": "Molecular Dynamics (Style)",
                "SPRING_K": 5.0, "COULOMB_K": 10.0, "COULOMB_CUTOFF": 30.0, 
                "DAMPING": 0.9, "DT": 0.005, "MAX_STEPS": 10000, "RMSD_THRESHOLD": 0.005,
                "PERCENTAGE_DROP_THRESHOLD": 0.1, "RMSD_WINDOW": 50,
                "ENABLE_PROGRESSIVE_SIMULATION": False,
                "PACKING_GEOMETRY": "Square",
                "PACKING_GRID_SIZE": 20.0,
                "SGLD_MIN_K": 20, "SGLD_K_PERCENT": 0.01,
                "SGLD_START_TEMP": 1.5, "SGLD_NOISE_SCALE": 1.0
            }
            
            # --- 1. Physics Engine Choice ---
            cb_engine = NoScrollComboBox()
            cb_engine.addItems(["Molecular Dynamics (Style)", "Monte Carlo (Style)"])
            initial_engine = globals().get("PHYSICS_ENGINE", "Molecular Dynamics (Style)")
            cb_engine.setCurrentText(initial_engine)
            lbl_engine = QLabel("Physics Engine:")
            lbl_engine.setFixedWidth(180)
            form_layout.addRow(lbl_engine, cb_engine)
            self.inputs["PHYSICS_ENGINE"] = cb_engine
            self.labels["PHYSICS_ENGINE"] = lbl_engine
            
            # --- 2. Existing Physics Sliders ---
            slider_settings = [
                {"key": "SPRING_K", "type": "float", "min": 1.0, "max": 20.0, "scale": 10.0, "decimals": 1, "default": 5.0},
                {"key": "COULOMB_K", "type": "float", "min": 1.0, "max": 30.0, "scale": 10.0, "decimals": 1, "default": 10.0},
                {"key": "COULOMB_CUTOFF", "type": "float", "min": 1.0, "max": 100.0, "scale": 10.0, "decimals": 1, "default": 30.0},
                {"key": "DAMPING", "type": "float", "min": 0.1, "max": 2.0, "scale": 100.0, "decimals": 2, "default": 0.9}
            ]
            
            for s in slider_settings:
                key = s["key"]
                display_name = key.replace('_', ' ').title()
                if key == "COULOMB_K": display_name = "Repulsion Constant"
                elif key == "COULOMB_CUTOFF": display_name = "Max Repulsion Cutoff"
                elif key == "SPRING_K": display_name = "Spring Constant"
                elif key == "DAMPING": display_name = "Damping Coefficient"
                    
                val_raw = globals().get(key, s["default"])
                
                ui_element = QWidget()
                ui_element.setObjectName("wrapper")
                h_lay = QHBoxLayout(ui_element)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                sl = NoScrollSlider(Qt.Orientation.Horizontal)
                
                try: val = float(val_raw)
                except: val = s["default"]
                
                sc = s["scale"]
                sl.setMinimum(int(s["min"] * sc))
                sl.setMaximum(int(s["max"] * sc))
                
                box = NoScrollDoubleSpinBox()
                box.setRange(-999999.0, 999999.0)
                box.setDecimals(s["decimals"])
                box.setFixedWidth(70)
                
                sl.setValue(int(val * sc))
                box.setValue(val)
                
                sl.valueChanged.connect(lambda v, b=box, scale=sc: b.setValue(v / scale))
                box.valueChanged.connect(lambda v, s=sl, scale=sc: s.setValue(int(v * scale)))
                
                h_lay.addWidget(sl)
                h_lay.addWidget(box)
                
                lbl = QLabel(f"{display_name}:")
                lbl.setFixedWidth(180)
                form_layout.addRow(lbl, ui_element)
                self.labels[key] = lbl
                self.inputs[key] = box
            
            # --- 3. Combined dt & max_steps row ---
            row_widget1 = QWidget()
            row_layout1 = QHBoxLayout(row_widget1)
            row_layout1.setContentsMargins(0, 0, 0, 0)
            row_layout1.setSpacing(10)
            
            # Step Size
            left_widget1 = QWidget()
            left_layout1 = QHBoxLayout(left_widget1)
            left_layout1.setContentsMargins(0, 0, 0, 0)
            lbl_dt = QLabel("Step Size:")
            lbl_dt.setFixedWidth(180)
            le_dt = QLineEdit(str(globals().get("DT", 0.005)))
            left_layout1.addWidget(lbl_dt)
            left_layout1.addWidget(le_dt)
            self.inputs["DT"] = le_dt
            self.labels["DT"] = lbl_dt
            
            # Max Steps
            right_widget1 = QWidget()
            right_layout1 = QHBoxLayout(right_widget1)
            right_layout1.setContentsMargins(0, 0, 0, 0)
            lbl_steps = QLabel("Max Steps:")
            lbl_steps.setFixedWidth(180)
            le_steps = QLineEdit(str(globals().get("MAX_STEPS", 10000)))
            right_layout1.addWidget(lbl_steps)
            right_layout1.addWidget(le_steps)
            self.inputs["MAX_STEPS"] = le_steps
            self.labels["MAX_STEPS"] = lbl_steps
            
            row_layout1.addWidget(left_widget1)
            row_layout1.addWidget(right_widget1)
            form_layout.addRow(row_widget1)
            
            # --- 4. Combined rmsd_threshold & percentage_drop_threshold row ---
            row_widget2 = QWidget()
            row_layout2 = QHBoxLayout(row_widget2)
            row_layout2.setContentsMargins(0, 0, 0, 0)
            row_layout2.setSpacing(10)
            
            # RMSD Threshold
            left_widget2 = QWidget()
            left_layout2 = QHBoxLayout(left_widget2)
            left_layout2.setContentsMargins(0, 0, 0, 0)
            lbl_rmsd = QLabel("RMSD Threshold:")
            lbl_rmsd.setFixedWidth(180)
            le_rmsd = QLineEdit(str(globals().get("RMSD_THRESHOLD", 0.005)))
            left_layout2.addWidget(lbl_rmsd)
            left_layout2.addWidget(le_rmsd)
            self.inputs["RMSD_THRESHOLD"] = le_rmsd
            self.labels["RMSD_THRESHOLD"] = lbl_rmsd
            
            # Min % Drop Threshold
            right_widget2 = QWidget()
            right_layout2 = QHBoxLayout(right_widget2)
            right_layout2.setContentsMargins(0, 0, 0, 0)
            lbl_drop = QLabel("Min % Drop Threshold:")
            lbl_drop.setFixedWidth(180)
            le_drop = QLineEdit(str(globals().get("PERCENTAGE_DROP_THRESHOLD", 0.1)))
            right_layout2.addWidget(lbl_drop)
            right_layout2.addWidget(le_drop)
            self.inputs["PERCENTAGE_DROP_THRESHOLD"] = le_drop
            self.labels["PERCENTAGE_DROP_THRESHOLD"] = lbl_drop
            
            row_layout2.addWidget(left_widget2)
            row_layout2.addWidget(right_widget2)
            form_layout.addRow(row_widget2)
            
            # --- 5. RMSD Window logscale slider + spinbox (10 to 1000) ---
            import math
            sl_window = NoScrollSlider(Qt.Orientation.Horizontal)
            sl_window.setMinimum(0)
            sl_window.setMaximum(100)
            
            box_window = NoScrollSpinBox()
            box_window.setRange(10, 1000)
            box_window.setFixedWidth(70)
            
            # Mapping functions
            def slider_to_val(x):
                return int(round(10.0 ** (1.0 + 2.0 * x / 100.0)))
                
            def val_to_slider(v):
                if v < 10: v = 10
                if v > 1000: v = 1000
                return int(round(100.0 * (math.log10(v) - 1.0) / 2.0))
                
            def on_slider_changed(x):
                val = slider_to_val(x)
                box_window.blockSignals(True)
                box_window.setValue(val)
                box_window.blockSignals(False)
                
            def on_box_changed(val):
                x = val_to_slider(val)
                sl_window.blockSignals(True)
                sl_window.setValue(x)
                sl_window.blockSignals(False)
                
            sl_window.valueChanged.connect(on_slider_changed)
            box_window.valueChanged.connect(on_box_changed)
            
            initial_window = int(globals().get("RMSD_WINDOW", 50))
            box_window.setValue(initial_window)
            sl_window.setValue(val_to_slider(initial_window))
            
            ui_window = QWidget()
            ui_window.setObjectName("wrapper")
            h_lay_window = QHBoxLayout(ui_window)
            h_lay_window.setContentsMargins(0, 0, 0, 0)
            h_lay_window.addWidget(sl_window)
            h_lay_window.addWidget(box_window)
            
            lbl_window = QLabel("RMSD Window:")
            lbl_window.setFixedWidth(180)
            form_layout.addRow(lbl_window, ui_window)
            self.inputs["RMSD_WINDOW"] = box_window
            self.labels["RMSD_WINDOW"] = lbl_window
            
            # --- 6. Combined Progressive Edge Annealing & Show Score Histogram row ---
            cb_row = QWidget()
            cb_layout = QHBoxLayout(cb_row)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            cb_layout.setSpacing(10)
            
            # Progressive Edge Annealing
            pea_container = QWidget()
            pea_layout = QHBoxLayout(pea_container)
            pea_layout.setContentsMargins(0, 0, 0, 0)
            lbl_prog = QLabel("Progressive Edge Annealing:")
            lbl_prog.setFixedWidth(180)
            cb_prog = QPushButton()
            cb_prog.setCheckable(True)
            cb_prog.setFixedSize(60, 28)
            pea_layout.addWidget(lbl_prog)
            pea_layout.addWidget(cb_prog)
            pea_layout.addStretch()
            self.inputs["ENABLE_PROGRESSIVE_SIMULATION"] = cb_prog
            self.labels["ENABLE_PROGRESSIVE_SIMULATION"] = lbl_prog
            
            def switch_toggle_style(checked, btn=cb_prog):
                if checked:
                    btn.setText("ON")
                    btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 14px; font-weight: bold; border: 1px solid #388E3C; }")
                else:
                    btn.setText("OFF")
                    btn.setStyleSheet("QPushButton { background-color: #e0e0e0; color: #333; border-radius: 14px; font-weight: bold; border: 1px solid #bdbdbd; }")
            
            cb_prog.toggled.connect(switch_toggle_style)
            initial_state = bool(globals().get("ENABLE_PROGRESSIVE_SIMULATION", False))
            cb_prog.setChecked(initial_state)
            switch_toggle_style(initial_state)
            
            # Packing Geometry Dropdown
            geom_container = QWidget()
            geom_layout = QHBoxLayout(geom_container)
            geom_layout.setContentsMargins(0, 0, 0, 0)
            lbl_geom = QLabel("Packing Geometry:")
            lbl_geom.setFixedWidth(180)
            cb_geom = NoScrollComboBox()
            cb_geom.addItems(["Square", "Circle"])
            
            from PyQt6.QtWidgets import QSizePolicy
            cb_geom.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            
            initial_geom = globals().get("PACKING_GEOMETRY", "Square")
            if initial_geom not in ["Square", "Circle"]:
                initial_geom = "Square"
            cb_geom.setCurrentText(initial_geom)
            
            geom_layout.addWidget(lbl_geom)
            geom_layout.addWidget(cb_geom)
            self.inputs["PACKING_GEOMETRY"] = cb_geom
            self.labels["PACKING_GEOMETRY"] = lbl_geom
            
            cb_layout.addWidget(pea_container)
            cb_layout.addWidget(geom_container)
            form_layout.addRow(cb_row)
            
            # Packing Grid Size Slider (Logarithmic Scale)
            pgs_widget = QWidget()
            pgs_widget.setObjectName("wrapper")
            pgs_layout = QHBoxLayout(pgs_widget)
            pgs_layout.setContentsMargins(0, 0, 0, 0)
            
            sl_pgs = NoScrollSlider(Qt.Orientation.Horizontal)
            sl_pgs.setMinimum(0)
            sl_pgs.setMaximum(1000)
            
            box_pgs = NoScrollDoubleSpinBox()
            box_pgs.setRange(1.0, 200.0)
            box_pgs.setDecimals(1)
            box_pgs.setFixedWidth(70)
            
            import math
            val_pgs = globals().get("PACKING_GRID_SIZE", 20.0)
            try: val_pgs = float(val_pgs)
            except: val_pgs = 20.0
            
            box_pgs.setValue(val_pgs)
            pct = math.log(max(1.0, min(val_pgs, 200.0))) / math.log(200.0)
            sl_pgs.setValue(int(pct * 1000.0))
            
            def update_from_slider(x):
                box_pgs.blockSignals(True)
                pct_val = x / 1000.0
                val = math.exp(pct_val * math.log(200.0))
                box_pgs.setValue(round(val, 1))
                box_pgs.blockSignals(False)
                
            def update_from_box(v):
                sl_pgs.blockSignals(True)
                v_clamped = max(1.0, min(v, 200.0))
                pct_val = math.log(v_clamped) / math.log(200.0)
                sl_pgs.setValue(int(pct_val * 1000.0))
                sl_pgs.blockSignals(False)
                
            sl_pgs.valueChanged.connect(update_from_slider)
            box_pgs.valueChanged.connect(update_from_box)
            
            pgs_layout.addWidget(sl_pgs)
            pgs_layout.addWidget(box_pgs)
            
            lbl_pgs = QLabel("Packing Grid Size:")
            lbl_pgs.setFixedWidth(180)
            
            form_layout.addRow(lbl_pgs, pgs_widget)
            self.inputs["PACKING_GRID_SIZE"] = box_pgs
            self.labels["PACKING_GRID_SIZE"] = lbl_pgs
            
            # --- 7. Combined Minimum K & Percentage K row (Monte Carlo only) ---
            k_row = QWidget()
            k_layout = QHBoxLayout(k_row)
            k_layout.setContentsMargins(0, 0, 0, 0)
            k_layout.setSpacing(10)
            
            # Minimum K
            min_k_container = QWidget()
            min_k_layout = QHBoxLayout(min_k_container)
            min_k_layout.setContentsMargins(0, 0, 0, 0)
            lbl_min_k = QLabel("Minimum K:")
            lbl_min_k.setFixedWidth(180)
            le_min_k = QLineEdit(str(globals().get("SGLD_MIN_K", 20)))
            min_k_layout.addWidget(lbl_min_k)
            min_k_layout.addWidget(le_min_k)
            self.inputs["SGLD_MIN_K"] = le_min_k
            self.labels["SGLD_MIN_K"] = lbl_min_k
            
            # Percentage K
            pct_k_container = QWidget()
            pct_k_layout = QHBoxLayout(pct_k_container)
            pct_k_layout.setContentsMargins(0, 0, 0, 0)
            lbl_pct_k = QLabel("Fraction K:")
            lbl_pct_k.setFixedWidth(180)
            le_pct_k = QLineEdit(str(globals().get("SGLD_K_PERCENT", 0.01)))
            pct_k_layout.addWidget(lbl_pct_k)
            pct_k_layout.addWidget(le_pct_k)
            self.inputs["SGLD_K_PERCENT"] = le_pct_k
            self.labels["SGLD_K_PERCENT"] = lbl_pct_k
            
            k_layout.addWidget(min_k_container)
            k_layout.addWidget(pct_k_container)
            form_layout.addRow(k_row)
            
            # --- 8. Combined Temperature parameters row (Monte Carlo only) ---
            temp_row = QWidget()
            temp_layout = QHBoxLayout(temp_row)
            temp_layout.setContentsMargins(0, 0, 0, 0)
            temp_layout.setSpacing(10)
            
            # Starting Temperature
            start_temp_container = QWidget()
            start_temp_layout = QHBoxLayout(start_temp_container)
            start_temp_layout.setContentsMargins(0, 0, 0, 0)
            lbl_start_temp = QLabel("Starting Temp:")
            lbl_start_temp.setFixedWidth(180)
            le_start_temp = QLineEdit(str(globals().get("SGLD_START_TEMP", 1.5)))
            start_temp_layout.addWidget(lbl_start_temp)
            start_temp_layout.addWidget(le_start_temp)
            self.inputs["SGLD_START_TEMP"] = le_start_temp
            self.labels["SGLD_START_TEMP"] = lbl_start_temp
            
            # Thermal Noise Scale
            noise_scale_container = QWidget()
            noise_scale_layout = QHBoxLayout(noise_scale_container)
            noise_scale_layout.setContentsMargins(0, 0, 0, 0)
            lbl_noise_scale = QLabel("Thermal Noise Scale:")
            lbl_noise_scale.setFixedWidth(180)
            le_noise_scale = QLineEdit(str(globals().get("SGLD_NOISE_SCALE", 1.0)))
            noise_scale_layout.addWidget(lbl_noise_scale)
            noise_scale_layout.addWidget(le_noise_scale)
            self.inputs["SGLD_NOISE_SCALE"] = le_noise_scale
            self.labels["SGLD_NOISE_SCALE"] = lbl_noise_scale
            
            temp_layout.addWidget(start_temp_container)
            temp_layout.addWidget(noise_scale_container)
            form_layout.addRow(temp_row)
            
            # Apply styling for disabled states to match other tabs
            disabled_lineedit_style = "QLineEdit:disabled { background-color: #f0f0f0; color: #888; }"
            disabled_label_style = "QLabel:disabled { color: #888; }"
            le_min_k.setStyleSheet(disabled_lineedit_style)
            le_pct_k.setStyleSheet(disabled_lineedit_style)
            le_start_temp.setStyleSheet(disabled_lineedit_style)
            le_noise_scale.setStyleSheet(disabled_lineedit_style)
            lbl_min_k.setStyleSheet(disabled_label_style)
            lbl_pct_k.setStyleSheet(disabled_label_style)
            lbl_start_temp.setStyleSheet(disabled_label_style)
            lbl_noise_scale.setStyleSheet(disabled_label_style)
            
            # --- Toggle Dependencies Function ---
            def update_engine_ui():
                is_mc = cb_engine.currentText() == "Monte Carlo (Style)"
                le_min_k.setEnabled(is_mc)
                le_pct_k.setEnabled(is_mc)
                lbl_min_k.setEnabled(is_mc)
                lbl_pct_k.setEnabled(is_mc)
                
                le_start_temp.setEnabled(is_mc)
                le_noise_scale.setEnabled(is_mc)
                lbl_start_temp.setEnabled(is_mc)
                lbl_noise_scale.setEnabled(is_mc)
                
            cb_engine.currentTextChanged.connect(update_engine_ui)
            update_engine_ui()
            
            main_layout.addLayout(form_layout)
            main_layout.addStretch()
            
            btn_reset = QPushButton("Reset to Default")
            btn_reset.setMinimumSize(140, 35)
            btn_reset.setStyleSheet("background-color: #e0e0e0; color: #333; font-weight: bold;")
            
            def reset_physics():
                for k, v in self.physics_defaults.items():
                    widget = self.inputs.get(k)
                    if widget:
                        if hasattr(widget, 'setCurrentText'):
                            widget.setCurrentText(str(v))
                        elif hasattr(widget, 'setValue'):
                            widget.setValue(v)
                        elif hasattr(widget, 'setChecked'):
                            widget.setChecked(v)
                        else:
                            widget.setText(str(v))
                update_engine_ui()
            
            btn_reset.clicked.connect(reset_physics)
            
            bottom_lay = QHBoxLayout()
            bottom_lay.addStretch()
            bottom_lay.addWidget(btn_reset)
            main_layout.addLayout(bottom_lay)
 
            self.tabs.addTab(tab, "Simulation && Physics")
            
        def create_directories_tab(self):
            tab = QWidget()
            layout = QFormLayout(tab)
            layout.setHorizontalSpacing(30)
            layout.setVerticalSpacing(10)
            
            # Added FASTA_DIR and HDF5_DIR
            keys = ["FASTA_DIR", "MSA_DIR", "HDF5_DIR", "SAVED_LAYOUT_DIR", "METADATA_DIR", "PRINT_SAVE_DIR", 
                    "FASTA_SPLIT_DIR", "CLUSTER_LABEL_DIR", "HEADER_LIST_DIR", "LOGO_DIR", "STRUCTURES_DIR"]
            
            for key in keys:
                container = QWidget()
                container.setObjectName("wrapper")
                h_lay = QHBoxLayout(container)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                val = globals().get(key, "")
                le = QLineEdit("" if val in [None, "None"] else str(val))
                btn = QPushButton("Browse...")
                
                def browse_dir(checked, line_edit=le):
                    folder = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text() or "")
                    if folder:
                        import os
                        line_edit.setText(os.path.normpath(folder))
                        
                btn.clicked.connect(browse_dir)
                
                h_lay.addWidget(le)
                h_lay.addWidget(btn)
                
                display_name = key.replace('_', ' ').title()
                display_name = display_name.replace('Msa', 'MSA')
                display_name = display_name.replace('Hdf5', 'Network')
                display_name = display_name.replace('Fasta Split', 'FASTA Split Saving')
                display_name = display_name.replace('Dir', 'Directory')
                
                lbl = QLabel(f"{display_name}:")
                layout.addRow(lbl, container)
                self.labels[key] = lbl
                self.inputs[key] = le

            # Bind the text changes to dynamically refresh the dropdowns in Tab 1
            self.inputs["FASTA_DIR"].textChanged.connect(lambda: self.refresh_combo(self.cb_fasta, "FASTA_DIR", ['.fasta']))
            self.inputs["MSA_DIR"].textChanged.connect(lambda: self.refresh_combo(self.cb_msa, "MSA_DIR", ['.fasta', '.h5']))
            self.inputs["HDF5_DIR"].textChanged.connect(lambda: self.refresh_combo(self.cb_hdf5, "HDF5_DIR", ['.h5']))
                
            self.tabs.addTab(tab, "Directories")

        def collect_data(self):
            data = {}
            from PyQt6.QtWidgets import QComboBox, QPushButton, QLineEdit
            for key, widget in self.inputs.items():
                
                # ---> NEW: Completely skip saving the target cache selection to JSON
                if key == "TARGET_CACHE_FILE":
                    continue
                    
                if isinstance(widget, QComboBox): 
                    val = widget.currentText()
                elif hasattr(widget, 'value'): 
                    val = str(widget.value())
                elif isinstance(widget, QPushButton) and widget.isCheckable(): 
                    val = widget.isChecked()                                   
                elif hasattr(widget, 'isChecked'): 
                    val = widget.isChecked()
                elif isinstance(widget, QLineEdit):
                    val = widget.text()
                else: 
                    val = str(widget)
                
                if not str(val).strip(): 
                    if key == "TOP_EDGE_PERCENT": val = "None"
                    elif key in ["MSA_FILE", "ALIGNMENT_REFERENCE"]: val = ""
                    else: continue
                
                if key == "NODE_FASTA_FILE": val = os.path.join(self.inputs["FASTA_DIR"].text(), val).replace("\\", "/") if val else ""
                elif key == "MSA_FILE": val = os.path.join(self.inputs["MSA_DIR"].text(), val).replace("\\", "/") if val else ""
                elif key == "INPUT_HDF5": val = os.path.join(self.inputs["HDF5_DIR"].text(), val).replace("\\", "/") if val else ""
                
                data[key] = val
            return data

        def save_settings(self):
            data = self.collect_data()
            try:
                os.makedirs("Input_Files", exist_ok=True)
                with open(SETTINGS_FILE, "w") as f: json.dump(data, f, indent=4)
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")
                return False

        def save_and_run(self):
            if self.save_settings():
                self.close()
                print("Launching SSN_Viewer.py...")
                
                import subprocess
                import sys
                import os
                
                env = os.environ.copy()
                selected_cache = self.cb_cache_file.currentText()
                
                # Check if the user opted to force a new calculation
                if selected_cache == "(New Layout Cache)":
                    custom_name = self.line_new_cache.text().strip()
                    
                    # Fallback to the greyed-out default if they left it blank
                    if not custom_name:
                        custom_name = self.line_new_cache.placeholderText()
                        
                    # Auto-append extension if forgotten
                    if not custom_name.endswith(".h5"):
                        custom_name += ".h5"
                        
                    env["SSN_TARGET_CACHE"] = custom_name
                    
                # Otherwise, load the existing file
                elif selected_cache:
                    env["SSN_TARGET_CACHE"] = selected_cache
                        
                # Use the project root (parent of src/) as cwd so all relative data paths resolve correctly
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                
                if sys.platform == "win32":
                    creationflags = subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0x10
                    subprocess.Popen(
                        f'cmd.exe /c ""{sys.executable}" src\\SSN_Viewer.py || pause"', 
                        env=env, 
                        creationflags=creationflags, 
                        cwd=script_dir
                    )
                else:
                    # GUI applications do not need a terminal window on macOS/Linux and run fine as detached processes
                    subprocess.Popen([sys.executable, os.path.join("src", "SSN_Viewer.py")], env=env, cwd=script_dir)

        def save_only(self):
            if self.save_settings():
                QMessageBox.information(self, "Success", "Settings saved successfully!")

    app = QApplication(sys.argv)
    
    # Set Application-wide Icon
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.ico")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        
    try:
        from SSN_Utils import force_light_palette
        force_light_palette(app)
    except Exception as e:
        print(f"Warning: Could not force light palette: {e}")
        app.setStyle("Fusion")
    window = ConfigGUI()
    window.show()
    sys.exit(app.exec())
