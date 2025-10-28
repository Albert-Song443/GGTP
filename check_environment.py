"""
Environment Check Script for GGTP
Checks all required dependencies are correctly installed
"""
import sys

def check_module(name, package_name=None, install_cmd=None):
    """Check if a module is installed."""
    if package_name is None:
        package_name = name
    try:
        module = __import__(name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"   ✗ {package_name}: NOT INSTALLED")
        if install_cmd:
            print(f"     Fix: {install_cmd}")
        return False

print("=" * 70)
print("GGTP Environment Check")
print("=" * 70)

# 1. System Info
print(f"\n1. System Information:")
print(f"   Python Version: {sys.version.split()[0]}")
print(f"   Platform: {sys.platform}")

# 2. Core Dependencies
print(f"\n2. Core Dependencies:")
all_ok = True
all_ok &= check_module('numpy', install_cmd='pip install numpy')
all_ok &= check_module('scipy', install_cmd='pip install scipy')
all_ok &= check_module('pandas', install_cmd='pip install pandas')
all_ok &= check_module('matplotlib', install_cmd='pip install matplotlib')
all_ok &= check_module('tqdm', install_cmd='pip install tqdm')
all_ok &= check_module('yaml', 'pyyaml', install_cmd='pip install pyyaml')

# 3. PyTorch
print(f"\n3. PyTorch:")
try:
    import torch
    print(f"   ✓ PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print(f"   ✗ PyTorch: NOT INSTALLED")
    print(f"     Fix: pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117")
    all_ok = False

# 4. PyTorch Geometric
print(f"\n4. Graph Neural Networks:")
all_ok &= check_module('torch_geometric', 'PyTorch Geometric', 
                       install_cmd='pip install torch-geometric')
all_ok &= check_module('torch_scatter', 
                       install_cmd='pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html')
all_ok &= check_module('torch_sparse',
                       install_cmd='pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu117.html')

# 5. GIS and Geometry
print(f"\n5. GIS and Geometry:")
all_ok &= check_module('shapely', install_cmd='pip install shapely')
all_ok &= check_module('geopandas', install_cmd='pip install geopandas')
all_ok &= check_module('fiona', install_cmd='pip install fiona')
all_ok &= check_module('rasterio', install_cmd='pip install rasterio')
all_ok &= check_module('pyogrio', install_cmd='pip install pyogrio')

# 6. Computer Vision
print(f"\n6. Computer Vision:")
all_ok &= check_module('cv2', 'opencv-python', install_cmd='pip install opencv-python')

# 7. Async and Utilities
print(f"\n7. Async I/O and Utilities:")
all_ok &= check_module('aiofiles', install_cmd='pip install aiofiles')
all_ok &= check_module('aioboto3', install_cmd='pip install aioboto3')
all_ok &= check_module('retry', install_cmd='pip install retry')
all_ok &= check_module('filelock', install_cmd='pip install filelock')

# 8. nuPlan Devkit
print(f"\n8. nuPlan Devkit:")
try:
    import nuplan
    print(f"   ✓ nuPlan devkit: installed")
    # Check if nuPlan modules can be imported
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    print(f"   ✓ nuPlan imports: working")
except ImportError as e:
    print(f"   ✗ nuPlan devkit: NOT INSTALLED or has issues")
    print(f"     Error: {e}")
    print(f"     Fix: cd .. && git clone https://github.com/motional/nuplan-devkit.git")
    print(f"          cd nuplan-devkit && pip install -e . && cd ../GGTP")
    all_ok = False

# 9. GGTP Modules
print(f"\n9. GGTP Modules:")
try:
    from ggtp_modules import GNNEncoder, GGTP_Decoder
    print(f"   ✓ ggtp_modules: OK")
except ImportError as e:
    print(f"   ✗ ggtp_modules: FAILED")
    print(f"     Error: {e}")
    all_ok = False

try:
    from adversarial_modules import adversarial_scenario_generation
    print(f"   ✓ adversarial_modules: OK")
except ImportError as e:
    print(f"   ✗ adversarial_modules: FAILED")
    print(f"     Error: {e}")
    all_ok = False

# 10. Quick Functional Test
print(f"\n10. Functional Tests:")
try:
    import torch
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    
    # Test GAT
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    x = torch.randn(2, 16)
    data = Data(x=x, edge_index=edge_index)
    conv = GATConv(16, 8, heads=1)
    out = conv(data.x, data.edge_index)
    
    print(f"   ✓ PyTorch Geometric test: PASSED")
except Exception as e:
    print(f"   ✗ PyTorch Geometric test: FAILED")
    print(f"     Error: {e}")
    all_ok = False

# Summary
print("\n" + "=" * 70)
if all_ok:
    print("✓ ALL CHECKS PASSED! Environment is ready for GGTP.")
    print("\nYou can now:")
    print("  1. Process data: python data_process.py --data_path ... --map_path ...")
    print("  2. Train model: python train_ggtp.py --train_set ... --valid_set ...")
else:
    print("✗ SOME CHECKS FAILED! Please install missing dependencies.")
    print("\nQuick fix:")
    print("  pip install -r requirements.txt")
    print("\nFor nuPlan devkit:")
    print("  cd .. && git clone https://github.com/motional/nuplan-devkit.git")
    print("  cd nuplan-devkit && pip install -e . && cd ../GGTP")

print("=" * 70)
