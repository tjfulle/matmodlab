#raise SystemExit("Material Model Lab GUI interface not yet completed")
import sys
import viz.mllab as mllab
window = mllab.MMLMaterialModelSelector()
sys.exit(window.configure_traits())
