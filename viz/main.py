raise SystemExit("Material Model Lab GUI interface not yet completed")
import viz.mllab as mllab
window = mllab.MaterialModelSelector(model_type="any")
sys.exit(window.configure_traits())
