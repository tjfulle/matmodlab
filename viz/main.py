#raise SystemExit("Material Model Lab GUI interface not yet completed")
import sys
import viz.mllab as mllab
def main(argv=None):
    window = mllab.MMLMaterialModelSelector()
    sys.exit(window.configure_traits())
if __name__ == "__main__":
    main()
