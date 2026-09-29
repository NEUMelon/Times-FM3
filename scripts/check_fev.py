try:
    import fev
    print("fev is installed, version:", getattr(fev, "__version__", "unknown"))
except ImportError as e:
    print("fev not installed:", e)
