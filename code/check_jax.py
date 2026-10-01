try:
    import jax
    print("jax ok:", jax.__version__)
except Exception as e:
    print("jax no:", e)