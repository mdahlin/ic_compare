from time import time
import numpy as np

import rs_agg_methods
import python.py_ic_funcs as py_agg_methods

NSIMS = 100_000
X = np.array([
    np.linspace(0., 99., NSIMS),
    np.linspace(0., 99., NSIMS),
    np.linspace(0., 99., NSIMS),
    np.linspace(0., 99., NSIMS)
]).T

Ctar = np.array(
    [
        [ 1.00, 0.50, 0.25, 0.05],
        [ 0.50, 1.00, 0.00, 0.30],
        [ 0.25, 0.00, 1.00, 0.00],
        [ 0.05, 0.30, 0.00, 1.00]
    ]
)

print("#" * 20)
print(" " * 20)
print("     IC REORDER    ")
print(" " * 20)
print("#" * 20)

start = time()
rs_res = rs_agg_methods.ic_reorder(X, Ctar)
print(f"Rust single run took {time() - start:.2f} seconds")
py_res = py_agg_methods.ICreorder(X, Ctar)
print(f"Python single run took {time() - start:.2f} seconds")

print("Target Matrix")
print(Ctar)

print("Rust Matrix")
print(np.corrcoef(rs_res, rowvar=False).round(2))

print("Python Matrix")
print(np.corrcoef(py_res, rowvar=False).round(2))

print(" " * 20)
print("#" * 20)
print(" " * 20)
print("    IC IPC REORDER   ")
print(" " * 20)
print("#" * 20)

start = time()
rs_res = rs_agg_methods.ic_ipc_reorder(X, Ctar)
print(f"Rust single run took {time() - start:.2f} seconds")
py_res = py_agg_methods.IC_IPCreorder(X, Ctar)
print(f"Python single run took {time() - start:.2f} seconds")

print("Target Matrix")
print(Ctar)

print("Rust Matrix")
print(np.corrcoef(rs_res, rowvar=False).round(2))

print("Python Matrix")
print(np.corrcoef(py_res, rowvar=False).round(2))
