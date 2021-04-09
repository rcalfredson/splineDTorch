import numpy as np
import threading

def open_phi_file():
    t = threading.Timer(0.2, open_phi_file)
    t.daemon = True
    t.start()
    np.load("./debug/phi_8.npy")
    print('loaded the phi.')

open_phi_file()
input()