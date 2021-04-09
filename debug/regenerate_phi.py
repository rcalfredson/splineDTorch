from splinedist.utils import phi_generator
import threading


def generate_dummy_phi():
    t = threading.Timer(0.7, generate_dummy_phi)
    t.daemon = True
    t.start()
    phi_generator(8, 82, debug=True)
    print('made a new phi.')


generate_dummy_phi()
input()