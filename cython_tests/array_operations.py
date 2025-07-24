import numpy as np
import numpy.typing as npt
import random
from arrayfun import where_and_wherenot

def where_and_wherenot_py(x: npt.NDArray[np.bool_])-> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    where = np.where(x)[0]
    wherenot = np.where(~x)[0]

    return where, wherenot

if __name__ == "__main__":
    import time
    nptime = []
    custime = []
    for _ in range(10000):
        mask = np.random.rand(20000) < 0.5
        t3 = time.time()
        a = np.where(mask)[0]
        d = np.where(~mask)[0]
        t4 = time.time()
        nptime.append(t4-t3)
        t1 = time.time()
        b,c = where_and_wherenot(mask)
        t2 = time.time()
        custime.append(t2-t1)
    

    print(f"Time for np: {np.mean(nptime)}")
    print(f"Time for custom: {np.mean(custime)}")
    print(a == b)
    print(c == d)
    print(mask)