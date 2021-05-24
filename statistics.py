import numpy as np
import random
import scipy.stats as st

from util import p2stars

WELCH = True

# returns t, p, na, nb
def ttest_rel(a, b, msg=None, min_n=2):
    return ttest(a, b, True, msg, min_n)


def ttest_ind(a, b, msg=None, min_n=2):
    return ttest(a, b, False, msg, min_n)


def ttest(a, b, paired, msg=None, min_n=2):
    if paired:
        abFinite = np.isfinite(a) & np.isfinite(b)
    a, b = (x[abFinite if paired else np.isfinite(x)] for x in (a, b))
    na, nb = len(a), len(b)
    if min(na, nb) < min_n:
        return np.nan, np.nan, na, nb
    with np.errstate(all="ignore"):
        t, p = st.ttest_rel(a, b) if paired else st.ttest_ind(a, b, equal_var=not WELCH)
    if msg:
        print("%spaired t-test -- %s:" % ("" if paired else "un", msg))
        print(
            "  n = %s means: %.3g, %.3g; t-test: p = %.5f, t = %.3f"
            % (
                "%d," % na if paired else "%d, %d;" % (na, nb),
                np.mean(a),
                np.mean(b),
                p,
                t,
            )
        )
        print("copyable output:")
        print("means: %.3g, %.3g" % (np.mean(a), np.mean(b)))
        print("p: %.5f; %s" % (p, p2stars(p)))
    return t, p, na, nb


a = np.array(
    [
        1.587,
        1.748,
        1.665,
        1.639,
        1.709,
        1.862,
        2.378,
        2.48,
        1.58,
        1.681,
    ]
)

b = np.array(
    [
        1.908,
        1.801,
        1.57,
        1.551,
        1.638,
        1.577,
        1.644,
        1.608,
        1.778,
        1.606,
        1.381,
        1.82,
        1.444,
        1.711,
        1.831,
        1.569,
        1.563,
        1.426,
        2.81,
        1.552,
        1.753,
        1.619,
        2.402,
    ]
)

# print("Quick test of ttest_rel")
# list1 = [4, 2, 1, 7, 5, 4, 3]
# list2 = [10, 45, 2, 30, 9, 6, 20]
# print(ttest_rel(np.asarray(list1), np.asarray(list2)))
# print(ttest_ind(np.asarray(list1), np.asarray(list2)))
# random.shuffle(list1)
# random.shuffle(list2)
# print("new list2:", list2)
# print(ttest_rel(np.asarray(list1), np.asarray(list2)))
# print(ttest_ind(np.asarray(list1), np.asarray(list2)))

# res = ttest_ind(a, b, msg=True)
# print("  p: %.3f; %s" % (res[1], p2stars(res[1])))
