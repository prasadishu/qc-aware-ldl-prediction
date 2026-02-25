import numpy as np

def friedewald(tc, tg, hdl):
    return tc - hdl - (tg / 5)

def martin(tc, tg, hdl):
    adjustable_factor = np.where(tg < 150, 5, 6)
    return tc - hdl - (tg / adjustable_factor)

def sampson(tc, tg, hdl):
    return (tc / 0.948) - (hdl / 0.971) - ((tg / 8.56) + (tg * (tc - hdl) / 2140)) - 9.44
