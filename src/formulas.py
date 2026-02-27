import numpy as np

def friedewald(tc, hdl, tg):
    return tc - hdl - (tg / 5)

def martin(tc, hdl, tg):
    adjustable_factor = 5  # simplified
    return tc - hdl - (tg / adjustable_factor)

def sampson(tc, hdl, tg):
    return tc - hdl - (0.16 * tg) + (0.0003 * tg**2)
