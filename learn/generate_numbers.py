import numpy as np

def generate_seq(raw_x,ending):
    x=[]

    width=len(str(ending))
    for xx in raw_x:
        y=xx / 3.0
        xx=str(xx).rjust(width,"0")
        xx=list(xx)
        xx=list( int(xxx) for xxx in xx)
        xx.append(y)
        x.append(xx)

    return np.array(x)



