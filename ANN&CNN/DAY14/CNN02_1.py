import numpy as np
y = np.array([0,2,0,1])
p = np.array([
    [.80, .10, .10],[.72, .18, .10],
    [.20, .65, .15],[.10, .75, .15]
])

pred, conf = p.argmax(1), p.max(1)
wrong = np.flatnonzero(pred != y)
right = np.flatnonzero(pred == y)
case = wrong[conf[wrong].argmax()]
control = right[conf[right].argmax()]
print("case/control:", case, control)
