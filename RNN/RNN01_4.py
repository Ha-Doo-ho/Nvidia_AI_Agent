import numpy as np
Wx, Wh, h = 1.0, 0.5, 0.0
sequence = [0.5, -0.2]
states = []
for t, x in enumerate(sequence, start=1):
    h = np.tanh(x * Wx + h * Wh)
    states.append(h)
    print(f"h{t} = {h:.3f}")

expected = [0.462117, 0.031049]
assert np.allclose(states, expected, atol=1e-3)
print("state check: PASS")