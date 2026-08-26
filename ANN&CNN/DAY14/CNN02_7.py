target = pred_target
before = float(
model(image, training=False)[0, target]
)
masked = image.numpy().copy()
masked[:, y0:y1, x0:x1, :] = 0.0
after = float(
model(masked, training=False)[0, target]
)
drop = before - after
print(f"before={before:.3f}")
print(f"after ={after:.3f}")
print(f"drop ={drop:.3f}")