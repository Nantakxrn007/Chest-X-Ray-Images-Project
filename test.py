import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# ตรวจสอบว่า TensorFlow เห็น GPU หรือไม่
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# แสดงรายละเอียด GPU ที่ TensorFlow เห็น
print("GPU Details:", tf.config.list_physical_devices('GPU'))

# ทดสอบว่า TensorFlow ใช้งาน GPU ได้จริงหรือไม่
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using GPU!")
else:
    print("TensorFlow is NOT using GPU.")
