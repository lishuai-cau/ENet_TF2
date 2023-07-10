from nets.ENet import enet
model = enet(n_classes=2,input_height=320, input_width=320)
model.load_weights("./logs/ep036-loss0.005-val_loss0.005.h5")
#保存带有计算图的h5模型
model.save("./weight/temp.h5")

# 使用kerasonnx 进行转换或 tf2onnx，两者都可能报错误，具体看根据版本尝试
# 使用kerasonnx转换方法
# import keras2onnx
# import onnx
# onnx_model = keras2onnx.convert_keras(model, model.name)
# temp_model_file = './onnx/Enet.onnx'
# onnx.save_model(onnx_model, temp_model_file)


# 使用tf2onnx转换方法
import tf2onnx
import keras
keras_model = keras.models.load_model('./weight/temp.h5')
model_proto, _ = tf2onnx.convert.from_keras(keras_model)
with open('test.onnx', 'wb') as f:
    f.write(model_proto.SerializeToString())
