import os
from tensorflow.python import pywrap_tensorflow
#ckpt_path = "./checkpoints_tf/model-143407-143407"
#ckpt_path = "./checkpoints/model.ckpt-0"
ckpt_path = "./pretrain_model_old/model-6-8243967"
reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print key
    #print reader.get_tensor(key)