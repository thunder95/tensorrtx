import paddle.fluid as fluid
from paddle.fluid.executor import _fetch_var as fetch_var
import struct
import numpy as np

weights_dir = "/home/hl/Downloads/PaddleVideo/action_paddle_stnet1220" #读取权重文件夹
weights_out = "../model_in/pd_stnet.wts" #输出的权重保存路径
fcs = {"fc_0.w_0": (1024, 4)}#paddle的fc权重需要转置
is_file_specified = False #是否指定模型和参数文件具体路径
model_filename = "" #对应model文件路径
params_filename = "" #对应params文件路径

place = fluid.CPUPlace()
exe = fluid.Executor(place)

inference_scope = fluid.core.Scope()
with fluid.scope_guard(inference_scope):

    # Load inference program and other target attributes
    if is_file_specified:
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
            weights_dir, exe, model_filename,params_filename)
    else:
        [inference_program, feed_target_names, fetch_targets
         ] = fluid.io.load_inference_model(weights_dir, exe)

    fetch_targets_names = [data.name for data in fetch_targets]

    feed_fetch_list = ["fetch", "feed"]
    # Load parameters
    weights  =  {}
    global_block = inference_program.global_block()
    for var_name in global_block.vars:
        var = global_block.var(var_name)
        if var_name not in feed_fetch_list \
                and var.persistable:

            weight = fetch_var(str(var.name), inference_scope)
            weights[str(var.name)] = weight.flatten().tolist()


    """
    input and output
    """

    print("input node: ", feed_target_names)
    print("output node: ", fetch_targets_names)
    print("weights len: ", len(weights))





    """
    save weights to file
    """
    w_len = len(weights)

    f = open(weights_out, 'w')
    f.write("{}\n".format(feed_target_names[0]))  # input
    f.write("{}\n".format(fetch_targets_names[0]))  # output
    f.write("{}\n".format(w_len))  # totlen len

    for wname in weights:
        wf_data = weights[wname]
        wf_len = len(wf_data)
        f.write("{} {}".format(wname, wf_len))  # summary for each weight
        print("save weight: ", wname, wf_len)

        if wname in fcs.keys():
            print("=====>converting fc weight ") #paddle的全连接权重需要单独处理
            aa = np.array(wf_data)
            aa = aa.reshape(fcs[wname])
            aa = aa.transpose()
            aa = list(aa.flatten())
            for v in aa:
                f.write(" ")
                f.write(struct.pack(">f", float(v)).hex())
            f.write("\n")
        else:

            for v in wf_data:
                f.write(" ")
                f.write(struct.pack(">f", float(v)).hex())
            f.write("\n")

    f.close()


