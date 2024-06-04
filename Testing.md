---
layout: default
title: Testing Tutorial
---
# TVM run model tutorial 
<br>


https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno.html#sphx-glr-how-to-deploy-models-deploy-model-on-adreno-py 
<br>


https://tvm.apache.org/docs/how_to/deploy/adreno.html 
<br>
<br>


## Build  
<br>


- https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno.html#sphx-glr-how-to-deploy-models-deploy-model-on-adreno-py  
<br>


- https://tvm.apache.org/docs/how_to/deploy/adreno.html  
<br>




./docker/build.sh ci_adreno 
<br>


docker tag tvm.ci_adreno ci_adreno
 <br>


export ADRENO_OPENCL=/home/newway/opt/Snapdragon/OpenCLML
 <br>


./tests/scripts/ci.py adreno -i 
<br> 
<br>



## Run 
<br>



https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno.html#sphx-glr-how-to-deploy-models-deploy-model-on-adreno-py <br>


bash <br> <br>





## Start container <br>




export ADRENO_OPENCL=/home/newway/opt/Snapdragon/OpenCLML <br>


./tests/scripts/ci.py adreno -i <br> <br>



## After the docker is built <br>



export ANDROID_SERIAL=832358d4 <br>


export TVM_HOME=/home/newway/Documents/Research/CNN/TVM/tvm <br>


export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} <br>


adb -s $ANDROID_SERIAL shell "mkdir /data/local/tmp/tvm_run" <br>


adb -s $ANDROID_SERIAL push {build-adreno-target/libtvm_runtime.so,build-adreno-target/tvm_rpc} /data/local/tmp/tvm_run <br>


find ${ANDROID_NDK_HOME} -name libc++_shared.so <br> <br>



## open a terminal within the docker <br>





export ANDROID_SERIAL=832358d4 <br>


python3 -m tvm.exec.rpc_tracker --port 9190 <br> <br>





## open another terminal out of the docker <br>


export ANDROID_SERIAL=832358d4 <br>


adb -s $ANDROID_SERIAL reverse tcp:9190 tcp:9190 <br>


adb -s $ANDROID_SERIAL forward tcp:5000 tcp:5000 <br>


adb -s $ANDROID_SERIAL forward tcp:5002 tcp:5001 <br>


adb -s $ANDROID_SERIAL forward tcp:5003 tcp:5002 <br>


adb -s $ANDROID_SERIAL forward tcp:5004 tcp:5003 <br>


adb -s $ANDROID_SERIAL shell LD_LIBRARY_PATH=/data/local/tmp/tvm_run /data/local/tmp/tvm_run/tvm_rpc server --host=0.0.0.0 --port=5000 --tracker=127.0.0.1:9190 --key=android --port-end=5100 <br> <br>





## open the third terminal within the docker <br>





export TVM_TRACKER_HOST=0.0.0.0 <br>


export TVM_TRACKER_PORT=9190 <br>


python -m tvm.exec.query_rpc_tracker --port 9190 <br>


export TVM_NDK_CC=/opt/android-sdk-linux/ndk/25.2.9519653/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++ <br>


python ./run_opencl.py <br> <br>



## TVM run_opencl.py file <br>





# https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno.html#sphx-glr-how-to-deploy-models-deploy-model-on-adreno-py <br>


# https://tvm.apache.org/docs/how_to/compile_models/from_onnx.html <br>


import os <br>
import time <br>
import torch <br>
import torchvision <br>
import numpy as np <br>
import onnx <br>
from tqdm import tqdm <br>
import tvm <br>
from tvm import te <br>
from tvm import relay, rpc <br>
from tvm.contrib import utils, ndk <br>
from tvm.contrib import graph_executor <br>
from tvm.relay.op.contrib import clml <br>
from tvm import autotvm <br>
from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision <br>
from tvm.contrib.debugger import debug_executor <br>
# https://discuss.tvm.apache.org/t/how-could-us-use-tvm-relay-transform-tomixedprecision/10465/4 <br>
# https://github.com/apache/tvm/blob/main/tests/python/unittest/test_runtime_profiling.py#L180-L196 <br>
# from tvm.relay.transform import mixed_precision # Register op attribute <br>


def benchmark_model(mod, params, input_data, <br>
                    model_name, input_name="input0",  <br>
                    round=50, is_tuning=False, tuning_trail=1024): <br>
    # Below are set of configuration that controls the behaviour of this script like <br>
    # local run or device run, target definitions,  dtype setting and auto tuning enablement. <br>
    # Change these settings as needed if required. <br>

    # Adreno devices are efficient with float16 compared to float32 <br>
    # Given the expected output doesn't effect by lowering precision <br>
    # it's advisable to use lower precision. <br>
    # We have a helper API to make the precision conversion simple and <br>
    # it supports dtype with "float16" and "float16_acc32" modes. <br>
    # Let's choose "float16" for calculation and "float32" for accumulation. <br>
    calculation_dtype = "float16" <br>
    acc_dtype = "float32" <br>

    # Auto tuning is compute intensive and time taking task, <br>
    # hence disabling for default run. Please enable it if required. <br>
    tune_log = f"adreno-{model_name}.log" <br>

    # Specify Adreno target before compiling to generate texture <br>
    # leveraging kernels and get all the benefits of textures <br>
    # Note: This generated example running on our x86 server for demonstration. <br>
    # If running it on the Android device, we need to <br>
    # specify its instruction set. Set :code:`local_demo` to False if you want <br>
    # to run this tutorial with a real device over rpc. <br>
    local_demo = False <br>

    # by default on CPU target will execute. <br>
    # select 'cpu', 'opencl' and 'opencl -device=adreno' 'opencl -device=mali' <br>
    test_target = "opencl -device=adreno" <br>

    # Change target configuration. <br>
    # Run `adb shell cat /proc/cpuinfo` to find the arch. <br>
    arch = "arm64" <br>
    target = tvm.target.Target("llvm -mtriple=%s-linux-android" % arch) <br>

    # To enable OpenCLML accelerated operator library. <br>
    enable_clml = False <br>

    if is_tuning: <br>
        from tvm.driver.tvmc.transform import apply_graph_transforms <br>
        mod = apply_graph_transforms( <br>
            mod, <br>
            { <br>
                "mixed_precision": True, <br>
                "mixed_precision_ops": ["nn.conv2d", "nn.dense"], <br>
                "mixed_precision_calculation_type": calculation_dtype, <br>
                "mixed_precision_acc_type": acc_dtype, <br>
            },  <br>
        )
    else: <br>
        # mod = tvm.relay.transform.EliminateCommonSubexpr()(mod) <br>
        # # mod = tvm.relay.transform.function_pass(mod, opt_level=3) <br>
        # mod = tvm.relay.transform.InferType()(mod) <br>
        # mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod) <br>
        # mod = tvm.relay.transform.FoldConstant()(mod) <br>
        
        # mod = tvm.relay.transform.InferType()(mod) <br>
        # mod = tvm.relay.transform.ToMixedPrecision()(mod) <br>
        
        # mod = tvm.relay.transform.EliminateCommonSubexpr()(mod) <br>
        # mod = tvm.relay.transform.FoldConstant()(mod) <br>
        # relay.build(mod, target=target, params=params) <br>
        pass <br>


    if local_demo: <br>
        target = tvm.target.Target("llvm") <br>
    elif test_target.find("opencl"): <br>
        target = tvm.target.Target(test_target, host=target) <br>


    # Get RPC related settings. <br>
    rpc_tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1") <br>
    rpc_tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190)) <br>
    key = "android"     <br>

    # Auto tuning is compute intensive and time taking task. <br>
    # It is set to False in above configuration as this script runs in x86 for demonstration. <br>
    # Please to set :code:`is_tuning` to True to enable auto tuning. <br>
    if is_tuning: <br>
        # Auto Tuning Stage 1: Extract tunable tasks <br>
        tasks = autotvm.task.extract_from_program( <br>
            mod, target=test_target, target_host=target, params=params <br>
        ) <br>


        # Auto Tuning Stage 2: Define tuning configuration <br>
        tmp_log_file = tune_log + ".tmp" <br>
        measure_option = autotvm.measure_option( <br>
            builder=autotvm.LocalBuilder( <br>
                build_func=ndk.create_shared, timeout=60 <br>
            ),  # Build the test kernel locally <br>
            runner=autotvm.RPCRunner(  # The runner would be on a remote device. <br>
                key,  # RPC Key <br>
                host=rpc_tracker_host,  # Tracker host <br>
                port=int(rpc_tracker_port),  # Tracker port <br>
                number=3,  # Number of runs before averaging <br>
                timeout=60,  # RPC Timeout <br>
            ), <br>
        ) <br>
        n_trial = tuning_trail # 1024  # Number of iteration of training before choosing the best kernel config <br>
        early_stopping = False  # Can be enabled to stop tuning while the loss is not minimizing. <br>


        # Auto Tuning Stage 3: Iterate through the tasks and tune. <br>
        from tvm.autotvm.tuner import XGBTuner <br>


        # for i, tsk in enumerate(reversed(tasks[:3])): <br>
        for i, tsk in enumerate(reversed(tasks)): <br>
            print("Task:", tsk) <br>
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks)) <br>


            # choose tuner <br>
            tuner = "xgb" <br>

            # create tuner <br>
            if tuner == "xgb": <br>
                tuner_obj = XGBTuner(tsk, loss_type="reg") <br>
            elif tuner == "xgb_knob": <br>
                tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob") <br>
            elif tuner == "xgb_itervar": <br>
                tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar") <br>
            elif tuner == "xgb_curve": <br>
                tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve") <br>
            elif tuner == "xgb_rank": <br>
                tuner_obj = XGBTuner(tsk, loss_type="rank") <br>
            elif tuner == "xgb_rank_knob": <br>
                tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob") <br>
            elif tuner == "xgb_rank_itervar": <br>
                tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar") <br>
            elif tuner == "xgb_rank_curve": <br>
                tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve") <br>
            elif tuner == "xgb_rank_binary": <br>
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary") <br>
            elif tuner == "xgb_rank_binary_knob": <br>
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob") <br>
            elif tuner == "xgb_rank_binary_itervar": <br>
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar") <br>
            elif tuner == "xgb_rank_binary_curve": <br>
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve") <br>
            elif tuner == "ga": <br>
                tuner_obj = GATuner(tsk, pop_size=50) <br>
            elif tuner == "random": <br>
                tuner_obj = RandomTuner(tsk) <br>
            elif tuner == "gridsearch": <br>
                tuner_obj = GridSearchTuner(tsk) <br>
            else: <br>
                raise ValueError("Invalid tuner: " + tuner) <br>

            tsk_trial = min(n_trial, len(tsk.config_space)) <br>
            tuner_obj.tune( <br>
                n_trial=tsk_trial, <br>
                early_stopping=early_stopping, <br>
                measure_option=measure_option, <br>
                callbacks=[ <br>
                    autotvm.callback.progress_bar(tsk_trial, prefix=prefix), <br>
                    autotvm.callback.log_to_file(tmp_log_file), <br>
                ], <br>
            ) <br>
        # Auto Tuning Stage 4: Pick the best performing configurations from the overall log. <br>
        autotvm.record.pick_best(tmp_log_file, tune_log) <br>


    if not local_demo and enable_clml: <br>
        mod = clml.partition_for_clml(mod, params) <br>

    if os.path.exists(tune_log): <br>
        with autotvm.apply_history_best(tune_log): <br>
            with tvm.transform.PassContext(opt_level=3): <br>
                print("Optimizing with tuned results...") <br>
                lib = relay.build(mod, target=target, params=params) <br>
    else: <br>
        with tvm.transform.PassContext(opt_level=3): <br>
            lib = relay.build(mod, target=target, params=params) <br>


    if local_demo: <br> 
        remote = rpc.LocalSession() <br>
    else: <br>
        tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port) <br>
        # When running a heavy model, we should increase the `session_timeout` <br>
        remote = tracker.request(key, priority=0, session_timeout=60*60) # in 1 hour <br>

    if local_demo: <br> 
        dev = remote.cpu(0) <br>
    elif test_target.find("opencl"): <br>
        dev = remote.cl(0) <br>
    else: <br>
        dev = remote.cpu(0) <br>

    temp = utils.tempdir() <br>
    dso_binary = "dev_lib_cl.so" <br>
    dso_binary_path = temp.relpath(dso_binary) <br>
    fcompile = ndk.create_shared if not local_demo else None <br>
    lib.export_library(dso_binary_path, fcompile=fcompile) <br>
    remote_path = "/data/local/tmp/tvm_run" + dso_binary <br>
    remote.upload(dso_binary_path) <br>
    rlib = remote.load_module(dso_binary) <br>
    # https://github.com/apache/tvm/blob/main/tests/python/unittest/test_runtime_profiling.py#L180-L196 <br>
    # Profiling <br>
    
    # TODO: <br>
    # gr = debug_executor.create(lib.get_graph_json(), rlib, dev) <br>
    # report = gr.profile(data=input_data) <br>
    # print(report) <br>
 
    m = graph_executor.GraphModule(rlib["default"](dev)) <br>

    if isinstance(input_data, dict): <br>
        for input_name, input_data in input_data.items(): <br>
            m.set_input(input_name, tvm.nd.array(input_data)) <br>
    else: <br>
        m.set_input(input_name, tvm.nd.array(input_data)) <br>
    m.run() <br>
    tvm_output = m.get_output(0) <br>
    
    # Useful functions <br>
    print(f"Benchmarking {model_name} ...") <br>
    print(lib.get_graph_json()) <br>
    print(m.benchmark(dev, number=round, repeat=1)) <br>
    return <br>
    
    prof_res = list() <br>
    for i in tqdm(range(round)): <br>
        start_time = time.time() <br>
        m.run() <br>
        end_time = time.time() <br>
        prof_res.append((end_time - start_time) * 1000) <br>
    prof_res = np.array(prof_res) <br>
    prof_min = np.min(prof_res) <br>
    prof_max = np.max(prof_res) <br>
    prof_avg = np.mean(prof_res) <br>
    print(f"TVM's average latency for model {model_name} is {prof_avg:.1f} in the range of [{prof_min:.1f}, {prof_max:.1f}] ms\n\n") <br>

    # print("Start to evaluate...") <br>
    # ftimer = m.module.time_evaluator("run", dev, number=round, repeat=1, min_repeat_ms=1000) <br>
    # prof_res = np.array(ftimer().results) * 1000 <br>
    # print(f"TVM latency for model {model_name} is {np.mean(prof_res):.1f} in the range of [{np.min(prof_res):.1f}, {np.max(prof_res):.1f}] ms") <br>
    # time.sleep(60) <br>


def profile_torch_vision_model(model_name, input_shape=[1, 3, 224, 224],  <br>
                               input_name="input0", round=50, is_tuning=False, <br>
                               tuning_trail=1024): <br>
    model = getattr(torchvision.models, model_name)(pretrained=False) <br>
    model = model.eval() <br>
    input_data = torch.randn(input_shape) <br>
    scripted_model = torch.jit.trace(model, input_data).eval() <br>
    shape_list = [(input_name, input_shape)] <br>
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list) <br>
    benchmark_model(mod, params, input_data, model_name, input_name, round, is_tuning, tuning_trail) <br>


def profile_onnx_model(model_path, model_name=None, input_shape=[1, 3, 224, 224],  <br>
                       input_name="input0", round=50, is_tuning=False, <br>
                       tuning_trail=1024, dtype="float"): <br>
    onnx_model = onnx.load(model_path) <br>
    if dtype == "float": <br>
        input_data = torch.randn(input_shape) <br>
    elif dtype == "int32": <br>
        input_data = torch.randint(1, 10, input_shape, dtype=torch.int32) <br>
    else: <br>
        raise ValueError(f"Unsupported dtype: {dtype}") <br>
    shape_dict = {input_name: input_shape} <br>
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict) <br>
    # model_name is equal to the file name of model_path <br>
    if model_name is None: <br>
        model_name = os.path.splitext(os.path.basename(model_path))[0] <br>
    benchmark_model(mod, params, input_data, model_name, input_name, round, is_tuning, tuning_trail) <br>


def profile_onnx_model_multi_input(model_path, input_info, model_name=None, round=50, is_tuning=False, <br>
                       tuning_trail=1024): <br>
    onnx_model = onnx.load(model_path) <br>
    shape_dict = dict() <br>
    data_dict = dict() <br>
    for (input_name, input_shape, dtype) in input_info: <br>
        if dtype == "float": <br>
            input_data = torch.randn(input_shape) <br>
        elif dtype == "int32": <br>
            input_data = torch.randint(1, 10, input_shape, dtype=torch.int32) <br>
        else: <br>
            raise ValueError(f"Unsupported dtype: {dtype}") <br>
        shape_dict[input_name] = input_shape <br>
        data_dict[input_name] = input_data <br>
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict) <br>
    # model_name is equal to the file name of model_path <br>
    if model_name is None: <br>
        model_name = os.path.splitext(os.path.basename(model_path))[0] <br>
    benchmark_model(mod, params, data_dict, model_name, None, round, is_tuning, tuning_trail) <br>

def profile_convnext_tiny(): <br>
    # model_name = "resnet18" # This is the original model <br>
    model_name = "convnext_tiny" <br>
    model_name = "regnet_y_3_2gf" <br>
    model_name = "resnext50_32x4d" <br>
    model = getattr(torchvision.models, model_name)(pretrained=True) <br>
    model = model.eval() <br>
    input_shape = [1, 3, 224, 224] <br>
    input_data = torch.randn(input_shape) <br>
    scripted_model = torch.jit.trace(model, input_data).eval() <br>
    input_name = "input0" <br>
    shape_list = [(input_name, input_shape)] <br>
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list) <br>
    benchmark_model(mod, params, input_data) <br>


is_tuning = False <br>
tuning_trail = 64 # 128 <br>

# profile_torch_vision_model("resnet18", is_tuning=is_tuning, tuning_trail=tuning_trail) # work <br>

# profile_torch_vision_model("convnext_tiny", is_tuning=is_tuning, tuning_trail=tuning_trail) # work <br>
# profile_torch_vision_model("regnet_y_3_2gf", is_tuning=is_tuning, tuning_trail=tuning_trail) # work <br>
# profile_torch_vision_model("resnext50_32x4d", is_tuning=is_tuning, tuning_trail=tuning_trail) # work <br>
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/yolov8n.onnx", input_name="images", input_shape=[1, 3, 640, 640], is_tuning=is_tuning,  tuning_trail=tuning_trail) # work <br>
# # Done tuning <br>

# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/autoformer.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning,  tuning_trail=tuning_trail) # work <br>
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/biformer.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail) <br>
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/crossformer.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail) <br>
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/cswin.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail) # work <br>
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/efficientvit.onnx", input_name="input.1", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work <br>
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/flatten-pvt.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail) # work <br>
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/smt.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work <br>
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/swin.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail) # work <br>
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/vit.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail) # work <br>


# profile_onnx_model("/home/harry/Documents/tvm/models/vit-base.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
<br>

# profile_onnx_model("/home/harry/Documents/tvm/models/vit-small.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
<br>

# profile_onnx_model("/home/harry/Documents/tvm/models/vit-tiny.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
<br>

# profile_onnx_model("/home/harry/Documents/tvm/models/cp_vit-base.onnx", input_name="input_1", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
<br>

# profile_onnx_model("/home/harry/Documents/tvm/models/cp_vit-small.onnx", input_name="input_1", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
<br>

# profile_onnx_model("/home/harry/Documents/tvm/models/cp_vit-tiny.onnx", input_name="input_1", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
<br>


# profile_onnx_model("/home/harry/Documents/tvm/models/efficientnet-b0.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
<br>





# Encoder <br>
# profile_onnx_model("/home/harry/Documents/tvm/models/model.onnx", input_name="input_ids", input_shape=[1,77], is_tuning=is_tuning, tuning_trail=tuning_trail, dtype="int32") # work
<br>

# profile_onnx_model("/home/harry/Documents/tvm/models/sd32_decoder.onnx", input_name="latent_sample", input_shape=[1, 4, 32, 32], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
<br>


unet_32_input_info = [ <br>
    ("sample", [1, 4, 32, 32], "float"), <br>
    ("timestep", [1], "int32"), <br>
    ("encoder_hidden_state", [1, 77, 768], "float"), <br>
] <br>
profile_onnx_model_multi_input("/home/harry/Documents/tvm/models/unet32/model.onnx", input_info=unet_32_input_info, is_tuning=is_tuning, tuning_trail=tuning_trail) # work
<br>
<br>



## Useful link <br>
https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno.html#sphx-glr-how-to-deploy-models-deploy-model-on-adreno-py


https://tvm.apache.org/docs/how_to/compile_models/from_onnx.html
<br><br>

# NCNN Tutorial <br>


# If you want to enable Vulkan, platform api version >= android-24 is needed <br>


https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android <br>
<br>

## Build for benchmark <br>




cmake -H. -Bbuild/build_android64                                              \ <br> 
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"  \ <br>
    -DANDROID_ABI="arm64-v8a"                                                  \ <br>
    -DANDROID_PLATFORM=android-24                                              \ <br>
    -DNCNN_VULKAN=ON                                                           \ <br>
    -DNCNN_THREADS=ON                                                          \ <br>
    -DCMAKE_BUILD_TYPE=Release                                                 \ <br>
    -DNCNN_BUILD_BENCHMARK=ON                                                   <br>
adb shell "mkdir -p /data/local/tmp/ncnn_run/attention"; <br>
cmake --build build/build_android64 --parallel 16; <br>
adb push build/build_android64/benchmark/benchncnn /data/local/tmp/ncnn_run; <br>

<br> <br>

## Build for convert <br>



cmake -H. -Bbuild/build_amd64                                                  \ <br>
    -DNCNN_BUILD_TOOLS=ON                                                      \ <br>
    -DNCNN_THREADS=ON                                                          \ <br>
    -DCMAKE_BUILD_TYPE=RelWithDebInfo                                          \ <br>
    -DNCNN_BUILD_BENCHMARK=ON                                                   <br>
cmake --build build/build_amd64 --parallel 16; <br>


<br>


## Convert model <br>





model_base_name="/home/newway/Documents/Research/DSP/DSPModels/fst/fsts_cutted"; build/build_amd64/tools/onnx/onnx2ncnn $model_base_name.onnx $model_base_name.param $model_base_name.bin
<br>

adb push $model_base_name.param /data/local/tmp/ncnn_run/attention/; adb push $model_base_name.bin /data/local/tmp/ncnn_run/attention/;
<br>

model_base_name="/home/newway/Documents/Research/LargeModel/CV/1.Other/regnet/regnet_y_3_2gf-nomodule-opt"; build/build_amd64/tools/onnx/onnx2ncnn $model_base_name.onnx $model_base_name.param $model_base_name.bin
<br>

adb push $model_base_name.param /data/local/tmp/ncnn_run/attention/; adb push $model_base_name.bin /data/local/tmp/ncnn_run/attention/;
<br>

model_base_name="/home/newway/Documents/Research/LargeModel/CV/1.Other/resnext/resnext50_32x4d-nomodule-opt"; build/build_amd64/tools/onnx/onnx2ncnn $model_base_name.onnx $model_base_name.param $model_base_name.bin
<br>

adb push $model_base_name.param /data/local/tmp/ncnn_run/attention/; adb push $model_base_name.bin /data/local/tmp/ncnn_run/attention/;
<br>

model_base_name="/home/newway/Documents/Research/LargeModel/CV/1.Other/convnext/convnext-tiny-nomodule-opt"; build/build_amd64/tools/onnx/onnx2ncnn $model_base_name.onnx $model_base_name.param $model_base_name.bin
<br>

adb push $model_base_name.param /data/local/tmp/ncnn_run/attention/; adb push $model_base_name.bin /data/local/tmp/ncnn_run/attention/;
<br>

model_base_name="/home/newway/Documents/Research/Models/ultralytics/yolov8n"; build/build_amd64/tools/onnx/onnx2ncnn $model_base_name.onnx $model_base_name.param $model_base_name.bin
<br>

adb push $model_base_name.param /data/local/tmp/ncnn_run/attention/; adb push $model_base_name.bin /data/local/tmp/ncnn_run/attention/;
<br><br>




## Run benchmark <br>


adb shell "/data/local/tmp/ncnn_run/benchncnn 30 4 2 0 0 param=/data/local/tmp/ncnn_run/attention/regnet_y_3_2gf-nomodule-opt.param shape=[224,224,3]";
<br>

adb shell "/data/local/tmp/ncnn_run/benchncnn 30 4 2 0 0 param=/data/local/tmp/ncnn_run/attention/resnext50_32x4d-nomodule-opt.param shape=[224,224,3]";
<br>

adb shell "/data/local/tmp/ncnn_run/benchncnn 30 4 2 0 0 param=/data/local/tmp/ncnn_run/attention/yolov8n.param shape=[640,640,3]";
<br>

<!-- LayerNorm does not support -->
<br>

adb shell "/data/local/tmp/ncnn_run/benchncnn 30 4 2 0 0 param=/data/local/tmp/ncnn_run/attention/convnext-tiny-nomodule-opt.param shape=[224,224,3]"
<br>
<br>

# TFLite Tutorial
<br>

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark
<br>

https://www.tensorflow.org/lite/android/lite_build
<br>
<br>

# Guidence for setting up a benchmark environment
<br>

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark
<br>

https://www.tensorflow.org/lite/android/lite_build
<br>

https://www.tensorflow.org/lite/performance/measurement
<br>

https://www.tensorflow.org/install/source
<br>

https://stackoverflow.com/questions/26333823/clang-doesnt-see-basic-headers
<br>

## Build
<br>

### For Android
<br>



shell
<br>

sudo apt install g++-12
<br><br>

# Make sure Android NDK Version == 25c
<br>

export ANDROID_NDK_HOME=/home/newway/Android/android-ndk-r25c
<br>

./configure
<br>

bazelisk build -c opt --config=android_arm64 tensorflow/lite/tools/benchmark:benchmark_model
<br>

bazelisk build -c opt --config=android_arm64  //tensorflow/lite/examples/label_image:label_image
<br>
<br>


### For Local
<br>




- bazelisk build -c opt --config=linux tensorflow/lite/tools/benchmark:benchmark_model
<br>

adb shell "mkdir -p /data/local/tmp/tflite_run/models"
<br>

adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/convnext_tiny_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 
--use_nnapi=true --nnapi_allow_fp16=false
<br>
<br>

## Prepare the model and executable
<br>

shell
<br>

# Prepare the models
<br>


adb shell "mkdir -p /data/local/tmp/tflite_run/models"
<br>

adb push ./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp/tflite_run
<br>

adb push /home/newway/Documents/Research/LargeModel/CV/2.TFLite/models/*.tflite /data/local/tmp/tflite_run/models
<br>
<br>



# SWin
<br>




adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/swin_tiny_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 --use_gpu=true --gpu_experimental_enable_quant=false
<br>
<br>


# ConvNext
<br>




adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/convnext_tiny_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 --use_gpu=true --gpu_experimental_enable_quant=false
<br>
<br>


# RegNet: succeed
<br>



adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/regnety032_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 --use_gpu=true --gpu_experimental_enable_quant=false
<br>
<br>

# ResNext50: succeed
<br>




adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/resnext50_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 --use_gpu=true --gpu_experimental_enable_quant=false
<br>

adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/resnext101_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 --use_gpu=true --gpu_experimental_enable_quant=false
<br>
