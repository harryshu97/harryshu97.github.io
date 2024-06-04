---
layout: default
title: testing tools
---
# TVM run model tutorial
https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno.html#sphx-glr-how-to-deploy-models-deploy-model-on-adreno-py
https://tvm.apache.org/docs/how_to/deploy/adreno.html
## Build 
- https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno.html#sphx-glr-how-to-deploy-models-deploy-model-on-adreno-py
- https://tvm.apache.org/docs/how_to/deploy/adreno.html

```
./docker/build.sh ci_adreno
docker tag tvm.ci_adreno ci_adreno
export ADRENO_OPENCL=/home/newway/opt/Snapdragon/OpenCLML
./tests/scripts/ci.py adreno -i
```


## Run
```
https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno.html#sphx-glr-how-to-deploy-models-deploy-model-on-adreno-py
bash
```

## Start container
```
export ADRENO_OPENCL=/home/newway/opt/Snapdragon/OpenCLML
./tests/scripts/ci.py adreno -i
```

## After the docker is built
```
export ANDROID_SERIAL=832358d4
export TVM_HOME=/home/newway/Documents/Research/CNN/TVM/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
adb -s $ANDROID_SERIAL shell "mkdir /data/local/tmp/tvm_run"
adb -s $ANDROID_SERIAL push {build-adreno-target/libtvm_runtime.so,build-adreno-target/tvm_rpc} /data/local/tmp/tvm_run
find ${ANDROID_NDK_HOME} -name libc++_shared.so
```

## open a terminal within the docker
```
export ANDROID_SERIAL=832358d4
python3 -m tvm.exec.rpc_tracker --port 9190
```

## open another terminal out of the docker
```
export ANDROID_SERIAL=832358d4
adb -s $ANDROID_SERIAL reverse tcp:9190 tcp:9190
adb -s $ANDROID_SERIAL forward tcp:5000 tcp:5000
adb -s $ANDROID_SERIAL forward tcp:5002 tcp:5001
adb -s $ANDROID_SERIAL forward tcp:5003 tcp:5002
adb -s $ANDROID_SERIAL forward tcp:5004 tcp:5003
adb -s $ANDROID_SERIAL shell LD_LIBRARY_PATH=/data/local/tmp/tvm_run /data/local/tmp/tvm_run/tvm_rpc server --host=0.0.0.0 --port=5000 --tracker=127.0.0.1:9190 --key=android --port-end=5100
```

## open the third terminal within the docker
```
export TVM_TRACKER_HOST=0.0.0.0
export TVM_TRACKER_PORT=9190
python -m tvm.exec.query_rpc_tracker --port 9190
export TVM_NDK_CC=/opt/android-sdk-linux/ndk/25.2.9519653/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++
python ./run_opencl.py
```

## TVM run_opencl.py file
```
# https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno.html#sphx-glr-how-to-deploy-models-deploy-model-on-adreno-py
# https://tvm.apache.org/docs/how_to/compile_models/from_onnx.html
import os
import time
import torch
import torchvision
import numpy as np
import onnx
from tqdm import tqdm
import tvm
from tvm import te
from tvm import relay, rpc
from tvm.contrib import utils, ndk
from tvm.contrib import graph_executor
from tvm.relay.op.contrib import clml
from tvm import autotvm
from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
from tvm.contrib.debugger import debug_executor
# https://discuss.tvm.apache.org/t/how-could-us-use-tvm-relay-transform-tomixedprecision/10465/4
# https://github.com/apache/tvm/blob/main/tests/python/unittest/test_runtime_profiling.py#L180-L196
# from tvm.relay.transform import mixed_precision # Register op attribute


def benchmark_model(mod, params, input_data, 
                    model_name, input_name="input0", 
                    round=50, is_tuning=False, tuning_trail=1024):
    # Below are set of configuration that controls the behaviour of this script like
    # local run or device run, target definitions,  dtype setting and auto tuning enablement.
    # Change these settings as needed if required.

    # Adreno devices are efficient with float16 compared to float32
    # Given the expected output doesn't effect by lowering precision
    # it's advisable to use lower precision.
    # We have a helper API to make the precision conversion simple and
    # it supports dtype with "float16" and "float16_acc32" modes.
    # Let's choose "float16" for calculation and "float32" for accumulation.
    calculation_dtype = "float16"
    acc_dtype = "float32"

    # Auto tuning is compute intensive and time taking task,
    # hence disabling for default run. Please enable it if required.
    tune_log = f"adreno-{model_name}.log"

    # Specify Adreno target before compiling to generate texture
    # leveraging kernels and get all the benefits of textures
    # Note: This generated example running on our x86 server for demonstration.
    # If running it on the Android device, we need to
    # specify its instruction set. Set :code:`local_demo` to False if you want
    # to run this tutorial with a real device over rpc.
    local_demo = False

    # by default on CPU target will execute.
    # select 'cpu', 'opencl' and 'opencl -device=adreno' 'opencl -device=mali'
    test_target = "opencl -device=adreno"

    # Change target configuration.
    # Run `adb shell cat /proc/cpuinfo` to find the arch.
    arch = "arm64"
    target = tvm.target.Target("llvm -mtriple=%s-linux-android" % arch)

    # To enable OpenCLML accelerated operator library.
    enable_clml = False

    if is_tuning:
        from tvm.driver.tvmc.transform import apply_graph_transforms
        mod = apply_graph_transforms(
            mod,
            {
                "mixed_precision": True,
                "mixed_precision_ops": ["nn.conv2d", "nn.dense"],
                "mixed_precision_calculation_type": calculation_dtype,
                "mixed_precision_acc_type": acc_dtype,
            },
        )
    else:
        # mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        # # mod = tvm.relay.transform.function_pass(mod, opt_level=3)
        # mod = tvm.relay.transform.InferType()(mod)
        # mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        # mod = tvm.relay.transform.FoldConstant()(mod)
        
        # mod = tvm.relay.transform.InferType()(mod)
        # mod = tvm.relay.transform.ToMixedPrecision()(mod)
        
        # mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        # mod = tvm.relay.transform.FoldConstant()(mod)
        # relay.build(mod, target=target, params=params)
        pass

    if local_demo:
        target = tvm.target.Target("llvm")
    elif test_target.find("opencl"):
        target = tvm.target.Target(test_target, host=target)


    # Get RPC related settings.
    rpc_tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
    rpc_tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
    key = "android"    

    # Auto tuning is compute intensive and time taking task.
    # It is set to False in above configuration as this script runs in x86 for demonstration.
    # Please to set :code:`is_tuning` to True to enable auto tuning.
    if is_tuning:
        # Auto Tuning Stage 1: Extract tunable tasks
        tasks = autotvm.task.extract_from_program(
            mod, target=test_target, target_host=target, params=params
        )

        # Auto Tuning Stage 2: Define tuning configuration
        tmp_log_file = tune_log + ".tmp"
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(
                build_func=ndk.create_shared, timeout=60
            ),  # Build the test kernel locally
            runner=autotvm.RPCRunner(  # The runner would be on a remote device.
                key,  # RPC Key
                host=rpc_tracker_host,  # Tracker host
                port=int(rpc_tracker_port),  # Tracker port
                number=3,  # Number of runs before averaging
                timeout=60,  # RPC Timeout
            ),
        )
        n_trial = tuning_trail # 1024  # Number of iteration of training before choosing the best kernel config
        early_stopping = False  # Can be enabled to stop tuning while the loss is not minimizing.

        # Auto Tuning Stage 3: Iterate through the tasks and tune.
        from tvm.autotvm.tuner import XGBTuner

        # for i, tsk in enumerate(reversed(tasks[:3])):
        for i, tsk in enumerate(reversed(tasks)):
            print("Task:", tsk)
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

            # choose tuner
            tuner = "xgb"

            # create tuner
            if tuner == "xgb":
                tuner_obj = XGBTuner(tsk, loss_type="reg")
            elif tuner == "xgb_knob":
                tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
            elif tuner == "xgb_itervar":
                tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
            elif tuner == "xgb_curve":
                tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
            elif tuner == "xgb_rank":
                tuner_obj = XGBTuner(tsk, loss_type="rank")
            elif tuner == "xgb_rank_knob":
                tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
            elif tuner == "xgb_rank_itervar":
                tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
            elif tuner == "xgb_rank_curve":
                tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
            elif tuner == "xgb_rank_binary":
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
            elif tuner == "xgb_rank_binary_knob":
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
            elif tuner == "xgb_rank_binary_itervar":
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
            elif tuner == "xgb_rank_binary_curve":
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
            elif tuner == "ga":
                tuner_obj = GATuner(tsk, pop_size=50)
            elif tuner == "random":
                tuner_obj = RandomTuner(tsk)
            elif tuner == "gridsearch":
                tuner_obj = GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            tsk_trial = min(n_trial, len(tsk.config_space))
            tuner_obj.tune(
                n_trial=tsk_trial,
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                    autotvm.callback.log_to_file(tmp_log_file),
                ],
            )
        # Auto Tuning Stage 4: Pick the best performing configurations from the overall log.
        autotvm.record.pick_best(tmp_log_file, tune_log)


    if not local_demo and enable_clml:
        mod = clml.partition_for_clml(mod, params)

    if os.path.exists(tune_log):
        with autotvm.apply_history_best(tune_log):
            with tvm.transform.PassContext(opt_level=3):
                print("Optimizing with tuned results...")
                lib = relay.build(mod, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)


    if local_demo:
        remote = rpc.LocalSession()
    else:
        tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
        # When running a heavy model, we should increase the `session_timeout`
        remote = tracker.request(key, priority=0, session_timeout=60*60) # in 1 hour

    if local_demo:
        dev = remote.cpu(0)
    elif test_target.find("opencl"):
        dev = remote.cl(0)
    else:
        dev = remote.cpu(0)

    temp = utils.tempdir()
    dso_binary = "dev_lib_cl.so"
    dso_binary_path = temp.relpath(dso_binary)
    fcompile = ndk.create_shared if not local_demo else None
    lib.export_library(dso_binary_path, fcompile=fcompile)
    remote_path = "/data/local/tmp/tvm_run" + dso_binary
    remote.upload(dso_binary_path)
    rlib = remote.load_module(dso_binary)
    # https://github.com/apache/tvm/blob/main/tests/python/unittest/test_runtime_profiling.py#L180-L196
    # Profiling
    
    # TODO:
    # gr = debug_executor.create(lib.get_graph_json(), rlib, dev)
    # report = gr.profile(data=input_data)
    # print(report)

    m = graph_executor.GraphModule(rlib["default"](dev))

    if isinstance(input_data, dict):
        for input_name, input_data in input_data.items():
            m.set_input(input_name, tvm.nd.array(input_data))
    else:
        m.set_input(input_name, tvm.nd.array(input_data))
    m.run()
    tvm_output = m.get_output(0)
    
    # Useful functions
    print(f"Benchmarking {model_name} ...")
    print(lib.get_graph_json())
    print(m.benchmark(dev, number=round, repeat=1))
    return
    
    prof_res = list()
    for i in tqdm(range(round)):
        start_time = time.time()
        m.run()
        end_time = time.time()
        prof_res.append((end_time - start_time) * 1000)
    prof_res = np.array(prof_res)
    prof_min = np.min(prof_res)
    prof_max = np.max(prof_res)
    prof_avg = np.mean(prof_res)
    print(f"TVM's average latency for model {model_name} is {prof_avg:.1f} in the range of [{prof_min:.1f}, {prof_max:.1f}] ms\n\n")

    # print("Start to evaluate...")
    # ftimer = m.module.time_evaluator("run", dev, number=round, repeat=1, min_repeat_ms=1000)
    # prof_res = np.array(ftimer().results) * 1000
    # print(f"TVM latency for model {model_name} is {np.mean(prof_res):.1f} in the range of [{np.min(prof_res):.1f}, {np.max(prof_res):.1f}] ms")
    # time.sleep(60)


def profile_torch_vision_model(model_name, input_shape=[1, 3, 224, 224], 
                               input_name="input0", round=50, is_tuning=False,
                               tuning_trail=1024):
    model = getattr(torchvision.models, model_name)(pretrained=False)
    model = model.eval()
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    benchmark_model(mod, params, input_data, model_name, input_name, round, is_tuning, tuning_trail)


def profile_onnx_model(model_path, model_name=None, input_shape=[1, 3, 224, 224], 
                       input_name="input0", round=50, is_tuning=False,
                       tuning_trail=1024, dtype="float"):
    onnx_model = onnx.load(model_path)
    if dtype == "float":
        input_data = torch.randn(input_shape)
    elif dtype == "int32":
        input_data = torch.randint(1, 10, input_shape, dtype=torch.int32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    # model_name is equal to the file name of model_path
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
    benchmark_model(mod, params, input_data, model_name, input_name, round, is_tuning, tuning_trail)


def profile_onnx_model_multi_input(model_path, input_info, model_name=None, round=50, is_tuning=False,
                       tuning_trail=1024):
    onnx_model = onnx.load(model_path)
    shape_dict = dict()
    data_dict = dict()
    for (input_name, input_shape, dtype) in input_info:
        if dtype == "float":
            input_data = torch.randn(input_shape)
        elif dtype == "int32":
            input_data = torch.randint(1, 10, input_shape, dtype=torch.int32)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        shape_dict[input_name] = input_shape
        data_dict[input_name] = input_data
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    # model_name is equal to the file name of model_path
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
    benchmark_model(mod, params, data_dict, model_name, None, round, is_tuning, tuning_trail)

def profile_convnext_tiny():
    # model_name = "resnet18" # This is the original model
    model_name = "convnext_tiny"
    model_name = "regnet_y_3_2gf"
    model_name = "resnext50_32x4d"
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    benchmark_model(mod, params, input_data)


is_tuning = False
tuning_trail = 64 # 128

# profile_torch_vision_model("resnet18", is_tuning=is_tuning, tuning_trail=tuning_trail) # work

# profile_torch_vision_model("convnext_tiny", is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_torch_vision_model("regnet_y_3_2gf", is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_torch_vision_model("resnext50_32x4d", is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/yolov8n.onnx", input_name="images", input_shape=[1, 3, 640, 640], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# # Done tuning

# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/autoformer.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/biformer.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail)
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/crossformer.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail)
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/cswin.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/efficientvit.onnx", input_name="input.1", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/flatten-pvt.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/smt.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/swin.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/newway/Documents/Research/CNN/TVM/tvm/vit.onnx", input_name="input", input_shape=[1, 3, 224, 224], round=50, is_tuning=is_tuning, tuning_trail=tuning_trail) # work


# profile_onnx_model("/home/harry/Documents/tvm/models/vit-base.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/harry/Documents/tvm/models/vit-small.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/harry/Documents/tvm/models/vit-tiny.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/harry/Documents/tvm/models/cp_vit-base.onnx", input_name="input_1", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/harry/Documents/tvm/models/cp_vit-small.onnx", input_name="input_1", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work
# profile_onnx_model("/home/harry/Documents/tvm/models/cp_vit-tiny.onnx", input_name="input_1", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work


# profile_onnx_model("/home/harry/Documents/tvm/models/efficientnet-b0.onnx", input_name="input", input_shape=[1, 3, 224, 224], is_tuning=is_tuning, tuning_trail=tuning_trail) # work






# Encoder
# profile_onnx_model("/home/harry/Documents/tvm/models/model.onnx", input_name="input_ids", input_shape=[1,77], is_tuning=is_tuning, tuning_trail=tuning_trail, dtype="int32") # work

# profile_onnx_model("/home/harry/Documents/tvm/models/sd32_decoder.onnx", input_name="latent_sample", input_shape=[1, 4, 32, 32], is_tuning=is_tuning, tuning_trail=tuning_trail) # work

unet_32_input_info = [
    ("sample", [1, 4, 32, 32], "float"),
    ("timestep", [1], "int32"),
    ("encoder_hidden_state", [1, 77, 768], "float"),
]
profile_onnx_model_multi_input("/home/harry/Documents/tvm/models/unet32/model.onnx", input_info=unet_32_input_info, is_tuning=is_tuning, tuning_trail=tuning_trail) # work

```

## Useful link
https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno.html#sphx-glr-how-to-deploy-models-deploy-model-on-adreno-py
https://tvm.apache.org/docs/how_to/compile_models/from_onnx.html

# NCNN Tutorial
# If you want to enable Vulkan, platform api version >= android-24 is needed

https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android

## Build for benchmark
```
cmake -H. -Bbuild/build_android64                                              \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"  \
    -DANDROID_ABI="arm64-v8a"                                                  \
    -DANDROID_PLATFORM=android-24                                              \
    -DNCNN_VULKAN=ON                                                           \
    -DNCNN_THREADS=ON                                                          \
    -DCMAKE_BUILD_TYPE=Release                                                 \
    -DNCNN_BUILD_BENCHMARK=ON                                                  
adb shell "mkdir -p /data/local/tmp/ncnn_run/attention";
cmake --build build/build_android64 --parallel 16;
adb push build/build_android64/benchmark/benchncnn /data/local/tmp/ncnn_run;
```

## Build for convert
```
cmake -H. -Bbuild/build_amd64                                                  \
    -DNCNN_BUILD_TOOLS=ON                                                      \
    -DNCNN_THREADS=ON                                                          \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo                                          \
    -DNCNN_BUILD_BENCHMARK=ON                                                  
cmake --build build/build_amd64 --parallel 16;
```

## Convert model
```
model_base_name="/home/newway/Documents/Research/DSP/DSPModels/fst/fsts_cutted"; build/build_amd64/tools/onnx/onnx2ncnn $model_base_name.onnx $model_base_name.param $model_base_name.bin
adb push $model_base_name.param /data/local/tmp/ncnn_run/attention/; adb push $model_base_name.bin /data/local/tmp/ncnn_run/attention/;

model_base_name="/home/newway/Documents/Research/LargeModel/CV/1.Other/regnet/regnet_y_3_2gf-nomodule-opt"; build/build_amd64/tools/onnx/onnx2ncnn $model_base_name.onnx $model_base_name.param $model_base_name.bin
adb push $model_base_name.param /data/local/tmp/ncnn_run/attention/; adb push $model_base_name.bin /data/local/tmp/ncnn_run/attention/;

model_base_name="/home/newway/Documents/Research/LargeModel/CV/1.Other/resnext/resnext50_32x4d-nomodule-opt"; build/build_amd64/tools/onnx/onnx2ncnn $model_base_name.onnx $model_base_name.param $model_base_name.bin
adb push $model_base_name.param /data/local/tmp/ncnn_run/attention/; adb push $model_base_name.bin /data/local/tmp/ncnn_run/attention/;

model_base_name="/home/newway/Documents/Research/LargeModel/CV/1.Other/convnext/convnext-tiny-nomodule-opt"; build/build_amd64/tools/onnx/onnx2ncnn $model_base_name.onnx $model_base_name.param $model_base_name.bin
adb push $model_base_name.param /data/local/tmp/ncnn_run/attention/; adb push $model_base_name.bin /data/local/tmp/ncnn_run/attention/;

model_base_name="/home/newway/Documents/Research/Models/ultralytics/yolov8n"; build/build_amd64/tools/onnx/onnx2ncnn $model_base_name.onnx $model_base_name.param $model_base_name.bin
adb push $model_base_name.param /data/local/tmp/ncnn_run/attention/; adb push $model_base_name.bin /data/local/tmp/ncnn_run/attention/;
```

## Run benchmark
```
adb shell "/data/local/tmp/ncnn_run/benchncnn 30 4 2 0 0 param=/data/local/tmp/ncnn_run/attention/regnet_y_3_2gf-nomodule-opt.param shape=[224,224,3]";
adb shell "/data/local/tmp/ncnn_run/benchncnn 30 4 2 0 0 param=/data/local/tmp/ncnn_run/attention/resnext50_32x4d-nomodule-opt.param shape=[224,224,3]";
adb shell "/data/local/tmp/ncnn_run/benchncnn 30 4 2 0 0 param=/data/local/tmp/ncnn_run/attention/yolov8n.param shape=[640,640,3]";
<!-- LayerNorm does not support -->
adb shell "/data/local/tmp/ncnn_run/benchncnn 30 4 2 0 0 param=/data/local/tmp/ncnn_run/attention/convnext-tiny-nomodule-opt.param shape=[224,224,3]"
```

# TFLite Tutorial
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark
https://www.tensorflow.org/lite/android/lite_build

# Guidence for setting up a benchmark environment
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark
https://www.tensorflow.org/lite/android/lite_build
https://www.tensorflow.org/lite/performance/measurement
https://www.tensorflow.org/install/source
https://stackoverflow.com/questions/26333823/clang-doesnt-see-basic-headers
## Build
### For Android
```
shell
sudo apt install g++-12
# Make sure Android NDK Version == 25c
export ANDROID_NDK_HOME=/home/newway/Android/android-ndk-r25c
./configure
bazelisk build -c opt --config=android_arm64 tensorflow/lite/tools/benchmark:benchmark_model
bazelisk build -c opt --config=android_arm64  //tensorflow/lite/examples/label_image:label_image
```

### For Local
```
- bazelisk build -c opt --config=linux tensorflow/lite/tools/benchmark:benchmark_model
adb shell "mkdir -p /data/local/tmp/tflite_run/models"
adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/convnext_tiny_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 --use_nnapi=true --nnapi_allow_fp16=false
```

## Prepare the model and executable
shell
# Prepare the models
```
adb shell "mkdir -p /data/local/tmp/tflite_run/models"
adb push ./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp/tflite_run
adb push /home/newway/Documents/Research/LargeModel/CV/2.TFLite/models/*.tflite /data/local/tmp/tflite_run/models
```

# SWin
```
adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/swin_tiny_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 --use_gpu=true --gpu_experimental_enable_quant=false
```

# ConvNext
```
adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/convnext_tiny_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 --use_gpu=true --gpu_experimental_enable_quant=false
```

# RegNet: succeed
```
adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/regnety032_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 --use_gpu=true --gpu_experimental_enable_quant=false
```

# ResNext50: succeed
```
adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/resnext50_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 --use_gpu=true --gpu_experimental_enable_quant=false
adb shell /data/local/tmp/tflite_run/benchmark_model --graph=/data/local/tmp/tflite_run/models/resnext101_float.tflite --warmup_runs=5 --num_runs=50 --num_threads=4 --use_gpu=true --gpu_experimental_enable_quant=false
```