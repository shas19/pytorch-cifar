
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import torch
from torchvision.transforms import Normalize
import torchvision
import torchvision.transforms as transforms
from models import *
from convert import *

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
BATCH_SIZE = 1
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def prepare_data():
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)

    return testloader

def build_model(testloader):
    print('==> Building model..')

    net = AlexNet2()
    net = net.to(device)
    # Use HPVM checkpoints
    assert os.path.isdir('model_params/pytorch'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./model_params/pytorch/alexnet2_cifar10.pth.tar')
    net.load_state_dict(checkpoint)

    net.eval()

    inputs, _ = next(iter(testloader))
    torch.onnx.export(net,  # model being run
                    inputs,  # model input (or a tuple for multiple inputs)
                    'alexnet_hpvm.onnx',  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
            )


def run_calibrator(testloader):
    print("Initialize calibrator")

    calibration_cache = "calibration_cache_alexnet.txt"
    step = 1
    calib_limit = 100
    onnx_filename = "alexnet_hpvm.onnx"
    engine_filename = "alexnet_eng.trt"
    json_file = "alexnet_out.json"

    calibrator = AlexNet2EntropyCalibrator(cache_file=calibration_cache, batch_size=step, 
        total_images=calib_limit, testloader=testloader)
    print("Build Engine")
    engine = get_engine(onnx_filename, engine_filename, batch_size=step, calibrator=calibrator)

    json_data = read_calibtable_txt2json(calibration_cache)
    dump_json(json_data, json_file)

class AlexNet2EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size, total_images, testloader):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0
        self.limit = total_images

        # Allocate enough memory for a whole batch.
        input_batch = next(iter(testloader))[0].numpy()# Model
        self.device_input = cuda.mem_alloc(1 * input_batch.nbytes)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.limit:
            return None

        current_batch = int(self.current_index / self.batch_size)
        print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        # loaded as RGB, convert to BGR
        # batch = np.ascontiguousarray(preprocess_input(np.load("data/ImageNet2012/x_val_%d_%d.npy" % (self.current_index, self.current_index+self.batch_size))))
        batch = input_batch
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        print('LOL ', dir(cache), type(cache), cache.shape)
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def get_engine(onnx_file_path, engine_file_path, batch_size, calibrator=None):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine(calibrator=None):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            # Parse model file
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            print('Completed parsing of ONNX file')
            
            print('Network inputs:')
            for i in range(network.num_inputs):
                tensor = network.get_input(i)
                print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)
                
            network.get_input(0).shape = [batch_size, 3, 32, 32]

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 20  # 256MiB
            if calibrator:
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calibrator

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_engine(network, config)

            print("Completed creating Engine. Writing file to: {}".format(engine_file_path))

            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(calibrator)

testloader = prepare_data()
build_model(testloader)
run_calibrator(testloader)