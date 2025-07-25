import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
from cuda import cudart

from utils import common
from image_batch import ImageBatcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = common.cuda_call(cudart.cudaMalloc(size))
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            common.memcpy_host_to_device(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """
    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30))
        # self.config.max_workspace_size = workspace * (2 ** 30)  # Deprecation

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path, end2end, conf_thres, iou_thres, max_det, **kwargs):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        v8 = kwargs['v8']
        v10 = kwargs['v10']
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        class_agnostic = not kwargs['no_class_agnostic']

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
        
        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                sys.exit(1)
        


        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
        print("Network Description")

        # #重新设置输入形状
        # for input in inputs:
        #     # 设置动态形状：将 batch 维度设为 -1 表示动态
        #     profile = self.builder.create_optimization_profile()
        #     min_shape = [1] + list(input.shape[1:])  # 最小尺寸
        #     opt_shape = [8] + list(input.shape[1:])  # 推荐尺寸
        #     max_shape = [32] + list(input.shape[1:]) # 最大尺寸

        #     profile.set_shape_input(input.name, min=min_shape, opt=opt_shape, max=max_shape)
        #     self.config.add_optimization_profile(profile)

        #     print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))


        for input in inputs:
            self.batch_size = input.shape[0]
            print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        assert self.batch_size > 0
        # self.builder.max_batch_size = self.batch_size  # This no effect for networks created with explicit batch dimension mode. Also DEPRECATED.

        if v10:
            try:
                for previous_output in outputs:
                    self.network.unmark_output(previous_output)
            except:
                previous_output = self.network.get_output(0)
                self.network.unmark_output(previous_output)
            # output [1, 300, 6]
            # 添加 TopK 层，在第二个维度上找到前 100 个最大值 [1, 100, 6]
            strides = trt.Dims([1,1,1])
            starts = trt.Dims([0,0,0])
            bs, num_boxes, temp = previous_output.shape
            shapes = trt.Dims([bs, num_boxes, 4])
            boxes = self.network.add_slice(previous_output, starts, shapes, strides)
            starts[2] = 4
            shapes[2] = 1
            # [0, 0, 4] [1, 300, 1] [1, 1, 1]
            obj_score = self.network.add_slice(previous_output, starts, shapes, strides)
            starts[2] = 5
            # [0, 0, 5] [1, 300, 1] [1, 1, 1]
            cls = self.network.add_slice(previous_output, starts, shapes, strides)
            outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
            print("YOLOv10 Modify")
            def squeeze(previous_output):
                reshape_dims = (bs, 300)
                previous_output = self.network.add_shuffle(previous_output.get_output(0))
                previous_output.reshape_dims    = reshape_dims
                return previous_output

            # 定义常量值和形状
            constant_value = 300.0
            constant_shape = (300,)
            constant_data = np.full(constant_shape, constant_value, dtype=np.float32)
            num = self.network.add_constant(constant_shape, trt.Weights(constant_data))
            num.get_output(0).name = "num"
            self.network.mark_output(num.get_output(0))
            boxes.get_output(0).name = "boxes"
            self.network.mark_output(boxes.get_output(0))
            obj_score= squeeze(obj_score)
            obj_score.get_output(0).name = "scores"
            self.network.mark_output(obj_score.get_output(0))
            cls = squeeze(cls)
            cls.get_output(0).name = "classes"
            self.network.mark_output(cls.get_output(0))

            for output in outputs:
                print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))

        if end2end and not v10:
            try:
                for previous_output in outputs:
                    self.network.unmark_output(previous_output)
            except:
                previous_output = self.network.get_output(0)
                self.network.unmark_output(previous_output)
            if  v8:
               # output [1, 84, 8400]
                strides = trt.Dims([1,1,1])
                starts = trt.Dims([0,0,0])
                previous_output = self.network.add_shuffle(previous_output)
                previous_output.second_transpose    = (0, 2, 1)
                # output [1, 8400, 84]
                bs, num_boxes, temp = previous_output.get_output(0).shape
                shapes = trt.Dims([bs, num_boxes, 4])
                # [0, 0, 0] [1, 8400, 4] [1, 1, 1]
                boxes = self.network.add_slice(previous_output.get_output(0), starts, shapes, strides)
                num_classes = temp -4
                starts[2] = 4
                shapes[2] = num_classes
                # [0, 0, 4] [1, 8400, 80] [1, 1, 1]
                scores = self.network.add_slice(previous_output.get_output(0), starts, shapes, strides)
                # scores = self.network.add_reduce(class_scores.get_output(0), op=trt.ReduceOperation.MAX, axes=1 << 2,  keep_dims=True)
            else:
                # output [1, 8400, 85]
                # slice boxes, obj_score, class_scores
                strides = trt.Dims([1,1,1])
                starts = trt.Dims([0,0,0])
                bs, num_boxes, temp = previous_output.shape
                shapes = trt.Dims([bs, num_boxes, 4])
                # [0, 0, 0] [1, 8400, 4] [1, 1, 1]
                boxes = self.network.add_slice(previous_output, starts, shapes, strides)
                num_classes = temp -5
                starts[2] = 4
                shapes[2] = 1
                # [0, 0, 4] [1, 8400, 1] [1, 1, 1]
                obj_score = self.network.add_slice(previous_output, starts, shapes, strides)
                starts[2] = 5
                shapes[2] = num_classes
                # [0, 0, 5] [1, 8400, 80] [1, 1, 1]
                scores = self.network.add_slice(previous_output, starts, shapes, strides)
                # scores = obj_score * class_scores => [bs, num_boxes, nc]
                scores = self.network.add_elementwise(obj_score.get_output(0), scores.get_output(0), trt.ElementWiseOperation.PROD)
            '''
            "plugin_version": "1",
            "background_class": -1,  # no background class
            "max_output_boxes": detections_per_img,
            "score_threshold": score_thresh,
            "iou_threshold": nms_thresh,
            "score_activation": False,
            "box_coding": 1,
            '''
            registry = trt.get_plugin_registry()
            assert(registry)
            creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
            assert(creator)
            fc = []
            fc.append(trt.PluginField("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("max_output_boxes", np.array([max_det], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("score_threshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            fc.append(trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            fc.append(trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("score_activation", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("class_agnostic", np.array([1 if class_agnostic else 0], dtype=np.int32), trt.PluginFieldType.INT32))

            fc = trt.PluginFieldCollection(fc)
            nms_layer = creator.create_plugin("nms_layer", fc)

            layer = self.network.add_plugin_v2([boxes.get_output(0), scores.get_output(0)], nms_layer)
            layer.get_output(0).name = "num"
            layer.get_output(1).name = "boxes"
            layer.get_output(2).name = "scores"
            layer.get_output(3).name = "classes"
            for i in range(4):
                self.network.mark_output(layer.get_output(i))


    def create_engine(self, engine_path, precision, calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        print("Building {} Engine in {}".format(precision, engine_path))
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        # TODO: Strict type is only needed If the per-layer precision overrides are used
        # If a better method is found to deal with that issue, this flag can be removed.

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                print("INT8 is not supported natively on this platform/device")
            else:
                if self.builder.platform_has_fast_fp16:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    self.config.set_flag(trt.BuilderFlag.FP16)
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    self.config.int8_calibrator.set_image_batcher(
                        ImageBatcher(calib_input, calib_shape, calib_dtype, max_num_images=calib_num_images,
                                     exact_batches=True))
        self.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  
        # with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
        with self.builder.build_serialized_network(self.network, self.config) as engine, open(engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(engine_path))
            f.write(engine)  # .serialize()

def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx, args.end2end, args.conf_thres, args.iou_thres, args.max_det, v8=args.v8, v10=args.v10, no_class_agnostic=args.no_class_agnostic)
    builder.create_engine(args.engine, args.precision, args.calib_input, args.calib_cache, args.calib_num_images,
                          args.calib_batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("-w", "--workspace", default=1, type=int, help="The max memory workspace size to allow in Gb, "
                                                                       "default: 1")
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument("--calib_cache", default="./calibration.cache",
                        help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    parser.add_argument("--calib_num_images", default=5000, type=int,
                        help="The maximum number of images to use for calibration, default: 5000")
    parser.add_argument("--calib_batch_size", default=8, type=int,
                        help="The batch size for the calibration process, default: 8")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="export the engine include nms plugin, default: False")
    parser.add_argument("--conf_thres", default=0.4, type=float,
                        help="The conf threshold for the nms, default: 0.4")
    parser.add_argument("--iou_thres", default=0.5, type=float,
                        help="The iou threshold for the nms, default: 0.5")
    parser.add_argument("--max_det", default=100, type=int,
                        help="The total num for results, default: 100")
    parser.add_argument("--v8", default=False, action="store_true",
                        help="use yolov8/9 model, default: False")
    parser.add_argument("--v10", default=False, action="store_true",
                        help="use yolov10 model, default: False")
    parser.add_argument("--no-class_agnostic", default=False, action="store_true",
                        help="Disable class-agnostic NMS (default: enabled)")
    args = parser.parse_args()
    print(args)
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision == "int8" and not (args.calib_input or os.path.exists(args.calib_cache)):
        parser.print_help()
        log.error("When building in int8 precision, --calib_input or an existing --calib_cache file is required")
        sys.exit(1)

    main(args)


