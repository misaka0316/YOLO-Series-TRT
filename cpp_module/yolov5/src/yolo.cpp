/**
 * @file yolo.cpp
 * @brief Implementation of the Yolo class for object detection using TensorRT.
 * 
 * This file contains the implementation of the Yolo class, which provides methods
 * for initializing the YOLO model, preprocessing input images, performing inference,
 * and postprocessing the results. The class is designed to work with TensorRT for
 * efficient inference on NVIDIA GPUs.
 * 
 * Features:
 * - Model initialization and engine loading.
 * - Image preprocessing with letterbox resizing.
 * - Conversion of images to blob format for inference.
 * - Object detection and bounding box drawing.
 * - Resource management and cleanup.
 * 
 * Dependencies:
 * - OpenCV for image processing.
 * - CUDA and TensorRT for GPU-based inference.
 * 
 * Usage:
 * 1. Initialize the Yolo object with the model path and output image path.
 * 2. Load the TensorRT engine using `load_engine()`.
 * 3. Perform inference on input images using `Infer()`.
 * 4. Draw detected objects on the image using `draw_objects()`.
 * 
 * @author [misaka]
 * @date [Date]
 */

#include "yolo.hpp"
#include "comm.hpp"

void Yolo::Init(char* model_path,char* output_path, bool is_log = false) {
    //查看CUDA设备是否可用
    // CHECK(cudaSetDevice(DEVICE));

	Yolo::is_log = is_log;
    // 获取可用的CUDA设备数量
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      cout << "No CUDA devices available.\n";
      std::abort();
    }

    // 打印可用的CUDA设备信息
	if (Yolo::is_log)
    cout << "Available CUDA devices: " << device_count << endl;
    for (int i = 0; i < device_count; ++i) {
      cudaDeviceProp device_prop;
      cudaGetDeviceProperties(&device_prop, i);
      cout << "Device " << i << ": " << device_prop.name << endl;
    }

    // 设置默认设备为0
    cudaSetDevice(0);
	if (Yolo::is_log)
    cout << "Using CUDA device 0.\n";

    //验证模型路径是否正确
    ifstream ifile(model_path, ios::in | ios::binary);
    if (!ifile) {
      cout << "read serialized file failed\n";
      std::abort(); 
    }

    Yolo::model_path = model_path;
    Yolo::output_image_path = output_image_path;

    //打印模型路径 输出路径
	if (Yolo::is_log){
    	cout << "model_path: " << model_path << endl;
    	cout << "output_image_path: " << output_image_path << endl;
	}
}

float Yolo::letterbox(
	const cv::Mat& image,
	cv::Mat& out_image,
	const cv::Size& new_shape = cv::Size(640, 640),
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(114, 114, 114),
	bool fixed_shape = false,
	bool scale_up = true) {
	cv::Size shape = image.size();
	float r = std::min(
		(float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);
	if (!scale_up) {
		r = std::min(r, 1.0f);
	}

	int newUnpad[2]{
		(int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

	cv::Mat tmp;
	if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
		cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
		// cv::resize(image, tmp, cv::Size(640, 640));
	} else {
		tmp = image.clone();
	}

	float dw = new_shape.width - newUnpad[0];
	float dh = new_shape.height - newUnpad[1];

	if (!fixed_shape) {
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}

	dw /= 2.0f;
	dh /= 2.0f;

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

	return 1.0f / r;
}

float* Yolo::blobFromImage(cv::Mat& img) {
  float* blob = new float[img.total() * 3];
  int channels = 3;
  int img_h = img.rows;
  int img_w = img.cols;
  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < img_h; h++) {
      for (size_t w = 0; w < img_w; w++) {
        blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c] / 255.0;
      }
    }
  }
  return blob;
}

void Yolo::draw_objects(const cv::Mat& img, float* Boxes, int* ClassIndexs, int* BboxNum) {
  for (int j = 0; j < BboxNum[0]; j++) {
    cv::Rect rect(Boxes[j * 4], Boxes[j * 4 + 1], Boxes[j * 4 + 2], Boxes[j * 4 + 3]);
    cv::rectangle(img, rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
    cv::putText(
        img,
        std::to_string(ClassIndexs[j]),
        cv::Point(rect.x, rect.y - 1),
        cv::FONT_HERSHEY_PLAIN,
        1.2,
        cv::Scalar(0xFF, 0xFF, 0xFF),
        2);
    cv::imwrite(Yolo::output_image_path, img);
  }
}

void Yolo::load_engine() {
	ifstream ifile(Yolo::model_path, ios::in | ios::binary);
	if (!ifile) {
		cout << "read serialized file failed\n";
		std::abort();
	}

	ifile.seekg(0, ios::end);
	const int mdsize = ifile.tellg();
	ifile.clear();
	ifile.seekg(0, ios::beg);
	vector<char> buf(mdsize);
	ifile.read(&buf[0], mdsize);
	ifile.close();
	if (Yolo::is_log) {
		cout << "model size: " << mdsize << endl;
	}
	
	runtime = nvinfer1::createInferRuntime(gLogger);
	initLibNvInferPlugins(&gLogger, "");
	engine = runtime->deserializeCudaEngine((void*)&buf[0], mdsize, nullptr);
	auto in_dims = engine->getBindingDimensions(engine->getBindingIndex("images"));

	iB = in_dims.d[0];
	iC = in_dims.d[1];
	iH = in_dims.d[2];
	iW = in_dims.d[3];

	//打印输入维度
	if (Yolo::is_log)
	cout << "input dims: " << iB << " " << iC << " " << iH << " " << iW << endl;

	in_size = 1;
	for (int j = 0; j < in_dims.nbDims; j++) {
		in_size *= in_dims.d[j];
	}
	auto out_dims1 = engine->getBindingDimensions(engine->getBindingIndex("num"));
	out_size1 = 1;
	for (int j = 0; j < out_dims1.nbDims; j++) {
		out_size1 *= out_dims1.d[j];
	}
	auto out_dims2 = engine->getBindingDimensions(engine->getBindingIndex("boxes"));
	out_size2 = 1;
	for (int j = 0; j < out_dims2.nbDims; j++) {
		out_size2 *= out_dims2.d[j];
	}
	auto out_dims3 = engine->getBindingDimensions(engine->getBindingIndex("scores"));
	out_size3 = 1;
	for (int j = 0; j < out_dims3.nbDims; j++) {
		out_size3 *= out_dims3.d[j];
	}
	auto out_dims4 = engine->getBindingDimensions(engine->getBindingIndex("classes"));
	out_size4 = 1;
	for (int j = 0; j < out_dims4.nbDims; j++) {
		out_size4 *= out_dims4.d[j];
	}
	if (Yolo::is_log) {
		//打印输出维度
		cout << "input size: " << in_size << endl;
		cout << "output num size: " << out_size1 << " "<< out_size2 <<" "<< out_size3 <<" "<< out_size4 << endl;
	}

	context = engine->createExecutionContext();
	if (!context) {
		cout << "create execution context failed\n";
		std::abort();
	}

	cudaError_t state;
	state = cudaMalloc(&buffs[0], in_size * sizeof(float));
	if (state) {
		cout << "allocate memory failed\n";
		std::abort();
	}
	state = cudaMalloc(&buffs[1], out_size1 * sizeof(int));
	if (state) {
		cout << "allocate memory failed\n";
		std::abort();
	}

	state = cudaMalloc(&buffs[2], out_size2 * sizeof(float));
	if (state) {
		cout << "allocate memory failed\n";
		std::abort();
	}

	state = cudaMalloc(&buffs[3], out_size3 * sizeof(float));
	if (state) {
		cout << "allocate memory failed\n";
		std::abort();
	}

	state = cudaMalloc(&buffs[4], out_size4 * sizeof(int));
	if (state) {
		cout << "allocate memory failed\n";
		std::abort();
	}

	state = cudaStreamCreate(&stream);
	if (state) {
		cout << "create stream failed\n";
		std::abort();
	}

	//打印buffs
	if (Yolo::is_log)
	cout << "buffs: " << in_size << " " << out_size1 << " " << out_size2 << " " << out_size3
		<< " " << out_size4 << endl;
	
}


PreprocessedImage  Yolo::preprocessed_input(std::string image_path) {
	

	// int width, int height, int channel, unsigned char* data, int target_width, int target_height

	PreprocessedImage preprocessed_image;

	cv::Mat original_img;
    preprocessed_image.original_img = cv::imread(image_path);

	
	// // 创建原始图像
	// original_img = cv::Mat(height, width, CV_MAKETYPE(CV_8U, preprocessed_image.original_img.channels()), preprocessed_image.original_img.data);

	// 预处理：调整大小并填充
	preprocessed_image.scale = letterbox(preprocessed_image.original_img, preprocessed_image.processed_img, {iW, iH}, 32, {114, 114, 114}, true);

	// 转换为 RGB 格式
	cv::cvtColor(preprocessed_image.processed_img, preprocessed_image.processed_img, cv::COLOR_BGR2RGB);

	// 转换为 blob 数据
	preprocessed_image.blob = blobFromImage(preprocessed_image.processed_img);
	preprocessed_image.img_w = original_img.cols;
	preprocessed_image.img_h = original_img.rows;

	return preprocessed_image;
}


det_images Yolo::processing(det_image output){

	det_images output_images;
	//转换输出结果
	output_images.BboxNum[0] = output.num_dets[0];
	// BboxNum[0] = num_dets[0];

	int img_w = output.img_w;
	int img_h = output.img_h;
	float scale = output.scale;

	int x_offset = (iW * scale - img_w) / 2;
	int y_offset = (iH * scale - img_h) / 2;
	for (size_t i = 0; i < output.num_dets[0]; i++) {
		float x0 = (output.det_boxes[i * 4]) * scale - x_offset;
		float y0 = (output.det_boxes[i * 4 + 1]) * scale - y_offset;
		float x1 = (output.det_boxes[i * 4 + 2]) * scale - x_offset;
		float y1 = (output.det_boxes[i * 4 + 3]) * scale - y_offset;
		x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
		y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
		x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
		y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
		output_images.Boxes[i * 4] = x0;
		output_images.Boxes[i * 4 + 1] = y0;
		output_images.Boxes[i * 4 + 2] = x1 - x0;
		output_images.Boxes[i * 4 + 3] = y1 - y0;
		output_images.ClassIndexs[i] = output.det_classes[i];
	}


	//打印Boxes 和class
	if (Yolo::is_log) {

		if (output.num_dets[0] > 0) {
			cout << "Detections found: " << output.num_dets[0] << endl;
		} else {
			cout << "No detections found." << endl;
		}

		for (size_t i = 0; i < output.num_dets[0]; i++) {
		cout << "Boxes: " << output_images.Boxes[i * 4] << " " << output_images.Boxes[i * 4 + 1] << " " << output_images.Boxes[i * 4 + 2] << " "
			<< output_images.Boxes[i * 4 + 3] << " "
			<< "ClassIndexs: " << output_images.ClassIndexs[i] << endl;
		}
	}

	return output_images;
}

det_image Yolo::inference(float* blob) {

	det_image output;
	static int* num_dets = new int[out_size1];
	static float* det_boxes = new float[out_size2];
	static float* det_scores = new float[out_size3];
	static int* det_classes = new int[out_size4];

	context->setTensorAddress("images", buffs[0]);
	context->setTensorAddress("num", buffs[1]);
	context->setTensorAddress("boxes", buffs[2]);
	context->setTensorAddress("scores", buffs[3]);
	context->setTensorAddress("classes", buffs[4]);

	//移动输入数据到设备
	cudaError_t state =
		cudaMemcpyAsync(buffs[0], &blob[0], in_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (state) {
		cout << "transmit to device failed\n";
		std::abort();
	}

	bool result = context->enqueueV3(stream);
	if (!result) {
		cout << "Inference failed\n";
		std::abort();
	}

	//等待流完成
	cudaStreamSynchronize(stream);

	//取输出结果
	state =
		cudaMemcpyAsync(output.num_dets, buffs[1], out_size1 * sizeof(int), cudaMemcpyDeviceToHost, stream);
	if (state) {
		cout << "transmit to host failed \n";
		std::abort();
	}
	state = cudaMemcpyAsync(
		output.det_boxes, buffs[2], out_size2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
	if (state) {
		cout << "transmit to host failed \n";
		std::abort();
	}
	state = cudaMemcpyAsync(
		output.det_scores, buffs[3], out_size3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
	if (state) {
		cout << "transmit to host failed \n";
		std::abort();
	}
	state = cudaMemcpyAsync(
		output.det_classes, buffs[4], out_size4 * sizeof(int), cudaMemcpyDeviceToHost, stream);
	if (state) {
		cout << "transmit to host failed \n";
		std::abort();
	}
	delete blob;
	output.img_h = preprocessed_image.img_h;
	output.img_w = preprocessed_image.img_w;
	output.scale = preprocessed_image.scale;
	return output;
}

void Yolo::Infer(std::string source_path) {

	// 判断source_path是图片还是文件夹路径
	struct stat path_stat;
	stat(source_path.c_str(), &path_stat);
	bool is_directory = S_ISDIR(path_stat.st_mode);


	if (iB <= 1) {
		if (is_directory) {
			cout << "source_path is a directory." << endl;
			// 处理文件夹中的所有图片
			std::vector<std::string> image_files;
			for (const auto& entry : std::filesystem::directory_iterator(source_path)) {
				if (entry.is_regular_file()) {
					std::string file_path = entry.path().string();
					// cout << "Processing file: " << file_path << endl;
					// 在这里可以调用推理处理每个文件
					image_files.push_back(file_path);
				}
			}
			
	
		} else {
			cout << "source_path is a file." << endl;
			// // 处理单个图片文件
			// cout << "Processing file: " << source_path << endl;
			// // 在这里可以调用推理处理该文件

			preprocessed_image = preprocessed_input(source_path);
			auto output = inference(preprocessed_image.blob);
			auto output_images = processing(output);
			draw_objects(preprocessed_image.original_img, output_images.Boxes, output_images.ClassIndexs, output_images.BboxNum);
		}
						
	}else{

	}

	
	// //根据iB的大小创建
	// cv::Mat img(aHeight, aWidth, CV_MAKETYPE(CV_8U, aChannel), aBytes);
	// cv::Mat pr_img;
	// float scale = letterbox(img, pr_img, {iW, iH}, 32, {114, 114, 114}, true);
	// cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB);
	// float* blob = blobFromImage(pr_img);

	// static int* num_dets = new int[out_size1];
	// static float* det_boxes = new float[out_size2];
	// static float* det_scores = new float[out_size3];
	// static int* det_classes = new int[out_size4];

	// cv::Mat img(aHeight, aWidth, CV_MAKETYPE(CV_8U, aChannel), aBytes);
	// cv::Mat pr_img;
	// float scale = letterbox(img, pr_img, {iW, iH}, 32, {114, 114, 114}, true);
	// cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB);
	// float* blob = blobFromImage(pr_img);

	
}

Yolo::~Yolo() {
  // cudaStreamSynchronize(stream);
  // if (buffs[0]) cudaFree(buffs[0]);
  // if (buffs[1]) cudaFree(buffs[1]);
  // if (buffs[2]) cudaFree(buffs[2]);
  // if (buffs[3]) cudaFree(buffs[3]);
  // if (buffs[4]) cudaFree(buffs[4]);
  // if (stream) cudaStreamDestroy(stream);
  // if (context) context->destroy();
  // if (engine) engine->destroy();
  // if (runtime) runtime->destroy();

  // cudaStreamSynchronize(stream);
  // cudaFree(buffs[0]);
  // cudaFree(buffs[1]);
  // cudaFree(buffs[2]);
  // cudaFree(buffs[3]);
  // cudaFree(buffs[4]);
  // cudaStreamDestroy(stream);
  // context->destroy();
  // engine->destroy();
  // runtime->destroy();
}