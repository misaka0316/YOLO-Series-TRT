/**
 * @file yolo.hpp
 * @brief Header file for the Yolo class, which provides functionalities for loading a YOLO model,
 *        performing inference, and processing images using TensorRT and OpenCV.
 * 
 * This file defines the Yolo class and its associated methods for object detection tasks.
 * It includes methods for loading the TensorRT engine, preprocessing images, running inference,
 * and visualizing detection results. The class also utilizes CUDA for GPU acceleration.
 * 
 * Dependencies:
 * - CUDA Runtime API
 * - OpenCV
 * - NVIDIA TensorRT
 * - Standard C++ libraries
 * 
 * @author [misaka]
 * @date [Date]
 */

#ifndef YOLO_HPP
#define YOLO_HPP

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"

#include <math.h>
#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>      // 提供 struct stat 和 S_ISDIR
#include <filesystem>      // 提供 std::filesystem



using nvinfer1::Dims2;
using nvinfer1::Dims3;
using nvinfer1::IBuilder;
using nvinfer1::IBuilderConfig;
using nvinfer1::ICudaEngine;
using nvinfer1::IExecutionContext;
using nvinfer1::IHostMemory;
using nvinfer1::ILogger;
using nvinfer1::INetworkDefinition;
using Severity = nvinfer1::ILogger::Severity;

using cv::Mat;
using std::array;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::ofstream;
using std::string;
using std::vector;

class Logger : public ILogger {
    public:
		void log(Severity severity, const char* msg) noexcept override {
		if (severity != Severity::kVERBOSE) {
			std::cout << msg << std::endl;
		}
		}
   };

struct PreprocessedImage {
	cv::Mat original_img;  // 原始图像
	cv::Mat processed_img; // 预处理后的图像
	float scale;           // 缩放比例
	float* blob;           // 转换后的 blob 数据
	int img_w;
	int img_h;
};

// struct det_image{

// 	int* num_dets;
// 	float* det_boxes;
// 	float* det_scores;
// 	int* det_classes;

// 	det_image(int out_size1, int out_size2, int out_size3, int out_size4) {
// 		num_dets = new int[out_size1];
// 		det_boxes = new float[out_size2];
// 		det_scores = new float[out_size3];
// 		det_classes = new int[out_size4];
// 	}

// 	// int* num_dets;
// 	// float* det_boxes;
// 	// float* det_scores;
// 	// int* det_classes;

// 	int img_w;
// 	int img_h;
// 	float scale;
// };

struct det_image {
    std::vector<int> num_dets;
    std::vector<float> det_boxes;
    std::vector<float> det_scores;
    std::vector<int> det_classes;

    int img_w;
    int img_h;
    float scale;

    // 默认构造函数
    det_image() = default;

    // 初始化函数
    void initialize(int out_size1, int out_size2, int out_size3, int out_size4) {
        num_dets.resize(out_size1);
        det_boxes.resize(out_size2);
        det_scores.resize(out_size3);
        det_classes.resize(out_size4);
    }
};

struct det_images{
	float* Boxes = new float[4000];
    int* BboxNum = new int[1];
    int* ClassIndexs = new int[1000];
};


class Yolo {
    public:
		Yolo() = default;
		void load_engine();
		float letterbox(
			const cv::Mat& image,
			cv::Mat& out_image,
			const cv::Size& new_shape,
			int stride,
			const cv::Scalar& color,
			bool fixed_shape,
			bool scale_up);
		float* blobFromImage(cv::Mat& img);
		void draw_objects(const cv::Mat& img, float* Boxes, int* ClassIndexs, int* BboxNum, std::string image_name);
		void Init(char* model_path, char* output_path, bool is_log);
		void Infer(std::string source_path);
		PreprocessedImage  preprocessed_input(std::string source_path);
		det_image inference(float* blob);
		det_images processing(det_image output);
		~Yolo();
   
    private:
		nvinfer1::ICudaEngine* engine = nullptr;
		nvinfer1::IRuntime* runtime = nullptr;
		nvinfer1::IExecutionContext* context = nullptr;
		cudaStream_t stream = nullptr;
		void* buffs[5];
		int iB, iC, iH, iW, in_size, out_size1, out_size2, out_size3, out_size4;
		Logger gLogger;
		std::string model_path;
		std::string output_path;
		bool is_log = false;
		std::vector<PreprocessedImage> preprocessed_images;
		PreprocessedImage preprocessed_image;
		
};


// int aWidth,
// int aHeight,
// int aChannel,
// unsigned char* aBytes,
// float* Boxes,
// int* ClassIndexs,
// int* BboxNum

#endif // YOLO_HPP