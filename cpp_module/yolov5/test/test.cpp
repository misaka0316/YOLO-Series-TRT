#include <iostream>
#include <yolo.hpp>

#include <fstream>
#include <json.hpp> // Include the JSON library

int main(int argc, char** argv) {
  if (std::string(argv[1]) == "-config_path") {
    char* config_path = argv[2];
    //打印配置文件路径
    std::cout << "Config path: " << config_path << std::endl;
    // Read JSON configuration file
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
      std::cerr << "Failed to open config file: " << config_path << std::endl;
      return -1;
    }

    nlohmann::json config;
    try {
      config_file >> config;
    } catch (const std::exception& e) {
      std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
      return -1;
    }

    // Extract model path and image path from JSON
    std::string model_path = config.value("model_path", "");
    std::string source_path = config.value("source_path", "");
    std::string output_path = config.value("output_path", "");
    std::string test_flag = config.value("test_flag", "");

    if (model_path.empty() || source_path.empty()) {
      std::cerr << "Invalid configuration: model_path or image_path is missing" << std::endl;
      return -1;
    }

    // float* Boxes = new float[4000];
    // int* BboxNum = new int[1];
    // int* ClassIndexs = new int[1000];

    Yolo yolo;
    yolo.Init(const_cast<char*>(model_path.c_str()),const_cast<char*>(output_path.c_str()), true); // Initialize the Yolo object with model path and logging enabled
    yolo.load_engine(); // Load the TensorRT engine

    
    // cv::Mat img;
    // img = cv::imread(image_path);

    yolo.Infer(source_path);

    if(test_flag == "true"){

    // for (int num = 0; num < 2; num++) {
    //   yolo.Infer(img.cols, img.rows, img.channels(), img.data, Boxes, ClassIndexs, BboxNum);
    // }

    //   auto start = std::chrono::system_clock::now();

    //   yolo.Infer(img.cols, img.rows, img.channels(), img.data, Boxes, ClassIndexs, BboxNum);

    //   auto end = std::chrono::system_clock::now();
    //   std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
      
    //   yolo.draw_objects(img, Boxes, ClassIndexs, BboxNum);
    }

    //打印完成推理
    std::cout << "--> Inference completed!" << std::endl;
  } else {
    std::cerr << "--> arguments not right!" << std::endl;
    std::cerr << "--> yolo -config_path ./config.json" << std::endl;
    return -1;
  }
}
