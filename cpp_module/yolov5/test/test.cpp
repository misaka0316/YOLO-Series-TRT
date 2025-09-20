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
    std::string log_test = config.value("log_test", "");
    std::string draw_objects = config.value("draw_objects", "");
    bool log_test_bool = (log_test == "true"); 
 
    if (model_path.empty() || source_path.empty()) {
      std::cerr << "Invalid configuration: model_path or image_path is missing" << std::endl;
      return -1;
    }

    Yolo yolo;
    yolo.Init(const_cast<char*>(model_path.c_str()),const_cast<char*>(output_path.c_str()), log_test_bool); // Initialize the Yolo object with model path and logging enabled
    yolo.load_engine(); // Load the TensorRT engine

    yolo.Infer(source_path);

    std::cout << "--> Inference completed!" << std::endl;
  } else {
    std::cerr << "--> arguments not right!" << std::endl;
    std::cerr << "--> yolo -config_path ./config.json" << std::endl;
    return -1;
  }
}
