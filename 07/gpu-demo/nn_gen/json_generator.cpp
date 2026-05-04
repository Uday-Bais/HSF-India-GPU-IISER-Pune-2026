#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <string>
#include <nlohmann/json.hpp>

void generate_random_model(const std::string& filename, 
                          unsigned num_input, 
                          unsigned num_node,
                          float weight_range = 1.0f,
                          float bias_range = 0.5f) {
    
    nlohmann::json j;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> weight_dist(-weight_range, weight_range);
    std::uniform_real_distribution<float> bias_dist(-bias_range, bias_range);
    std::uniform_real_distribution<float> norm_dist(0.1f, 2.0f);  // For mean/std normalization
    
    j["num_input"] = num_input;
    j["num_node"] = num_node;

    // Generate mean values
    std::vector<float> mean(num_input);
    for (unsigned i = 0; i < num_input; i++) {
        mean[i] = norm_dist(gen);
    }
    j["mean"] = mean;
    
    // Generate std values
    std::vector<float> std_dev(num_input);
    for (unsigned i = 0; i < num_input; i++) {
        std_dev[i] = norm_dist(gen);
    }
    j["std"] = std_dev;
    
    // Generate weights1 (2D array)
    std::vector<std::vector<float>> weights1(num_node, std::vector<float>(num_input));
    for (unsigned i = 0; i < num_node; i++) {
        for (unsigned j = 0; j < num_input; j++) {
            weights1[i][j] = weight_dist(gen);
        }
    }
    j["weights1"] = weights1;
    
    // Generate bias1
    std::vector<float> bias1(num_node);
    for (unsigned i = 0; i < num_node; i++) {
        bias1[i] = bias_dist(gen);
    }
    j["bias1"] = bias1;
    
    // Generate weights2
    std::vector<float> weights2(num_node);
    for (unsigned i = 0; i < num_node; i++) {
        weights2[i] = weight_dist(gen);
    }
    j["weights2"] = weights2;
    
    // Generate bias2
    j["bias2"] = bias_dist(gen);
    
    std::ofstream file(filename);
    file << std::setw(4) << j << std::endl;
    file.close();
    
    std::cout << "Generated random model: " << filename << std::endl;
    std::cout << "- Input size: " << num_input << std::endl;
    std::cout << "- Hidden nodes: " << num_node << std::endl;
    std::cout << "- Weight range: [-" << weight_range << ", " << weight_range << "]" << std::endl;
    std::cout << "- Bias range: [-" << bias_range << ", " << bias_range << "]" << std::endl;
}

int main(int argc, char* argv[]) {
    unsigned num_input = 4;
    unsigned num_node = 8;
    std::string filename = "random_model.json";
    
    if (argc > 1) num_input = std::stoi(argv[1]);
    if (argc > 2) num_node = std::stoi(argv[2]);
    if (argc > 3) filename = argv[3];
    
    std::cout << "Random Neural Network Model Generator" << std::endl;
    std::cout << "====================================" << std::endl;
    
    generate_random_model(filename, num_input, num_node);
    
    return 0;
}
