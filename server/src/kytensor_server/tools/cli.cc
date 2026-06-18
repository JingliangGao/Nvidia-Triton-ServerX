// Copyright 2026. All Rights Reserved.
// Description: kytensor CLI tool for server management and status check

#include "../src/kytensor_client.h"
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <iomanip>

namespace kc = kytensor::client;

void print_usage() {
    std::cout << "Usage: kytensor-cli <command> [args]\n\n"
              << "Commands:\n"
              << "  live                          Check if the server is live\n"
              << "  ready                         Check if the server is ready\n"
              << "  model-ready <model> [version] Check if a specific model is ready\n"
              << "  load <model> [version]        Load a model\n"
              << "  unload <model>                Unload a model\n"
              << "  config <model> [version]      Get model configuration\n"
              << "  index                         Get model repository index\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string command = argv[1];
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error create_err = kc::KyTensorClient::Create(&client);
    if (!create_err.IsOk()) {
        std::cerr << "Failed to create client: " << create_err.Message() << std::endl;
        return 1;
    }

    const int timeout = 5000; // 5 seconds default timeout

    if (command == "live") {
        bool live = false;
        kc::Error err = client->IsServerLive(&live, timeout);
        if (err.IsOk()) {
            std::cout << "Server Live: " << (live ? "YES" : "NO") << std::endl;
        } else {
            std::cerr << "Error checking live status: " << err.Message() << std::endl;
            return 1;
        }
    } else if (command == "ready") {
        bool ready = false;
        kc::Error err = client->IsServerReady(&ready, timeout);
        if (err.IsOk()) {
            std::cout << "Server Ready: " << (ready ? "YES" : "NO") << std::endl;
        } else {
            std::cerr << "Error checking ready status: " << err.Message() << std::endl;
            return 1;
        }
    } else if (command == "model-ready") {
        if (argc < 3) {
            std::cerr << "Usage: kytensor-cli model-ready <model> [version]" << std::endl;
            return 1;
        }
        std::string model = argv[2];
        std::string version = (argc >= 4) ? argv[3] : "";
        bool ready = false;
        kc::Error err = client->IsModelReady(&ready, model, version, timeout);
        if (err.IsOk()) {
            std::cout << "Model [" << model << ":" << (version.empty() ? "latest" : version)
                      << "] Ready: " << (ready ? "YES" : "NO") << std::endl;
        } else {
            std::cerr << "Error checking model ready status: " << err.Message() << std::endl;
            return 1;
        }
    } else if (command == "load") {
        if (argc < 3) {
            std::cerr << "Usage: kytensor-cli load <model> [version]" << std::endl;
            return 1;
        }
        std::string model = argv[2];
        std::string version = (argc >= 4) ? argv[3] : "";
        kc::Error err = client->LoadModel(model, version, {}, 1000 * 60);
        if (err.IsOk()) {
            std::cout << "Successfully loaded model: " << model << " (version: " << (version.empty() ? "latest" : version) << ")" << std::endl;
        } else {
            std::cerr << "Error loading model: " << err.Message() << std::endl;
            return 1;
        }
    } else if (command == "unload") {
        if (argc < 3) {
            std::cerr << "Usage: kytensor-cli unload <model>" << std::endl;
            return 1;
        }
        std::string model = argv[2];
        kc::Error err = client->UnloadModel(model, timeout);
        if (err.IsOk()) {
            std::cout << "Successfully unloaded model: " << model << std::endl;
        } else {
            std::cerr << "Error unloading model: " << err.Message() << std::endl;
            return 1;
        }
    } else if (command == "config") {
        if (argc < 3) {
            std::cerr << "Usage: kytensor-cli config <model> [version]" << std::endl;
            return 1;
        }
        std::string model = argv[2];
        std::string version = (argc >= 4) ? argv[3] : "";
        inference::ModelConfigResponse response;
        kc::Error err = client->ModelConfig(&response, model, version, timeout);
        if (err.IsOk()) {
            std::cout << "Model Config for " << model << ":" << (version.empty() ? "latest" : version) << "\n";
            std::cout << "Successfully retrieved model config." << std::endl;
        } else {
            std::cerr << "Error retrieving model config: " << err.Message() << std::endl;
            return 1;
        }
    } else if (command == "index") {
        inference::RepositoryIndexResponse response;
        kc::Error err = client->ModelRepositoryIndex(&response, timeout);
        if (err.IsOk()) {
            std::cout << "Model Repository Index retrieved successfully." << std::endl;
        } else {
            std::cerr << "Error retrieving repository index: " << err.Message() << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        print_usage();
        return 1;
    }

    return 0;
}
