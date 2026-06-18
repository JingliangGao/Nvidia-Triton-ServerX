
// Copyright 2025. All Rights Reserved.
// Author: Arthur Bin
// Date: 2025-10-16
// Description: kytensor client infer
#include <getopt.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <iomanip>

#include "../src/kytensor_client.h"

namespace kc = kytensor::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    kc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

const int SYNC_TEST_COUNT = 1000;
std::vector<uint64_t> sync_rtt_times;

void print_statistics(const std::vector<uint64_t>& times, const std::string& label) {
    if (times.empty()) {
        std::cout << label << ": No data collected\n";
        return;
    }

    std::vector<uint64_t> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    uint64_t sum = std::accumulate(sorted_times.begin(), sorted_times.end(), 0ULL);
    double average = static_cast<double>(sum) / sorted_times.size();

    uint64_t min_time = sorted_times.front();
    uint64_t max_time = sorted_times.back();
    uint64_t median = sorted_times[sorted_times.size() / 2];

    // Calculate 95th percentile
    size_t percentile_95_index = static_cast<size_t>(sorted_times.size() * 0.95);
    uint64_t percentile_95 = sorted_times[percentile_95_index];

    // Calculate 99th percentile
    size_t percentile_99_index = static_cast<size_t>(sorted_times.size() * 0.99);
    uint64_t percentile_99 = sorted_times[percentile_99_index];

    std::cout << "\n" << label << " Statistics:\n";
    std::cout << "  Count: " << sorted_times.size() << "\n";
    std::cout << "  Average (us): " << std::fixed << std::setprecision(2) << average << "\n";
    std::cout << "  Min (us): " << min_time << "\n";
    std::cout << "  Max (us): " << max_time << "\n";
    std::cout << "  Median (us): " << median << "\n";
    std::cout << "  95th Percentile (us): " << percentile_95 << "\n";
    std::cout << "  99th Percentile (us): " << percentile_99 << "\n";

    // Calculate throughput (requests per second)
    if (max_time > min_time) {
        double duration_seconds = static_cast<double>(max_time - min_time) / 1000000.0;
        double throughput = sorted_times.size() / duration_seconds;
        std::cout << "  Throughput (req/s): " << std::fixed << std::setprecision(2) << throughput << "\n";
    }
}

void
ValidateShapeAndDatatype(
    const std::string& name, std::shared_ptr<kc::InferResult> result)
{
  std::vector<int64_t> shape;
  FAIL_IF_ERR(
      result->Shape(name, &shape), "unable to get shape for '" + name + "'");
  // Validate shape
  if ((shape.size() != 2) || (shape[0] != 1) || (shape[1] != 16)) {
    std::cerr << "error: received incorrect shapes for '" << name << "'"
              << std::endl;
    exit(1);
  }
  std::string datatype;
  FAIL_IF_ERR(
      result->Datatype(name, &datatype),
      "unable to get datatype for '" + name + "'");
  // Validate datatype
  if (datatype.compare("INT32") != 0) {
    std::cerr << "error: received incorrect datatype for '" << name
              << "': " << datatype << std::endl;
    exit(1);
  }
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  uint32_t client_timeout = 0;
  size_t repeat_cnt = 1;

  // {name, has_arg, *flag, val}
  static struct option long_options[] = {
      {"ssl", 0, 0, 0},
      {"root-certificates", 1, 0, 1},
      {"private-key", 1, 0, 2},
      {"certificate-chain", 1, 0, 3}};

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(argc, argv, "vu:t:r:", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'r':
        repeat_cnt = std::stoi(optarg);
        break;
      case 't':
        client_timeout = std::stoi(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "simple";
  std::string model_version = "";

  std::unique_ptr<kc::KyTensorClient> client;
  FAIL_IF_ERR(kc::KyTensorClient::Create(&client, verbose), err);

    // Create the data for the two input tensors. Initialize the first
    // to unique integers and the second to all ones.
    std::vector<int32_t> input0_data(16);
    std::vector<int32_t> input1_data(16);
    for (size_t i = 0; i < 16; ++i) {
      input0_data[i] = i;
      input1_data[i] = 1;
    }

    std::vector<int64_t> shape{1, 16};

    // Initialize the inputs with the data.
    kc::InferInput* input0;
    kc::InferInput* input1;

    FAIL_IF_ERR(
        kc::InferInput::Create(&input0, "INPUT0:0", shape, "INT32"),
        "unable to get INPUT0");
    std::shared_ptr<kc::InferInput> input0_ptr;
    input0_ptr.reset(input0);
    FAIL_IF_ERR(
        kc::InferInput::Create(&input1, "INPUT1:0", shape, "INT32"),
        "unable to get INPUT1");
    std::shared_ptr<kc::InferInput> input1_ptr;
    input1_ptr.reset(input1);

    FAIL_IF_ERR(
        input0_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&input0_data[0]),
            input0_data.size() * sizeof(int32_t)),
        "unable to set data for INPUT0");
    FAIL_IF_ERR(
        input1_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&input1_data[0]),
            input1_data.size() * sizeof(int32_t)),
        "unable to set data for INPUT1");

    // Generate the outputs to be requested.
    kc::InferRequestedOutput* output0;
    kc::InferRequestedOutput* output1;

    FAIL_IF_ERR(
        kc::InferRequestedOutput::Create(&output0, "OUTPUT0:0"),
        "unable to get 'OUTPUT0'");
    std::shared_ptr<kc::InferRequestedOutput> output0_ptr;
    output0_ptr.reset(output0);
    FAIL_IF_ERR(
        kc::InferRequestedOutput::Create(&output1, "OUTPUT1:0"),
        "unable to get 'OUTPUT1'");
    std::shared_ptr<kc::InferRequestedOutput> output1_ptr;
    output1_ptr.reset(output1);


    // The inference settings. Will be using default for now.
    kc::InferOptions options(model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = client_timeout;

    std::vector<kc::InferInput*> inputs = {input0_ptr.get(), input1_ptr.get()};
    std::vector<const kc::InferRequestedOutput*> outputs = {
        output0_ptr.get(), output1_ptr.get()};

    kc::InferResult* results;
    FAIL_IF_ERR(
        client->Infer(
            &results, options, inputs, outputs),
        "unable to run model");
    std::shared_ptr<kc::InferResult> results_ptr;
    results_ptr.reset(results);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat_cnt; ++i) {
      auto send_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      FAIL_IF_ERR(
        client->Infer(
            &results, options, inputs, outputs),
        "unable to run model");

      auto receive_time = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      uint64_t rtt = receive_time - send_time;
      sync_rtt_times.push_back(rtt);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Validate the results...
    ValidateShapeAndDatatype("OUTPUT0:0", results_ptr);
    ValidateShapeAndDatatype("OUTPUT1:0", results_ptr);

    // Get pointers to the result returned...
    int32_t* output0_data;
    size_t output0_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "OUTPUT0:0", (const uint8_t**)&output0_data, &output0_byte_size),
        "unable to get result data for 'OUTPUT0'");
    if (output0_byte_size != 64) {
      std::cerr << "error: received incorrect byte size for 'OUTPUT0': "
                << output0_byte_size << std::endl;
      exit(1);
    }

    int32_t* output1_data;
    size_t output1_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "OUTPUT1:0", (const uint8_t**)&output1_data, &output1_byte_size),
        "unable to get result data for 'OUTPUT1'");
    if (output1_byte_size != 64) {
      std::cerr << "error: received incorrect byte size for 'OUTPUT1': "
                << output1_byte_size << std::endl;
      exit(1);
    }

    for (size_t i = 0; i < 16; ++i) {
      std::cout << input0_data[i] << " + " << input1_data[i] << " = "
                << *(output0_data + i) << std::endl;
      std::cout << input0_data[i] << " - " << input1_data[i] << " = "
                << *(output1_data + i) << std::endl;

      if ((input0_data[i] + input1_data[i]) != *(output0_data + i)) {
        std::cerr << "error: incorrect sum" << std::endl;
        exit(1);
      }
      if ((input0_data[i] - input1_data[i]) != *(output1_data + i)) {
        std::cerr << "error: incorrect difference" << std::endl;
        exit(1);
      }
    }

    // Get full response
    std::cout << results_ptr->DebugString() << std::endl;

    std::cout << "PASS : Infer" << std::endl;
    print_statistics(sync_rtt_times, "Synchronous Messages");
    std::cout << "Total time for " << repeat_cnt << " SyncInfer operations: "
            << duration.count() << " us" << std::endl;
    double throughput = repeat_cnt / (static_cast<double>(duration.count()) / 1000000.0);
    std::cout << "Throughput (req/s): " << std::fixed << std::setprecision(2) << throughput << "\n";

    return 0;
}
