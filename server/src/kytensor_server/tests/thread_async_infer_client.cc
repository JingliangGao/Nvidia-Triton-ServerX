// Copyright 2025. All Rights Reserved.
// Author: Arthur Bin
// Date: 2025-10-16
// Description: kytensor client thread async infer
#include <unistd.h>

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <thread>
#include <vector>
#include <atomic>

#include "../src/kytensor_client.h"

namespace kc = kytensor::client;

// Test configuration
const int REQUESTS_PER_THREAD = 500;  // Each thread will send this many requests

// Global variables for statistics
std::vector<uint64_t> thread0_rtt_times;
std::vector<uint64_t> thread1_rtt_times;
std::mutex client_mutex;   // Mutex to protect the shared client access

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
}

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    kc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << "\t-r <requests per thread>" << std::endl;
  std::cerr << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  uint32_t client_timeout = 0;
  int requests_per_thread = REQUESTS_PER_THREAD;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vt:r:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 't':
        client_timeout = std::stoi(optarg);
        break;
      case 'r':
        requests_per_thread = std::stoi(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  std::cout << "Running two-threaded async inference test with "
            << requests_per_thread << " requests per thread..." << std::endl;

  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "simple";
  std::string model_version = "";

  // Create a single kytensor client to be shared by both threads
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

  std::condition_variable cv;
  size_t done_cnt = 0;
  // Record the start time for total AsyncInfer operations
  auto total_start_time = std::chrono::high_resolution_clock::now();

  // Create two threads that each send async requests using the same client
  std::vector<uint64_t> send_time0(requests_per_thread);
  std::thread thread0([&]() {
      std::cout << "Thread 0 starting to send " << requests_per_thread << " requests..." << std::endl;
      for (int i = 0; i < requests_per_thread; ++i) {
        send_time0[i] = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // Use mutex to protect the shared client during AsyncInfer calls
        {
          std::lock_guard<std::mutex> lk(client_mutex);
          FAIL_IF_ERR(
              client->AsyncInfer(
                  [&, i](kc::InferResult* result) {
                    {
                      std::shared_ptr<kc::InferResult> result_ptr;
                      result_ptr.reset(result);
                      done_cnt++;
                      auto receive_time = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                      auto rtt_us = receive_time - send_time0[i];

                      thread0_rtt_times.push_back(rtt_us);
                    }
                    cv.notify_all();
                  },
                  options, inputs, outputs),
              "unable to run model");
        }
      }
      std::cout << "Thread 0 completed sending requests" << std::endl;
  });

  std::vector<uint64_t> send_time1(requests_per_thread);
  std::thread thread1([&]() {
      std::cout << "Thread 1 starting to send " << requests_per_thread << " requests..." << std::endl;
      std::vector<uint64_t> send_time(requests_per_thread);
      for (int i = 0; i < requests_per_thread; ++i) {
        send_time1[i] = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // Use mutex to protect the shared client during AsyncInfer calls
        {
          std::lock_guard<std::mutex> lk(client_mutex);
          FAIL_IF_ERR(
              client->AsyncInfer(
                  [&, i](kc::InferResult* result) {
                    {
                      std::shared_ptr<kc::InferResult> result_ptr;
                      result_ptr.reset(result);
                      done_cnt++;
                      auto receive_time = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                      auto rtt_us = receive_time - send_time1[i];

                      thread1_rtt_times.push_back(rtt_us);
                    }
                    cv.notify_all();
                  },
                  options, inputs, outputs),
              "unable to run model");
        }
      }
      std::cout << "Thread 1 completed sending requests" << std::endl;
  });

  // Wait until all callbacks are invoked
  {
    std::unique_lock<std::mutex> lk(client_mutex);
    cv.wait(lk, [&]() {
      if (done_cnt >= requests_per_thread * 2) {
        return true;
      } else {
        return false;
      }
    });
  }
  if (done_cnt == requests_per_thread * 2) {
    std::cout << "All done" << std::endl;
  } else {
    std::cerr << "Done cnt: " << done_cnt
              << " does not match repeat cnt: " << requests_per_thread * 2 << std::endl;
    exit(1);
  }
  // Wait for both threads to complete
  thread0.join();
  thread1.join();
  // Record the end time for total AsyncInfer operations
  auto total_end_time = std::chrono::high_resolution_clock::now();

  std::cout << "Both threads completed" << std::endl;

  // Calculate total time for all AsyncInfer operations
  print_statistics(thread0_rtt_times, "Thread 0 Async Infer RTT");
  print_statistics(thread1_rtt_times, "Thread 1 Async Infer RTT");

  auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
    total_end_time - total_start_time);
  int total_requests = requests_per_thread * 2;
  std::cout << "Total time for " << total_requests << " AsyncInfer operations: "
          << total_duration.count() << " us" << std::endl;
  double throughput = total_requests / (static_cast<double>(total_duration.count()) / 1000000.0);
  std::cout << "Throughput (req/s): " << std::fixed << std::setprecision(2) << throughput << "\n";
  std::cout << "PASS : Two-threaded Async Infer with Single Client" << std::endl;

  return 0;
}