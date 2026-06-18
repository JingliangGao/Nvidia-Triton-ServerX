// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <iomanip>
#include <numeric>

#include "../src/kytensor_client.h"

namespace kc = kytensor::client;

using ResultMap =
    std::map<std::string, std::vector<std::shared_ptr<kc::InferResult>>>;
using ResultList = std::vector<std::shared_ptr<kc::InferResult>>;

// Global mutex to synchronize the threads
std::mutex mutex_;
std::condition_variable cv_;

// Test configuration

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
  std::cerr << "\t-u <URL for inference service and its gRPC port>"
            << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;
  std::cerr << "\t-r <the number of inference requests>" << std::endl;
  std::cerr << "\t-s <the number of inference response to generate per request>"
            << std::endl;
  std::cerr << "\t-o <data offset>" << std::endl;
  std::cerr << "\t-d <delay time between each response>" << std::endl;
  std::cerr << "\t-w <wait time before releasing the request>" << std::endl;
  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8001");
  int request_count = 1;
  int repeat_count = 1;
  int data_offset = 100;
  uint32_t delay_time = 1000;
  uint32_t wait_time = 500;
  std::vector<uint64_t> stream_rtt_times;
  std::unordered_map<int, uint64_t> stream_send_times;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:H:r:s:o:d:w:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        //http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
      case 'r':
        request_count = std::stoi(optarg);
        break;
      case 's':
        repeat_count = std::stoi(optarg);
        break;
      case 'o':
        data_offset = std::stoi(optarg);
        break;
      case 'd':
        delay_time = std::stoi(optarg);
        break;
      case 'w':
        wait_time = std::stoi(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  kc::Error err;

  // We use the custom "repeat_int32" model which takes 3 inputs and
  // 1 output. For a single request the model will generate 'repeat_count'
  // responses. See is src/backends/backend/examples/repeat.cc.
  std::string model_name = "repeat_int32";
  std::atomic<int32_t> received_response(0);

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<kc::KyTensorClient> client;
  FAIL_IF_ERR(
        kc::KyTensorClient::Create(&client, verbose),
      "unable to create grpc client");

  ResultMap result_map;

  // Note that client side statistics should be disabled in case of
  // of decoupled model.
  FAIL_IF_ERR(
      client->StartStream(
          [&](kc::InferResult* result) {
            {
              std::shared_ptr<kc::InferResult> result_ptr(result);
              std::lock_guard<std::mutex> lk(mutex_);
              std::string request_id;
              result->Id(&request_id);
              int id = std::stoi(request_id);
              auto sit = stream_send_times.find(id);
              if (sit != stream_send_times.end()) {
                  auto receive_time = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                  auto rtt_us = receive_time - sit->second;
                  stream_rtt_times.push_back(rtt_us);
                  stream_send_times.erase(sit);
              }

              auto it = result_map.find(request_id);
              if (it == result_map.end()) {
                result_map[request_id] = ResultList();
              }
              result_map[request_id].push_back(result_ptr);
              received_response++;
            }
            cv_.notify_all();
          },
          false /*enable_stats*/, 0 /* stream_timeout */),
      "unable to establish a streaming connection to server");

  // Prepare the data for the tensors
  std::vector<int32_t> in_data;
  std::vector<uint32_t> delay_data;
  std::vector<uint32_t> wait_data;
  for (int i = 0; i < repeat_count; i++) {
    in_data.push_back(data_offset + i);
    delay_data.push_back(delay_time);
  }
  wait_data.push_back(wait_time);

  // Initialize the inputs with the data.
  kc::InferInput* in;
  std::vector<int64_t> shape{repeat_count};
  FAIL_IF_ERR(
      kc::InferInput::Create(&in, "IN", shape, "INT32"),
      "unable to create 'IN'");
  std::shared_ptr<kc::InferInput> in_ptr(in);
  FAIL_IF_ERR(in_ptr->Reset(), "unable to reset 'IN'");
  FAIL_IF_ERR(
      in_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&in_data[0]),
          sizeof(int32_t) * repeat_count),
      "unable to set data for 'IN'");

  kc::InferInput* delay;
  FAIL_IF_ERR(
      kc::InferInput::Create(&delay, "DELAY", shape, "UINT32"),
      "unable to create 'DELAY'");
  std::shared_ptr<kc::InferInput> delay_ptr(delay);
  FAIL_IF_ERR(delay_ptr->Reset(), "unable to reset 'DELAY'");
  FAIL_IF_ERR(
      delay_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&delay_data[0]),
          sizeof(uint32_t) * repeat_count),
      "unable to set data for 'DELAY'");

  kc::InferInput* wait;
  shape[0] = 1;
  FAIL_IF_ERR(
      kc::InferInput::Create(&wait, "WAIT", shape, "UINT32"),
      "unable to create 'WAIT'");
  std::shared_ptr<kc::InferInput> wait_ptr(wait);
  FAIL_IF_ERR(wait_ptr->Reset(), "unable to reset 'WAIT'");
  FAIL_IF_ERR(
      wait_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&wait_data[0]), sizeof(uint32_t)),
      "unable to set data for 'WAIT'");

  std::vector<kc::InferInput*> inputs = {
      in_ptr.get(), delay_ptr.get(), wait_ptr.get()};

  kc::InferOptions options(model_name);

  auto total_start_time = std::chrono::high_resolution_clock::now();
  for (int id = 0; id < request_count; id++) {
    options.request_id_ = std::to_string(id);
    // Send inference request to the inference server.
    auto send_time = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    stream_send_times[id] = send_time;
    FAIL_IF_ERR(
        client->AsyncStreamInfer(options, inputs), "unable to run model");
  }

  // Wait until all callbacks are invoked
  {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&]() {
      if (received_response >= (repeat_count * request_count)) {
        return true;
      } else {
        return false;
      }
    });
  }

  auto total_end_time = std::chrono::high_resolution_clock::now();
  print_statistics(stream_rtt_times, "Stream Inference RTT");
  auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
    total_end_time - total_start_time);
  std::cout << "Total time for " << request_count << " StreamInfer operations: "
          << total_duration.count() << " us" << std::endl;
  double throughput = request_count / (static_cast<double>(total_duration.count()) / 1000000.0);
  std::cout << "Throughput (req/s): " << std::fixed << std::setprecision(2) << throughput << "\n";

  for (int i = 0; i < request_count; i++) {
    std::string id(std::to_string(i));
    if (repeat_count == 0) {
      auto it = result_map.find(id);
      if (it != result_map.end()) {
        std::cerr << "received unexpected response for request id " << id
                  << std::endl;
        exit(1);
      }
    } else {
      int32_t expected_output = data_offset;
      auto it = result_map.find(id);
      if (it == result_map.end()) {
        std::cerr << "response for request id " << id << " not received"
                  << std::endl;
        exit(1);
      }
      if (it->second.size() != (uint32_t)repeat_count) {
        std::cerr << "expected " << repeat_count << " many responses, got "
                  << it->second.size() << std::endl;
        exit(1);
      }
      for (auto this_result : it->second) {
        int32_t* output_data;
        size_t output_byte_size;
        FAIL_IF_ERR(
            this_result->RawData(
                "OUT", (const uint8_t**)&output_data, &output_byte_size),
            "unable to get result data for 'OUT'");
        if (output_byte_size != 4) {
          std::cerr << "error: received incorrect byte size for 'OUT': "
                    << output_byte_size << std::endl;
          exit(1);
        }
        if (*output_data != expected_output) {
          std::cerr << "error: incorrect result returned, expected "
                    << expected_output << ", got " << *output_data << std::endl;
          exit(1);
        }
        expected_output++;
      }
    }
  }

  return 0;
}