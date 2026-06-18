// Copyright 2025. All Rights Reserved.
// Author: Arthur Bin
// Date: 2025-10-16
// Description: kytensor client perfermance test
#include "../src/kytensor_client.h"
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

using namespace kytensor::client;

// Test configuration
const int SYNC_TEST_COUNT = 1000;
const int ASYNC_TEST_COUNT = 1000;
const int CONCURRENT_ASYNC_COUNT = 100; // Number of concurrent async requests

// Statistics
std::vector<uint64_t> sync_rtt_times;
std::vector<uint64_t> async_rtt_times;
int async_responses_received = 0;
std::mutex async_mutex;
std::unordered_map<int, uint64_t> async_send_times;
double sync_throughput;
double async_throughput;

// Helper function to calculate statistics
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

// Synchronous message test
void test_sync_performance(std::unique_ptr<KyTensorClient>& client) {
    std::cout << "Starting synchronous message performance test...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < SYNC_TEST_COUNT; ++i) {
        auto send_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        //client.SyncSendMessage("sync test message");
        bool live = false;
        //Error err = client.IsServerLive(&live, 5000);
        inference::RepositoryIndexResponse response;
        Error err = client->ModelRepositoryIndex(&response, 5000);

        auto receive_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        uint64_t rtt = receive_time - send_time;
        sync_rtt_times.push_back(rtt);

        // Small delay to avoid overwhelming the server
        if (i % 100 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Synchronous test completed in " << duration.count() << " us\n";
    std::cout << "Average RTT: " << (duration.count() / SYNC_TEST_COUNT) << " microseconds\n";
    sync_throughput = SYNC_TEST_COUNT / (double)duration.count() * 1000000;
}

// Asynchronous message test
void test_async_performance(std::unique_ptr<KyTensorClient>& client) {
    std::cout << "Starting asynchronous message performance test...\n";

    auto callback = [&](const EmptyPtr& res) {
        std::lock_guard<std::mutex> lock(async_mutex);
        async_responses_received++;

        // In a real test, we would record the RTT here
        // For now, we'll just count responses
        auto it = async_send_times.find(res->id());
        if (it != async_send_times.end()) {
            auto receive_time = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            auto rtt_us = receive_time - it->second;
            async_rtt_times.push_back(rtt_us);
            async_send_times.erase(it);
        }
    };

    auto start_time = std::chrono::high_resolution_clock::now();

    // Send all async messages
    for (int i = 0; i < ASYNC_TEST_COUNT; ++i) {
        auto send_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        async_send_times[i] = send_time;
        client->AsyncSendMessage(callback, i);

        // Small delay to avoid overwhelming the server
        if (i % 100 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Wait for all responses
    int wait_count = 0;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(async_mutex);
            if (async_responses_received >= ASYNC_TEST_COUNT) {
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;

        // Timeout after 30 seconds
        if (wait_count > 3000) {
            std::cout << "Timeout waiting for async responses\n";
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Asynchronous test completed in " << duration.count() << " us\n";
    std::cout << "Responses received: " << async_responses_received << "/" << ASYNC_TEST_COUNT << "\n";
    async_throughput = ASYNC_TEST_COUNT / (double)duration.count() * 1000000;
}

int main() {
    std::unique_ptr<KyTensorClient> client;

    Error err = KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        return -1;
    }

    // Warm up
    std::cout << "Warming up...\n";
    client->SyncSendMessage("warmup");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Run tests
    test_sync_performance(client);
    test_async_performance(client);

    // Print statistics
    print_statistics(sync_rtt_times, "Synchronous Messages");
    std::cout << "  Throughput (req/s): " << std::fixed << std::setprecision(2) << sync_throughput << "\n";
    print_statistics(async_rtt_times, "Asynchronous Messages");
    std::cout << "  Throughput (req/s): " << std::fixed << std::setprecision(2) << async_throughput << "\n";

    std::cout << "Performance tests completed\n";
    return 0;
}