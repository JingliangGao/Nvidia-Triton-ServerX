#pragma once

#include "llamacpp_queue.h"

namespace triton::backend::llamacpp
{

class LlamaCppResponse {
public:
    // for keeping track of all tasks waiting for the result
    std::unordered_set<int> waiting_task_ids;

    // the main result queue
    std::vector<ServerTaskResultPtr> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

    bool recv_running = true;

    //method
    void add_waiting_task_id(int id_task);
    void add_waiting_tasks(const std::vector<ServerTask> & tasks);
    void remove_waiting_task_id(int id_task);
    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks);
    void recv_terminate();
    ServerTaskResultPtr recv();
    ServerTaskResultPtr recv(const std::unordered_set<int> & id_tasks);
    ServerTaskResultPtr recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout);
    // single-task version of recv()
    ServerTaskResultPtr recv(int id_task);
    void send(ServerTaskResultPtr && result);
};

} // namespace triton::backend::llamacpp