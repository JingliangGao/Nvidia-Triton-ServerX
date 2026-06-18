#include "llamacpp_response.h"
#include "llamacpp_utils.h"
#include <cassert>

namespace triton::backend::llamacpp
{

// add the id_task to the list of tasks waiting for response
void LlamaCppResponse::add_waiting_task_id(int id_task) {
    SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.insert(id_task);
}

void LlamaCppResponse::add_waiting_tasks(const std::vector<ServerTask> & tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & task : tasks) {
        SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", task.id, (int) waiting_task_ids.size());
        waiting_task_ids.insert(task.id);
    }
}

// when the request is finished, we can remove task associated with it
void LlamaCppResponse::remove_waiting_task_id(int id_task) {
    SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.erase(id_task);
    // make sure to clean up all pending results
    queue_results.erase(
        std::remove_if(queue_results.begin(), queue_results.end(), [id_task](const ServerTaskResultPtr & res) {
            return res->id == id_task;
        }),
        queue_results.end());
}

void LlamaCppResponse::remove_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & id_task : id_tasks) {
        SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());
        waiting_task_ids.erase(id_task);
    }
}

// This function blocks the thread until there is a response for this id_task
ServerTaskResultPtr LlamaCppResponse::recv() {
    recv_running = true;
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&]{
            return !queue_results.empty() || !recv_running;
        });

        if (recv_running == false) {
            return nullptr;
        }

        ServerTaskResultPtr res = std::move(queue_results[0]);
        queue_results.erase(queue_results.begin());
        return res;
    }

    // should never reach here
}

// This function blocks the thread until there is a response for this id_task
ServerTaskResultPtr LlamaCppResponse::recv(const std::unordered_set<int> & id_tasks) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&]{
            return !queue_results.empty();
        });

        for (size_t i = 0; i < queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                ServerTaskResultPtr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }
    }

    // should never reach here
}

ServerTaskResultPtr LlamaCppResponse::recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (int i = 0; i < (int) queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                ServerTaskResultPtr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }

        std::cv_status cr_res = condition_results.wait_for(lock, std::chrono::seconds(timeout));
        if (cr_res == std::cv_status::timeout) {
            return nullptr;
        }
    }

    // should never reach here
}

// single-task version of recv()
ServerTaskResultPtr LlamaCppResponse::recv(int id_task) {
    std::unordered_set<int> id_tasks = {id_task};
    return recv(id_tasks);
}

// end the start_loop routine
void LlamaCppResponse::recv_terminate() {
    std::unique_lock<std::mutex> lock(mutex_results);
    recv_running = false;
    condition_results.notify_all();
}

// Send a new result to a waiting id_task
void LlamaCppResponse::send(ServerTaskResultPtr && result) {
    SRV_DBG("sending result for task id = %d\n", result->id);

    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto & id_task : waiting_task_ids) {
        if (result->id == id_task) {
            SRV_DBG("task id = %d pushed to result queue\n", result->id);

            queue_results.emplace_back(std::move(result));
            condition_results.notify_all();
            return;
        }
    }
}

} // namespace triton::backend::llamacpp