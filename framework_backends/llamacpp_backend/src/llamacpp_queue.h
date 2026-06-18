#pragma once

#include "llamacpp_utils.h"
#include "llamacpp_common.h"
#include "llamacpp_worker.h"
#include "llamacpp_task.h"

#include "llamacpp/json-schema-to-grammar.h"

#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <unordered_map>
#include <tuple>
#include <mutex>
#include <set>
#include <condition_variable>
#include <unordered_set>
#include <deque>

namespace triton::backend::llamacpp
{

enum ServerState {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

class LlamaCppQueue {

public:
    int id = 0;
    bool running;

    // queues
    std::deque<ServerTask> queue_tasks;
    std::deque<ServerTask> queue_tasks_deferred;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(ServerTask)> callback_new_task;
    std::function<void(void)>       callback_update_workers;

    //method

    // Add a new task to the end of the queue
    int post(ServerTask task, bool front = false);
    // multi-task version of post()
    int post(std::vector<ServerTask> & tasks, bool front = false);
    // Add a new task, but defer until one worker is available
    void defer(ServerTask task);
    // Get the next id for creating anew task
    int get_new_id();
    // Register function to process a new task
    void on_new_task(std::function<void(ServerTask)> callback);
    // Register the function to be called when all workers data is ready to be processed
    void on_update_workers(std::function<void(void)> callback);
    // Call when the state of one worker is changed
    void pop_deferred_task();
    // end the start_loop routine
    void terminate();
    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into worker)
     * - Check if multitask is finished
     * - Update all worker
     */
    void start_loop();

    bool abort_clenup_pending_task(int id_target);
private:
void cleanup_pending_task(int id_target);
};

} // namespace triton::backend::llamacpp