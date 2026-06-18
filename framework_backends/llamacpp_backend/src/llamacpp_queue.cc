#include "llamacpp_queue.h"
#include "llamacpp_utils.h"

namespace triton::backend::llamacpp
{

// Add a new task to the end of the queue
int LlamaCppQueue::post(ServerTask task, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    GGML_ASSERT(task.id != -1);
    // if this is cancel task make sure to clean up pending tasks
    if (task.type == SERVER_TASK_TYPE_CANCEL || task.type == SERVER_TASK_TYPE_ABORT) {
        cleanup_pending_task(task.id_target);
    }
    QUE_DBG("new task, id = %d, front = %d\n", task.id, front);
    if (front) {
        queue_tasks.push_front(std::move(task));
    } else {
        queue_tasks.push_back(std::move(task));
    }
    condition_tasks.notify_one();
    return task.id;
}

int LlamaCppQueue::post(std::vector<ServerTask> & tasks, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    for (auto & task : tasks) {
        if (task.id == -1) {
            task.id = id++;
        }
        // if this is cancel task make sure to clean up pending tasks
        if (task.type == SERVER_TASK_TYPE_CANCEL || task.type == SERVER_TASK_TYPE_ABORT) {
            cleanup_pending_task(task.id_target);
        }
        QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int) tasks.size(), front);
        if (front) {
            queue_tasks.push_front(std::move(task));
        } else {
            queue_tasks.push_back(std::move(task));
        }
    }
    condition_tasks.notify_one();
    return 0;
}

// Add a new task, but defer until one worker is available
void LlamaCppQueue::defer(ServerTask task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    QUE_DBG("defer task, id = %d\n", task.id);
    queue_tasks_deferred.push_back(std::move(task));
    condition_tasks.notify_one();
}

// Get the next id for creating anew task
int LlamaCppQueue::get_new_id() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    int new_id = id++;
    return new_id;
}

// Register function to process a new task
void LlamaCppQueue::on_new_task(std::function<void(ServerTask)> callback) {
    callback_new_task = std::move(callback);
}

// Register the function to be called when all workers data is ready to be processed
void LlamaCppQueue::on_update_workers(std::function<void(void)> callback) {
    callback_update_workers = std::move(callback);
}

// Call when the state of one worker is changed
void LlamaCppQueue::pop_deferred_task() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!queue_tasks_deferred.empty()) {
        queue_tasks.emplace_back(std::move(queue_tasks_deferred.front()));
        queue_tasks_deferred.pop_front();
    }
    condition_tasks.notify_one();
}

// end the start_loop routine
void LlamaCppQueue::terminate() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    running = false;
    condition_tasks.notify_all();
}

/**
 * Main loop consists of these steps:
 * - Wait until a new task arrives
 * - Process the task (i.e. maybe copy data into worker)
 * - Check if multitask is finished
 * - Update all workers
 */
void LlamaCppQueue::start_loop() {
    running = true;

    while (true) {
        QUE_DBG("%s", "processing new tasks\n");

        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (!running) {
                QUE_DBG("%s", "terminate\n");
                return;
            }
            if (queue_tasks.empty()) {
                lock.unlock();
                break;
            }
            ServerTask task = queue_tasks.front();
            queue_tasks.pop_front();
            lock.unlock();

            QUE_DBG("processing task, id = %d\n", task.id);
            callback_new_task(std::move(task));
        }

        // all tasks in the current loop is processed, workers data is now ready
        QUE_DBG("%s", "update workers\n");

        callback_update_workers();

        QUE_DBG("%s", "waiting for new tasks\n");
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (!running) {
                QUE_DBG("%s", "terminate\n");
                return;
            }
            if (queue_tasks.empty()) {
                condition_tasks.wait(lock, [&]{
                    return (!queue_tasks.empty() || !running);
                });
            }
        }
    }
}

void LlamaCppQueue::cleanup_pending_task(int id_target) {
    // no need lock because this is called exclusively by post()
    auto rm_func = [id_target](const ServerTask & task) {
        return task.id == id_target;
    };
    queue_tasks.erase(
        std::remove_if(queue_tasks.begin(),          queue_tasks.end(),          rm_func),
        queue_tasks.end());
    queue_tasks_deferred.erase(
        std::remove_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), rm_func),
        queue_tasks_deferred.end());
}

bool LlamaCppQueue::abort_clenup_pending_task(int id_target) {
    // no need lock because this is called exclusively by post()
    bool removed = false;
    for (auto it = queue_tasks.begin(); it != queue_tasks.end(); ) {
        if (it->id == id_target) {
            it = queue_tasks.erase(it);
            removed = true;
        } else {
            ++it;
        }
    }
    for (auto it = queue_tasks_deferred.begin(); it != queue_tasks_deferred.end(); ) {
        if (it->id == id_target) {
            it = queue_tasks_deferred.erase(it);
            removed = true;
        } else {
            ++it;
        }
    }
    return removed;
}

} // namespace triton::backend::llamacpp