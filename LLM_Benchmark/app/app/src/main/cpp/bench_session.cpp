#include "bench_session.h"
#include <android/log.h>
#include <sstream>
#include "MNN/expr/ExecutorScope.hpp"

#define LOG_TAG "MNNBench"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace bench {

BenchSession::BenchSession(const std::string& model_dir, const std::string& config_json)
    : model_dir_(model_dir), config_json_(config_json) {
}

BenchSession::~BenchSession() {
    if (llm_) {
        Llm::destroy(llm_);
        llm_ = nullptr;
    }
}

bool BenchSession::load() {
    std::string config_path = model_dir_ + "/config.json";
    LOGD("Creating LLM from: %s", config_path.c_str());

    llm_ = Llm::createLLM(config_path);
    if (!llm_) {
        LOGE("Failed to create LLM");
        return false;
    }

    // Apply user settings from the app's Settings screen before loading
    if (!config_json_.empty() && config_json_ != "{}") {
        LOGD("Applying config: %s", config_json_.c_str());
        llm_->set_config(config_json_);
    }

    loaded_ = llm_->load();
    if (!loaded_) {
        LOGE("Failed to load model");
        Llm::destroy(llm_);
        llm_ = nullptr;
        return false;
    }

    LOGD("Model loaded successfully");
    return true;
}

void BenchSession::setSystemPrompt(const std::string& prompt) {
    system_prompt_ = prompt;
    LOGD("System prompt set: %.50s...", prompt.c_str());
}

void BenchSession::ensureSystemPrompt() {
    if (system_prompt_.empty()) return;
    // Insert system prompt at the beginning if not already there
    if (history_.empty() || history_[0].first != "system") {
        history_.insert(history_.begin(), {"system", system_prompt_});
    } else {
        // Update existing system prompt if changed
        history_[0].second = system_prompt_;
    }
}

const LlmContext* BenchSession::generate(const std::string& prompt,
                                          const ProgressCallback& cb) {
    if (!isReady()) return nullptr;

    ensureSystemPrompt();
    history_.push_back({"user", prompt});

    std::ostringstream oss;
    llm_->response(history_, &oss, "<eop>", 0); // prefill

    std::string accumulated;
    while (!llm_->stoped()) {
        llm_->generate(1);
        auto* ctx = llm_->getContext();
        std::string current = oss.str();
        std::string delta = current.substr(accumulated.size());
        accumulated = current;

        if (!delta.empty() && cb) {
            bool cont = cb(delta, false);
            if (!cont) break;
        }
    }

    if (cb) cb("", true); // signal end

    auto* ctx = llm_->getContext();

    // Add assistant response to history
    history_.push_back({"assistant", accumulated});

    return ctx;
}

const LlmContext* BenchSession::generateWithImage(const std::string& prompt,
                                                   const std::string& image_path,
                                                   const ProgressCallback& cb) {
    if (!isReady()) return nullptr;

    // Build prompt with image tag for VLM models
    // The MNN LLM engine handles image loading internally via <img> tags
    std::string vlm_prompt = "<img>" + image_path + "</img>" + prompt;

    ensureSystemPrompt();
    history_.push_back({"user", vlm_prompt});

    std::ostringstream oss;
    llm_->response(history_, &oss, "<eop>", 0); // prefill

    std::string accumulated;
    while (!llm_->stoped()) {
        llm_->generate(1);
        std::string current = oss.str();
        std::string delta = current.substr(accumulated.size());
        accumulated = current;

        if (!delta.empty() && cb) {
            bool cont = cb(delta, false);
            if (!cont) break;
        }
    }

    if (cb) cb("", true);

    auto* ctx = llm_->getContext();
    history_.push_back({"assistant", accumulated});

    return ctx;
}

void BenchSession::reset() {
    if (llm_) {
        llm_->reset();
    }
    history_.clear();
}

} // namespace bench
