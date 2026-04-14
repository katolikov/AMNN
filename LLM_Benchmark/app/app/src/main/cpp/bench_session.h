#pragma once

#include <string>
#include <functional>
#include <vector>
#include "llm/llm.hpp"

namespace bench {

using MNN::Transformer::Llm;
using MNN::Transformer::LlmContext;
using MNN::Transformer::ChatMessages;

// Callback: receives token text and whether generation ended. Return false to stop.
using ProgressCallback = std::function<bool(const std::string& token, bool is_eop)>;

class BenchSession {
public:
    BenchSession(const std::string& model_dir, const std::string& config_json);
    ~BenchSession();

    bool load();
    bool isReady() const { return llm_ != nullptr && loaded_; }

    // Set system prompt for chat context.
    void setSystemPrompt(const std::string& prompt);

    // Text-only inference. Returns context with timing metrics.
    const LlmContext* generate(const std::string& prompt, const ProgressCallback& cb);

    // VLM inference with image. Returns context with timing metrics.
    const LlmContext* generateWithImage(const std::string& prompt,
                                        const std::string& image_path,
                                        const ProgressCallback& cb);

    void reset();

private:
    void ensureSystemPrompt();

    std::string model_dir_;
    std::string config_json_;
    std::string system_prompt_;
    Llm* llm_ = nullptr;
    bool loaded_ = false;
    ChatMessages history_;
};

} // namespace bench
