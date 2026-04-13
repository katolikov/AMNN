#include "enum_maps.hpp"
#include <cctype>
#include <cstdio>
#include <cstdlib>

const std::unordered_map<std::string, int>& gpu_mode_map() {
    static const std::unordered_map<std::string, int> m = {
        {"MNN_GPU_TUNING_NONE",    1 << 0},
        {"MNN_GPU_TUNING_HEAVY",   1 << 1},
        {"MNN_GPU_TUNING_WIDE",    1 << 2},
        {"MNN_GPU_TUNING_NORMAL",  1 << 3},
        {"MNN_GPU_TUNING_FAST",    1 << 4},
        {"MNN_GPU_MEMORY_BUFFER",  1 << 6},
        {"MNN_GPU_MEMORY_IMAGE",   1 << 7},
        {"MNN_GPU_RECORD_OP",      1 << 8},
        {"MNN_GPU_RECORD_BATCH",   1 << 9},
    };
    return m;
}

const std::unordered_map<std::string, int>& session_mode_map() {
    static const std::unordered_map<std::string, int> m = {
        {"Session_Debug", 0}, {"Session_Release", 1},
        {"Session_Input_Inside", 2}, {"Session_Input_User", 3},
        {"Session_Output_Inside", 4}, {"Session_Output_User", 5},
        {"Session_Resize_Direct", 6}, {"Session_Resize_Defer", 7},
        {"Session_Backend_Fix", 8}, {"Session_Backend_Auto", 9},
        {"Session_Memory_Collect", 10}, {"Session_Memory_Cache", 11},
        {"Session_Codegen_Disable", 12}, {"Session_Codegen_Enable", 13},
        {"Session_Resize_Check", 14}, {"Session_Resize_Fix", 15},
        {"Module_Forward_Separate", 16}, {"Module_Forward_Combine", 17},
    };
    return m;
}

const std::unordered_map<std::string, int>& hint_mode_map() {
    static const std::unordered_map<std::string, int> m = {
        {"MAX_TUNING_NUMBER", 0}, {"STRICT_CHECK_MODEL", 1},
        {"MEM_ALLOCATOR_TYPE", 2}, {"WINOGRAD_MEMORY_LEVEL", 3},
        {"GEOMETRY_COMPUTE_MASK", 4}, {"DYNAMIC_QUANT_OPTIONS", 5},
        {"CPU_LITTLECORE_DECREASE_RATE", 6}, {"QKV_QUANT_OPTIONS", 7},
        {"KVCACHE_SIZE_LIMIT", 8}, {"OP_ENCODER_NUMBER_FOR_COMMIT", 9},
        {"KVCACHE_INFO", 10}, {"MMAP_FILE_SIZE", 11},
        {"USE_CACHED_MMAP", 12}, {"INIT_THREAD_NUMBER", 13},
        {"CPU_CORE_IDS", 14}, {"CPU_SME2_INSTRUCTIONS", 15},
        {"CPU_ENABLE_KLEIDIAI", 16},
    };
    return m;
}

int resolve(const std::unordered_map<std::string, int>& table,
            const std::string& name, const char* context) {
    auto it = table.find(name);
    if (it != table.end()) return it->second;
    if (!name.empty() && (isdigit(name[0]) || name[0] == '-'))
        return atoi(name.c_str());
    fprintf(stderr, "[BENCH] WARNING: unknown %s name \"%s\"\n", context, name.c_str());
    return 0;
}
