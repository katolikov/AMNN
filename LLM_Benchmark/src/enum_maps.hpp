#pragma once
//  enum_maps.hpp — String→Int mapping tables for MNN enums

#include <string>
#include <unordered_map>

const std::unordered_map<std::string, int>& gpu_mode_map();
const std::unordered_map<std::string, int>& session_mode_map();
const std::unordered_map<std::string, int>& hint_mode_map();

int resolve(const std::unordered_map<std::string, int>& table,
            const std::string& name, const char* context);
