#pragma once
#include <sentencepiece_processor.h>
#include <string>
#include <vector>
#include <stdexcept>

class SPTokenizer {
private:
    sentencepiece::SentencePieceProcessor sp;
    int pad_id;
    int eos_id;

public:
    SPTokenizer(const std::string &model_path);
    std::vector<int> encode(const std::string &text, bool add_eos = true);
    std::string decode(const std::vector<int> &ids);
    int get_pad_id() const;
    int get_eos_id() const;
};


