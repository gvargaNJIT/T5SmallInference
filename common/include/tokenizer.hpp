#ifndef T5_TOKENIZER_HPP
#define T5_TOKENIZER_HPP

#include <string>
#include <vector>
#include <sentencepiece_processor.h>

namespace t5 {

class Tokenizer {
public:
    Tokenizer() : pad_id_(0), eos_id_(1), unk_id_(2) {}

    bool load_vocab(const std::string& spm_path);

    std::vector<int> encode(const std::string& text);

    std::string decode(const std::vector<int>& ids);

    int pad_token_id() const { return pad_id_; }
    int eos_token_id() const { return eos_id_; }
    int unk_token_id() const { return unk_id_; }

    size_t vocab_size() const { return sp_.GetPieceSize(); }

private:
    sentencepiece::SentencePieceProcessor sp_;

    int pad_id_;
    int eos_id_;
    int unk_id_;
};

}
#endif
