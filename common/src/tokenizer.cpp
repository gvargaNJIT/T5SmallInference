#include "tokenizer.hpp"
#include <iostream>

namespace t5 {

bool Tokenizer::load_vocab(const std::string& spm_path) {
    auto status = sp_.Load(spm_path);
    if (!status.ok()) {
        std::cerr << "SentencePiece load error: " << status.ToString() << "\n";
        return false;
    }
    if (sp_.PieceToId("<pad>") != 0) std::cerr << "Warning: <pad> ID != 0\n";
    if (sp_.PieceToId("</s>") != 1) std::cerr << "Warning: </s> ID != 1\n";
    if (sp_.PieceToId("<unk>") != 2) std::cerr << "Warning: <unk> ID != 2\n";

    pad_id_ = sp_.PieceToId("<pad>");
    eos_id_ = sp_.PieceToId("</s>");
    unk_id_ = sp_.PieceToId("<unk>");

    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    sp_.Encode(text, &ids);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string out;
    sp_.Decode(ids, &out);
    return out;
}

}
