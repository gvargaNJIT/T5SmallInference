#include "tokenizer.hpp"
#include <stdexcept>

SPTokenizer::SPTokenizer(const std::string &model_path) {
    auto status = sp.Load(model_path);
    if (!status.ok()) {
        throw std::runtime_error(
            "Failed to load SentencePiece model: " +
            std::string(status.message()));
    }

    pad_id = sp.PieceToId("<pad>");
    eos_id = sp.PieceToId("</s>");

    if (pad_id < 0 || eos_id < 0) {
        throw std::runtime_error(
            "Tokenizer does not contain <pad> or </s> pieces.");
    }
}

std::vector<int> SPTokenizer::encode(const std::string &text, bool add_eos) {
    std::vector<int> ids;
    auto status = sp.Encode(text, &ids);

    if (!status.ok()) {
        throw std::runtime_error(
            "SentencePiece encode failed: " +
            std::string(status.message()));
    }

    if (add_eos) {
        ids.push_back(eos_id);
    }

    return ids;
}

std::string SPTokenizer::decode(const std::vector<int> &ids) {
    std::string out;
    auto status = sp.Decode(ids, &out);

    if (!status.ok()) {
        throw std::runtime_error(
            "SentencePiece decode failed: " +
            std::string(status.message()));
    }

    return out;
}

int SPTokenizer::get_pad_id() const {
    return pad_id;
}

int SPTokenizer::get_eos_id() const {
    return eos_id;
}