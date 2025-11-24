#!/usr/bin/env python3
import argparse
from config import get_t5_config
from cases.tensor_cases import TensorCases
from cases.linear_cases import LinearCases
from cases.embedding_cases import EmbeddingCases
from cases.layernorm_cases import LayerNormCases
from cases.feedforward_cases import FeedForwardCases
from cases.attention_cases import AttentionCases


def main():
    output = "test_cases"
    config = get_t5_config()

    TensorCases(output).generate()
    LinearCases(output, config).generate()
    EmbeddingCases(output, config).generate()
    LayerNormCases(output, config).generate()
    FeedForwardCases(output, config).generate()
    AttentionCases(output, config).generate()

    print("\nAll test cases generated successfully!")


if __name__ == "__main__":
    main()
