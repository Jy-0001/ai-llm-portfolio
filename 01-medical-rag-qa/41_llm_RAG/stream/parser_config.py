import argparse
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        default="/root/llm_RAG/DeepSpeedExamples-master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/output/actor",
                        help="Directory containing trained actor model")
    
    parser.add_argument("--path1",
                        type=str,
                        default="/root/llm_RAG/DeepSpeedExamples-master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output",
                        help="Directory containing step1 trained actor model")

    parser.add_argument("--path2",
                        type=str,
                        default="/root/llm_RAG/new_data/RAG_data/dialog.jsonl",
                        help="File of dialog data")

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()
    return args

