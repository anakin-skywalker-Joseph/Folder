import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
