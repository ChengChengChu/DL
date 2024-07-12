import numpy as np
from utils import *

def main():
    parser = ArgumentParser()
    args  = set_arguments(parser)
   
            
def set_arguments():
    parser.add_argument("--lr", type=float, default=10e-5)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_iters", type=int, default=2)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()    