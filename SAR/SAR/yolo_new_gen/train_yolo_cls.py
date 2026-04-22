import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="11")
    parser.add_argument("--variant", type=str, default="n")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    cmd = f"python3 train_yolo.py --task classify --data dataset_yolo_cls --version {args.version} --variant {args.variant} --epochs {args.epochs} --batch {args.batch}"
    print(f"Running Classification: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()
