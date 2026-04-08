import argparse


def main():
    parser = argparse.ArgumentParser(description="Validate trained tracker")
    parser.add_argument("--checkpoint", type=str, default="checkpoints\\best.pth")
    args = parser.parse_args()

    print("Validation script started")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()