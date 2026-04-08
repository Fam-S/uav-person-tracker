import argparse
import yaml


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="UAV Person Tracker Inference")
    parser.add_argument("--config", type=str, default="configs\\tracker.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print("Inference config loaded successfully:")
    for k, v in config.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()