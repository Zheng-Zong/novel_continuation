import os
import yaml


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        config_data = yaml.safe_load(file)
        environment = config_data.get('environment', {})
        for key, value in environment.items():
            os.environ[key] = value
        return config_data.get('config', {})


config = load_config(file_path='config.yaml')


if __name__ == '__main__':
    print("Environment Variables:")
    for key, value in os.environ.items():
        print(f"{key}: {value}")

    print("\nConfig:")
    for key, value in config.items():
        print(f"{key}: {value}")