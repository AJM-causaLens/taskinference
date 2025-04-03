import os
import argparse
import json
from experiment_utils import get_env_properties, make_vae, make_task_inference_model

def get_config(config_path):
    files = os.listdir(config_path)
    args_files = [file for file in files if file.endswith('.json')]
    vae_files = [file for file in files if file.endswith('.pt')]
    best_vae = max(vae_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    assert len(args_files) == 1, f"Expected 1 args file, got {len(args_files)}"

    args_path = os.path.join(config_path, args_files[0])
    vae_path = os.path.join(config_path, best_vae)
    with open(args_path, 'r') as f:
        args_dict = json.load(f)

    # Convert dictionary to argparse.Namespace
    args = argparse.Namespace(**args_dict)
    input_dim, action_dim, _ = get_env_properties(args.env_name)
    vae = make_vae(input_dim, action_dim, args)

    vae.load_model(vae_path)

    return args, vae



def get_args_from_path(model_path):
    """
    Given the path to a trained VAE model, find and load the corresponding JSON config file.

    Args:
        model_path (str): Path to the VAE model (e.g., 'trained_vae/point_mass_easy/run_10/model_700.pth').

    Returns:
        argparse.Namespace: The parsed configuration.
    """
    # Get the directory of the provided model path
    config_path = os.path.dirname(model_path)

    # Get all files in the directory
    files = os.listdir(config_path)

    # Find the JSON config file
    args_files = [file for file in files if file.endswith('.json')]
    assert len(args_files) == 1, f"Expected 1 args file, got {len(args_files)}"

    # Construct the full path to the JSON config file
    args_path = os.path.join(config_path, args_files[0])

    # Load the configuration
    with open(args_path, 'r') as f:
        args_dict = json.load(f)

    # Convert dictionary to argparse.Namespace
    args = argparse.Namespace(**args_dict)

    return args

def get_config_from_path(model_path):
    """
    Given the path to a trained VAE model, find and load the corresponding JSON config file.

    Args:
        model_path (str): Path to the VAE model (e.g., 'trained_vae/point_mass_easy/run_10/model_700.pth').

    Returns:
        tuple: (args, vae) where args is the parsed configuration, and vae is the loaded model.
    """
    # Get the directory of the provided model path
    config_path = os.path.dirname(model_path)

    # Get all files in the directory
    files = os.listdir(config_path)

    # Find the JSON config file
    args_files = [file for file in files if file.endswith('.json')]
    assert len(args_files) == 1, f"Expected 1 args file, got {len(args_files)}"

    # Construct the full path to the JSON config file
    args_path = os.path.join(config_path, args_files[0])

    # Load the configuration
    with open(args_path, 'r') as f:
        args_dict = json.load(f)

    # Convert dictionary to argparse.Namespace
    args = argparse.Namespace(**args_dict)

    args.pretrained_path = model_path

    # Get VAE properties
    input_dim, action_dim, _ = get_env_properties(args.env_name)
    if "task_decoder" in model_path:
        vae = make_task_inference_model(input_dim, action_dim, args)
    else:
        vae = make_vae(input_dim, action_dim, args)

    # Load the model using the provided path
    vae.load_model(model_path)

    return args, vae
