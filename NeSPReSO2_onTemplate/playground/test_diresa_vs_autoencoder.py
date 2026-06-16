import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from diresa.models import build_diresa
from diresa.loss import mse_dist_loss, LatentCovLoss
from diresa.callback import LossWeightAnnealing
import keras.backend as K
import tensorflow as tf
from keras import layers, Input, Model
import keras

# Reuse plotting and evaluation functions from test_autoencoders.py
from test_autoencoders import (
    load_data, plot_combined_losses, plot_combined_profiles, calculate_captured_variance
)

# Import classic autoencoder
from model.model import Autoencoder

# --- Masked Encoder and Decoder for DIRESA ---
def masked_encoder_model(input_shape, output_shape, units=40):
    # input_shape = (data_dim,)
    concat_input = Input(shape=(input_shape[0]*2,))
    data = layers.Lambda(lambda x: x[:, :input_shape[0]])(concat_input)
    mask = layers.Lambda(lambda x: x[:, input_shape[0]:])(concat_input)
    x_masked = layers.Multiply()([data, layers.Lambda(lambda m: 1. - m)(mask)])
    y = layers.Dense(units=units, activation="relu")(x_masked)
    y = layers.Dense(units=units // 2, activation="relu")(y)
    y = layers.Dense(output_shape, activation="linear")(y)
    return Model(concat_input, y, name="Encoder")

def masked_decoder_model(input_shape, output_shape, units=40):
    latent = Input(shape=(input_shape,))
    y = layers.Dense(units=units // 2, activation="relu")(latent)
    y = layers.Dense(units=units, activation="relu")(y)
    y = layers.Dense(output_shape, activation="linear")(y)
    return Model(latent, y, name="Recon")

# --- Custom Masked MSE Loss for DIRESA ---
def masked_mse_loss(y_true, y_pred):
    # y_true: (batch, data_dim + mask_dim)
    # y_pred: (batch, data_dim)
    data_dim = tf.shape(y_pred)[-1]
    data = y_true[:, :data_dim]
    mask = y_true[:, data_dim:]
    valid_mask = 1. - mask  # mask: 1 for masked, 0 for valid
    squared_error = tf.square(data - y_pred) * valid_mask
    sum_error = tf.reduce_sum(squared_error)
    num_valid = tf.reduce_sum(valid_mask)
    return tf.where(num_valid > 0, sum_error / num_valid, 0.0)

def train_diresa_model(model, train, train_twin, val, val_twin, epochs, batch_size, cov_weight=None):
    # Compile model
    if cov_weight is not None:
        model.compile(
            loss=['MSE', LatentCovLoss(cov_weight), mse_dist_loss],
            loss_weights=[1., 1., 1.],
            optimizer="adam",
        )
    else:
        model.compile(
            loss=['MSE', LatentCovLoss(), mse_dist_loss],
            loss_weights=[1., 3., 1.5],
            optimizer="adam",
        )
    # Annealing callback if cov_weight is used
    callbacks = []
    if cov_weight is not None:
        callbacks = [LossWeightAnnealing(cov_weight, target_loss=0.0001, anneal_step=0.2, start_epoch=3)]
    # Fit
    history = model.fit(
        (train, train_twin), (train, train, train),
        validation_data=((val, val_twin), (val, val, val)),
        epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2, callbacks=callbacks
    )
    return history

def train_masked_diresa_model(model, train, train_twin, val, val_twin, epochs, batch_size, cov_weight=None):
    # Compile model with custom masked MSE loss
    if cov_weight is not None:
        model.compile(
            loss=[masked_mse_loss, LatentCovLoss(cov_weight), mse_dist_loss],
            loss_weights=[1., 1., 1.],
            optimizer="adam",
        )
    else:
        model.compile(
            loss=[masked_mse_loss, LatentCovLoss(), mse_dist_loss],
            loss_weights=[1., 3., 1.5],
            optimizer="adam",
        )
    callbacks = []
    if cov_weight is not None:
        callbacks = [LossWeightAnnealing(cov_weight, target_loss=0.0001, anneal_step=0.2, start_epoch=3)]
    # Fit
    history = model.fit(
        [train, train_twin], [train, train, train],
        validation_data=([val, val_twin], [val, val, val]),
        epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2, callbacks=callbacks
    )
    return history

def main():
    # Set device for torch (for classic autoencoder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    torch.manual_seed(42)
    np.random.seed(42)

    # Data path and parameters
    data_path = '/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/NeSPReSO_v2_GoM_sat/preprocessed_satellite_data.pkl'
    parameter_names = ['salinity_PSAL', 'temperature_TEMP']

    # Only use 2 encoding dimensions and 2 layer layouts
    encoding_dims = [16, 64]  # Drop the middle one
    layer_layouts = [
        ([512, 128, 32], [32, 128, 512]),
        ([128], [128]),
    ]
    layout_names = [
        '[512, 128, 32]',
        '[128]'
    ]
    
    # Load data file to get available parameters
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Available parameters: {list(data['outputs'].keys())}")

    for parameter_name in parameter_names:
        print(f"\n{'='*50}")
        print(f"Processing parameter: {parameter_name}")
        print(f"{'='*50}")
        param_data, mask = load_data(data_path, parameter_name)
        num_samples = len(param_data)
        indices = torch.randperm(num_samples)
        train_size = int(0.8 * num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_dataset = TensorDataset(param_data[train_indices], mask[train_indices])
        val_dataset = TensorDataset(param_data[val_indices], mask[val_indices])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        num_vis_profiles = 4
        vis_indices = torch.randperm(len(val_dataset))[:num_vis_profiles]
        vis_data = torch.stack([val_dataset[i][0] for i in vis_indices])
        vis_mask = torch.stack([val_dataset[i][1] for i in vis_indices])

        for layout, layout_name in zip(layer_layouts, layout_names):
            encoder_layers, decoder_layers = layout
            print(f"\nTesting with layer layout: encoder={encoder_layers}, decoder={decoder_layers}")
            all_train_losses = []
            all_val_losses = []
            all_reconstructions = []
            all_diresa_train_losses = []
            all_diresa_val_losses = []
            all_diresa_reconstructions = []
            for dim in encoding_dims:
                print(f"\n  Encoding dimension: {dim}")
                # Classic Masked Autoencoder
                model_save_path = f'saved_models/best_masked_ae_{parameter_name}_dim{dim}_layout{layout_name}.pth'
                masked_ae = Autoencoder(
                    encoding_dim=dim,
                    encoder_layers=encoder_layers,
                    decoder_layers=decoder_layers,
                    input_dim=param_data.shape[1]
                ).to(device)
                from test_autoencoders import train_autoencoder
                if os.path.exists(model_save_path):
                    print(f"Loading saved model from {model_save_path}")
                    masked_ae.load_state_dict(torch.load(model_save_path, map_location=device))
                    train_losses, val_losses = [None], [None]  # Placeholder, not available when loading
                else:
                    train_losses, val_losses = train_autoencoder(
                        masked_ae, train_loader, val_loader,
                        num_epochs=20, device=device,
                        model_name=f'masked_ae_{parameter_name}_dim{dim}_layout{layout_name}'
                    )
                all_train_losses.append(train_losses)
                all_val_losses.append(val_losses)
                masked_ae.eval()
                with torch.no_grad():
                    vis_data_dev = vis_data.to(device)
                    vis_mask_dev = vis_mask.to(device)
                    masked_recon = masked_ae(vis_data_dev, vis_mask_dev)
                all_reconstructions.append(masked_recon.cpu())
                captured_var = calculate_captured_variance(vis_data, masked_recon.cpu(), vis_mask)
                print(f"    Masked AE captured variance: {captured_var:.2f}%")

                # DIRESA Masked Model
                input_shape = (param_data.shape[1],)
                encoder = masked_encoder_model(input_shape, dim, units=encoder_layers[0] if len(encoder_layers) > 0 else 40)
                decoder = masked_decoder_model(dim, input_shape[0], units=decoder_layers[0] if len(decoder_layers) > 0 else 40)
                from diresa.models import diresa_model
                x = Input(shape=(input_shape[0]*2,))
                x_twin = Input(shape=(input_shape[0]*2,))
                diresa = diresa_model(x=x, x_twin=x_twin, encoder=encoder, decoder=decoder)
                cov_weight = tf.Variable(0., dtype=tf.float32)
                # Prepare data for DIRESA (numpy arrays)
                train_np = param_data[train_indices].cpu().numpy()
                train_mask_np = mask[train_indices].cpu().numpy().astype(np.float32)
                val_np = param_data[val_indices].cpu().numpy()
                val_mask_np = mask[val_indices].cpu().numpy().astype(np.float32)
                # Concatenate data and mask for input
                train_concat = np.concatenate([train_np, train_mask_np], axis=1)
                val_concat = np.concatenate([val_np, val_mask_np], axis=1)
                # For twin input, shuffle each batch
                train_twin = np.copy(train_concat)
                np.random.shuffle(train_twin)
                val_twin = np.copy(val_concat)
                np.random.shuffle(val_twin)
                # Train Masked DIRESA
                history = train_masked_diresa_model(
                    diresa, train_concat, train_twin, val_concat, val_twin,
                    epochs=20, batch_size=32, cov_weight=cov_weight
                )
                # Losses
                diresa_train_loss = history.history['loss']
                diresa_val_loss = history.history['val_loss']
                all_diresa_train_losses.append(diresa_train_loss)
                all_diresa_val_losses.append(diresa_val_loss)
                # Encoder/decoder for reconstruction
                keras.config.enable_unsafe_deserialization()
                from diresa.toolbox import encoder_decoder
                compress_model, decode_model = encoder_decoder(diresa)
                vis_data_np = vis_data.cpu().numpy()
                vis_mask_np = vis_mask.cpu().numpy().astype(np.float32)
                vis_concat = np.concatenate([vis_data_np, vis_mask_np], axis=1)
                latent = compress_model.predict(vis_concat)
                predict = decode_model.predict(latent)
                # Restore original values for masked points (like in PyTorch)
                predict[vis_mask_np == 1] = vis_data_np[vis_mask_np == 1]
                predict_tensor = torch.tensor(predict)
                all_diresa_reconstructions.append(predict_tensor)
                captured_var_diresa = calculate_captured_variance(vis_data, predict_tensor, vis_mask)
                print(f"    Masked DIRESA captured variance: {captured_var_diresa:.2f}%")

            # Plot combined results for this layout
            plot_combined_losses(all_train_losses, all_val_losses, f"{parameter_name} Masked AE (layout {layout_name})", encoding_dims)
            plot_combined_profiles(vis_data, all_reconstructions, vis_mask, f"{parameter_name} Masked AE (layout {layout_name})", encoding_dims, num_vis_profiles)
            plot_combined_losses(all_diresa_train_losses, all_diresa_val_losses, f"{parameter_name} Masked DIRESA (layout {layout_name})", encoding_dims)
            plot_combined_profiles(vis_data, all_diresa_reconstructions, vis_mask, f"{parameter_name} Masked DIRESA (layout {layout_name})", encoding_dims, num_vis_profiles)

if __name__ == "__main__":
    main() 