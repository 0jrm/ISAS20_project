import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA
import h5py

# Add the parent directory to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate')
from model.model import Autoencoder

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        
    def forward(self, pred, target, mask):
        """
        Compute MSE loss only on unmasked values
        
        Args:
            pred: predicted values
            target: target values
            mask: boolean mask (True for NaN/masked values, False for valid values)
        """
        # Convert mask to float and invert it (1 for valid, 0 for masked)
        valid_mask = (~mask).float()
        
        # # Print debugging information for first batch
        # if pred.shape[0] > 0:
        #     print("\nLoss debugging information:")
        #     print(f"Prediction range: [{pred[0].min():.3f}, {pred[0].max():.3f}]")
        #     print(f"Target range: [{target[0].min():.3f}, {target[0].max():.3f}]")
        #     print(f"Number of valid points: {torch.sum(valid_mask[0])}")
        #     print(f"Number of masked points: {torch.sum(mask[0])}")
        #     print(f"Valid locations: {torch.where(valid_mask[0])[0]}")
        #     print(f"Masked locations: {torch.where(mask[0])[0]}")
        
        # Compute squared error
        squared_error = (pred - target) ** 2
        
        # Apply mask and compute mean only over valid values
        masked_error = squared_error * valid_mask
        sum_error = torch.sum(masked_error)
        num_valid = torch.sum(valid_mask)
        
        # Avoid division by zero
        if num_valid == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # # Print loss components
        # print(f"Sum error: {sum_error:.3f}")
        # print(f"Number of valid points: {num_valid}")
        # print(f"Final loss: {sum_error / num_valid:.3f}")
            
        return sum_error / num_valid

def load_data(data_path, parameter_name):
    """Load and prepare the preprocessed data for a specific parameter
    
    Args:
        data_path (str): Path to the preprocessed data file
        parameter_name (str): Name of the parameter to load (e.g., 'temperature_TEMP' or 'salinity_PSAL')
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get parameter data and mask
    param_data = data['outputs'][parameter_name]
    mask = data['masks'][parameter_name]
    
    # # Print debugging information
    # print("\nData loading debugging information:")
    # print(f"Salinity shape: {param_data.shape}")
    # print(f"Mask shape: {mask.shape}")
    # print(f"Number of masked points in first profile: {torch.sum(~mask[0])}")
    # print(f"Masked locations in first profile: {torch.where(~mask[0])[0]}")
    
    # Convert to float32 for better performance
    param_data = param_data.float()
    mask = mask.bool()
    
    return param_data, mask

def train_autoencoder(model, train_loader, val_loader, num_epochs, device, model_name):
    """Train the autoencoder"""
    criterion = MaskedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        # for batch_x, batch_mask in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        for batch_x, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x, batch_mask)
            
            # # Print input and output ranges for debugging
            # if epoch == 0 and train_loss == 0:
            #     print("\nTraining debugging information:")
            #     print(f"Input range: [{batch_x[0].min():.3f}, {batch_x[0].max():.3f}]")
            #     print(f"Output range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")
            #     print(f"Number of valid points: {torch.sum(~batch_mask[0])}")
            #     print(f"Number of masked points: {torch.sum(batch_mask[0])}")
            
            loss = criterion(outputs, batch_x, batch_mask)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_mask in val_loader:
                batch_x = batch_x.to(device)
                batch_mask = batch_mask.to(device)
                
                outputs = model(batch_x, batch_mask)
                loss = criterion(outputs, batch_x, batch_mask)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), f'saved_models/best_{model_name}.pth')
        
        # if (epoch + 1) % 10 == 0:
            # print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Epoch {epoch+1}/{num_epochs}: T. Loss: {train_loss:.2e} | V. Loss: {val_loss:.2e}')
    
    return train_losses, val_losses

def plot_losses(train_losses, val_losses, model_name):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Losses')
    plt.legend()
    # plt.savefig(f'{model_name}_losses.png')
    plt.show()
    # plt.close()

def plot_profiles(original, reconstructed, mask, model_name, num_profiles=4):
    """Plot original vs reconstructed profiles for a few examples
    
    Args:
        original: Original data tensor
        reconstructed: Reconstructed data tensor
        mask: Boolean mask (True for masked values)
        model_name: Name of the model for the plot title
        num_profiles: Number of profiles to plot (default: 4)
    """
    # Convert tensors to numpy arrays
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    mask = mask.cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_profiles, 1, figsize=(8, 4*num_profiles))
    
    # Plot each profile
    for i in range(num_profiles):
        ax = axes[i]
        # Plot original profile
        ax.plot(original[i], range(len(original[i])), 'b-', label='Original', alpha=0.7)
        # Plot reconstructed profile
        ax.plot(reconstructed[i], range(len(reconstructed[i])), 'r--', label='Reconstructed', alpha=0.7)
        # Plot masked points (where mask is True, i.e., NaN values)
        masked_points = np.where(mask[i])[0]
        if len(masked_points) > 0:
            ax.scatter(original[i][masked_points], masked_points, 
                      color='gray', alpha=0.3, label='Masked')
        
        ax.set_title(f'Profile {i+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Depth Level')
        ax.invert_yaxis()  # Invert y-axis to show depth increasing downward
    
    # Add legend to the first subplot
    axes[0].legend()
    plt.suptitle(f'{model_name} - Original vs Reconstructed Profiles')
    plt.tight_layout()
    
    # plt.savefig(f'{model_name}_profiles.png')
    plt.show()
    # plt.close()

def calculate_captured_variance(original, reconstructed, mask):
    """
    Calculate the variance captured by the reconstruction, ignoring masked values.
    
    Args:
        original: Original data tensor
        reconstructed: Reconstructed data tensor
        mask: Boolean mask (True for masked values)
    
    Returns:
        float: Percentage of variance captured
    """
    # Convert mask to float and invert it (1 for valid, 0 for masked)
    valid_mask = (~mask).float()
    
    # Calculate total variance of original data (only on valid points)
    original_mean = (original * valid_mask).sum() / valid_mask.sum()
    total_variance = ((original - original_mean) ** 2 * valid_mask).sum() / valid_mask.sum()
    
    # Calculate reconstruction error variance
    error_variance = ((original - reconstructed) ** 2 * valid_mask).sum() / valid_mask.sum()
    
    # Calculate captured variance percentage
    captured_variance = (1 - error_variance / total_variance) * 100
    
    return captured_variance

def plot_combined_profiles(original, reconstructions, mask, parameter_name, encoding_dims, num_profiles=4):
    """Plot original vs reconstructed profiles for all encoding dimensions in a grid
    
    Args:
        original: Original data tensor
        reconstructions: List of reconstructed data tensors for each encoding dimension
        mask: Boolean mask (True for masked values)
        parameter_name: Name of the parameter being plotted
        encoding_dims: List of encoding dimensions used
        num_profiles: Number of profiles to plot (default: 4)
    """
    # Convert tensors to numpy arrays
    original = original.cpu().numpy()
    reconstructions = [r.cpu().numpy() for r in reconstructions]
    mask = mask.cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_profiles, len(encoding_dims), figsize=(4*len(encoding_dims), 4*num_profiles))
    
    # Plot each profile
    for i in range(num_profiles):
        # Plot reconstructions for each encoding dimension
        for j, (recon, dim) in enumerate(zip(reconstructions, encoding_dims)):
            ax = axes[i, j]
            ax.plot(original[i], range(len(original[i])), 'b-', label='Original', alpha=0.7)
            ax.plot(recon[i], range(len(recon[i])), 'r--', label='Reconstructed', alpha=0.7)
            masked_points = np.where(mask[i])[0]
            if len(masked_points) > 0:
                ax.scatter(original[i][masked_points], masked_points, 
                          color='gray', alpha=0.3, label='Masked')
            ax.set_title(f'Profile {i+1} - Dim {dim}')
            ax.set_xlabel('Value')
            if j == 0:  # Only show y-label for first column
                ax.set_ylabel('Depth Level')
            ax.invert_yaxis()
    
    # Add legend to the first subplot
    axes[0, 0].legend()
    plt.suptitle(f'{parameter_name} - Original vs Reconstructed Profiles for Different Encoding Dimensions')
    plt.tight_layout()
    plt.show()

def plot_combined_losses(train_losses_list, val_losses_list, parameter_name, encoding_dims):
    """Plot training and validation losses for all encoding dimensions
    
    Args:
        train_losses_list: List of training losses for each encoding dimension
        val_losses_list: List of validation losses for each encoding dimension
        parameter_name: Name of the parameter being plotted
        encoding_dims: List of encoding dimensions used
    """
    plt.figure(figsize=(10, 6))
    
    # Plot losses for each encoding dimension
    for i, (train_losses, val_losses, dim) in enumerate(zip(train_losses_list, val_losses_list, encoding_dims)):
        plt.semilogy(train_losses, label=f'Train (dim={dim})', alpha=0.7)
        plt.semilogy(val_losses, label=f'Val (dim={dim})', linestyle='--', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'{parameter_name} - Training and Validation Losses for Different Encoding Dimensions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def pca_reconstruct(train_data, train_mask, test_data, test_mask, n_components):
    """
    Fit PCA on unmasked values of train_data, then reconstruct test_data using top n_components.
    Masked values are ignored in fitting and reconstruction is only evaluated on unmasked values.
    Args:
        train_data: (N, D) tensor
        train_mask: (N, D) tensor (True for masked/NaN)
        test_data: (M, D) tensor
        test_mask: (M, D) tensor
        n_components: int
    Returns:
        reconstructed: (M, D) tensor
    """
    # Convert to numpy
    train_data_np = train_data.cpu().numpy()
    train_mask_np = train_mask.cpu().numpy()
    test_data_np = test_data.cpu().numpy()
    test_mask_np = test_mask.cpu().numpy()
    
    # Only use rows with at least some unmasked values for PCA fit
    valid_rows = ~np.all(train_mask_np, axis=1)
    X_train = train_data_np[valid_rows]
    mask_train = train_mask_np[valid_rows]
    # For PCA, fill masked values with the mean of each feature (column)
    col_means = np.nanmean(np.where(mask_train, np.nan, X_train), axis=0)
    # Fix: Replace NaN means (from all-masked columns) with 0
    col_means = np.where(np.isnan(col_means), 0, col_means)
    X_train_filled = np.where(mask_train, col_means, X_train)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_train_filled)
    
    # For test/vis data, fill masked values with train means
    test_filled = np.where(test_mask_np, col_means, test_data_np)
    # Project and reconstruct
    X_proj = pca.transform(test_filled)
    X_recon = pca.inverse_transform(X_proj)
    # Return as torch tensor
    return torch.tensor(X_recon, dtype=test_data.dtype, device=test_data.device)

def plot_overlay_profiles_ae_pca(original, ae_recons, pca_recons, mask, parameter_name, encoding_dims, depth_list, num_profiles=4, depth_limit=1000):
    """Overlay original (black), AE (red), and PCA (blue) for each profile and encoding dim."""
    original = original.cpu().numpy()
    ae_recons = [r.cpu().numpy() for r in ae_recons]
    pca_recons = [r.cpu().numpy() for r in pca_recons]
    mask = mask.cpu().numpy()
    ncols = len(encoding_dims)
    fig, axes = plt.subplots(num_profiles, ncols, figsize=(4*ncols, 4*num_profiles), sharey=True)
    if num_profiles == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)
    for i in range(num_profiles):
        for j, dim in enumerate(encoding_dims):
            ax = axes[i, j]
            ax.plot(original[i], depth_list, color='b', label='Original', alpha=0.8, linewidth=1.5)
            ax.plot(ae_recons[j][i], depth_list, color='r', linestyle='-', label='AE', alpha=0.6)
            ax.plot(pca_recons[j][i], depth_list, color='k', linestyle='-', label='PCA', alpha=0.6)
            masked_points = np.where(mask[i])[0]
            if len(masked_points) > 0:
                ax.scatter(original[i][masked_points], depth_list[masked_points], color='gray', alpha=0.3, label='Masked')
            ax.set_title(f'Profile {i+1} - Dim {dim}')
            ax.set_xlabel('Value')
            if j == 0:
                ax.set_ylabel('Depth')
            ax.set_ylim(depth_limit, 0)  # Set y-axis limit and invert
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend()
    plt.suptitle(f'{parameter_name} - Overlay: Original (black), AE (red), PCA (blue)')
    plt.tight_layout()
    plt.show()

def plot_residuals_grid(original, ae_recons, pca_recons, mask, encoding_dims, depth_list, num_profiles=4):
    """Plot residuals (original-recon) for AE and PCA, all profiles in translucent gray, mean=0 as dashed blue line, one subplot per technique/encoding dim. All subplots share the same x-axis range. Arranged in a 3x2 grid."""
    original = original.cpu().numpy()
    ae_recons = [r.cpu().numpy() for r in ae_recons]
    pca_recons = [r.cpu().numpy() for r in pca_recons]
    mask = mask.cpu().numpy()
    depth_arr = np.array(depth_list)  # ensure numpy array
    n_methods = 2
    n_dims = len(encoding_dims)
    n_subplots = n_methods * n_dims
    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharey=True)
    axes = axes.flatten()
    # Gather residuals to determine global x-limits
    all_residuals = []
    for j in range(n_dims):
        for recons in [ae_recons[j], pca_recons[j]]:
            for i in range(num_profiles):
                valid = ~mask[i]
                if np.any(valid):
                    residual = (original[i] - recons[i])[valid]
                    all_residuals.append(residual)
    if all_residuals:
        global_min = min([r.min() for r in all_residuals])
        global_max = max([r.max() for r in all_residuals])
    else:
        global_min, global_max = -1, 1
    plot_idx = 0
    for j, dim in enumerate(encoding_dims):
        for method_idx, (recons, method_name, color) in enumerate([
            (ae_recons[j], 'AE', 'r'), (pca_recons[j], 'PCA', 'b')]):
            if plot_idx >= nrows * ncols:
                break
            ax = axes[plot_idx]
            for i in range(num_profiles):
                valid = ~mask[i]
                if not np.any(valid):
                    continue
                residual = (original[i] - recons[i])[valid]
                depth_valid = depth_arr[valid]
                # Plot as a line to maintain profile structure
                ax.plot(residual, depth_valid, color='gray', alpha=0.3)
            ax.axvline(0, color='b', linestyle='--', linewidth=1, label='Mean=0')
            ax.set_title(f'{method_name} Residuals (dim={dim})')
            ax.set_xlabel('Residual')
            if plot_idx == 0:
                ax.set_ylabel('Depth')
            ax.set_xlim(global_min, global_max)
            plot_idx += 1
    # Hide any unused subplots
    for idx in range(plot_idx, nrows * ncols):
        fig.delaxes(axes[idx])
        # Invert y-axis once (shared among axes)
    axes[0].invert_yaxis()
    axes[0].legend()
    plt.suptitle('Residuals: Original - Reconstruction (gray: profiles, blue dashed: mean=0)')
    plt.tight_layout()
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    data_path = '/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/NeSPReSO_v2_GoM_sat/preprocessed_satellite_data.pkl'
    
    # Get available parameters from the data file
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    available_params = list(data['outputs'].keys())
    print(f"Available parameters: {available_params}")
    
    # Get the depth list from another file
    depth_list = h5py.File('/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/NeSPReSO_v2_GoM_sat/profiles_NeSPReSO_v2_GoM.h5', 'r')['model']['DEPH'][0,:]
    
    # List of parameters to process
    parameter_names = ['salinity_PSAL', 'temperature_TEMP']
    print(f"\nProcessing parameters: {parameter_names}")
    
    # Process each parameter
    for parameter_name in parameter_names:
        print(f"\n{'='*50}")
        print(f"Processing parameter: {parameter_name}")
        print(f"{'='*50}")
        
        param_data, mask = load_data(data_path, parameter_name)
        
        # Create random indices for train/val/test split (70/20/10)
        num_samples = len(param_data)
        indices = torch.randperm(num_samples)
        train_size = int(0.7 * num_samples)
        val_size = int(0.2 * num_samples)
        test_size = num_samples - train_size - val_size
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # Create datasets using the random indices
        train_dataset = TensorDataset(param_data[train_indices], mask[train_indices])
        val_dataset = TensorDataset(param_data[val_indices], mask[val_indices])
        test_dataset = TensorDataset(param_data[test_indices], mask[test_indices])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Select random profiles for visualization from test set
        num_vis_profiles = 4
        vis_indices = torch.randperm(len(test_dataset))[:num_vis_profiles]
        vis_data = torch.stack([test_dataset[i][0] for i in vis_indices])
        vis_mask = torch.stack([test_dataset[i][1] for i in vis_indices])
        # For residuals, use all test profiles
        test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
        test_mask = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
        
        # Test both autoencoders
        encoding_dims = [16, 32, 64]  # Different bottleneck sizes to test
        
        # Define different layer layouts to test
        layer_layouts = [
            ([512, 128, 32], [32, 128, 512]),
            ([128, 674], [64, 128]),
            ([128], [128]),
        ]

        layout_names = [
            '[512, 128, 32]',
            '[128, 64]',
            '[128]'
        ]

        for layout, layout_name in zip(layer_layouts, layout_names):
            encoder_layers, decoder_layers = layout
            print(f"\nTesting with layer layout: encoder={encoder_layers}, decoder={decoder_layers}")
            all_train_losses = []
            all_val_losses = []
            all_reconstructions = []
            all_pca_reconstructions = []
            all_pca_variances = []
            for dim in encoding_dims:
                print(f"\n  Encoding dimension: {dim}")
                # Masked Autoencoder
                masked_ae = Autoencoder(
                    encoding_dim=dim,
                    encoder_layers=encoder_layers,
                    decoder_layers=decoder_layers,
                    input_dim=param_data.shape[1]
                ).to(device)
                train_losses, val_losses = train_autoencoder(
                    masked_ae, train_loader, val_loader, 
                    num_epochs=20, device=device, 
                    model_name=f'masked_ae_{parameter_name}_dim{dim}_layout{layout_name}'
                )
                all_train_losses.append(train_losses)
                all_val_losses.append(val_losses)
                masked_ae.eval()
                with torch.no_grad():
                    vis_data_device = vis_data.to(device)
                    vis_mask_device = vis_mask.to(device)
                    masked_recon = masked_ae(vis_data_device, vis_mask_device).cpu()
                    # For residuals, reconstruct all test profiles
                    test_data_device = test_data.to(device)
                    test_mask_device = test_mask.to(device)
                    masked_recon_test = masked_ae(test_data_device, test_mask_device).cpu()
                all_reconstructions.append(masked_recon)
                # PCA reconstruction and variance
                pca_recon = pca_reconstruct(param_data[train_indices], mask[train_indices], vis_data, vis_mask, dim)
                pca_recon_test = pca_reconstruct(param_data[train_indices], mask[train_indices], test_data, test_mask, dim)
                all_pca_reconstructions.append(pca_recon)
                # Variance (optional, not used in plots)
                captured_var = calculate_captured_variance(vis_data, masked_recon, vis_mask)
                pca_var = calculate_captured_variance(vis_data, pca_recon, vis_mask)
                all_pca_variances.append(pca_var)
                print(f"    Autoencoder captured variance: {captured_var:.2f}%")
                print(f"    PCA captured variance: {pca_var:.2f}%")
                if dim == encoding_dims[0]:
                    all_reconstructions_test = [masked_recon_test]
                    all_pca_reconstructions_test = [pca_recon_test]
                else:
                    all_reconstructions_test.append(masked_recon_test)
                    all_pca_reconstructions_test.append(pca_recon_test)
            # Plot combined results for this layout (autoencoder and PCA)
            plot_combined_losses(all_train_losses, all_val_losses, f"{parameter_name} (layout {layout_name})", encoding_dims)
            plot_overlay_profiles_ae_pca(vis_data, all_reconstructions, all_pca_reconstructions, vis_mask, f"{parameter_name} (layout {layout_name})", encoding_dims, depth_list, num_vis_profiles)
            plot_residuals_grid(test_data, all_reconstructions_test, all_pca_reconstructions_test, test_mask, encoding_dims, depth_list, num_profiles=len(test_data))

if __name__ == "__main__":
    main() 