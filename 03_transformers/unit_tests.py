import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from patch_embed import img_to_patch, PositionalEncoding, PatchEmbedding
# from model import ClassToken, SelfAttention, MultiHeadSelfAttention, MultiLayerPerceptron, TransformerEncoder, ClassificationHead, VisionTransformer
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns



def test_img_to_patch(img_to_patch):
    # read image and resize to 128
    image = Image.open('data/images/car.png').resize((128, 128))

    # convert to array
    image = np.array(image)

    # Parameters for patching
    P = 16  # patch size
    C = 3   # number of channels (RGB)

    # split image into patches 
    flatten_patches = img_to_patch(image, P, C)

    # Output information about the patches
    print('Image shape: ', image.shape)  # width, height, channel
    print('Number of patches: {} with resolution ({}, {})'.format(flatten_patches.shape[1], P, P))
    print('Flattened patches shape: ', flatten_patches.shape)
    
def display_img_and_patches(img_to_patch, heatmap=False):
    # Display image and patches side-by-side
    image = Image.open('data/images/car.png').resize((128, 128))
    image = np.array(image)

    # Parameters for patching
    P = 16  # patch size
    C = 3   # number of channels (RGB)

    # split image into patches 
    flatten_patches = img_to_patch(image, P, C)
    if not heatmap:
        fig = plt.figure(figsize=(12, 6))  # Set figure size for better visibility

        # Create a grid for the main image and the patches
        gridspec = fig.add_gridspec(1, 2, width_ratios=[1, 2])  # Allocate more space to patches
        ax1 = fig.add_subplot(gridspec[0])
        ax1.set_title('Original Image')
        ax1.axis('off')  # Hide axes for a cleaner look

        # Display the original image
        ax1.imshow(image)

        # Create a subgrid for displaying patches (8x8 grid of patches)
        subgridspec = gridspec[1].subgridspec(8, 8, hspace=0.05, wspace=0.05)  # Adjust spacing

        # Display each patch in the grid
        for i in range(8):
            for j in range(8):
                ax = fig.add_subplot(subgridspec[i, j])
                ax.axis('off')  # Hide axes for each patch
                patch_index = i * 8 + j
                # Assuming 'flatten_patches' is correctly reshaped 
                patch_image = flatten_patches[0][patch_index].reshape(C, P, P).permute(1, 2, 0) # Reshape each patch
                ax.imshow(patch_image.numpy().astype(int))  # Convert to uint8 for display
        plt.show()
    
    else:
        # Display the first 10 flattened patches, considering up to 25 values each
        heat_map = flatten_patches[0, :10, :25].numpy().astype(int)  # Adjust indexing if needed based on data structure

        # Create y-tick labels representing each patch
        yticklabels = ['Patch {}'.format(i + 1) for i in range(10)]

        # Setup the plot
        plt.figure(figsize=(16, 10))  # Set an appropriate figure size for clarity
        sns.set_style("whitegrid")  # Set a background style for better visibility

        # Create a heatmap using seaborn
        ax = sns.heatmap(heat_map,  
                        cmap=sns.light_palette("purple", as_cmap=True),  # Aesthetic color choice
                        annot=True,  # Optional: turn on to see values in each cell
                        fmt="d",  # Formatting option if annotations are enabled
                        xticklabels=False,  # Disable x-tick labels for clarity
                        yticklabels=yticklabels,  # Apply custom y-tick labels
                        linewidths=0.5,  # Adjust line widths for better separation
                        linecolor='white'  # Line color for separation lines
                        )

        ax.set_title('First 10 Flattened Patches')  # Title of the heatmap
        plt.xlabel('Feature Values')  # Label for the x-axis
        plt.ylabel('Patches')  # Label for the y-axis
        plt.show()  # Display the heatmap
        
def test_patch_embeddings(PatchEmbedding):
    # Dimensionality of patch embeddings
    D = 768
    P = 16  # Example patch size
    C = 3   # Number of channels (e.g., RGB)

    PE = PatchEmbedding(P, C, D)
    
    # Simulate some data
    B = 10  # Batch size
    N = 20  # Number of patches per image
    flatten_patches = torch.randn(B, N, C * P * P)

    # Generate patch embeddings
    patch_embeddings = PE(flatten_patches)

    # Assert the shape of the output
    assert patch_embeddings.shape == (B, N, D), "Output shape is incorrect"
    print("Output shape is correct:", patch_embeddings.shape)
    
def test_class_token(ClassToken):
    embedding_dim = 768
    cls_token = ClassToken(embedding_dim)
    x = torch.randn(2, 5, embedding_dim)
    output = cls_token(x)

    assert output.shape == (2, 6, embedding_dim), "ClassToken output shape is incorrect"
    print("ClassToken Test Passed: Output shape is correct.")
    
def test_positional_encoding(PositionalEncoding):
    batch_size = 10
    num_patches = 100  # Assuming a 10x10 grid of patches
    embedding_dim = 768

    # Create dummy input tensor
    x = torch.randn(batch_size, num_patches, embedding_dim)

    # Create positional encoding layer
    positional_encoding = PositionalEncoding(num_patches, embedding_dim)

    # Apply positional encoding
    encoded_x = positional_encoding(x)

    print("Shape of encoded features:", encoded_x.shape)
    assert encoded_x.shape == (batch_size, num_patches, embedding_dim), "Output shape is incorrect"
    print ("Positional Encoding Test Passed: Output shape is correct.")
    

def test_self_attention(SelfAttention):
    batch_size = 2
    seq_length = 5
    embedding_dim = 768
    key_dim = 64

    model = SelfAttention(embedding_dim, key_dim)
    x = torch.randn(batch_size, seq_length, embedding_dim)
    output = model(x)

    assert output.shape == (batch_size, seq_length, embedding_dim), "SelfAttention output shape is incorrect"
    print("SelfAttention Test Passed: Output shape is correct.")
    
def test_multi_head_self_attention(MultiHeadSelfAttention):
    batch_size = 2
    seq_length = 5
    embedding_dim = 768
    num_heads = 12

    model = MultiHeadSelfAttention(embedding_dim, num_heads)
    x = torch.randn(batch_size, seq_length, embedding_dim)
    output = model(x)

    assert output.shape == (batch_size, seq_length, embedding_dim), "MultiHeadSelfAttention output shape is incorrect"
    print("MultiHeadSelfAttention Test Passed: Output shape is correct.")
    
def test_mlp(MultiLayerPerceptron):
    batch_size = 2
    seq_length = 5
    embedding_dim = 768
    hidden_dim = 3072

    model = MultiLayerPerceptron(embedding_dim, hidden_dim)
    x = torch.randn(batch_size, seq_length, embedding_dim)
    output = model(x)

    assert output.shape == (batch_size, seq_length, embedding_dim), "MLP output shape is incorrect"
    print("MLP Test Passed: Output shape is correct.")


def test_transformer_encoder(TransformerEncoder):
    batch_size = 2
    seq_length = 5
    embedding_dim = 768
    num_heads = 12
    hidden_dim = 3072
    dropout_prob = 0.1

    model = TransformerEncoder(embedding_dim, num_heads, hidden_dim, dropout_prob)
    x = torch.randn(batch_size, seq_length, embedding_dim)
    output = model(x)

    assert output.shape == (batch_size, seq_length, embedding_dim), "TransformerEncoder output shape is incorrect"
    print("TransformerEncoder Test Passed: Output shape is correct.")
    
def test_classification_head(ClassificationHead):
    batch_size = 2
    seq_length = 5
    embedding_dim = 768
    num_classes = 10

    model = ClassificationHead(embedding_dim, num_classes)
    x = torch.randn(batch_size, seq_length, embedding_dim)
    output = model(x)

    assert output.shape == (batch_size, seq_length, num_classes), "ClassificationHead output shape is incorrect"
    print("ClassificationHead Test Passed: Output shape is correct.")

def test_vision_transformer(VisionTransformer):
    batch_size = 2
    image_size = 224
    channel_size = 3
    patch_size = 16
    num_layers = 12
    embedding_dim = 768
    num_heads = 12
    hidden_dim = 3072
    dropout_prob = 0.1
    num_classes = 10

    model = VisionTransformer(patch_size, image_size, channel_size, num_layers, embedding_dim, num_heads, hidden_dim, dropout_prob, num_classes)
    x = torch.randn(batch_size, channel_size, image_size, image_size)
    output = model(x)

    assert output.shape == (batch_size, num_classes), "VisionTransformer output shape is incorrect"
    print("VisionTransformer Test Passed: Output shape is correct.")
