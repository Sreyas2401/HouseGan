import numpy as np
import matplotlib.pyplot as plt

def visualize_sdf_map(data_path):
    # Read data from the file
    data = np.loadtxt(data_path)

    # Extract coordinates and signed distances
    x_coordinates = data[:, 0]
    y_coordinates = data[:, 1]
    signed_distances = data[:, 2]

    # Create an image grid
    x_min, x_max = min(x_coordinates), max(x_coordinates)
    y_min, y_max = min(y_coordinates), max(y_coordinates)
    image_size = (500, 500)  # Adjust as needed
    image = np.zeros(image_size)

    # Plot points on the image grid based on signed distances
    for x, y, sdf in zip(x_coordinates, y_coordinates, signed_distances):
        # Map signed distance to color
        color = 1.0 if sdf > 0 else 0.0  # White for positive, black for negative
        # Map coordinates to image grid
        x_pixel = int((x - x_min) / (x_max - x_min) * (image_size[1] - 1))
        y_pixel = int((y - y_min) / (y_max - y_min) * (image_size[0] - 1))
        # Paint the pixel
        image[y_pixel, x_pixel] = color

    # Display the SDF map
    plt.imshow(image, cmap='gray', extent=[x_min, x_max, y_min, y_max])
    plt.colorbar()
    plt.title('SDF Map')
    plt.show()

if __name__ == "__main__":
    val = input("Enter shape name: ")
    data_file = f'..//datasets/val/{val}.txt'
    visualize_sdf_map(data_file)
