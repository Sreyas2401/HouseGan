import numpy as np
import matplotlib.pyplot as plt

def visualize_sdf_map(data_path):
    data = np.loadtxt(data_path)

    x_coordinates = data[:, 0]
    y_coordinates = data[:, 1]
    signed_distances = data[:, 2]

    x_min, x_max = min(x_coordinates), max(x_coordinates)
    y_min, y_max = min(y_coordinates), max(y_coordinates)
    image_size = (500, 500, 3)  
    image = np.zeros(image_size)

    for x, y, sdf in zip(x_coordinates, y_coordinates, signed_distances):
        color = [1.0, 0.0, 0.0] if sdf > 0 else [0.0, 0.0, 0.0]  
        x_pixel = int((x - x_min) / (x_max - x_min) * (image_size[1] - 1))
        y_pixel = int((y - y_min) / (y_max - y_min) * (image_size[0] - 1))
        image[y_pixel, x_pixel] = color

    plt.imshow(image, cmap='gray', extent=[x_min, x_max, y_min, y_max])
    plt.colorbar()
    plt.title('SDF Map')
    plt.show()

if __name__ == "__main__":
    val = input("Enter shape name: ")
    data_file = f'..//datasets/val/{val}.txt'
    visualize_sdf_map(data_file)
