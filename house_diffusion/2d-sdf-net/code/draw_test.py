import matplotlib.pyplot as plt

# Read the x, y coordinates from your file
x = []
y = []
with open('../shapes/raw/quarter_circle.txt', 'r') as file:
    for line in file:
        coords = line.split()
        x.append(float(coords[0]))
        y.append(float(coords[1]))

# Plot the coordinates
plt.plot(x, y, 'o', markersize=1)  # 'o' for points, adjust markersize as needed
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of X, Y coordinates')
plt.grid(True)
plt.show()
