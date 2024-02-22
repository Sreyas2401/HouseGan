from shape import Polygon

class SurfaceArea:
    def __init__(self, name):
        self.shape_name = name
        self.shape_path = '../shapes/raw/'
        self.shape = Polygon()  
        self.vertices = self.shape.load_list(self.shape_path, self.shape_name)
        print(self.vertices)

    def calc_polygon_area(self):
        vertices = self.vertices
        n = len(vertices)
        if n < 3:
            raise ValueError("A polygon must have at least 3 vertices")

        # Apply the Shoelace Formula
        area = 0.5 * sum((vertices[i][0] * vertices[(i + 1) % n][1] - vertices[(i + 1) % n][0] * vertices[i][1])
                        for i in range(n))

        return abs(area)

if __name__ == '__main__':
    print('Enter shape name:')
    shape_name = input()
    area_calculator = SurfaceArea(shape_name)
    area = area_calculator.calc_polygon_area()
    print(f'Surface Area: {area}')
