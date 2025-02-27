import numpy as np

def write_obj(points: np.ndarray, file_path: str):
    if not isinstance(points, np.ndarray) or points.shape[1] != 3:
        raise ValueError("points must be (n, 3) numpy.ndarray")
    
    with open(file_path, 'w') as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
    print(f"OBJ saved in {file_path}")



def generate_xyz(min, max):
    random_xyz = np.random.uniform(low = min, high=max, size=3)
    x, y, z = np.sort(random_xyz)[::-1]
    return x, y, z

def sample_points_on_cuboid_surface(x, y, z, num_points, noise=0.0):
    areas = np.array([x*y, x*y, x*z, x*z, y*z, y*z])
    total_area = 2 * (x*y + x*z + y*z)
    probs = areas / total_area

    face_indices = np.random.choice(6, size=num_points, p=probs)
    points = np.zeros((num_points, 3))
    
    mask = (face_indices == 0) | (face_indices == 1)
    n = np.sum(mask)
    points[mask, 0] = np.random.uniform(-x, x, size=n)
    points[mask, 1] = np.random.uniform(-y, y, size=n)
    points[mask, 2] = np.where(face_indices[mask] == 0, z, -z)
    
    mask = (face_indices == 2) | (face_indices == 3)
    n = np.sum(mask)
    points[mask, 0] = np.random.uniform(-x, x, size=n)
    points[mask, 2] = np.random.uniform(-z, z, size=n)
    points[mask, 1] = np.where(face_indices[mask] == 2, y, -y)
    
    mask = (face_indices == 4) | (face_indices == 5)
    n = np.sum(mask)
    points[mask, 1] = np.random.uniform(-y, y, size=n)
    points[mask, 2] = np.random.uniform(-z, z, size=n)
    points[mask, 0] = np.where(face_indices[mask] == 4, x, -x)
    
    points += noise * np.random.randn(*points.shape)
    return points

if __name__ == "__main__":
    # test the code 
    points = sample_points_on_cuboid_surface(*generate_xyz(1, 3), 1024, noise=0.03)
    write_obj(points, "datasets/test_points.obj")