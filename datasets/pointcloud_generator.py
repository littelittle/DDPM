import numpy as np
from scipy.spatial.transform import Rotation as R

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

def add_random_rotation(point_cloud: np.ndarray):
    R_M = R.random().as_matrix()
    return point_cloud@R_M, R_M

def face_generator(x, y, z):
    """
    Generate the six rectangular faces of a cuboid given its dimensions x, y, and z.
    Returns a list of numpy arrays, each containing the four vertices of a face, and the rotation matrix.
    Remember, the rotation matrix is mutiplied on the right, which means that we need to retrive the row of the matrix
    to get the correct axis.
    """
    # Define the 8 vertices of the cuboid
    vertices = np.array([
        [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],  # Bottom face
        [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]   # Top face
    ])
    
    # centralize to (0,0,0)
    vertices = vertices - np.mean(vertices, axis=0)

    # add random rotation
    vertices, R_M = add_random_rotation(vertices)

    # Define the six faces using vertex indices
    faces = [
        [vertices[3], vertices[2], vertices[1], vertices[0]],  # Bottom face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
        [vertices[4], vertices[7], vertices[3], vertices[0]],  # Left face
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # Right face
    ]
    
    return [np.array(face) for face in faces], R_M


if __name__ == "__main__":
    # test the code 
    points = sample_points_on_cuboid_surface(*generate_xyz(1, 3), 1024, noise=0.03)
    write_obj(points, "datasets/test_points.obj")