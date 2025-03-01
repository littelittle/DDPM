import numpy as np
from datasets.pointcloud_generator import *
from concurrent.futures import ThreadPoolExecutor


def get_render(point_cloud, distance, r, num_rays, step=0.2, bounds=0.1):
    r_s = np.random.uniform(low=0, high=r, size=num_rays)
    theta = np.random.uniform(low=0, high=2*np.pi, size=num_rays)
    start_points = np.stack([r_s*np.cos(theta), r_s*np.sin(theta), np.full_like(theta, -3)], axis=-1)
    directions = np.stack([r_s*np.cos(theta), r_s*np.sin(theta), np.full_like(theta, distance-3)], axis=-1)/(distance-3)
    step_size = int(6//step)
    renderd_points = []
    write_obj(start_points, "datasets/start_points.obj")
    for i in range(step_size):
        start_points += directions*step
        for j in range(start_points.shape[0]-len(renderd_points)):
            if np.min(np.linalg.norm(point_cloud-start_points[j,:], axis=-1))<bounds:
                renderd_points.append(start_points[j,:].copy())
                start_points[j,:] = start_points[start_points.shape[0]-len(renderd_points)-1,:]

    renderd_points = np.stack(renderd_points)
    return renderd_points

def ray_generator(r, num_rays, ray_origin=np.array([0, 0, 6])):
    '''
    generate ray_dirs like normal distribution
    '''
    r_s = np.random.uniform(low=0, high=r, size=num_rays)
    theta = np.random.uniform(low=0, high=2*np.pi, size=num_rays)
    directions = np.stack([r_s*np.cos(theta), r_s*np.sin(theta), np.full_like(theta, ray_origin[2]-3)], axis=-1)
    # normalize directions
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    return ray_origin, directions

def ray_rectangle_intersection(ray_origin, ray_dir, vertices, j):
    v0, v1, v2, v3 = vertices
    u = v1 - v0
    v_vec = v2 - v1
    n = np.cross(u, v_vec)
    norm = np.linalg.norm(n)
    if norm < 1e-10:
        return None  
    n = n / norm
    denominator = np.dot(n, ray_dir)
    if abs(denominator) < 1e-10:
        return None  
    t = np.dot(n, v0 - ray_origin) / abs(denominator)
    if t > 0:
        return None  
    p = ray_origin + t * ray_dir

    u_axis = (v1 - v0) / np.linalg.norm(v1 - v0)
    v_axis = np.cross(n, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    def project(point):
        offset = point - v0
        x = np.dot(offset, u_axis)
        y = np.dot(offset, v_axis)
        return np.array([x, y])
    
    vertices_2d = [project(v) for v in vertices]
    p_2d = project(p)

    tri1 = [vertices_2d[0], vertices_2d[1], vertices_2d[2]]
    tri2 = [vertices_2d[0], vertices_2d[2], vertices_2d[3]]

    def is_in_triangle(point, tri):
        a, b, c = tri
        ab = b - a
        bc = c - b
        ca = a - c
        cross0 = (point[0]-a[0])*ab[1] - (point[1]-a[1])*ab[0]
        cross1 = (point[0]-b[0])*bc[1] - (point[1]-b[1])*bc[0]
        cross2 = (point[0]-c[0])*ca[1] - (point[1]-c[1])*ca[0]
        return (cross0 <= 1e-8 and cross1 <= 1e-8 and cross2 <= 1e-8) or \
               (cross0 >= -1e-8 and cross1 >= -1e-8 and cross2 >= -1e-8)
    
    in_tri1 = is_in_triangle(p_2d, tri1)
    in_tri2 = is_in_triangle(p_2d, tri2)

    return p if (in_tri1 or in_tri2) else None

def ray_tracing(r, num_rays, max_points=100):
    # generate all the rays
    ray_origin, ray_dir = ray_generator(r, num_rays)
    # generate all the faces
    faces, rotation = face_generator(*generate_xyz(1, 3))
    # initialize rendered points list
    renderd_points = [] 
    for i in range(num_rays):
        for j in range(6):
            p = ray_rectangle_intersection(ray_origin, ray_dir[i,:], faces[j], j)
            if p is not None:
                renderd_points.append(p)
                break
        if len(renderd_points) >= max_points:
            break

    renderd_points = np.stack(renderd_points)
    if renderd_points.shape[0] < 100:
        # repeat sampling
        n_repeat = int(np.ceil(100 / renderd_points.shape[0]))
        renderd_points = np.tile(renderd_points, (n_repeat, 1))[:100]
    return np.stack(renderd_points), rotation

def process_ray(i, ray_origin, ray_dir, faces):
    for j in range(6):
        p = ray_rectangle_intersection(ray_origin, ray_dir[i, :], faces[j], j)
        if p is not None:
            return p
    return None

def ray_tracing_parallel(r, num_rays):
    ray_origin, ray_dir = ray_generator(r, num_rays)
    faces, rotation = face_generator(*generate_xyz(1, 3))
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda i: process_ray(i, ray_origin, ray_dir, faces), range(num_rays)))
    
    renderd_points = [p for p in results if p is not None]
    return np.stack(renderd_points), rotation

################# the relationship between radius and hit rate ############
def plot_rh_curve():
    import matplotlib.pyplot as plt
    hit_rate_list = []
    repeat_time = 15
    r_list = np.arange(0.5, 2, 0.1)
    for r in r_list:
        rate = 0
        for _ in range(repeat_time):
            rate += len(ray_tracing(r, 512))/512/repeat_time
        hit_rate_list.append(rate)
    plt.plot(r_list, hit_rate_list)
    plt.xlabel("radius")
    plt.ylabel("rate")
    plt.show()


if __name__ == "__main__":
    #  simple test
    # plot_rh_curve()
    rendered_points, _ = ray_tracing(2, 512)
    write_obj(rendered_points, 'datasets/test_render.obj')