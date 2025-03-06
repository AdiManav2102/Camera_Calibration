import os, cv2, argparse
import os.path as osp
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import scipy.optimize
import matplotlib.pyplot as plt

def load_images(data_dir):
    
    images = []
    try:
        for img_name in natsorted(os.listdir(data_dir)):
            img = cv2.imread(osp.join(data_dir, img_name))
            if img is not None:
                img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append([img_name, img, img_grayscale])
    except Exception as e:
        print(e)

    return images

def world_coords(len, width, sq_size):
    
    world_coord_x, world_coord_y = np.meshgrid(range(len), range(width))
    world_coord = np.hstack((world_coord_x.reshape(-1, 1), world_coord_y.reshape(-1, 1)))
    world_coord = world_coord * sq_size
    
    return world_coord # Shape: (len * width, 2)

def checkerboard_coords(images, p_size, data_dir):
    
    corners = []
    results_dir = osp.join("Results", "Detected_Corners")
    if not osp.exists(results_dir): os.makedirs(results_dir)

    pbar = tqdm(enumerate(images), total=len(images), desc="Detecting Corners")
    for ind, image in pbar:
        image_copy = image[1].copy()
        ret, corner = cv2.findChessboardCorners(image[2], p_size, None)
        if ret:
            corner = np.squeeze(corner)
            # Save the image with the corners drawn
            cv2.drawChessboardCorners(image_copy, p_size, corner, ret)
            cv2.imwrite(osp.join(results_dir, image[0]), image_copy)
        else:
            continue

        corners.append([image[0], corner])
    
    # Format: [[img_name, corners], [img_name, corners], ...]
    # Corners: shape ((length * width), 1, 2) 
    return corners 

def find_homography(world_coords, img_coords):
    
    if world_coords.shape != img_coords.shape:
        raise ValueError("The shape of world coordinates and image coordinates must match.")
    
    A = []
    for i in range(world_coords.shape[0]):
        X1, Y1 = world_coords[i, :]
        x2, y2 = img_coords[i, :]
        
        A.append([-X1, -Y1, -1, 0, 0, 0, x2*X1, x2*Y1, x2])
        A.append([0, 0, 0, -X1, -Y1, -1, y2*X1, y2*Y1, y2])
    
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    H = np.reshape(V[-1], (3, 3))
    H = (1 / H.item(8)) * H 
    return H


def find_homography_matrices(img_corners, world_coords):
    
    homography_matrices = []
    for corner_info in img_corners:
        img_name = corner_info[0]
        corners = corner_info[1]
        H = find_homography(world_coords, corners)

        homography_matrices.append([img_name, H])
    
    # Format: [[img_name, homography_matrix], [img_name, homography_matrix], ...]
    return homography_matrices


def compute_b_vector(homography_matrices):
    
    V = [] # Shape is (2 * len(homography_matrices), 6)

    def compute_vij(H, i, j):
        
        vij = np.array([
            H[i][0] * H[j][0], 
            H[i][0] * H[j][1] + H[i][1] * H[j][0], 
            H[i][1] * H[j][1], 
            H[i][2] * H[j][0] + H[i][0] * H[j][2], 
            H[i][2] * H[j][1] + H[i][1] * H[j][2], 
            H[i][2] * H[j][2]
        ])
        return vij.T
    
    for homography_info in homography_matrices:
        H = homography_info[1].T
        v11 = compute_vij(H, 0, 0)
        v12 = compute_vij(H, 0, 1)
        v22 = compute_vij(H, 1, 1)
        V.append(v12)
        V.append(v11 - v22)

    V = np.array(V)

    if V.size == 0:
        raise ValueError("V matrix is empty. Check homographies and corner detection.")

    print(f"V matrix shape: {V.shape}")  # Debugging

    # Compute the b vector
    U, S, Vt = np.linalg.svd(V, full_matrices=True)
    b_vector = Vt[-1, :]

    return b_vector

    
def extract_intrinsic_matrix(b):
    
    v0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] ** 2) # v0 Prindipal point in y direction
    arb_scale = b[5] - (b[3] ** 2 + v0 * (b[1] * b[3] - b[0] * b[4])) / b[0] # scale_factor
    u_scale_factor = np.sqrt(arb_scale / b[0]) # alpha
    v_scale_factor = np.sqrt((arb_scale * b[0]) / (b[0] * b[2] - b[1] ** 2)) # beta
    skew = -1 * b[1] * (u_scale_factor ** 2) * v_scale_factor / arb_scale # gamma
    u0 = (skew * v0 / v_scale_factor) - (b[3] * (u_scale_factor ** 2)) / arb_scale # u0 Principal point in x direction

    # Intrinsic matrix
    K = np.array([
        [u_scale_factor, skew, u0],
        [0, v_scale_factor, v0],
        [0, 0, 1]
    ])

    return K

def extract_extrinsic_matrix(K, homography_matrices):
    
    R = []
    for homography_info in homography_matrices:
        H = homography_info[1]
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        scale_factor = 1 / np.linalg.norm(np.linalg.inv(K) @ h1, ord=2)
        r1 = (scale_factor * np.dot(np.linalg.inv(K), h1)).reshape(-1, 1)
        r2 = (scale_factor * np.dot(np.linalg.inv(K), h2)).reshape(-1, 1)
        t = (scale_factor * np.dot(np.linalg.inv(K), h3)).reshape(-1, 1)

        # Transformation matrix
        coord_transform_matrix = np.hstack((r1, r2, t))
        R.append([homography_info[0], coord_transform_matrix])
    return R


def get_optimization_parameters(A, distortion_vec):

    return np.array([A[0][0], A[0][1], A[1][1], A[0][2], A[1][2], distortion_vec.flatten()[0], distortion_vec.flatten()[1]])


def objective_function(x0, R, world_coords, img_corners):

    total_error, _, _ = compute_projection_error(x0, R, world_coords, img_corners)
    return total_error

def compute_projection_error(x0, R, world_coords, img_corners):
    
    u0 = x0[3]
    v0 = x0[4]
    k1 = x0[5]
    k2 = x0[6]

    # Compute the intrinsic matrix
    A = np.array([
        [x0[0], x0[1], u0],
        [0, x0[2], v0],
        [0, 0, 1]
    ])

    total_error = 0
    all_reprojected_corners = []
    individual_img_errors = []

    for i, corner_info in enumerate(img_corners):
        image_name = corner_info[0]
        extrinsic_matrix = R[i][1]
        image_error= 0
        reprojected_corners = []

        total_transformation_matrix = A @ extrinsic_matrix

        for j, corner in enumerate(corner_info[1]):

            # Fetch the ground truth image coordinates
            image_gt_corner = corner.reshape(-1, 1)
            image_gt_corner = np.vstack((image_gt_corner, 1))

            # Convert the world coordinates to homogeneous coordinates
            world_coord = np.hstack((world_coords[j], 1)).reshape(-1, 1)

            # Camera coordinate
            camera_coord = extrinsic_matrix @ world_coord
            x = camera_coord[0] / camera_coord[2]
            y = camera_coord[1] / camera_coord[2]

            # Pixel coordinate
            pixel_coord = total_transformation_matrix @ world_coord
            u = pixel_coord[0] / pixel_coord[2]
            v = pixel_coord[1] / pixel_coord[2]

            u_hat = u + (u - u0) * (k1 * (x**2 + y**2) + k2 * (x**2 + y**2)**2)
            v_hat = v + (v - v0) * (k1 * (x**2 + y**2) + k2 * (x**2 + y**2)**2)

            image_projected_corner = np.array([u_hat, v_hat]).reshape(-1, 1)
            image_projected_corner = np.vstack((image_projected_corner, 1))

            reprojected_corners.append(image_projected_corner)

            image_error += np.linalg.norm(image_projected_corner - image_gt_corner, ord=2)
        
        image_error = image_error / len(corner_info[1])
        individual_img_errors.append([image_name, image_error])

        total_error += image_error / len(img_corners)
        all_reprojected_corners.append(reprojected_corners)

    # return np.array(total_error)
    return np.array([total_error, 0, 0, 0, 0, 0, 0]), all_reprojected_corners, individual_img_errors

def print_projection_error_table(before_errors, after_errors):
    """Prints a well-formatted table for per-image projection error."""
    
    print("\n" + "="*50)
    print(f"{'Image':<10}{'Before':<10}{'After'}")
    print("="*50)

    for i, (before, after) in enumerate(zip(before_errors, after_errors), start=1):
        print(f"{i:<10}{before[1]:<10.4f}{after[1]:.4f}")

    print("="*50)

def undistort_image(image, A, distortion):
    
    dist = distortion
    h, w = image.shape[:2]
    dst = cv2.undistort(image, A, dist)
    return dst

def log_error(error):
    
    with open("error_logs.txt", "w") as f:
        for err in error:
            f.write(f"{err[0]}: Before Error: {err[1]}, After Error: {err[2]}\n") 

# def visualize_error_distribution(errors_before, errors_after):
#     """Visualizes and saves the reprojection error distribution before and after optimization."""
#     output_dir = "Results"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, len(errors_before)+1), errors_before, 'b-', label='Before Optimization')
#     plt.plot(range(1, len(errors_after)+1), errors_after, 'r-', label='After Optimization')
#     plt.xlabel('Image Number')
#     plt.ylabel('Reprojection Error (pixels)')
#     plt.title('Reprojection Error Distribution')
#     plt.legend()
#     plt.grid(True)
    
#     save_path = os.path.join(output_dir, 'error_distribution.png')
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Reprojection error distribution plot saved at: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="Data", help="Path to calibration images")
    args = parser.parse_args()
    data = args.data

    PATTERN_SIZE = (9, 6)
    calibration_images = load_images(osp.join("Data", "Calibration_Imgs"))
    world_coordinates = world_coords(PATTERN_SIZE[0], PATTERN_SIZE[1], 21.5) # 9x6 checkerboard with 21.5mm square size
    img_corners = checkerboard_coords(calibration_images, PATTERN_SIZE, data)
    homography_matrices = find_homography_matrices(img_corners, world_coordinates)
    vec_b = compute_b_vector(homography_matrices)

    A = extract_intrinsic_matrix(vec_b)
    print(f"Initial Intrinsic Matrix:\n{A}\n")

    R = extract_extrinsic_matrix(A, homography_matrices) # Format: [[img_name, extrinsic_matrix], [img_name, extrinsic_matrix], ...]
    distortion = np.array([0, 0]).reshape(-1, 1)

    x0 = get_optimization_parameters(A, distortion)
    print(f"Initial Optimization Parameters: {x0}\n")

    # x = scipy.optimize.minimize(fun=objective_function, x0=x0, method="Powell", args=(R, world_coordinates, img_corners))
    x = scipy.optimize.least_squares(fun=objective_function, x0=x0, method="lm", args=(R, world_coordinates, img_corners), max_nfev=3000, verbose=2)
    _u_scale_factor, _arb_scale, _v_scale_factor, _u0, _v0, _k1, _k2 = x.x

    A_optimized = np.array([
        [_u_scale_factor, _arb_scale, _u0],
        [0, _v_scale_factor, _v0],
        [0, 0, 1]
    ])

    distortion_optimized = np.array([_k1, _k2, 0, 0, 0])
    print(f"Optimized Intrinsic Matrix:\n{A_optimized}")

    before_reprojection_error, _, before_individual_image_error = compute_projection_error(x0, R, world_coordinates, img_corners)
    after_reprojection_error, reprojected_points, after_individual_image_error = compute_projection_error(x.x, R, world_coordinates, img_corners)

    mean_reprojection_error = np.mean([error[1] for error in after_individual_image_error])
    print(f"\nMean Re-projection Error after Optimization: {mean_reprojection_error:.4f}")
    print_projection_error_table(before_individual_image_error, after_individual_image_error)

    # Extract the per-image errors for visualization
    # errors_before = [error[1] for error in before_individual_image_error]
    # errors_after = [error[1] for error in after_individual_image_error]

    # # Call visualization function
    # visualize_error_distribution(errors_before, errors_after)

    results_dir = osp.join("Results", "Reprojected_Corners")
    if not osp.exists(results_dir): os.makedirs(results_dir)

    for i, img_info in tqdm(enumerate(calibration_images)):
        img_name = img_info[0]
        img_copy = img_info[1].copy()
        reprojected_img = undistort_image(img_copy, A_optimized, distortion_optimized)
        
        for corner in reprojected_points[i]:
            cv2.circle(reprojected_img, (int(corner[0].item()), int(corner[1].item())), 11, (0, 0, 255), 4)
            cv2.circle(reprojected_img, (int(corner[0].item()), int(corner[1].item())), 3, (0, 255, 0), -1)

        cv2.imwrite(osp.join(results_dir, img_name), reprojected_img)
    
    # Log the reprojection error for each image
    error_logs = []
    for i, error in enumerate(before_individual_image_error):
        error_logs.append([error[0], error[1], after_individual_image_error[i][1]])
    
    log_error(error_logs)

if __name__ == "__main__":
    main()
