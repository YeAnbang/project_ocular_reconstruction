from face_point_cloud_landmark import PointCloudLandmark
import numpy as np

def generate_landmarks(color_model_path):
    '''
	generate 3D landmarks on a .pkl model with color, the figure in the scan should face 
	the +z direciton. its head should align to y axis
	input: path to the pkl model
    '''
    scan = PointCloudLandmark(color_model_path)
    index, landmarks_3d_numpy = scan.get_landmark_byPLY()

    print(landmarks_3d_numpy)
    np.save("./data/landmarks_3d_51_points",landmarks_3d_numpy[17:])
    return landmarks_3d_numpy[17:]


if __name__ == "__main__":
    generate_landmarks()
    print("landmarks save to ./data/landmarks_3d_51_points")