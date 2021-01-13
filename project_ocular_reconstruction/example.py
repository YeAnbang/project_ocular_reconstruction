from generate_landmarks import generate_landmarks
from find_sampler_indexes import find_sampler_indexes
from fit_scan_region import *
import numpy as np
from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model

color_model_path = "./data/RCHigh_trimmed_color.ply"
scan_path_interested = "./data/RCHigh_trimmed_ocular_region.obj"
scan_path_unsure = "./data/RCHigh_trimmed.obj"
scan_lmk_path = './data/landmarks_3d_51_points.npy'
sample_index_path = "./data/sample_index.npy"
scan_unit = 'm' 
model_path = './models/female_model.pkl'

def fitting_scan_regional(scan_path,sample_index_path,scan_lmk_path,scan_unit,model_path, sample_index=None, lmk_3d = None, max_points_to_sample=30000):
    '''
    Param
        # input scan
        scan_path = './data/RCHigh_trimmed.obj'
        #sample_index
        sample_index_path = "./data/sample_index.npy"
        # landmarks of the scan
        scan_lmk_path = './data/landmarks_3d_51_points.npy'
        # measurement unit of landmarks ['m', 'cm', 'mm', 'NA'] 
        # When using option 'NA', the scale of the scan will be estimated by rigidly aligning model and scan landmarks
        scan_unit = 'm' 
        # model
        model_path = './models/female_model.pkl' # change to 'female_model.pkl' or 'male_model.pkl', if gender is known
    
    Output:
        writing the generated model to 
    '''
    

    scan = Mesh(filename=scan_path)
    print("loaded scan from:", scan_path)
    if scan_lmk_path!=None:
        lmk_3d = np.load(scan_lmk_path)
    if scan_unit == 'cm': 
        lmk_3d = lmk_3d*100
    print(lmk_3d)
    print(lmk_3d.shape)
    print("loaded scan landmark from:", scan_lmk_path)

    model = load_model(model_path)       # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    print("loaded model from:", model_path)

    # landmark embedding
    lmk_emb_path = './models/flame_static_embedding.pkl' 
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    print("loaded lmk embedding")

    # scale scans and scan landmarks to be in the same local coordinate systems as the FLAME model
    if scan_unit.lower() == 'na':
        print('No scale specifiec - compute approximate scale based on the landmarks')
        scale_factor = compute_approx_scale(lmk_3d, model, lmk_face_idx, lmk_b_coords)
        print('Scale factor: %f' % scale_factor)
    else:
        scale_factor = get_unit_factor('m') / get_unit_factor(scan_unit)
        print('Scale factor: %f' % scale_factor)        

    scan.v[:] *= scale_factor
    lmk_3d[:] *= scale_factor

    if sample_index_path!=None:
        sample_index = np.load(sample_index_path)

    # to make the program faster, we sample max_points_to_sample points from that region randomly
    sample_index = np.array(random.sample(sample_index.tolist(), int(min(max_points_to_sample, sample_index.shape[0]))))

    print("load interested point to predict shape from: %s"%sample_index_path)
    # output
    output_dir = './output'
    safe_mkdir(output_dir)

    # weights
    weights = {}
    # scan vertex to model surface distance term
    weights['s2m']   = 4.0   #2.0
    # landmark term
    weights['lmk']   = 1e-2
    # shape regularizer (weight higher to regularize face shape more towards the mean)
    weights['shape'] = 5e-5	#1e-4
    # expression regularizer (weight higher to regularize facial expression more towards the mean)
    weights['expr']  = 5e-5	#1e-4
    # regularization of head rotation around the neck and jaw opening (weight higher for more regularization)
    weights['pose']  = 1e-3
    # Parameter of the Geman-McClure robustifier (higher weight for a larger bassin of attraction which makes it less robust to outliers)
    gmo_sigma = 1e-4

    # optimization options
    import scipy.sparse as sp
    opt_options = {}
    opt_options['disp']    = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3']     = 1e-4
    opt_options['maxiter'] = 2000
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver

    # run fitting
    mesh_v, mesh_f, parms = fit_scan(   scan=scan,                                             # input scan
                                        lmk_3d=lmk_3d,                                         # input landmark 3d
                                        model=model,                                           # model
                                        lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords,  # landmark embedding
                                        weights=weights,                                       # weights for the objectives
                                        gmo_sigma=gmo_sigma,                                   # parameter of the regularizer
                                        sample_index = sample_index,                                     
                                        shape_num=300, expr_num=100, opt_options=opt_options ) # options

    # write result
    output_path = join( output_dir, 'fit_region_scan_result.obj' )
    write_simple_obj( mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False )
    print('output mesh saved to: ', output_path) 

    # output scaled scan for reference (output scan fit and the scan should be spatially aligned)
    output_path = join( output_dir, 'scan_scaled.obj' )    
    write_simple_obj( mesh_v=scan.v, mesh_f=scan.f, filepath=output_path, verbose=False )

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    landmarks = generate_landmarks(color_model_path)
    sample_index = find_sampler_indexes(scan_path_interested, scan_path_unsure, is_save=True)
    fitting_scan_regional(scan_path_unsure,None,None,scan_unit,model_path,sample_index=sample_index,lmk_3d=landmarks)
