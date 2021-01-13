import numpy as np
import chumpy as ch
from os.path import join

from psbody.mesh import Mesh

def find_sampler_indexes(scan_path_interested, scan_path_unsure, is_save=True):
    '''
    Param:
        scan_path_interested: path to the .obj model with ocular region removed. the obj file should contain only vertices
        scan_path_unsure: path to the .obj model with unsure region (eyes) removed. the obj file should contain only vertices
    Return:
        An array of indexes of vertices on the ocular region excluding all unsure parts
    '''
    scan_path_interested = "./data/RCHigh_trimmed_ocular_region.obj"
    scan_path_unsure = "./data/RCHigh_trimmed.obj"
    
    scan_remove_interested_region = Mesh(filename=scan_path_interested)
    print("loaded scan from:", scan_path_interested)
    
    scan_remove_unsure_region = Mesh(filename=scan_path_unsure)
    print("loaded scan from:", scan_path_unsure)
    
    num_v_unsure = len(scan_remove_unsure_region.v)
    num_v_interested = len(scan_remove_interested_region.v)
    
    print(type(scan_remove_unsure_region.v))
    #print(scan_remove_unsure_region.v[:100])
    print(num_v_interested , num_v_unsure)
    print(scan_remove_unsure_region.v.size, scan_remove_interested_region.v.size)
    print(scan_remove_unsure_region.v.shape[0], scan_remove_interested_region.v.shape[0])
    index_to_sample = []
    #v_remove_unsure_region = scan_remove_unsure_region.v.tolist()
    #v_remove_interested_region = scan_remove_interested_region.v.tolist()
    scan_remove_unsure_region.v = scan_remove_unsure_region.v*1000000
    scan_remove_interested_region.v = scan_remove_interested_region.v*1000000
    hash_dict = {}
    print("the original face model have %d vertices"%num_v_unsure)
    for i in range(num_v_unsure):
        hash_dict[scan_remove_unsure_region.v[i].tostring()]=0
    print("the mask have %d vertices"%num_v_interested)
    for i in range(num_v_interested):
        hash_dict[scan_remove_interested_region.v[i].tostring()]+=1
    print("writing to dict: done")
    for i in range(num_v_unsure):
        if hash_dict[scan_remove_unsure_region.v[i].tostring()]==0:
            index_to_sample.append(i)
        if hash_dict[scan_remove_unsure_region.v[i].tostring()]>1:
            print(i,hash_dict[scan_remove_unsure_region.v[i].tostring()],scan_remove_unsure_region.v[i])
            break
    #print(index_to_sample)

    print("find %d vertices in interested region!"%len(index_to_sample))
    assert len(index_to_sample)==num_v_unsure-num_v_interested, "please use a *.obj file without color!"
    if is_save:
        np.save("./data/sample_index",np.asarray(index_to_sample))
        print("save sample indexes to ./data/sample_index.npy")
    return np.asarray(index_to_sample)



if __name__ == "__main__":
    scan_path_interested = "./data/RCHigh_trimmed_ocular_region.obj"
    scan_path_unsure = "./data/RCHigh_trimmed.obj"
    find_sampler_indexes(scan_path_interested, scan_path_unsure, is_save=True)