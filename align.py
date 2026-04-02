import json
import math
import numpy as np
import os

# WGS84 to ECEF config
a = 6378137.0
b = 6356752.314245
e2 = 1 - (b**2 / a**2)

def geodetic_to_ecef(lat, lon, alt):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)
    return np.array([x, y, z])

def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    ecef0 = geodetic_to_ecef(lat0, lon0, alt0)
    dx = x - ecef0[0]
    dy = y - ecef0[1]
    dz = z - ecef0[2]

    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    u = clat * clon * dx + clat * slon * dy + slat * dz

    return np.array([e, n, u])

def umeyama(src, dst):
    num = src.shape[0]
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    
    src_c = src - src_mean
    dst_c = dst - dst_mean

    src_var = np.mean(np.sum(src_c**2, axis=1))
    
    H = (src_c.T @ dst_c) / num
    U, D, V = np.linalg.svd(H)
    
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[2, 2] = -1

    R_mat = V.T @ S @ U.T
    
    trace_D = np.sum(D * np.diag(S))
    c = trace_D / src_var

    t = dst_mean - c * (R_mat @ src_mean)
    
    return c, R_mat, t

def main():
    # Load GPS
    with open(r'c:\cg\Project\python\slam3r\SLAM3R\tmp\gps_path.json', 'r') as f:
        gps_data = json.load(f)
        
    num_pts = len(gps_data)
    lat0 = gps_data[0]['lat']
    lon0 = gps_data[0]['lon']
    alt0 = gps_data[0]['alt']
    
    target_enu = []
    for pt in gps_data:
        ecef = geodetic_to_ecef(pt['lat'], pt['lon'], pt['alt'])
        enu = ecef_to_enu(ecef[0], ecef[1], ecef[2], lat0, lon0, alt0)
        target_enu.append(enu)
    
    target_enu = np.array(target_enu)
    
    # Load SLAM poses
    src_local = []
    with open(r'c:\cg\Project\python\slam3r\SLAM3R\tmp\tmp713rmb5nslam3r_gradio_demo\scene_traj.txt', 'r') as f:
        lines = f.read().strip().split('\n')
        for line in lines[:num_pts]:
            if not line.strip(): continue
            vals = list(map(float, line.strip().split()))
            src_local.append([vals[3], vals[7], vals[11]])
            
    src_local = np.array(src_local)
    
    # Apply user rotation from Phase 1: X=-90, Z=90
    rx = math.radians(-90)
    rz = math.radians(90)
    
    Mx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])
    Mz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # UserRot = mz * my * mx (my is identity)
    UserRot = Mz @ Mx
    corrected_src = (UserRot @ src_local.T).T
    
    # Umeyama registration
    c, R_mat, t = umeyama(corrected_src, target_enu)
    
    transform_enu = np.eye(4)
    transform_enu[:3, :3] = c * R_mat
    transform_enu[:3, 3] = t
    
    out_data = {
        "anchor": {"lon": lon0, "lat": lat0, "alt": alt0},
        "transform_enu": transform_enu.T.flatten().tolist()  # Column-major for Cesium
    }
    
    out_path = r'c:\cg\Project\python\slam3r\SLAM3R\tmp\phase3_transform.json'
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
        
    print(f"Alignment calculated.")
    print(f"Scale parameter (1 unit in SLAM = X meters): {c:.4f}")
    
    transformed_src = (c * (R_mat @ corrected_src.T)).T + t
    err = np.linalg.norm(transformed_src - target_enu, axis=1)
    print(f"Mean Registration Error: {np.mean(err):.2f} meters")

if __name__ == '__main__':
    main()
