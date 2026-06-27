from os import environ
environ["OMP_NUM_THREADS"] = "8"
environ["MKL_NUM_THREADS"] = "8"
environ["NUMEXPR_NUM_THREADS"] = "8"

import numpy as np
from numba import njit

# NOTE: 惨痛教训！Numba就是一个C编译器垃圾，千万别写任何的Numpy高级函数！

@njit(cache=True, fastmath=True)
def Translation(transBatch: np.ndarray):
    T: np.ndarray = np.zeros((transBatch.shape[0], 4, 4), dtype=np.float32)
    T[:, 0, 0] = 1.
    T[:, 1, 1] = 1.
    T[:, 2, 2] = 1.
    T[:, 3, 3] = 1.
    T[:, 0, 3] = transBatch[:, 0]
    T[:, 1, 3] = transBatch[:, 1]
    T[:, 2, 3] = transBatch[:, 2]
    return T

@njit(cache=True, fastmath=True)
def kTranslation(float3: np.ndarray):
    T: np.ndarray = np.zeros((4, 4), dtype=np.float32)
    T[0, 0] = 1.0
    T[1, 1] = 1.0
    T[2, 2] = 1.0
    T[3, 3] = 1.0
    T[0, 3] = float3[0]
    T[1, 3] = float3[1]
    T[2, 3] = float3[2]
    return T

@njit(cache=True, fastmath=True)
def RotationX(thetaBatch: np.ndarray):
    R: np.ndarray = np.zeros((thetaBatch.shape[0], 4, 4), dtype=np.float32)
    R[:, 0, 0] = 1.
    R[:, 1, 1] = np.cos(thetaBatch)
    R[:, 1, 2] = -np.sin(thetaBatch)
    R[:, 2, 1] = np.sin(thetaBatch)
    R[:, 2, 2] = np.cos(thetaBatch)
    R[:, 3, 3] = 1.
    return R

@njit(cache=True, fastmath=True)
def RotationY(thetaBatch: np.ndarray):
    R: np.ndarray = np.zeros((thetaBatch.shape[0], 4, 4), dtype=np.float32)
    R[:, 0, 0] = np.cos(thetaBatch)
    R[:, 0, 2] = np.sin(thetaBatch)
    R[:, 1, 1] = 1.
    R[:, 2, 0] = -np.sin(thetaBatch)
    R[:, 2, 2] = np.cos(thetaBatch)
    R[:, 3, 3] = 1.
    return R

@njit(cache=True, fastmath=True)
def RotationZ(thetaBatch: np.ndarray):
    R: np.ndarray = np.zeros((thetaBatch.shape[0], 4, 4), dtype=np.float32)
    R[:, 0, 0] = np.cos(thetaBatch)
    R[:, 0, 1] = -np.sin(thetaBatch)
    R[:, 1, 0] = np.sin(thetaBatch)
    R[:, 1, 1] = np.cos(thetaBatch)
    R[:, 2, 2] = 1.
    R[:, 3, 3] = 1.
    return R

@njit(cache=True, fastmath=True)
def Scaling(scaleBatch: np.ndarray):
    S: np.ndarray = np.zeros((scaleBatch.shape[0], 4, 4), dtype=np.float32)
    S[:, 0, 0] = scaleBatch[:, 0]
    S[:, 1, 1] = scaleBatch[:, 1]
    S[:, 2, 2] = scaleBatch[:, 2]
    S[:, 3, 3] = 1.
    return S

@njit(cache=True, fastmath=True)
def normalize(v: np.ndarray):
    return v / np.linalg.norm(v)

@njit(cache=True, fastmath=True)
def cameraZAxis(right: np.ndarray, up: np.ndarray, forward: np.ndarray):
    R: np.ndarray = np.zeros((4, 4), dtype=np.float32)
    R[0, :3] = right
    R[1, :3] = up
    R[2, :3] = forward
    R[3, 3] = 1.0
    return R

@njit(cache=True, fastmath=True)
def Projection(fov: float, aspectRatio: float, near: float, far: float):
    top: float = near * np.tan(fov / 2)
    bottom: float = -top
    right: float = top * aspectRatio
    left: float = -right
    P: np.ndarray = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2 * near / (right - left)
    P[0, 2] = (right + left) / (right - left)
    P[1, 1] = 2 * near / (top - bottom)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = -1.0 # Inter
    return P

@njit(cache=True, fastmath=True)
def Model(scaleBatch: np.ndarray, thetaBatch: np.ndarray, positionBatch: np.ndarray):
    S: np.ndarray = Scaling(scaleBatch)
    Rx: np.ndarray = RotationX(thetaBatch[:, 0])
    Ry: np.ndarray = RotationY(thetaBatch[:, 1])
    Rz: np.ndarray = RotationZ(thetaBatch[:, 2])
    T: np.ndarray = Translation(positionBatch)
    N: int = scaleBatch.shape[0]
    M: np.ndarray = np.zeros((N, 4, 4), dtype=np.float32)
    for index in range(N): # Numba 不支持3D乘法 优化交给JIT
        M[index] = T[index] @ Rz[index] @ Ry[index] @ Rx[index] @ S[index]
    return M

@njit(cache=True, fastmath=True)
def View(cameraPosition: np.ndarray, targetPosition: np.ndarray, upVector: np.ndarray):
    forward: np.ndarray = normalize(targetPosition - cameraPosition)

    T: np.ndarray = kTranslation(-cameraPosition)
    rightVector: np.ndarray = normalize(np.cross(upVector, forward))
    newUpVector: np.ndarray = np.cross(forward, rightVector)
    forwardVector: np.ndarray = -forward

    R: np.ndarray = cameraZAxis(rightVector, newUpVector, forwardVector)
    V: np.ndarray = R @ T
    return V

@njit(cache=True, fastmath=True)
def Vertex(
    P: np.ndarray, # Projection(fov, aspectRatio, near, far)
    cameraPosition: np.ndarray, targetPosition: np.ndarray, upVector: np.ndarray,
    scaleBatch: np.ndarray, thetaBatch: np.ndarray, positionBatch: np.ndarray
    # Batch: (N, 3)
):
    V: np.ndarray = View(cameraPosition, targetPosition, upVector)
    M: np.ndarray = Model(scaleBatch, thetaBatch, positionBatch)
    PV: np.ndarray = P @ V

    N: int = M.shape[0]
    MVP: np.ndarray = np.zeros((N, 4, 4), dtype=np.float32)
    for index in range(N):
        MVP[index] = PV @ M[index]
    return MVP

# @njit(cache=True, fastmath=True)
# def Cutting(posVector: np.ndarray, MVP: np.ndarray):
#     # Vector: (N, 3), MVP: (4, 4) -> cuttedVector: (N, 4)    posVector: (N, 3)
#     N = posVector.shape[0]
#     posHomo = np.empty((N, 4), dtype=np.float32)
#     posHomo[:, :3] = posVector
#     posHomo[:, 3] = 1.0
#     posClip = posHomo @ MVP.T
#     x, y, z, w = posClip[:, 0], posClip[:, 1], posClip[:, 2], posClip[:, 3]
#     visible = (w > 0) & (x >= -w) & (x <= w) & (y >= -w) & (y <= w) & (z >= -w) & (z <= w)
#     return posClip[visible]

# @njit(cache=True, fastmath=True)
# def Applied(posVector: np.ndarray, MVP: np.ndarray):
#     # Vector: (N, 3), MVP: (4, 4) -> posVector: (N, 4)
#     N = posVector.shape[0]
#     posHomo = np.empty((N, 4), dtype=np.float32)
#     posHomo[:, :3] = posVector
#     posHomo[:, 3] = 1.0
#     posVector = posHomo @ MVP.T
#     return posVector

@njit(cache=True, fastmath=True)
def clipEdge(p1: np.ndarray, p2: np.ndarray, plane: int, sign: float, out: np.ndarray) -> np.ndarray:
    axis: int = plane if plane < 3 else -1
    if plane < 3:
        d1: float = p2[axis] - p1[axis]
        d2: float = p2[3] - p1[3]
        denom: float = d1 - sign * d2
        t: float = (sign * p1[3] - p1[axis]) / denom
    else:
        denom: float = p2[3] - p1[3]
        t: float = -p1[3] / denom
    
    t: float = max(0.0, min(1.0, t))
    out[0] = p1[0] + t * (p2[0] - p1[0])
    out[1] = p1[1] + t * (p2[1] - p1[1])
    out[2] = p1[2] + t * (p2[2] - p1[2])
    out[3] = p1[3] + t * (p2[3] - p1[3])

@njit(cache=True, fastmath=True)
def clipPolygon(polygon: np.ndarray, plane: int, sign: float, tmpEdge: np.ndarray) -> np.ndarray:
    N: int = len(polygon)
    output: np.ndarray = np.empty((N + 1, 4), dtype=np.float32)
    count: int = 0
    
    if plane == 0:
        axis: int = 0
    elif plane == 1:
        axis = 1
    elif plane == 2:
        axis = 2
    else:
        axis = -1
    
    prev: np.ndarray = polygon[-1]
    if axis >= 0:
        if sign > 0:
            prev_inside: bool = prev[axis] <= prev[3]
        else:
            prev_inside: bool = prev[axis] >= -prev[3]
    else:
        prev_inside: bool = prev[3] > 0
    
    for index in range(N):
        curr: np.ndarray = polygon[index]
        
        if axis >= 0:
            if sign > 0:
                curr_inside: bool = curr[axis] <= curr[3]
            else:
                curr_inside: bool = curr[axis] >= -curr[3]
        else:
            curr_inside: bool = curr[3] > 0
        
        if prev_inside and curr_inside:
            output[count] = curr
            count += 1
        elif prev_inside and not curr_inside:
            clipEdge(prev, curr, plane, sign, tmpEdge)
            output[count] = tmpEdge
            count += 1
        elif not prev_inside and curr_inside:
            clipEdge(prev, curr, plane, sign, tmpEdge)
            output[count] = tmpEdge
            count += 1
            output[count] = curr
            count += 1
        
        prev: np.ndarray = curr
        prev_inside: bool = curr_inside
    
    return output[:count]

@njit(cache=True, fastmath=True)
def clipTriangle(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, 
    tmpEdge: np.ndarray, outTriangles: np.ndarray
) -> int:
    buffers: np.ndarray = np.empty((2, 8, 4), dtype=np.float32)
    
    buffers[0, 0] = v0
    buffers[0, 1] = v1
    buffers[0, 2] = v2
    poly_len: int = 3
    src_idx: int = 0
    dst_idx: int = 1
    
    planes: np.ndarray = np.array([
        (0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0),
        (2, -1.0), (2, 1.0), (3, 0.0)
    ], dtype=np.float32)
    
    for idx in range(7):
        plane: int = int(planes[idx, 0])
        sign: float = planes[idx, 1]
        
        src: np.ndarray = buffers[src_idx, :poly_len, :]
        dst: np.ndarray = buffers[dst_idx]
        
        new_poly: np.ndarray = clipPolygon(src, plane, sign, tmpEdge)
        poly_len = len(new_poly)
        
        if poly_len < 3:
            return 0
        
        dst[:poly_len] = new_poly
        src_idx, dst_idx = dst_idx, src_idx
    
    num_triangles: int = poly_len - 2
    src: np.ndarray = buffers[src_idx, :poly_len, :]
    
    for i in range(num_triangles):
        outTriangles[i, 0] = src[0]
        outTriangles[i, 1] = src[i + 1]
        outTriangles[i, 2] = src[i + 2]
    
    return num_triangles

@njit(cache=True, fastmath=True)
def Cutting(triangle: np.ndarray, MVP: np.ndarray, tmpEdge: np.ndarray, outTriangles: np.ndarray) -> int:
    posHomo: np.ndarray = np.empty((3, 4), dtype=np.float32)
    posHomo[:, :3] = triangle
    posHomo[:, 3] = 1.0
    posClip: np.ndarray = posHomo @ MVP.T
    return clipTriangle(posClip[0], posClip[1], posClip[2], tmpEdge, outTriangles)

# @njit(cache=True, fastmath=True)
# def Cutting(posVector: np.ndarray):
#     # posVector: (N, 4)
#     x, y, z, w = posVector[:, 0], posVector[:, 1], posVector[:, 2], posVector[:, 3]
#     visible = (w > 0) & (x >= -w) & (x <= w) & (y >= -w) & (y <= w) & (z >= -w) & (z <= w)
#     return posVector[visible]

@njit(cache=True, fastmath=True)
def Map(
    viewPort: np.ndarray,
    cuttedVector: np.ndarray,
    outScreen: np.ndarray
):
    vx: float = viewPort[0]
    vy: float = viewPort[1]
    vw: float = viewPort[2]
    vh: float = viewPort[3]
    
    for index in range(3):
        w: float = cuttedVector[index, 3]
        inv_w: float = 1.0 / w
        
        x_ndc: float = cuttedVector[index, 0] * inv_w
        y_ndc: float = cuttedVector[index, 1] * inv_w
        z_ndc: float = cuttedVector[index, 2] * inv_w
        
        outScreen[index, 0] = (x_ndc + 1.0) * 0.5 * vw + vx
        outScreen[index, 1] = (1.0 - y_ndc) * 0.5 * vh + vy
        outScreen[index, 2] = (z_ndc + 1.0) * 0.5
        outScreen[index, 3] = w

# @njit(cache=True, fastmath=True)
# def meshgrid(xs: np.ndarray, ys: np.ndarray):
#     nx, ny = len(xs), len(ys)
#     xx: np.ndarray = np.empty((ny, nx), dtype=xs.dtype)
#     yy: np.ndarray = np.empty((ny, nx), dtype=ys.dtype)
#     for i in range(ny):
#         for j in range(nx):
#             xx[i, j] = xs[j]
#             yy[i, j] = ys[i]
#     return xx, yy


@njit(cache=True, fastmath=True)
def rasterTriangle(
    vectorA: np.ndarray, vectorB: np.ndarray, vectorC: np.ndarray,
    color: np.ndarray, framebuffer: np.ndarray, zbuffer: np.ndarray
): # Vector: (4,) color: (3,) framebuffer: (H, W, 3), zbuffer: (H, W)
    height, width = framebuffer.shape[:2]

    x0, y0, z0, w0 = vectorA
    x1, y1, z1, w1 = vectorB
    x2, y2, z2, w2 = vectorC

    edge1_x, edge1_y = x1 - x0, y1 - y0
    edge2_x, edge2_y = x2 - x0, y2 - y0
    area = edge1_x * edge2_y - edge1_y * edge2_x

    if area <= 0:
        return

    xmin: int = max(0, int(min(x0, x1, x2)))
    xmax: int = min(width - 1, int(max(x0, x1, x2)))
    ymin: int = max(0, int(min(y0, y1, y2)))
    ymax: int = min(height - 1, int(max(y0, y1, y2)))

    if xmin > xmax or ymin > ymax:
        return

    A01: float = y0 - y1
    B01: float = x1 - x0
    A12: float = y1 - y2
    B12: float = x2 - x1
    A20: float = y2 - y0
    B20: float = x0 - x2

    inv_area: float = 1.0 / area
    z0_inv: float = z0 / w0
    z1_inv: float = z1 / w1
    z2_inv: float = z2 / w2
    inv_w0: float = 1.0 / w0
    inv_w1: float = 1.0 / w1
    inv_w2: float = 1.0 / w2


    for y in range(ymin, ymax + 1):
        wy01: float = (y - y0) * B01
        wy12: float = (y - y1) * B12
        wy20: float = (y - y2) * B20
        w01: float = (xmin - x0) * A01 + wy01
        w12: float = (xmin - x1) * A12 + wy12
        w20: float = (xmin - x2) * A20 + wy20

        for x in range(xmin, xmax + 1):
            if w01 >= 0 and w12 >= 0 and w20 >= 0:
                alpha: float = w12 * inv_area
                beta: float = w20 * inv_area
                gamma: float = w01 * inv_area
                
                inv_w: float = alpha * inv_w0 + beta * inv_w1 + gamma * inv_w2
                z_num: float = alpha * z0_inv + beta * z1_inv + gamma * z2_inv
                z: float = z_num / inv_w
                
                if z < zbuffer[y, x]:
                    zbuffer[y, x] = z
                    framebuffer[y, x] = color
            
            w01 += A01
            w12 += A12
            w20 += A20