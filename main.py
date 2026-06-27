# ----------------------------------------------------------------

from util import logger
from os import environ, path, mkdir, path
environ["PYSDL2_DLL_PATH"] = path.dirname(__file__)

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning, module="sdl2")

if not path.exists("screenshots"):
    mkdir("screenshots")

# ----------------------------------------------------------------

from time import time
from typing import Tuple

import numpy as np
import sdl2 as sd
import ctypes
from PIL import Image

from numba import njit

sdl2_version = sd.SDL_version()
sd.SDL_GetVersion(sdl2_version)
logger.info(f"SDL2 Version: {sdl2_version.major}.{sdl2_version.minor}.{sdl2_version.patch}")

from platform import processor
logger.info(f"程序运行在处理器 - {processor()} 上。")

# ----------------------------------------------------------------

from loader import test
from shader import Vertex, Projection, Cutting, Map, rasterTriangle, normalize

@njit(cache=True, fastmath=True)
def remove(data: np.ndarray, tick: int):
    grid: np.ndarray = data[tick].copy()
    shape: Tuple[int, int, int] = grid.shape
        
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                if grid[x, y, z] == 1:
                    if (
                        x > 0 and x < shape[0]-1 and
                        y > 0 and y < shape[1]-1 and
                        z > 0 and z < shape[2]-1 and
                        grid[x-1, y, z] == 1 and
                        grid[x+1, y, z] == 1 and
                        grid[x, y-1, z] == 1 and
                        grid[x, y+1, z] == 1 and
                        grid[x, y, z-1] == 1 and
                        grid[x, y, z+1] == 1
                    ):
                        grid[x, y, z] = 0
    return grid

class Cuber:
    def __init__(self):
        result = test()
        self.data: np.ndarray = result["result"]  # shape: (num_frames, 32, 32, 32)
        self.cache: np.ndarray = None
        self.tick: int = 0
        self.update()
        logger.info(f"Cube 加载完毕。{self.data.shape=}")

    def add(self, value: int):
        self.tick += value
        if self.tick >= self.data.shape[0]:
            self.tick = self.data.shape[0] - 1
        if self.tick < 0:
            self.tick = 0
        self.update()

    def update(self):
        self.cache = remove(self.data, self.tick)

cuber: Cuber = Cuber()

# ----------------------------------------------------------------

width, height = 1920, 1080
fov: float = np.pi/3
aspect: float = width / height
near, far = 0.1, 100.0

P: np.ndarray = Projection(fov, aspect, near, far)
up: np.ndarray = np.array([0, 1, 0], dtype=np.float32)
viewport: np.ndarray = np.array([0, 0, width, height], dtype=np.float32)

# ----------------------------------------------------------------

# COLOR: np.ndarray = np.array([255, 133, 51], dtype=np.uint8)

@njit(cache=True, fastmath=True)
def GetCube(cell: np.ndarray):
    indices: Tuple = np.where(cell == 1)
    positions: np.ndarray = np.stack(indices, axis=1)
    N: int = positions.shape[0]
    rotations: np.ndarray = np.zeros((N, 3), dtype=np.float32)
    scales: np.ndarray = np.ones((N, 3), dtype=np.float32)
    colors: np.ndarray = np.zeros((N, 3), dtype=np.uint8)
    max_z = max(np.max(positions[:, 2]) if N > 0 else 1, 1.0)
    
    for i in range(N):
        z = positions[i, 2]
        t = z / max_z
        r = int(255 * (1 - t))
        b = int(255 * t)
        g = int(255 * (1 - abs(2*t - 1)))
        colors[i] = [r, g, b]
    return positions, rotations, scales, colors

# ----------------------------------------------------------------

@njit(cache=True, fastmath=True)
def GetTarget(
    cameraPos: np.ndarray, yaw: float, pitch: float
) -> np.ndarray:

    forward_x: float = np.cos(yaw) * np.cos(pitch)
    forward_y: float = np.sin(pitch)
    forward_z: float = np.sin(yaw) * np.cos(pitch)

    target: np.ndarray = cameraPos + np.array([forward_x, forward_y, forward_z], dtype=np.float32)
    return target

# ----------------------------------------------------------------

cube_vertices: np.ndarray = np.array([
    [-0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5, -0.5,  0.5],
    [-0.5, -0.5,  0.5],
    [-0.5,  0.5, -0.5],
    [ 0.5,  0.5, -0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5]
], dtype=np.float32)

cube_faces: np.ndarray = np.array([
    [3, 2, 6],
    [3, 6, 7],
    [0, 4, 5],
    [0, 5, 1],
    [0, 7, 4],
    [0, 3, 7],
    [1, 5, 6],
    [1, 6, 2],
    [4, 6, 5],
    [4, 7, 6],
    [0, 1, 2],
    [0, 2, 3]
], dtype=np.int32)

# ----------------------------------------------------------------

@njit(cache=True, fastmath=True)
def GetFrame(
    cameraPos: np.ndarray, target: np.ndarray, cell: np.ndarray,
    framebuffer: np.ndarray, zbuffer: np.ndarray, tmpEdge: np.ndarray, 
    outTriangles: np.ndarray, outScreen: np.ndarray
):
    framebuffer[:, :, :] = 0.0
    zbuffer[:, :] = np.inf

    positions, rotations, scales, colors = GetCube(cell)
    MVPbatch: np.ndarray = Vertex(P, cameraPos, target, up, scales, rotations, positions)
    
    tri_vertices: np.ndarray = np.empty((3, 3), dtype=np.float32)
    
    for index in range(positions.shape[0]):
        mvp: np.ndarray = MVPbatch[index]
        cube_color: np.ndarray = colors[index]
        
        for face in cube_faces:
            tri_vertices[0] = cube_vertices[face[0]]
            tri_vertices[1] = cube_vertices[face[1]]
            tri_vertices[2] = cube_vertices[face[2]]
            
            num_triangles: int = Cutting(tri_vertices, mvp, tmpEdge, outTriangles)
            
            for tri_idx in range(num_triangles):
                tri: np.ndarray = outTriangles[tri_idx]
                Map(viewport, tri, outScreen)
                rasterTriangle(
                    outScreen[0], outScreen[1], outScreen[2],
                    cube_color, framebuffer, zbuffer
                )

@njit(cache=True, fastmath=True)
def movingVector(cameraPos: np.ndarray, target: np.ndarray, up: np.ndarray):
    forward: np.ndarray = normalize(target - cameraPos)
    right: np.ndarray = normalize(np.cross(forward, up))
    return forward, right

# ----------------------------------------------------------------

def main():
    window = sd.SDL_CreateWindow(
        b"Window",
        sd.SDL_WINDOWPOS_CENTERED,
        sd.SDL_WINDOWPOS_CENTERED,
        width, height,
        sd.SDL_WINDOW_SHOWN
    )

    if not window:
        logger.info(f"SDL_CreateWindow Error: {sd.SDL_GetError()}")
        sd.SDL_Quit()
        return

    logger.info("窗口创建完毕。")

# ----------------------------------------------------------------

    renderer = sd.SDL_CreateRenderer(
        window,
        -1, 
        sd.SDL_RENDERER_ACCELERATED | sd.SDL_RENDERER_PRESENTVSYNC
    )

    if not renderer:
        logger.info(f"SDL_CreateRenderer Error: {sd.SDL_GetError()}")
        sd.SDL_DestroyWindow(window)
        sd.SDL_Quit()
        return

    logger.info("渲染器创建完毕。")

# ----------------------------------------------------------------

    texture = sd.SDL_CreateTexture(
        renderer,
        sd.SDL_PIXELFORMAT_RGB24,
        sd.SDL_TEXTUREACCESS_STREAMING,
        width, height
    )

    if not texture:
        logger.info(f"SDL_CreateTexture Error: {sd.SDL_GetError()}")
        sd.SDL_DestroyRenderer(renderer)
        sd.SDL_DestroyWindow(window)
        sd.SDL_Quit()
        return

    logger.info("纹理创建完毕。")

# ----------------------------------------------------------------

    running: bool = True
    event = sd.SDL_Event()

    frames_count: int = 0
    last_time: float = time()

    sd.mouse.SDL_SetRelativeMouseMode(sd.SDL_TRUE)

    cameraPos: np.ndarray = np.array([0, 0, 0], dtype=np.float32)
    target: np.ndarray = np.array([1, 1, 1], dtype=np.float32)

    mouseRate: float = 0.003
    strideRate: float = 0.1
    minSpeed: float = 0.01
    maxSpeed: float = 3.0

    direction: np.ndarray = normalize(np.array([1, 1, 1], dtype=np.float32) - cameraPos)
    yaw: float = np.atan2(direction[2], direction[0])
    pitch: float = np.asin(direction[1])

    framebuffer: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
    zbuffer: np.ndarray = np.zeros((height, width), dtype=np.float32)
    tmpEdge = np.empty(4, dtype=np.float32)
    outTriangles = np.empty((12, 3, 4), dtype=np.float32)
    outScreen = np.empty((3, 4), dtype=np.float32)


    logger.info("请确保输入法为英文模式，否则键盘事件可能无效")
    logger.info("鼠标滚轮调整步幅，WASD移动，F2截图")
    logger.info("RF调整时间步，即第四维度，R前进，F后退")
    logger.info("如果直接卡到关闭不了的话就在终端^C关闭")
    logger.info("如果比较卡可以离远点，降低渲染压力")
    logger.info("Q键复位到初始蓝色小方块")
    logger.info("正在进行jit编译，稍等")

    try:
        while running:
            while sd.SDL_PollEvent(ctypes.byref(event)) != 0:
                if event.type == sd.SDL_QUIT:
                    running = False

                elif event.type == sd.SDL_KEYDOWN:
                    if event.key.keysym.sym == sd.SDLK_ESCAPE:
                        running = False
                    elif event.key.keysym.sym == sd.SDLK_r:
                        cuber.add(+1)
                    elif event.key.keysym.sym == sd.SDLK_f:
                        cuber.add(-1)
                    elif event.key.keysym.sym == sd.SDLK_F2:
                        filename: str = f"screenshots/screenshot_{time()}.png"
                        Image.fromarray(framebuffer).save(filename)
                        logger.info(f"截图已保存: {filename}")
                    elif event.key.keysym.sym == sd.SDLK_q:
                        yaw: float = np.atan2(direction[2], direction[0])
                        pitch: float = np.asin(direction[1])
                        cameraPos: np.ndarray = np.array([0, 0, 0], dtype=np.float32)
                        target: np.ndarray = GetTarget(cameraPos, yaw, pitch)

                elif event.type == sd.SDL_MOUSEMOTION:
                    xrel = event.motion.xrel
                    yrel = event.motion.yrel

                    yaw -= xrel * mouseRate
                    pitch -= yrel * mouseRate

                    if pitch > np.pi / 2:
                        pitch = np.pi / 2 - 0.001
                    if pitch < -np.pi / 2:
                        pitch = -np.pi / 2 + 0.001

                    target: np.ndarray = GetTarget(cameraPos, yaw, pitch)

                elif event.type == sd.SDL_MOUSEWHEEL:
                    if event.wheel.y > 0:
                        strideRate *= 1.2
                    elif event.wheel.y < 0:
                        strideRate *= 0.8

                    if strideRate > maxSpeed:
                        strideRate = maxSpeed
                    if strideRate < minSpeed:
                        strideRate = minSpeed

            # ----------------------------------------------------------------

            keys = sd.SDL_GetKeyboardState(None)
            forward, right = movingVector(cameraPos, target, up)
                        
            if keys[sd.SDL_SCANCODE_W]:
                cameraPos = cameraPos + forward * strideRate
                target = target + forward * strideRate
            if keys[sd.SDL_SCANCODE_S]:
                cameraPos = cameraPos - forward * strideRate
                target = target - forward * strideRate
            if keys[sd.SDL_SCANCODE_A]:
                cameraPos = cameraPos + right * strideRate
                target = target + right * strideRate
            if keys[sd.SDL_SCANCODE_D]:
                cameraPos = cameraPos - right * strideRate
                target = target - right * strideRate

            # ----------------------------------------------------------------

            # RGB
            GetFrame(
                cameraPos, target, cuber.cache, framebuffer, zbuffer, tmpEdge, outTriangles, outScreen
            )
            # Lock
            pixels = ctypes.c_void_p()
            pitchs = ctypes.c_int()
            sd.SDL_LockTexture(texture, None, ctypes.byref(pixels), ctypes.byref(pitchs))

            # Copy
            ctypes.memmove(pixels, framebuffer.ctypes.data, width * height * 3)

            # UnLock
            sd.SDL_UnlockTexture(texture)

            sd.SDL_RenderClear(renderer)
            sd.SDL_RenderCopy(renderer, texture, None, None)
            sd.SDL_RenderPresent(renderer)

            # -----------------------------

            frames_count += 1

            if time() - last_time >= 1.0:
                logger.info(f"FPS: {frames_count}")
                frames_count = 0
                last_time = time()

    except KeyboardInterrupt:
        running = False

# ----------------------------------------------------------------

    # 资源释放
    sd.SDL_DestroyTexture(texture)
    sd.SDL_DestroyRenderer(renderer)
    sd.SDL_DestroyWindow(window)
    sd.SDL_Quit()

    logger.info("窗口已关闭，程序退出。")

# ----------------------------------------------------------------

if __name__ == "__main__":
    main()