# CLImFractal

Realtime fractal explorer, using GPU compute via OpenCL and a Dear ImGUI interface.

Currently frames are passed from OpenCL -> RAM -> OpenGL, but the idea would be to use GLCL interop to directly write to OpenGL buffers from OpenCL, never leaving the GPU.

![alt text](gallery/1.png)

![alt text](gallery/2.png)

The files `mandel.cl`, `mandelstructs.h` and `mandelutils.c` should be kept with the binary, as the OpenCL kernels are compiled at runtime from these.

## Building

Requirements:

- Dear ImGUI
- Stb (used for image loading)
- GLFW (ImGUI backend)

If not on global paths, the `IMGUI_DIR`, `STB_DIR` and `OPENCL_INCLUDE_PATH` should be updated in the `Makefile`, then simply run `make -j <Ncores>`.

## Todo

* Images seem to load flipped horizontally
* Viewport resolution selection
* Save/load sets of compute params
* Arbitrary precision?
