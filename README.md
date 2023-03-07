# CLImFractal

OpenCL fractals with Dear ImGUI. 

Currently frames are passed from OpenCL -> RAM -> OpenGL, but the idea would be to use GLCL interop to directly write to OpenGL buffers from OpenCL, never leaving the GPU.

