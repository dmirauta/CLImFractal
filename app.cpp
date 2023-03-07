#include <filesystem>
#include <cmath>

#include "app.h"
#include "imgui.h"

std::string App::title = "CLImFractal";

App::App()
{
    N = 600;
    M = 800;
    pix = new unsigned char[N*M*3];
    start = high_resolution_clock::now();

    vector<string> source_files{"mandel.cl"}; // this time importing its own deps
    vector<string> kernel_names{"escape_iter", "min_prox", "orbit_trap", "map_img", "apply_log_int", "apply_log_fpn"};
    string ocl_include_path = filesystem::current_path(); // include path required for cl file importing own deps
    ecl.load_kernels(source_files, kernel_names, "-I "+ocl_include_path);
    ecl.no_block = true;

    prox1 = new SynchronisedArray<double>(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    prox2 = new SynchronisedArray<double>(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    prox3 = new SynchronisedArray<double>(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    param = new SynchronisedArray<MPParam>(ecl.context);

    compute_begin(); // start computing first frame

}

App::~App()
{
    delete [] pix;
    delete prox1;
    delete prox2;
    delete prox3;
    delete param;
}

void App::compute_begin()
{
    (*param)[0].mandel = mandel ? 1 : 0;
    (*param)[0].c = {(FPN) cre, (FPN) cim};
    (*param)[0].view_rect = {viewport_center.re-viewport_deltas.re,
                             viewport_center.re+viewport_deltas.re,
                             viewport_center.im-viewport_deltas.im,
                             viewport_center.im+viewport_deltas.im};
    (*param)[0].MAXITER = MAXITER;

    // Run kernel(s)
    (*param)[0].PROXTYPE = 3;
    ecl.apply_kernel("min_prox", *prox1, *param);
    
    (*param)[0].PROXTYPE = 5;
    ecl.apply_kernel("min_prox", *prox2, *param);
    
    (*param)[0].PROXTYPE = 6;
    ecl.apply_kernel("min_prox", *prox3, *param);

    // ecl.queue.finish();
    // ecl.apply_kernel("apply_log_fpn", *prox1); // this needs read write access...
    // ecl.apply_kernel("apply_log_fpn", *prox2);
    // ecl.apply_kernel("apply_log_fpn", *prox3);

    double s;
    for (int i=0; i<N*M; i++)
    {
        s = prox1->cpu_buff[i] + prox2->cpu_buff[i] + prox3->cpu_buff[i];
        pix[3*i]   = 255*(prox1->cpu_buff[i]/s);
        pix[3*i+1] = 255*(prox2->cpu_buff[i]/s);
        pix[3*i+2] = 255*(prox3->cpu_buff[i]/s);
    }
}

void App::compute_join()
{
    ecl.queue.finish();
}

void App::render()
{
    // ImGui::ShowDemoWindow();

    compute_join(); // should probably be outside of rendering code, but would come immediately before and after anyway, so keeping inside own scripts (main is from imgui examples)

    viewport.set(pix, M, N);

    ImGui::Begin("Viewport");

        ImGuiIO& io = ImGui::GetIO();

        // Pan
        float deltaX = 0, deltaY = 0;
        if(ImGui::IsWindowHovered() && ImGui::IsMousePosValid() && ImGui::IsMouseDown(1))
        {
            deltaX = 2*io.MouseDelta.x/(float) N;
            deltaY = 2*io.MouseDelta.y/(float) M;
            viewport_center.re -= viewport_deltas.re * deltaX;
            viewport_center.im -= viewport_deltas.im * deltaY;
            // ImGui::Text("Mouse delta (screen fraction): (%f, %f)", deltaX, deltaY);
        } 

        // Zoom
        if (io.MouseWheel > 0)
        {
            viewport_deltas.re /= 1.07;
            viewport_deltas.im /= 1.07;
        } else if (io.MouseWheel < 0) {
            viewport_deltas.re *= 1.07;
            viewport_deltas.im *= 1.07;
        }

        ImGui::Text("FPS %f (currently copying frames from OpenCL -> RAM -> OpenGL)", ImGui::GetIO().Framerate);
        ImGui::Text("Center: (%lg) + (%lg)i", viewport_center.re, viewport_center.im);
        ImGui::Text("Box dims: (%lg) x (%lg)", 2*viewport_deltas.re, 2*viewport_deltas.im);
        ImGui::Image((void*)(intptr_t)viewport.tex_id, ImVec2(M, N));

    ImGui::End();

    ImGui::Begin("Controlls");

        ImGui::Text("Inputs:");
        ImGui::Text("Pan: Right click and drag in viewport");
        ImGui::Text("Zoom: Mouse wheel");

        ImGui::Text("\nParams:");
        ImGui::Checkbox("Compute mandelbrot (else Julia)", &mandel);

        if (!mandel)
        {
            ImGui::Text("Julia constant:");
            ImGui::SliderFloat("CRE", &cre, -2, 2);
            ImGui::SliderFloat("CIM", &cim, -2, 2);
        }

        ImGui::SliderFloat("log10 MAXITER", &MAXITERpow, 0, 4);
        MAXITER = pow(10, MAXITERpow);
        ImGui::Text("MAXITER: %d", MAXITER);
    ImGui::End();

    compute_begin(); // start computing next frame

}
