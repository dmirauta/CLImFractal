#include <filesystem>

#include "app.h"
#include "imgui.h"

std::string App::title = "Testing";

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
    (*param)[0].mandel = 1;
    (*param)[0].view_rect = {re0, re1, im0, im1};
    (*param)[0].MAXITER = MAXITER;

    // Run kernel(s)
    (*param)[0].PROXTYPE = 1;
    ecl.apply_kernel("min_prox", *prox1, *param);
    // ecl.apply_kernel("apply_log_fpn", prox1);

    (*param)[0].PROXTYPE = 2;
    ecl.apply_kernel("min_prox", *prox2, *param);
    // ecl.apply_kernel("apply_log_fpn", prox2);

    (*param)[0].PROXTYPE = 4;
    ecl.apply_kernel("min_prox", *prox3, *param);

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
    //ImGui::ShowDemoWindow();

    compute_join(); // should probably be outside of rendering code, but would come immediately before and after anyway, so keeping inside own scripts (main is from imgui examples)

    viewport.set(pix, M, N);

    ImGui::Begin("Viewport");
    ImGui::Text("FPS %f (currently copying frames from OpenCL -> RAM -> OpenGL)", ImGui::GetIO().Framerate);
    ImGui::Image((void*)(intptr_t)viewport.tex_id, ImVec2(M, N));
    ImGui::End();

    ImGui::Begin("Params");
    ImGui::SliderFloat("Re0", &re0, -2.0f, 0.5f);
    ImGui::SliderFloat("Re1", &re1, -2.0f, 0.5f);
    ImGui::SliderFloat("Im0", &im0, -1.0f, 1.0f);
    ImGui::SliderFloat("Im1", &im1, -1.0f, 1.0f);
    ImGui::SliderInt("MAXITER", &MAXITER, 1, 10000);
    ImGui::End();

    compute_begin(); // start computing next frame

}
