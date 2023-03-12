#include <filesystem>
#include <cmath>

#include "app.h"
#include "imgui.h"

std::string App::title = "CLImFractal";

App::App()
{
    N = 600;
    M = 800;

    vector<string> source_files{"mandel.cl"}; // this time importing its own deps
    vector<string> kernel_names{"escape_iter", "escape_iter_fpn", "min_prox", 
                                "orbit_trap", "map_img", "apply_log_int", 
                                "apply_log_fpn", "pack", "pack_norm",
                                "map_sines"};
    string ocl_include_path = filesystem::current_path(); // include path required for cl file importing own deps
    ecl.load_kernels(source_files, kernel_names, "-I "+ocl_include_path);
    ecl.no_block = true;

    prox1 = new SynchronisedArray<double>(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    prox2 = new SynchronisedArray<double>(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    prox3 = new SynchronisedArray<double>(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    pix   = new SynchronisedArray<Pixel>(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    param = new SynchronisedArray<FParam>(ecl.context);

}

App::~App()
{
    delete pix;
    delete prox1;
    delete prox2;
    delete prox3;
    delete param;
}

void App::escape_iter(SynchronisedArray<double> *prox)
{
    if (compute_enabled)
    {
        running_gpu_job = true;

        ecl.apply_kernel("escape_iter_fpn", *prox, *param);
        
    }

}

void App::min_prox(SynchronisedArray<double> *prox, int PROXTYPE)
{
    if (compute_enabled)
    {
        running_gpu_job = true;

        SynchronisedArray<int> pt(ecl.context);
        pt[0] = PROXTYPE;

        ecl.apply_kernel("min_prox", *prox, *param, pt);
        
    }

}

void App::fields_to_RGB(bool norm = false)
{
    if (compute_enabled)
    {
        running_gpu_job = true;

        if (norm)
        {
            ecl.apply_kernel("pack_norm", *prox1, *prox2, *prox3, *pix);
        } else {
            // should still normalise individual fields here
            ecl.apply_kernel("pack", *prox1, *prox2, *prox3, *pix);
        }
    }
}

void App::map_sines(FPN f1, FPN f2, FPN f3)
{
    if (compute_enabled)
    {
        running_gpu_job = true;

        SynchronisedArray<Freqs> freqs(ecl.context);
        freqs[0] = {f1, f2, f3};

        ecl.apply_kernel("map_sines", *prox1, *pix, freqs);

    }
}

void App::compute_join()
{
    if (running_gpu_job)
    {
        ecl.queue.finish();
        running_gpu_job = false;
    }       
}

void App::render()
{
    // ImGui::ShowDemoWindow();

    compute_join(); // should probably be outside of rendering code, but would come immediately before and after anyway, so keeping inside own scripts (main is from imgui examples)

    show_viewport();
    controlls_tab(); // queing gpu jobs in here

}

void App::show_viewport()
{
    viewport.set(pix->cpu_buff, M, N);

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
}

void App::controlls_tab()
{
    ImGui::Begin("Controlls");

        if (ImGui::Button("Toggle compute active"))
            compute_enabled = !compute_enabled;

        ImGui::Text("Inputs:");
        ImGui::Text("Pan: Right click and drag in viewport");
        ImGui::Text("Zoom: Mouse wheel");

        ImGui::Text("\nGeneral Params:");
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

        ImGui::Text("\nMode:");
        ImGui::RadioButton("Single field", &compute_mode, ComputeMode::SingleField);
        ImGui::RadioButton("Dual field - Image map", &compute_mode, ComputeMode::DualField);
        ImGui::RadioButton("Tri field - RGB", &compute_mode, ComputeMode::TriField);

        // Update general params
        (*param)[0].mandel = mandel ? 1 : 0;
        (*param)[0].c = {(FPN) cre, (FPN) cim};
        (*param)[0].view_rect = {viewport_center.re-viewport_deltas.re,
                                    viewport_center.re+viewport_deltas.re,
                                    viewport_center.im-viewport_deltas.im,
                                    viewport_center.im+viewport_deltas.im};
        (*param)[0].MAXITER = MAXITER;

        // start computing next frame // could interfere with subsequent OpenGL calls?
        switch (compute_mode)
        {
            case ComputeMode::SingleField:
            {
                static int field = 0;
                static int pt = 1;
                handle_field("Field", prox1, &field, &pt);

                static float f1 = 1;
                static float f2 = 2;
                static float f3 = 3;
                ImGui::Text("Cmap frequencies:");
                ImGui::SliderFloat("f1", &f1, 0.1, 10);
                ImGui::SliderFloat("f2", &f2, 0.1, 10);
                ImGui::SliderFloat("f3", &f3, 0.1, 10);
                map_sines(f1, f2, f3);
                break;
            }
            case ComputeMode::TriField:
            {
                static int field_1 = 0;
                static int pt1 = 1;
                handle_field("R Field", prox1, &field_1, &pt1);
                static int field_2 = 0;
                static int pt2 = 1;
                handle_field("G Field", prox2, &field_2, &pt2);
                static int field_3 = 0;
                static int pt3 = 1;
                handle_field("B Field", prox3, &field_3, &pt3);

                static bool nc = false;
                ImGui::Checkbox("Normalise colors", &nc);
                fields_to_RGB(nc);
                break;
            }
            default:
                ImGui::Text("Selected mode not implemented.");
                break;
        }


    ImGui::End();
}

void App::handle_field(string field_name, SynchronisedArray<double> *prox, int *field, int *proxtype)
{
    ImGui::Combo(field_name.c_str(), field, "Iters\0Proximity\0\0");

    switch(*field)
    {
        case 0:
            escape_iter(prox);
            break;
        case 1:
        {
            string fn = field_name+" PROXTYPE"; // sliders seem to get linked if they do not have unique names
            ImGui::SliderInt(fn.c_str(), proxtype, 1, 7);

            min_prox(prox, *proxtype);
            break;

        }
            
        default:
            ImGui::Text("Selected field not implemented.");
            break;
    }

}
