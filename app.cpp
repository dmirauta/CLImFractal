#include "app.h"
#include "imgui.h"

void make_img(unsigned char* pix, int N, int M, bool flip)
{
    int k=0;
    for (int i=0; i<M; i++)
    {
        bool set_blue = flip ? i<M/2 : i>M/2;
        for (int j=0; j<N; j++)
        {
            k = i*N+j;
            pix[3*k]   = 255*set_blue;
            pix[3*k+1] = 0;
            pix[3*k+2] = 255*(1-set_blue);

        }
    }
}

std::string App::title = "Testing";

App::App()
{
    N = 600;
    M = 800;
    pix = new unsigned char[N*M*3];
    start = high_resolution_clock::now();
}

App::~App()
{
    delete [] pix;
}

void App::render()
{
    //ImGui::ShowDemoWindow();

    auto now = high_resolution_clock::now();
    int seconds_since_start = duration_cast<seconds>(now - start).count();
    make_img(pix, N, M, seconds_since_start%2);
    viewport.set(pix, M, N);

    ImGui::Begin("Viewport");
    ImGui::Text("FPS %f", ImGui::GetIO().Framerate);
    ImGui::Image((void*)(intptr_t)viewport.tex_id, ImVec2(M, N));
    ImGui::End();
}
