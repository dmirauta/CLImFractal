#include "app.h"
#include "imgui.h"

void make_img(unsigned char* pix, int N, int M)
{
    int k=0;
    for (int i=0; i<M; i++)
    {
        for (int j=0; j<N; j++)
        {
            k = i*N+j;
            if ( i<M/2 )
            {
                pix[3*k]   = 255;
                pix[3*k+1] = 0;
                pix[3*k+2] = 0;
            } else {
                pix[3*k]   = 0;
                pix[3*k+1] = 0;
                pix[3*k+2] = 255;
            }
        }
    }
}

std::string App::title = "Testing";

App::App()
{
    N = 600;
    M = 800;
    pix = new unsigned char[N*M*3];
    make_img(pix, N, M);
    display.set(pix, M, N);
}

App::~App()
{
    delete [] pix;
}

void App::render()
{
    //ImGui::ShowDemoWindow();

    ImGui::Begin("OpenGL Texture Text");
    ImGui::Text("pointer = %p", display.tex_id);
    ImGui::Image((void*)(intptr_t)display.tex_id, ImVec2(M, N));
    ImGui::End();
}
