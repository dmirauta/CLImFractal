#include "app.h"
#include "imgui.h"

void make_img(unsigned char* pix, int N, int M, int blocksX, int blocksY)
{
    int deltaX = M/blocksX;
    int deltaY = N/blocksY;
    for (int i=0; i<M; i++)
    {
        for (int j=0; j<N; j++)
        {
            if ( (i/deltaY)%2 && (j/deltaX)%2 )
            {
                pix[3*i]   = 255;
                pix[3*i+1] = 0;
                pix[3*i+2] = 0;
            } else {
                pix[3*i]   = 0;
                pix[3*i+1] = 0;
                pix[3*i+2] = 255;
            }
        }
    }
}

App::App()
{
    N = 600;
    M = 800;
    pix = new unsigned char[N*M*3];
    make_img(pix, N, M, 3, 3);
    display.set(pix, M, N);
}

App::~App()
{
    delete [] pix;
}

void App::render()
{
    ImGui::Begin("OpenGL Texture Text");
    ImGui::Text("Yay?");
    ImGui::Image((void*)(intptr_t)display.tex_id, ImVec2(M, N));
    ImGui::End();
}
