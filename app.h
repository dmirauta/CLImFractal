#pragma once

#include <iostream>
#include <GLFW/glfw3.h>

#include <chrono>
using namespace std::chrono;

#include "easy_cl.hpp"
#include "mandelstructs.h"

using namespace std;

class Texture
// based on snippets on https://github-wiki-see.page/m/ocornut/imgui/wiki/Image-Loading-and-Displaying-Examples
{
    public:
        GLuint tex_id = 0;

        Texture()
        {
            // Create a OpenGL texture identifier
            glGenTextures(1, &tex_id);
            glBindTexture(GL_TEXTURE_2D, tex_id); // set glTex funcs to apply to this texture

            // Setup filtering parameters for display
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same
        }

        void set(unsigned char* image_data, int image_width, int image_height)
        // Upload pixels into texture
        {
            #if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
                glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            #endif
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data); // can only set once? or at least will require setting to same size?
        }

};

class App 
{
    public:
        int N;
        int M;
        static string title;
        Texture viewport;
        unsigned char * pix;
        time_point<high_resolution_clock> start;

        EasyCL ecl;
        SynchronisedArray<double>  *prox1;
        SynchronisedArray<double>  *prox2;
        SynchronisedArray<double>  *prox3;
        SynchronisedArray<MPParam> *param;

        float re0=-2.0f, re1=0.5f, im0=-1.0f, im1=1.0f;
        int MAXITER = 100;

        App();
        ~App();

        void compute_begin();
        void compute_join();
        void render();
};
