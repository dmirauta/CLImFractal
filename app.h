#pragma once

#include <iostream>
#include <GLFW/glfw3.h>

// #include <chrono> // for timing
// using namespace std::chrono;

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

        void set(void* image_data, int image_width, int image_height)
        // Upload pixels into texture
        {
            #if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
                glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            #endif
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data); // can only set once? or at least will require setting to same size?
        }

};

enum ComputeMode { SingleField = 0, DualField = 1, TriField = 2 };

class App 
{
    public:
        int N;
        int M;
        static string title;

        Texture viewport;
        // unsigned char * pix;

        EasyCL ecl;
        SynchronisedArray<double>  *prox1;
        SynchronisedArray<double>  *prox2;
        SynchronisedArray<double>  *prox3;
        SynchronisedArray<FParam>  *param;
        SynchronisedArray<Pixel>   *pix;

        Complex viewport_center = {-0.75, 0};
        Complex viewport_deltas = {1.25, 1};
        int MAXITER;
        int compute_mode = ComputeMode::TriField;
        float MAXITERpow = 2, cre=-0.85, cim=0.6;
        bool mandel = true;

        bool compute_enabled = false;
        bool running_gpu_job = false;

        App();
        ~App();

        // job types
        void min_prox(SynchronisedArray<double> *prox, int PROXTYPE);
        void escape_iter(SynchronisedArray<double> *prox);
        void fields_to_RGB(bool normalise);
        void map_sines(FPN f1, FPN f2, FPN f3);

        void compute_join();
        void render();
        void show_viewport();
        void controlls_tab();
        void handle_field(string field_name, SynchronisedArray<double> *prox, int *field, int *proxtype);
};
