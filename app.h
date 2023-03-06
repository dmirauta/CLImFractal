#pragma once

#include <iostream>
#include <GLFW/glfw3.h>

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
        static std::string title;
        Texture display;
        unsigned char * pix;

        App();
        ~App();

        void render();
};
