#pragma once

#include <GLFW/glfw3.h>
#include <iostream>

// #include <chrono> // for timing
// using namespace std::chrono;

#include "../mandelstructs.h"
#include "easy_cl.hpp"

using namespace std;

class Texture
// based on snippets on
// https://github-wiki-see.page/m/ocornut/imgui/wiki/Image-Loading-and-Displaying-Examples
{
public:
  GLuint tex_id = 0;

  Texture() {
    // Create a OpenGL texture identifier
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D,
                  tex_id); // set glTex funcs to apply to this texture

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                    GL_CLAMP_TO_EDGE); // This is required on WebGL for non
                                       // power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same
  }

  void set(void *image_data, int image_width, int image_height)
  // Upload pixels into texture
  {
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE,
                 image_data); // can only set once? or at least will require
                              // setting to same size?
  }
};

enum ComputeMode { SingleField = 0, DualField = 1, TriField = 2 };

struct FieldUIState {
  int field = 0;
  int proxtype = 1;
  bool real = false;
  // unfortunately cannot just use Box struct, since imgui sliders work on float
  float box_bot = 0;
  float box_top = 0.5;
  float box_left = 0;
  float box_right = 0.5;
};

class App {
public:
  int N;
  int M;
  static string title;
  vector<string> mimgs;
  string migs_opts = "";

  Texture viewport;

  EasyCL ecl;

  SynchronisedArray<FPN> *field1;
  SynchronisedArray<FPN> *field2;
  SynchronisedArray<FPN> *field3;

  SynchronisedArray<FParam> *param;
  SynchronisedArray<Pixel> *pix;

  Complex viewport_center = {-0.75, 0};
  Complex viewport_deltas = {1.25, 1.25};
  int MAXITER = 100;
  int compute_mode = ComputeMode::SingleField;
  float MAXITERpow = 2, cre = -0.85, cim = 0.6;
  bool mandel = true;

  bool compute_enabled = false;

  string default_recurse_func = "inline Complex_t f(Complex_t z, Complex_t c)\n\
{\n\
    return complex_add(complex_pow(z, 2), c);\n\
}";

  const static size_t func_buff_size = 512;
  char func_buff[func_buff_size];

  App();
  ~App();

  // gpu jobs
  void min_prox(SynchronisedArray<FPN> *prox, int PROXTYPE);
  void escape_iter(SynchronisedArray<FPN> *prox);
  void orbit_trap(SynchronisedArray<FPN> *prox, float bb, float bt, float bl,
                  float br, bool real);
  void map_sines(FPN f1, FPN f2, FPN f3);
  void map_img(string img_file);
  void fields_to_RGB(bool normalise);

  void compute_join();
  bool compile_kernels(string new_func);
  void render();
  void reset_view();
  void show_viewport();
  void controlls_tab();
  void handle_field(string field_name, SynchronisedArray<FPN> *prox,
                    FieldUIState *state);
};
