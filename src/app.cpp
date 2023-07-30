#include <cmath>
#include <filesystem>
#include <regex>
namespace fs = std::filesystem;

#include "app.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "imgui.h"
#include "stb_image.h"

std::string App::title = "CLImFractal";

App::App() {
  N = 600;
  M = 800;

  strcpy(func_buff, default_recurse_func.c_str());

  // ecl._verbose = true;
  compile_kernels("");
  ecl.no_block = true;

  field1 = new SynchronisedArray<FPN>(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
  field2 = new SynchronisedArray<FPN>(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
  field3 = new SynchronisedArray<FPN>(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
  pix = new SynchronisedArray<Pixel>(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
  param = new SynchronisedArray<FParam>(ecl.context);

  for (const auto &entry : fs::directory_iterator("mimg")) {
    string s = entry.path();
    regex r(".*\\.(?:png|jpg)");
    smatch m;
    if (regex_search(s, m, r)) {
      mimgs.push_back(s);
      migs_opts += s;
      migs_opts.push_back('\0');
    }
  }
  migs_opts.push_back('\0');
}

App::~App() {
  delete pix;
  delete field1;
  delete field2;
  delete field3;
  delete param;
}

bool App::compile_kernels(string new_func) {
  vector<string> source_files{"mandelstructs.h", "mandelutils.c", "mandel.cl"};
  vector<string> kernel_names{
      "escape_iter",   "escape_iter_fpn", "min_prox", "orbit_trap",
      "orbit_trap_re", "orbit_trap_im",   "map_img",  "map_img2",
      "apply_log_int", "apply_log_fpn",   "pack",     "pack_norm",
      "map_sines"};
  string build_options =
      "-I " + string(fs::current_path()) + " -D EXTERNAL_CONCAT";
#ifdef USE_FLOAT
  build_options += " -D USE_FLOAT";
#endif
  return ecl.load_kernels(source_files, kernel_names, build_options,
                          "//>>(.|\n)*//<<", new_func);
}

void App::escape_iter(SynchronisedArray<FPN> *field) {
  if (compute_enabled)
    ecl.apply_kernel("escape_iter_fpn", *field, *param);
}

void App::min_prox(SynchronisedArray<FPN> *field, int PROXTYPE) {
  if (compute_enabled) {
    SynchronisedArray<int> pt(ecl.context);
    pt[0] = PROXTYPE;

    ecl.apply_kernel("min_prox", *field, *param, pt);
  }
}

void App::orbit_trap(SynchronisedArray<FPN> *field, float bb, float bt,
                     float bl, float br, bool real) {
  if (compute_enabled) {
    SynchronisedArray<Box> _box(ecl.context);
    _box[0] = {bb, bt, bl, br};
    string kernel = real ? "orbit_trap_re" : "orbit_trap_im";
    ecl.apply_kernel(kernel, *field, *param, _box);
  }
}

void App::map_sines(FPN f1, FPN f2, FPN f3) {
  if (compute_enabled) {
    SynchronisedArray<Freqs> freqs(ecl.context);
    freqs[0] = {f1, f2, f3};

    ecl.apply_kernel("map_sines", *field1, *pix, freqs);
  }
}

void App::map_img(string img_file) {
  if (compute_enabled) {
    int w;
    int h;
    int comp;
    unsigned char *image = stbi_load(img_file.c_str(), &w, &h, &comp, STBI_rgb);

    // if (comp!=3) // STBI_rgb should ensure this?
    //     cout << "Warning: Expecting an RGB image (3 channels, got " << comp
    //     <<")\n"; // may well crash if fewer or load wrong if greater?

    SynchronisedArray<Pixel> img(ecl.context, CL_MEM_READ_ONLY, {w, h});
    for (int i = 0; i < w; i++) {
      for (int j = 0; j < h; j++) {
        int k = j * w + i;
        img[i, j] = {image[3 * k], image[3 * k + 1], image[3 * k + 2]}; // <--
      }
    }

    SynchronisedArray<ImDims> dims(ecl.context, CL_MEM_READ_ONLY);
    dims[0] = {w, h};

    ecl.apply_kernel("map_img2", *field1, *field2, img, *pix, dims);

    delete image;
  }
}

void App::fields_to_RGB(bool norm = false) {
  string kernel = norm ? "pack_norm" : "pack";
  ecl.apply_kernel(kernel, *field1, *field2, *field3, *pix);
}

void App::compute_join() { ecl.queue.finish(); }

void App::render() {
  // ImGui::ShowDemoWindow();

  compute_join(); // should probably be outside of rendering code, but would
                  // come immediately before and after anyway, so keeping inside
                  // own scripts (main is from imgui examples)

  show_viewport();
  controlls_tab(); // queing gpu jobs in here
}

void App::reset_view() {
  if (mandel) {
    viewport_center = {-0.75, 0};
    viewport_deltas = {1.25, 1.25};
  } else {
    viewport_center = {0, 0};
    viewport_deltas = {2, 2};
  }
}

void App::show_viewport() {
  viewport.set(pix->cpu_buff, M, N);

  ImGui::Begin("Viewport");

  ImGuiIO &io = ImGui::GetIO();

  // Pan
  float deltaX = 0, deltaY = 0;
  if (ImGui::IsWindowHovered() && ImGui::IsMousePosValid() &&
      ImGui::IsMouseDown(1)) {
    deltaX = 2 * io.MouseDelta.x / (float)N;
    deltaY = 2 * io.MouseDelta.y / (float)M;
    viewport_center.re -= viewport_deltas.re * deltaX;
    viewport_center.im -= viewport_deltas.im * deltaY;
    // ImGui::Text("Mouse delta (screen fraction): (%f, %f)", deltaX, deltaY);
  }

  // Zoom
  if (io.MouseWheel > 0) {
    viewport_deltas.re /= 1.1;
    viewport_deltas.im /= 1.1;
  } else if (io.MouseWheel < 0) {
    viewport_deltas.re *= 1.1;
    viewport_deltas.im *= 1.1;
  }

  ImGui::Text("FPS %f (currently copying frames from OpenCL -> RAM -> OpenGL)",
              ImGui::GetIO().Framerate);
  ImGui::Text("Center: (%lg) + (%lg)i", viewport_center.re, viewport_center.im);
  ImGui::Text("Box dims: (%lg) x (%lg)", 2 * viewport_deltas.re,
              2 * viewport_deltas.im);

  if (ImGui::Button("Reset view"))
    reset_view();

  ImGui::Image((void *)(intptr_t)viewport.tex_id, ImVec2(M, N));

  ImGui::End();
}

void App::controlls_tab() {
  ImGui::Begin("Controlls");

  if (ImGui::Button("Toggle compute active"))
    compute_enabled = !compute_enabled;

  ImGui::Text("Inputs:");
  ImGui::Text("Pan: Right click and drag in viewport");
  ImGui::Text("Zoom: Mouse wheel");

  bool recompile = ImGui::Button("Recompile");
  ImGui::SameLine();
  if (ImGui::Button("Reset")) {
    strcpy(func_buff, default_recurse_func.c_str());
    compile_kernels(func_buff);
  }

  ImGui::InputTextMultiline("Recursed function:", &func_buff[0],
                            func_buff_size);

  static bool success;

  if (recompile)
    success = compile_kernels(func_buff);

  if (!success)
    ImGui::Text(ecl.cl_error.c_str());

  ImGui::Text("\nGeneral Params:");

  if (ImGui::Checkbox("Compute mandelbrot (else Julia)", &mandel))
    reset_view();

  if (!mandel) {
    ImGui::Text("Julia constant:");
    ImGui::SliderFloat("CRE", &cre, -2, 2);
    ImGui::SliderFloat("CIM", &cim, -2, 2);
  }

  ImGui::SliderFloat("log10 MAXITER", &MAXITERpow, 0, 4);
  MAXITER = pow(10, MAXITERpow);
  ImGui::Text("MAXITER: %d", MAXITER);

  ImGui::Text("\nMode:");
  ImGui::RadioButton("Single field", &compute_mode, ComputeMode::SingleField);
  ImGui::RadioButton("Dual field - Image map", &compute_mode,
                     ComputeMode::DualField);
  ImGui::RadioButton("Tri field - RGB", &compute_mode, ComputeMode::TriField);

  // Update general params
  (*param)[0].mandel = mandel ? 1 : 0;
  (*param)[0].c = {(FPN)cre, (FPN)cim};
  (*param)[0].view_rect = {viewport_center.re - viewport_deltas.re,
                           viewport_center.re + viewport_deltas.re,
                           viewport_center.im - viewport_deltas.im,
                           viewport_center.im + viewport_deltas.im};
  (*param)[0].MAXITER = MAXITER;

  // start computing next frame // could interfere with subsequent OpenGL calls?
  switch (compute_mode) {
  case ComputeMode::SingleField: {
    static FieldUIState state;
    handle_field("Field", field1, &state);

    static float f1 = 1;
    static float f2 = 2;
    static float f3 = 3;
    ImGui::Text("Cmap frequencies:");
    ImGui::SliderFloat("f1", &f1, 0.01, 100); // make logarithmic?
    ImGui::SliderFloat("f2", &f2, 0.01, 100);
    ImGui::SliderFloat("f3", &f3, 0.01, 100);
    map_sines(f1, f2, f3);
    break;
  }
  case ComputeMode::DualField: {

    static int file_idx = 0;
    ImGui::Combo("Mimg", &file_idx, migs_opts.c_str());

    static FieldUIState stateU;
    handle_field("U Field", field1, &stateU);
    static FieldUIState stateV;
    handle_field("V Field", field2, &stateV);

    map_img(mimgs[file_idx]);
    break;
  }
  case ComputeMode::TriField: {
    static FieldUIState stateR;
    handle_field("R Field", field1, &stateR);
    static FieldUIState stateG;
    handle_field("G Field", field2, &stateG);
    static FieldUIState stateB;
    handle_field("B Field", field3, &stateB);

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

void App::handle_field(string field_name, SynchronisedArray<FPN> *field,
                       FieldUIState *state) {
  ImGui::Combo(field_name.c_str(), &state->field,
               "Iters\0Proximity\0Orbit trap\0\0");

  switch (state->field) {
  case 0:
    escape_iter(field);
    break;
  case 1: {
    string fn = field_name + " PROXTYPE"; // sliders seem to get linked if they
                                          // do not have unique names
    ImGui::SliderInt(fn.c_str(), &state->proxtype, 1, 7);

    min_prox(field, state->proxtype);
    break;
  }
  case 2: {
    string fn = field_name + " IS REAL"; // sliders seem to get linked if they
                                         // do not have unique names
    ImGui::Checkbox(fn.c_str(), &state->real);

    ImGui::SliderFloat("trap bot", &state->box_bot, -2,
                       2); // linking may actually be desired here
    ImGui::SliderFloat("trap top", &state->box_top, -2, 2);
    ImGui::SliderFloat("trap left", &state->box_left, -2, 2);
    ImGui::SliderFloat("trap right", &state->box_right, -2, 2);

    orbit_trap(field, state->box_bot, state->box_top, state->box_left,
               state->box_right, state->real);
    break;
  }
  default:
    ImGui::Text("Selected field not implemented.");
    break;
  }
}
