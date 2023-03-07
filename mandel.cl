#include "mandelutils.c"

__kernel void apply_log_int(__global int *res_g)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    res_g[i*M+j] = log((float) res_g[i*M+j]);
}

__kernel void apply_log_fpn(__global FPN *res_g)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    res_g[i*M+j] = log((float) res_g[i*M+j]);
}

__kernel void escape_iter(__global int *res_g,
                          __global EIParam_t *param)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {param->view_rect.left + j*(param->view_rect.right-param->view_rect.left)/M,
                   param->view_rect.bot  + i*(param->view_rect.top  -param->view_rect.bot )/N};

    Complex_t _c = param->mandel ? p : param->c;

    res_g[i*M+j] = _escape_iter(p, _c, param->MAXITER);
}

__kernel void min_prox(__global FPN *res_g,
                       __global MPParam_t *param)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {param->view_rect.left + j*(param->view_rect.right-param->view_rect.left)/M,
                   param->view_rect.bot  + i*(param->view_rect.top  -param->view_rect.bot )/N};

    Complex_t _c = param->mandel ? p : param->c;

    res_g[i*M+j] = _minprox(p, _c, param->MAXITER, param->PROXTYPE);
}

__kernel void orbit_trap(__global Complex_t *res_g,
                         __global OTParam_t *param)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {param->view_rect.left + j*(param->view_rect.right-param->view_rect.left)/M,
                   param->view_rect.bot  + i*(param->view_rect.top  -param->view_rect.bot )/N};

    Complex_t _c = param->mandel ? p : param->c;

    res_g[i*M+j] = _orbit_trap(p, _c, param->trap, param->MAXITER);
}

__kernel void map_img   (__global Complex_t *res_g, // result of orbit trap
                         __global Pixel_t   *sim_g, // sample image
                         __global Pixel_t   *mim_g, // mapped image
                         __global ImDims_t  *dims)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    int _i = (int) ( ((float) (dims->imH-1)) * res_g[i*M+j].im );
    int _j = (int) ( ((float) (dims->imW-1)) * res_g[i*M+j].re );

    mim_g[i*M+j] = sim_g[_i*dims->imW + _j];

}
