#ifdef USE_FLOAT
    typedef float FPN;
#else
    typedef double FPN;
#endif

typedef struct Complex {
    FPN re;
    FPN im;
} Complex_t;

typedef struct Box {
    FPN left;
    FPN right;
    FPN bot;
    FPN top;
} Box_t;

typedef struct Pixel {
   unsigned char r;
   unsigned char g;
   unsigned char b;
} Pixel_t;

typedef struct EIParam {
// General fract iter params
    int            mandel; // mandel or julia
    Complex_t      c;      // not given when mandel selected
    Box_t          view_rect;

// escape iter param
    int            MAXITER;
} EIParam_t;

typedef struct MPParam {
// General fract iter params
    int            mandel; // mandel or julia
    Complex_t      c;      // not given when mandel selected
    Box_t          view_rect;

// min prox param
    int           MAXITER;
    int           PROXTYPE;
} MPParam_t;

typedef struct OTParam {
// General fract iter params
    int            mandel; // mandel or julia
    Complex_t      c;      // not given when mandel selected
    Box_t          view_rect;

// orbit trap param
    Box_t      trap;
    int        MAXITER;
} OTParam_t;

typedef struct ImDims {
    int        imH;
    int        imW;
} ImDims_t;
