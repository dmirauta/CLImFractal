// Header only lib from https://github.com/dmirauta/easy-opencl-cpp

#pragma once

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

////////////////////////////////////////////////////////////////////////////
//// Arrays

class Dims
{
    public:
        int x;
        int y;
        int z;

        Dims(int nx=1, int ny=1, int nz=1)
        {
            x = nx;
            y = ny;
            z = nz;
        }

        Dims(std::initializer_list<int> args)
        {

            if (args.size()>=1) {
                x = args.begin()[0];
            } else {
                x = 1;
            }

            if (args.size()>=2) {
                y = args.begin()[1];
            } else {
                y = 1;
            }

            if (args.size()>=3) {
                z = args.begin()[2];
            } else {
                z = 1;
            }

        }

        void operator=(const Dims& d)
        {
            x=d.x;
            y=d.y;
            z=d.z;
        }


};

// To simplify some function prototypes, that don't need template knowledge
class AbstractSynchronisedArray
{
    public:
        Dims dims;
        int items;

        cl::Buffer gpu_buff;
        cl_mem_flags mem_flags;

        virtual void to_gpu(cl::CommandQueue &queue) = 0;

        virtual void from_gpu(cl::CommandQueue &queue) = 0;

};

template<typename T>
class SynchronisedArray : public AbstractSynchronisedArray
{
    public:

        int buffsize;
        bool no_copy_back; // dont copy back even if not read only

        T* cpu_buff;

        SynchronisedArray() {};
        
        SynchronisedArray(cl::Context &context, cl_mem_flags flags, Dims dimensions)
        {
            mem_flags = flags;
            no_copy_back = false;

            dims = dimensions;
            items = dims.x * dims.y * dims.z;

            buffsize = sizeof(T)*items;
            cpu_buff = new T[items];
            gpu_buff = cl::Buffer(context, flags, buffsize);
        }

        SynchronisedArray(cl::Context &context, Dims dimensions={})
            : SynchronisedArray(context, CL_MEM_READ_WRITE, dimensions) {}

        ~SynchronisedArray()
        {
            delete [] cpu_buff;
        }

        void to_gpu(cl::CommandQueue &queue)
        {
            if (mem_flags!=CL_MEM_WRITE_ONLY) // otherwise gpu will not need to read it, no need to copy to
                queue.enqueueWriteBuffer(gpu_buff, CL_TRUE, 0, buffsize, cpu_buff);
        }

        void from_gpu(cl::CommandQueue &queue)
        {
            if (mem_flags!=CL_MEM_READ_ONLY && !no_copy_back) // if either mem_flags==CL_MEM_READ_ONLY or no_copy_back, we skip
                queue.enqueueReadBuffer(gpu_buff, CL_TRUE, 0, buffsize, cpu_buff);
        }

        T& operator[](std::size_t i)
        {
            assert(i<dims.x);
            return cpu_buff[i];
        }

        T& operator[](std::size_t i, std::size_t j) // requires -std=c++23
        {
            assert(i<dims.x);
            assert(j<dims.y);
            return cpu_buff[i*dims.y + j];
        }

        T& operator[](std::size_t i, std::size_t j, std::size_t k)
        {
            assert(i<dims.x);
            assert(j<dims.y);
            assert(k<dims.z);
            return cpu_buff[ (i*dims.y + j)*dims.z + k ];
        }

};

////////////////////////////////////////////////////////////////////////////
//// Helper functions

inline void to_gpu(cl::CommandQueue &queue, cl::Kernel &kernel, int firstargnum) { }

template<typename... ASArrays>
void to_gpu(cl::CommandQueue &queue,
            cl::Kernel &kernel,
            int firstargnum,
            AbstractSynchronisedArray& first_arr,
            ASArrays&... arrs)
{
    first_arr.to_gpu(queue);
    kernel.setArg(firstargnum, first_arr.gpu_buff);
    to_gpu(queue, kernel, firstargnum+1, arrs...);
}


inline void from_gpu(cl::CommandQueue &queue) { }

template<typename... ASArrays>
void from_gpu(cl::CommandQueue &queue,
            AbstractSynchronisedArray& first_arr,
            ASArrays&... arrs)
{
    first_arr.from_gpu(queue);
    from_gpu(queue, arrs...);
}

////////////////////////////////////////////////////////////////////////////
//// Main class

class EasyCL
{
    public:
        cl::Context context;
        cl::Device device;
        cl::CommandQueue queue;

        std::map<std::string, cl::Kernel> kernels;

        bool _verbose;

        EasyCL(bool verbose=false)
        {
            _verbose = verbose;

            // based on https://github.com/Dakkers/OpenCL-examples/blob/master/example01/main.cpp

            // get all platforms (drivers), e.g. NVIDIA
            std::vector<cl::Platform> all_platforms;
            cl::Platform::get(&all_platforms);

            if (all_platforms.size()==0) {
                std::cout<<" No platforms found. Check OpenCL installation!\n";
                exit(1);
            }
            cl::Platform default_platform=all_platforms[0];

            std::vector<cl::Device> all_devices;
            default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
            if(all_devices.size()==0){
                std::cout<<" No devices found. Check OpenCL installation!\n";
                exit(1);
            }

            device=all_devices[0];
            context = cl::Context({device});
            queue = cl::CommandQueue(context, device);

            if (_verbose)
            {
                std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
                std::cout << "Using device: "<<device.getInfo<CL_DEVICE_NAME>()<<"\n";
            }
        }

        void load_kernels(std::vector<std::string> source_files,
                          std::vector<std::string> kernel_names,
                          std::string build_options="")
        // based on https://github.com/Dakkers/OpenCL-examples/blob/master/example01/main.cpp
        {
            cl::Program::Sources sources; // cannot actually push_back multiple sources to this?

            std::string kernel_code = "";
            for(auto source_file : source_files)
            {
                // https://stackoverflow.com/a/62772405
                const std::ifstream input_stream(source_file, std::ios_base::binary);

                if (input_stream.fail()) {
                    throw std::runtime_error("Failed to open file");
                }

                std::stringstream buffer;
                buffer << input_stream.rdbuf();

                kernel_code += buffer.str();
            }
            sources.push_back({kernel_code.c_str(), kernel_code.length()});

            if (_verbose)
            {
                std::cout << "Build options:\n"
                        << build_options << "\n"
                        << "Source:\n"
                        << kernel_code;
            }

            cl::Program program(context, sources);
            if (program.build({device}, build_options.c_str()) != CL_SUCCESS) {
                std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
                exit(1);
            }

            for(auto kernel_name : kernel_names)
            {
                kernels[kernel_name] = cl::Kernel(program, kernel_name.c_str());
            }

        }

        template<typename... ASArrays>
        void apply_kernel(std::string kernel_name, 
                          AbstractSynchronisedArray& first_arr,
                          ASArrays&... arrs)
        {
            cl::NDRange global_dims;
            if (first_arr.dims.z>1)
            {
                global_dims = cl::NDRange(first_arr.dims.x, first_arr.dims.y, first_arr.dims.z);
            } else if (first_arr.dims.y>1) {
                global_dims = cl::NDRange(first_arr.dims.x, first_arr.dims.y);
            } else if (first_arr.dims.x>1) {
                global_dims = cl::NDRange(first_arr.dims.x);
            } else {
                std::cout << "Invalid global dims in apply_kernel? (based on input data)\n";
                exit(1);
            }

            to_gpu(queue, kernels[kernel_name], 0, first_arr, arrs...);

            queue.enqueueNDRangeKernel(kernels[kernel_name],
                                    cl::NullRange,  // offset
                                    global_dims,
                                    cl::NullRange); // local  dims (warps/workgroups)

            from_gpu(queue, first_arr, arrs...);

            queue.finish(); // blocking
        }


};
