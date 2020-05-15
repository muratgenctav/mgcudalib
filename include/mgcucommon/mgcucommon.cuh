#ifndef MGCUCOMMON_H
#define MGCUCOMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <string>
#include <sstream>
#include <exception>

#define MAX_THREADS 1024

#define checkCudaErrors(val) mgcu::checkCuda( (val), #val, __FILE__, __LINE__)

namespace mgcu
{
    /**
     * Class for exception handling.
     */
    class mgcuError: public std::exception
    {
    public:
        mgcuError(const std::string& msg) : e_msg(msg) { }

        virtual const char * what() const throw()
        {
            return e_msg.c_str();
        }

    private:
        const std::string e_msg;
    };

    template<typename T>
    void checkCuda(T err, const char* const func, const char* const file, const int line) {
        if (err != cudaSuccess) {
            std::ostringstream os;
            os << "CUDA error at: " << file << ":" << line << std::endl
               << cudaGetErrorString(err) << " " << func << std::endl;
            
            const string msg = os.str();
            throw (mgcuError(msg));
        }
    }
}

#endif