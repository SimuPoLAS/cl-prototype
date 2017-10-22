#include <iostream>
#include <vector>
#include <string>

#include <CL/cl.h>

std::string source = R"(
typedef struct __attribute__ ((packed)) _node
{
    int value;
    int children[8];
} node;

__kernel void SAXPY (__global float* a, __global float* b, __global node* nodes)
{
    const int i = get_global_id(0);
    //int childID = nodes[0].children[0];
    int childID = 0;
    b[i] += a[i] + nodes[childID].value;
}
)";

int main(int argc, char const *argv[])
{
    // get platforms
    cl_uint platformIdCount = 0;
    clGetPlatformIDs (0, nullptr, &platformIdCount);

    std::vector<cl_platform_id> platformIds (platformIdCount);
    clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);

    // check platformcount
    if (platformIds.size() == 0)
    {
        std::cout << "no platforms found" << '\n';
        exit(1);
    }

    std::cout << "platforms found: " << platformIdCount << '\n';

    // get devices
    cl_uint deviceIdCount = 0;
    clGetDeviceIDs
    (
        platformIds[0],
        CL_DEVICE_TYPE_ALL,
        0,
        nullptr,
        &deviceIdCount
    );

    std::vector<cl_device_id> deviceIds(deviceIdCount);
    clGetDeviceIDs
    (
        platformIds[0],
        CL_DEVICE_TYPE_ALL,
        deviceIdCount,
        deviceIds.data(),
        nullptr
    );

    // check devices count
    if (deviceIds.size() == 0)
    {
        std::cout << "no devices found" << '\n';
        exit(1);
    }

    std::cout << "devices found: " << deviceIdCount << '\n';

    std::cout << "devices: " << '\n';
    // nameing devices
    for(size_t i = 0; i < deviceIdCount; i++)
    {
        size_t size = 0;
        clGetDeviceInfo(deviceIds[i], CL_DEVICE_NAME, 0, nullptr, &size);

        std::string result;
        result.resize(size);
        clGetDeviceInfo
        (
            deviceIds[i],
            CL_DEVICE_NAME,
            size,
            const_cast<char*>(result.data()),
            nullptr
        );

        std::cout << std::to_string(i) << ": " << result << '\n';
    }

    // create context
    const cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platformIds[0]),
        0, 0
    };

    cl_int error;
    cl_context context = clCreateContext
    (
        contextProperties,
        deviceIdCount,
        deviceIds.data(),
        nullptr,
        nullptr,
        &error
    );

    if(error != CL_SUCCESS)
    {
        std::cout << "errors occured at context creation" << '\n';
        exit(1);
    }

    std::cout << "context created" << '\n';

    // create program
    size_t lengths[] = { source.size() };
    const char* sources[] = { source.c_str() };

    cl_program program = clCreateProgramWithSource
    (
        context,
        1,
        sources,
        lengths,
        &error
    );

    if (error != CL_SUCCESS)
    {
        std::cout << "errors occured at program creation" << '\n';
        exit(1);
    }

    std::cout << "program created" << '\n';

    // build program
    error = clBuildProgram
    (
        program,
        deviceIdCount,
        deviceIds.data(),
        nullptr,
        nullptr,
        nullptr
    );

    if (error != CL_SUCCESS)
    {
        std::cout << "errors occured at program building" << '\n';
        exit(1);
    }

    std::cout << "program built" << '\n';

    // create kernel
    cl_kernel kernel = clCreateKernel(program, "SAXPY", &error);

    if (error != CL_SUCCESS)
    {
        std::cout << "errors occured at kernel creation" << '\n';
        exit(1);
    }

    std::cout << "kernel created" << '\n';

    // prepare some test data
    const size_t testDataSize = 5;
    std::vector<float> a(testDataSize), b(testDataSize);
    for (size_t i = 0; i < testDataSize; i++)
    {
        a [i] = static_cast<float> (5);
        b [i] = static_cast<float> (6 + i);
    }

    // create buffers

    // buffer a creation
    cl_mem aBuffer = clCreateBuffer
    (
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * testDataSize,
        a.data(),
        &error
    );

    if (error != CL_SUCCESS)
    {
        std::cout << "errors occured at buffer a creation" << '\n';
        exit(1);
    }

    std::cout << "buffer a created" << '\n';

    // buffer b creation
    cl_mem bBuffer = clCreateBuffer
    (
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * testDataSize,
        b.data(),
        &error
    );

    if (error != CL_SUCCESS)
    {
        std::cout << "errors occured at buffer b creation" << '\n';
        exit(1);
    }

    std::cout << "buffer b created" << '\n';

    // prepare data for n buffer
    typedef struct __attribute__ ((packed)) _node
    {
        int value;
        int children[8];
    } node;

    node nodes[] = { node { 10, { 1 } }, node { 20, { } } };

    // buffer n creation
    cl_mem nBuffer = clCreateBuffer
    (
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(node) * sizeof(nodes),
        nodes,
        &error
    );

    if (error != CL_SUCCESS)
    {
        std::cout << "errors occured at buffer n creation" << '\n';
        exit(1);
    }

    std::cout << "buffer n created" << '\n';

    // create command queue
    // TODO: use clCreateCommandQueueWithProperties in future
    cl_command_queue queue = clCreateCommandQueue
    (
        context,
        deviceIds[0],
        0,
        &error
    );

    if (error != CL_SUCCESS)
    {
        std::cout << "errors occured at command queue creation" << '\n';
        exit(1);
    }

    std::cout << "command queue created" << '\n';

    // set args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &nBuffer);

    // run code
    const size_t globalWorkSize[] = { testDataSize, 0, 0 };

    error = clEnqueueNDRangeKernel
    (
        queue,
        kernel,
        1,
        nullptr,
        globalWorkSize,
        nullptr,
        0,
        nullptr,
        nullptr
    );

    if (error != CL_SUCCESS)
    {
        std::cout << "errors occured at running code" << '\n';
        exit(1);
    }

    std::cout << "command run" << '\n';

    // reading results
    error = clEnqueueReadBuffer
    (
        queue,
        bBuffer,
        CL_TRUE,
        0,
        sizeof(float) * testDataSize,
        b.data(),
        0,
        nullptr,
        nullptr
    );

    if (error != CL_SUCCESS)
    {
        std::cout << "errors occured at reading results" << '\n';
        exit(1);
    }

    std::cout << "results read" << '\n';

    std::cout << "results:" << '\n';

    for (size_t i = 0; i < testDataSize; i++)
    {
        std::cout << std::to_string(b[i]) << '\n';
    }

    clReleaseCommandQueue(queue);

    clReleaseMemObject(bBuffer);
    clReleaseMemObject(aBuffer);

    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseContext(context);

    return 0;
}
