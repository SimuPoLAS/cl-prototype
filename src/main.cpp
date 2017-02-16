#include <iostream>
#include <vector>

#include <CL/cl.h>

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

    if (deviceIds.size() == 0)
    {
        std::cout << "no devices found" << '\n';
        exit(1);
    }

    std::cout << "devices found: " << deviceIdCount << '\n';

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

    clReleaseContext(context);

    return 0;
}
