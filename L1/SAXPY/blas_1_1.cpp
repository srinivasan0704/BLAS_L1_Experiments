#include "hip/hip_runtime.h"
#include <iostream>
#include <vector>
#include "hip/hip_runtime_api.h"
#include "rocblas.h"
#include <fstream>

using namespace std;

// Kernel functions go here ...


__global__ void vecAdd(float *a, float *b, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        b[id] = a[id] + b[id];
}




int main()
{
    rocblas_int n = 2<<10;
    float alpha = 10.0;

    float elapsedTime; 
    hipEvent_t start, stop;

    vector<float> hx(n);
    vector<float> hz(n);

    vector<float> hxx(n);
    vector<float> hzz(n);

    int numDevices = 0 ;
    hipGetDeviceCount(&numDevices);
    cout<<"No of devices="<<numDevices<<"\n";

    int deviceID = 1 ;
    //hipSetDevice(deviceID);
    hipGetDevice(&deviceID);
    cout<<"Device id ="<<deviceID<<"\n";

    float* dx;
    float* dy;

    float* dxx;
    float* dyy;

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // allocate memory on device
    hipMalloc(&dx, n * sizeof(float));
    hipMalloc(&dy, n * sizeof(float));


    hipMalloc(&dxx, n * sizeof(float));
    hipMalloc(&dyy, n * sizeof(float));


    // Initial Data on CPU,
    srand(1);
    for( int i = 0; i < n; ++i )
    {
        hx[i] =i;  //generate a integer number between [1, 10]
        hz[i] =i;  //generate a integer number between [1, 10]

        hxx[i]=i;
        hzz[i]=i;
    }

    // copy array from host memory to device memory
    hipMemcpy(dx, hx.data(), sizeof(float) * n, hipMemcpyHostToDevice);
    hipMemcpy(dy, hz.data(), sizeof(float) * n, hipMemcpyHostToDevice);

    // call rocBLAS function
    

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
 //rocblas_status status = rocblas_sscal(handle, n, &alpha, dx, 1);
    rocblas_status status=rocblas_saxpy(handle, n, &alpha, dx, 1, dy, 1);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);
    hipDeviceSynchronize();
    hipMemcpy(hx.data(), dx, sizeof(float) * n, hipMemcpyDeviceToHost);
    hipMemcpy(hz.data(), dy, sizeof(float) * n, hipMemcpyDeviceToHost);
    cout<<"Exec_time ="<<elapsedTime<<"\n";




    
    // check status for errors
    if(status == rocblas_status_success)
    {
        cout << "status == rocblas_status_success" << endl;
    }
    else
    {
        cout << "rocblas failure: status = " << status << endl;
    }




    // copy output from device memory to host memory




 ofstream MyFile("data/rocblas_saxpy.txt");


     for(int i=0; i < hz.size(); i++)
    {
     //   std::cout << hy.at(i) << ' ';
      MyFile<<hz.at(i) << ' '<<endl;
    
    }

MyFile.close();


hipFree(dx);
hipFree(dy);
hipFree(dxx);
hipFree(dyy);
rocblas_destroy_handle(handle);



return 0;
}