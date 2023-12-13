#include "stdlib.h"
#include "stdio.h"
#include "sys/time.h"

const double G = 6.67408e-11; // gravitational constant in m^3 kg^-1 s^-2
const double L = 1.0; // box width in m
const double epsilon = 1e-3*L; // softening parameter in m
const double softening2 = epsilon*epsilon;

struct Vector3d
{
    double x;
    double y;
    double z;
};

__global__ void update_acceleration(Vector3d* acc, Vector3d* pos, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i != j)
            {
                double dx = pos[j].x - pos[i].x;
                double dy = pos[j].y - pos[i].y;
                double dz = pos[j].z - pos[i].z;
                double r2 = dx*dx + dy*dy + dz*dz;
                double denominator = (r2 + softening2)*sqrt(r2);
                acc[i].x += G*dx / denominator;
                acc[i].y += G*dy / denominator;
                acc[i].z += G*dz / denominator;
            }
        }
    }
}

__global__ void update_state(Vector3d* pos, Vector3d* vel, Vector3d* acc, double dt, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        pos[i].x += vel[i].x*dt;
        pos[i].y += vel[i].y*dt;
        pos[i].z += vel[i].z*dt;
        vel[i].x += acc[i].x*dt;
        vel[i].y += acc[i].y*dt;
        vel[i].z += acc[i].z*dt;
    }
}

struct OutputEntry
{
    double t;
    size_t index;
    Vector3d pos;
    Vector3d vel;
};

__global__ void output_states(OutputEntry* out_states, Vector3d* pos, Vector3d* vel, int n,
    int step_index, double t)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int step_offset = step_index*n;
    if (i < n)
    {
        out_states[step_offset + i].t = t;
        out_states[step_offset + i].index = (size_t)i;
        out_states[step_offset + i].pos.x = pos[i].x;
        out_states[step_offset + i].pos.y = pos[i].y;
        out_states[step_offset + i].pos.z = pos[i].z;
        out_states[step_offset + i].vel.x = vel[i].x;
        out_states[step_offset + i].vel.y = vel[i].y;
        out_states[step_offset + i].vel.z = vel[i].z;
    }
}

//__global__ void print_output_states(OutputEntry* out_states, int n, int n_steps)
//{
//    int i = blockIdx.x*blockDim.x + threadIdx.x;
//    if (i < n)
//    {
//        for (int j = 0; j < n_steps; ++j)
//        {
//            OutputEntry* entry = &out_states[j*n + i];
//            printf("i = %d, t = %.16e, index = %lu, pos.x = %.16e, pos.y = %.16e, pos.z = %.16e, vel.x = %.16e, vel.y = %.16e, vel.z = %.16e\n",
//                i, entry->t, entry->index,
//                entry->pos.x, entry->pos.y, entry->pos.z,
//                entry->vel.x, entry->vel.y, entry->vel.z);
//        }
//    }
//}

double cuda_euler_loop(Vector3d* pos, Vector3d* vel, int n, double dt, double max_time,
    OutputEntry** out_states, size_t* out_nstates)
{
    Vector3d* d_pos;
    Vector3d* d_vel;
    Vector3d* d_acc;
    cudaMalloc(&d_pos, n*sizeof(Vector3d));
    cudaMalloc(&d_vel, n*sizeof(Vector3d));
    cudaMalloc(&d_acc, n*sizeof(Vector3d));
    cudaMemcpy(d_pos, pos, n*sizeof(Vector3d), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, vel, n*sizeof(Vector3d), cudaMemcpyHostToDevice);
    int n_reserve_steps = (int)(max_time / dt) * 2;
    OutputEntry* d_out_states;
    cudaMalloc(&d_out_states, n_reserve_steps*n*sizeof(OutputEntry));
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    int n_steps = 0;
    struct timeval start;
    gettimeofday(&start, NULL);
    for (double t = 0.0; t < max_time; t += dt)
    {
        update_acceleration<<<num_blocks, block_size>>>(d_acc, d_pos, n);
        update_state<<<num_blocks, block_size>>>(d_pos, d_vel, d_acc, dt, n);

        output_states<<<num_blocks, block_size>>>(d_out_states, d_pos, d_vel, n, n_steps, t);
        ++n_steps;
    }
    struct timeval end;
    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + 1e-6*(end.tv_usec - start.tv_usec);
    OutputEntry* out_states_host = (OutputEntry*)malloc(n_steps*n*sizeof(OutputEntry));
    cudaMemcpy(out_states_host, d_out_states, n_steps*n*sizeof(OutputEntry), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos, d_pos, n*sizeof(Vector3d), cudaMemcpyDeviceToHost);
    cudaMemcpy(vel, d_vel, n*sizeof(Vector3d), cudaMemcpyDeviceToHost);
    cudaFree(d_out_states);
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc);

    *out_states = out_states_host;
    *out_nstates = n_steps*n;

    return elapsed_time;
}

void output_results(const char* filename, OutputEntry* out_states, size_t nstates)
{
    FILE* fp = fopen(filename, "w");
    for (size_t i = 0; i < nstates; ++i)
    {
        OutputEntry* entry = &out_states[i];
        fprintf(fp, "%lf,%lu,%lf,%lf,%lf,%lf,%lf,%lf\n",
            entry->t, entry->index,
            entry->pos.x, entry->pos.y, entry->pos.z,
            entry->vel.x, entry->vel.y, entry->vel.z);
    }
    fclose(fp);
}

int N = 1 << 3;
int output_to_file = 0;

int main(int argc, char** argv)
{
    if (argc > 1) { N = atoi(argv[1]); }
    if (argc > 2) { output_to_file = atoi(argv[2]); }

    // initialize state
    Vector3d* pos = (Vector3d*)malloc(N*sizeof(Vector3d));
    Vector3d* vel = (Vector3d*)malloc(N*sizeof(Vector3d));
    for (int i = 0; i < N; ++i)
    {
        pos[i].x = L*rand() / RAND_MAX;
        pos[i].y = L*rand() / RAND_MAX;
        pos[i].z = L*rand() / RAND_MAX;
        vel[i].x = 0.0;
        vel[i].y = 0.0;
        vel[i].z = 0.0;
    }

    // run simulation
    double max_time = sqrt(L*L*L / (G * N));
    double dt = 1e-3*max_time;
    OutputEntry* out_states;
    size_t nstates;
    struct timeval start;
    gettimeofday(&start, NULL);
    double time_on_gpu = cuda_euler_loop(pos, vel, N, dt, max_time, &out_states, &nstates);
    struct timeval end;
    gettimeofday(&end, NULL);
    double total_seconds = (end.tv_sec - start.tv_sec) + 1e-6*(end.tv_usec - start.tv_usec);
    printf("N = %d, time on GPU = %lfs, total time = %lfs\n", N, time_on_gpu, total_seconds);

    // output results
    if (output_to_file)
    {
        char filename[256];
        sprintf(filename, "cudaoutput_%d_.csv", N);
        output_results(filename, out_states, nstates);
    }
    free(out_states);

    // free state
    free(pos);
    free(vel);

    return 0;
}