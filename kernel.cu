
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__
void addArrays(int* A, int* B, int* C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	C[i] = A[i] + B[i];
}

int main(void) {
	int N = 1024;

	int *A, *B, *C;

	// Crea los buffer, con Unified Memory, para los datos de entrada y salida
	cudaMallocManaged(&A, N * sizeof(int));
	cudaMallocManaged(&B, N * sizeof(int));
	cudaMallocManaged(&C, N * sizeof(int));

	// Inicializa los buffer del host con los valores de entrada
	for (int i = 0; i < N; i++) {
		A[i] = i; //0,1,2,...,1023
		B[i] = N - i; //1023,1022,...,0
	}

	// Ejecuta la kernel en la GPU (4 bloques * 256 hilos = 1024 elementos calculados)
	addArrays <<<4, 256>>> (A, B, C);

	// Espera que termine la kernel
	cudaDeviceSynchronize();

	// Presenta el resultado
	for (int i = 0; i < N; i++) {
		printf("Resultados %d: (%d + %d = %d)\n", i, A[i], B[i], C[i]);
	}

	// Libera los recursos
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

}
