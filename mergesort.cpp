#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>

#include <iostream>
#include <algorithm>
// Add this head file OpenMP library for parallelism
#include <omp.h>

/**
 * Helper routine: check if array is sorted correctly by compared with the default sort
 */
bool isSorted(int ref[], int data[], const size_t size)
{
	std::sort(ref, ref + size);
	for (size_t idx = 0; idx < size; ++idx)
	{
		if (ref[idx] != data[idx])
		{
			return false;
		}
	}
	return true;
}

/**
 * sequential merge step without parallel by halves, just copy from original code!
 */
void MsMergeSequential(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin)
{
	long left = begin1;
	long right = begin2;

	long idx = outBegin;

	while (left < end1 && right < end2)
	{
		if (in[left] <= in[right])
		{
			out[idx] = in[left];
			left++;
		}
		else
		{
			out[idx] = in[right];
			right++;
		}
		idx++;
	}

	while (left < end1)
	{
		out[idx] = in[left];
		left++, idx++;
	}

	while (right < end2)
	{
		out[idx] = in[right];
		right++, idx++;
	}
}

/**
 * 1.Parallel merge algorithm by using double tasks for each merge step and moving towards the middle separately
 * 2.There is also a cutOffMerge threshold value to control if using the parallel version. When the sub array
 * is big enough we can use parallel merge for acceleration. If too small, parallelization by OpenMp tasks will
 * in turn increase the total overhead for managing.
 */
void MsMergeParallel(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin)
{
	// Divide the array into two halves and merge from different direction
	long left = begin1;
	long right = end2 - 1;
	long outLeft = outBegin;
	long outRight = outBegin + (end1 - begin1) + (end2 - begin2) - 1; // the is the last index of out array
	long mid = (outLeft + outRight) / 2;							  // Define the middle point for stopping

#pragma omp parallel // Creat parallel region
	{
#pragma omp single // use single thread multiple tasks
		{
// Task 1: merge arrays from left (smallest value) to middle point.
#pragma omp task shared(out, in)
			{
				long l = left;	  // start of left sub-array
				long r = begin2;  // start of right sub-array
				long o = outLeft; // start point of out-array
				while (l < end1 && r < end2 && o <= mid)
				{ // add another condition that stopping at the middle point when moving right
					if (in[l] <= in[r])
					{
						out[o++] = in[l++];
					}
					else
					{
						out[o++] = in[r++];
					}
				}
				// merge the remaining part elements in case one sub array finished all
				while (l < end1 && o <= mid)
				{
					out[o++] = in[l++];
				}
				while (r < end2 && o <= mid)
				{
					out[o++] = in[r++];
				}
			}

// Task 2: merge arrays from right (largest value) to middle point.
#pragma omp task shared(out, in)
			{
				long l = end1 - 1; // end of left sub-array
				long r = end2 - 1; // end of right sub-array
				long o = outRight; // end point of out-array
				while (l >= begin1 && r >= begin2 && o > mid)
				{ // add another condition that stopping at the middle point when moving left
					if (in[l] >= in[r])
					{
						out[o--] = in[l--];
					}
					else
					{
						out[o--] = in[r--];
					}
				}
				// merge the remaining part elements in case one sub array finished all
				while (l >= begin1 && o > mid)
				{
					out[o--] = in[l--];
				}
				while (r >= begin2 && o > mid)
				{
					out[o--] = in[r--];
				}
			}

#pragma omp taskwait // Wait to synchronize double tasks
		}
	}
}

/**
 * Parallel MergeSort with parallel by multiple tasks and cut-off standard
 */
void MsSequential(int *array, int *tmp, bool inplace, long begin, long end, int depth, long cutOffSort, long cutOffMerge)
{
	if ((end - begin) <= cutOffSort)
	{
		// If below cut-off (which means the sub-array is very small), perform a default sequential sort instead, more suitable
		if (inplace)
		{
			std::sort(array + begin, array + end);
		}
		else
		{
			std::copy(array + begin, array + end, tmp + begin);
			std::sort(tmp + begin, tmp + end);
		}
		return;
	}

	const long half = (begin + end) / 2;

	// Parallel the recursive calls by shared tasks when depth allows parallelism
	if (depth > 0)
	{
// Paralleled share tasks
#pragma omp task shared(array, tmp)
		MsSequential(array, tmp, !inplace, begin, half, depth - 1, cutOffSort, cutOffMerge); // depth - 1
// Paralleled share tasks
#pragma omp task shared(array, tmp)
		MsSequential(array, tmp, !inplace, half, end, depth - 1, cutOffSort, cutOffMerge); // depth - 1

#pragma omp taskwait // Wait to synchronize tasks
	}
	else
	{
		// Do original sequential algorithm once reaching the depth limitation
		// its value will be unchanged anymore, which means alway zero
		MsSequential(array, tmp, !inplace, begin, half, depth, cutOffSort, cutOffMerge);
		MsSequential(array, tmp, !inplace, half, end, depth, cutOffSort, cutOffMerge);
	}

	// Merge the results according the cutOffMerge threshold value, we only merge them in parallel when big enough
	bool mergeByParallel = (end - begin) >= cutOffMerge;
	if (mergeByParallel)
	{
		// Parallel: if the arrays to merge is large enough which means we can benefit from parallel
		if (inplace)
		{
			MsMergeParallel(array, tmp, begin, half, half, end, begin);
		}
		else
		{
			MsMergeParallel(tmp, array, begin, half, half, end, begin);
		}
	}
	else
	{
		// Sequential: if the arrays are too small to parallel them (overload for too many tasks to manage)
		if (inplace)
		{
			MsMergeSequential(array, tmp, begin, half, half, end, begin);
		}
		else
		{
			MsMergeSequential(tmp, array, begin, half, half, end, begin);
		}
	}
}

/**
 * Parallel MergeSort - creates the parallel region here
 */
void MsSerial(int *array, int *tmp, const size_t size, int maxDepth, long cutOffSort, long cutOffMerge)
{
	/*
	We can set the desired number of threads if need to see their influence, default is using all
	*/
	// int num_threads = 2;
	// omp_set_num_threads(num_threads);

// Create pool of threads and start with one of them.
#pragma omp parallel // parallel region
	{
#pragma omp single // Ensure only one thread starts tasks
		{
			MsSequential(array, tmp, true, 0, size, maxDepth, cutOffSort, cutOffMerge);
		}
	}
}

/**
 * Program entry point
 */
int main(int argc, char *argv[])
{
	// Variables to measure the elapsed time
	struct timeval t1, t2;
	double etime;

	// Expect one command line argument: array size
	if (argc != 2)
	{
		printf("Usage: MergeSort.exe <array size> \n");
		printf("\n");
		return EXIT_FAILURE;
	}
	else
	{
		const size_t stSize = strtol(argv[1], NULL, 10);
		int *data = (int *)malloc(stSize * sizeof(int));
		int *tmp = (int *)malloc(stSize * sizeof(int));
		int *ref = (int *)malloc(stSize * sizeof(int));

		printf("Initialization...\n");

		srand(95);
		for (size_t idx = 0; idx < stSize; ++idx)
		{
			data[idx] = (int)(stSize * (double(rand()) / RAND_MAX));
		}
		std::copy(data, data + stSize, ref);

		double dSize = (stSize * sizeof(int)) / 1024 / 1024;
		printf("Sorting %zu elements of type int (%f MiB)...\n", stSize, dSize);

		// Start measuring time
		gettimeofday(&t1, NULL);

		// Example parameters: maxDepth = 4, cutOffSort = 16, cutoffMerge = 2048, input datasize = 1000000000
		// 1. If depth is too large, high level of parallel but also high tasks pressure, worse performance;
		// -  If depth is too low we haven't fully utilized the whole threads for acceleration.
		// 2. High cutOffSort, to big sub array to sort sequentially, low speed;
		// -  Low cutOffSort too many small arrays to deal with
		// 3. High cutOffMerge like 8192, arrays to merge at lower levels can't benefit from this parallelism;
		// -  Low cutOffSort, too many small paralleled tasks to manage which leading bad performance.
		const long depth = 4;		// suitable value according to my laptop's threads number which is 16
		const long cutOffSort = 16; // between 16-32 is good for sequential algorithm to deal with
		// shouldn't be too small to benefit from parallel without too much tasks control overhead
		// (larger than cutOffSort), because complexity is just O(n), whereas the default sort is O(n log_n)
		const long cutOffMerge = 2048;
		// Perform the mergesort algorithm
		MsSerial(data, tmp, stSize, depth, cutOffSort, cutOffMerge);

		gettimeofday(&t2, NULL);

		// Calculate elapsed time
		etime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
		etime = etime / 1000;

		printf("done, took %f sec. Verification...", etime);
		if (isSorted(ref, data, stSize))
		{
			printf(" successful.\n");
			printf("With parameter depth: %ld, cutOffSort: %ld, cutOffMerge: %ld\n", depth, cutOffSort, cutOffMerge);
		}
		else
		{
			printf(" FAILED.\n");
		}

		free(data);
		free(tmp);
		free(ref);
	}

	return EXIT_SUCCESS;
}
