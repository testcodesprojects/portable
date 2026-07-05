inline __attribute__((__always_inline__))
double stiles_sparse_ddot(int n, double *__restrict v, double *__restrict a, int *__restrict idx)
{
	// computes sum_i v[i] * a[idx[i]]
	
	if (n <= 32) {
		double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;			
		int unroll = 8;							
		int m = n & ~(unroll - 1);					
//#pragma omp simd reduction(+: s0, s1, s2, s3)
		for (int i = 0; i < m; i += unroll) {				
			s0 += v[i + 0] * a[idx[i + 0]];				
			s1 += v[i + 1] * a[idx[i + 1]];				
			s2 += v[i + 2] * a[idx[i + 2]];				
			s3 += v[i + 3] * a[idx[i + 3]];				

			s0 += v[i + 4] * a[idx[i + 4]];				
			s1 += v[i + 5] * a[idx[i + 5]];				
			s2 += v[i + 6] * a[idx[i + 6]];				
			s3 += v[i + 7] * a[idx[i + 7]];				
		}								
		for (int i = m; i < n; i++) {					
			s0 += v[i] * a[idx[i]];					
		}								
		return (s0 + s1) + (s2 + s3);
	} else {
		return cblas_ddoti(n, v, idx, a);
	}
}
