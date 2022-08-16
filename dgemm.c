/*
		use gcc -lpthread -fopenmp -mavx
idea from P&H's COD book, MIPS 5th edition.However some explanations on the book seems to be wrong...
I strongly believe it must be B * A rather than A * B!!!! And the comment of the matrix index in the book also fails

Let's first consider the question: What are the datas that will affect the value of cij?
The answer is easy: all the datas in the row i of A and the column j of B

Take b11 as an further observation.
Since `b11` will affect both c11 c12 c13 c14..., we could copy it 4 times as a pack and multiply it to a11, a12, a13, a14 them all at one time
(that's why __mm256_broadcast_sd is applied to b,which also indicates why it should be B * A)

we use the rule of row combination for calculation: row k of C is equal to some row combinations of B

          K ---->		     I  ---->			 I  ---->
   J	| b11 b12 b13  |        K   | a11 a12 a13 |         J	|b11 * row1 of A + b12 * row2 of A + b13 * row 3 of A |
   |	| b21 b22 b23  |    *   |   | a21 a22 a23 |	=   |	|...						      |
   V	| b31 b32 b33  |        V   | a31 a32 a33 |  	    V	|...		  				      |		
	       B		   	   A                       			 C
	     K + NJ		       I + 4X + NK				    I + 4X + NJ
	  
These are the datas that will be used during a K-loop:

	 BBBB   -------------------------- ----------------------------------------> 	 
	 \   /				  |					   |
	  \ / 				  |					  c1   c2   c3   c4   
	  |B B B B ...	|		|AAAA AAAA AAAA AAAA	...	|	|CCCC CCCC CCCC CCCC ...	|	c on these location will be affected in this loop
	  |		|		|				|	|		    		|
	  |		|		|				|	|		    		|
	 	 B				A			  			C
After K++:
	   BBBB
 	   \  /
 	    \/
	  |B B B B ...  |		|			...	|	|CCCC CCCC CCCC CCCC ...	|	we are still working on the same cs, since b is still on the old row
	  |             |		|AAAA AAAA AAAA AAAA    ...	|	|				|	
	  |             |	 	|				|	|		    		|
	 	 B				A			  			C
*/
#include <x86intrin.h>
#include<stdio.h>
#include<time.h>
#define UNROLL (4)		//loop unroll. reduce loop times so fewer branch prediction failures can happen
#define BLOCKSIZE 32		//keep the data currently processing in cache
#define size 512
void do_block (int n, int si, int sj, int sk,double *A, double *B, double *C)
{
    for ( int i = si; i < si+BLOCKSIZE; i+=UNROLL*4 )
	for ( int j = sj; j < sj+BLOCKSIZE; j++ ) 
	{
	    __m256d c[4];
	    for ( int x = 0; x < UNROLL; x++ )
		c[x] = _mm256_load_pd(C+i+x*4+j*n);	
	    for( int k = sk; k < sk+BLOCKSIZE; k++ )		
	    {							
		__m256d b = _mm256_broadcast_sd(B+k+j*n);		/*broadcast: a -> aaaa*/
		for (int x = 0; x < UNROLL; x++)
		    c[x] = _mm256_add_pd(c[x], _mm256_mul_pd(_mm256_load_pd(A+n*k+x*4+i), b));		
	    }
	for ( int x = 0; x < UNROLL; x++ )
	    _mm256_store_pd(C+i+x*4+j*n, c[x]);		
	}
}
void dgemm (int n, double* A, double* B, double* C)
{
    #pragma omp parallel for
	for ( int sj = 0; sj < n; sj += BLOCKSIZE )
	    for ( int si = 0; si < n; si += BLOCKSIZE )
		for ( int sk = 0; sk < n; sk += BLOCKSIZE )
		    do_block(n, si, sj, sk, A, B, C);
}
int main(){
	
	double a[size][size] __attribute__ ((aligned (32))),		//vmovapd needs the data to be aligned
	       b[size][size] __attribute__ ((aligned (32))),
	       c[size][size] __attribute__ ((aligned (32)));
	
	for(int i=0;i<size;i++)
	    for(int j=0;j<size;j++)
	    {
	        a[i][j]=i*size+j+1;					//initialize the values, with some simple values
	        b[i][j]=a[i][j];
	        c[i][j]=0;
	    }
	double time=0;
	for(int i=0;i<20;i++)
	{
		time_t start = clock();
		dgemm(size,a,b,c);
		time_t end = clock();
		double temp = (double)(end - start);
		printf("%.4f\n",temp);
		time += temp;
	}
	printf("average: %.4f\n",time/20);
}
