/*
 * simple code for computing the KL-divergence objective function and gradient
 * from "Neighbourhood Components Analysis" Goldberger et al, NIPS04
 *
 * charless fowlkes
 * fowlkes@cs.berkeley.edu
 * 2005-02-23
 *
 */

#include <string.h>
#include <math.h>
#include <mex.h>

void 
mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // check number of arguments
    if (nlhs < 2) {
        mexErrMsgTxt("Too few output arguments.");
    }
    if (nlhs >= 3) {
        mexErrMsgTxt("Too many output arguments.");
    }
    if (nrhs < 4) {
        mexErrMsgTxt("Too few input arguments.");
    }
    if (nrhs >= 5) {
        mexErrMsgTxt("Too many input arguments.");
    }

    // get arguments
    double* A = mxGetPr(prhs[0]);
    int ID = mxGetN(prhs[0]);
    int OD = mxGetM(prhs[0]);
    double* X = mxGetPr(prhs[1]);
    if (mxGetN(prhs[1]) != ID) { mexErrMsgTxt("data (X) dimension  does not match (A) input dimension"); }
    int N = mxGetM(prhs[1]);

    double* Y = mxGetPr(prhs[2]);
    int K = mxGetN(prhs[2]);
    if (mxGetM(prhs[2]) != N) { mexErrMsgTxt("different #of class labels (Y) and point coordinates (X)"); } 

    double* AXT = mxGetPr(prhs[3]);
    if (mxGetN(prhs[3]) != N) { mexErrMsgTxt("AX has wrong # colums"); } 
    if (mxGetM(prhs[3]) != OD) { mexErrMsgTxt("AX has wrong # rows"); } 

    printf("pts=%d ",N);
    printf("classes=%d ",K);
    printf("indim=%d ",ID);
    printf("outdim=%d \n",OD);

    ////// set up output arguments
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL); 
    //plhs[1] = mxCreateDoubleMatrix(1,ID*OD,mxREAL); 
    plhs[1] = mxCreateDoubleMatrix(ID,ID,mxREAL); 
    double* F = mxGetPr(plhs[0]);
    double* M = mxGetPr(plhs[1]);

    //compute exp(-D2)
    printf("compute exp\n");
    double* ED2 = new double[N*N];
    for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
      {
        double d2 = 0;
        for (int k = 0; k < OD; k++)
        {
          d2 = d2 + (AXT[i*OD+k] - AXT[j*OD+k])*(AXT[i*OD+k] - AXT[j*OD+k]) ;
        }
        ED2[i*N+j] = exp(-d2);
      }
    }

    //compute softmax function P 
    printf("compute softmax function\n");
    double* P = new double[N*N];
    for (int j = 0; j < N; j++)
    {
      for (int i = 0; i < N; i++)
      {
        if (i == j)
        {
          P[j*N+i] = 0;
        }
        else
        {
          double den = 0;
          for (int k = 0; k < N; k++)
          {
            if (k != i)
            {
              den = den + ED2[i*N+k];
            }
          }
          P[j*N+i] = ED2[j*N+i] / den;
        }
      }
    }

    //compute classification probability pi and total objective F
    printf("compute obj\n");
    double* pi = new double[N];
    F[0] = 0;
    for (int i = 0; i < N; i++)
    {
      int ci = -1;
      for (int k = 0; k < K; k++)
      {
        if (Y[k*N+i] != 0)
        {
          ci = k; 
        }
      }
      pi[i] = 0; //probability of drawing a neighbor in our same class
      for (int j = 0; j < N; j++)
      {
        if (Y[ci*N+j] != 0)
        {
          pi[i] = pi[i] + P[j*N+i];
        }
      }
      F[0] = F[0] + log(pi[i]);
    }

    //now compute the gradient
    printf("compute gradient\n");
    //double* M = new double[ID*ID];
    memset(M,0,ID*ID*sizeof(double));
    for (int i = 0; i < N; i++)
    {
      //add in first sum
      for (int k = 0; k < N; k++)
      {
        for (int m = 0; m < ID; m++)
        {
          for (int n = 0; n < ID; n++)
          {
            M[m*ID+n] = M[m*ID+n] + P[k*N+i]*(X[m*N+i] - X[m*N+k])*(X[n*N+i] - X[n*N+k]);
          }
        }
      }

      //subtract off second sum (only over class of point i)
      int ci = -1;
      for (int k = 0; k < K; k++)
      {
        if (Y[k*N+i] != 0)
        {
          ci = k; 
        }
      }
      for (int j = 0; j < N; j++)
      {
        if (Y[ci*N+j] != 0)
        {
          for (int m = 0; m < ID; m++)
          {
            for (int n = 0; n < ID; n++)
            {
              M[m*ID+n] = M[m*ID+n] - (1/pi[i])*P[j*N+i]*(X[m*N+i] - X[m*N+j])*(X[n*N+i] - X[n*N+j]);
            }
          }
        }
      }
    }
      
    delete[] ED2;
    delete[] P;
    delete[] pi;
}



