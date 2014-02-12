#include <math.h>
#include "mex.h"

// small value, used to avoid division by zero
#define eps 0.0001

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

// main function:
// takes a double color image and a bin size 
// returns HOG features
mxArray *process(const mxArray *mximage, const mxArray *mxsbin) {
  double *im = (double *)mxGetPr(mximage);
  const int *dims = mxGetDimensions(mximage);
  if (mxGetNumberOfDimensions(mximage) != 3 ||
      dims[2] != 3 ||
      mxGetClassID(mximage) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Invalid input");

  int sbin = (int)mxGetScalar(mxsbin);

  // memory for caching orientation histograms & their norms
  // blocks=size+2 if use default settings in train.m
  int blocks[2];
  blocks[0] = (int)((double)dims[0]/(double)sbin + 0.5);
  blocks[1] = (int)((double)dims[1]/(double)sbin + 0.5);
  // 18 orientations
  double *color_sum = (double *)mxCalloc(blocks[0]*blocks[1]*3, sizeof(double));
  double *color_sumsq = (double *)mxCalloc(blocks[0]*blocks[1]*3, sizeof(double));
  double *color_weight = (double *)mxCalloc(blocks[0]*blocks[1], sizeof(double));
  int out[3];
  out[0] = blocks[0];
  out[1] = blocks[1];
  out[2] = 6;
  mxArray *mxfeat = mxCreateNumericArray(3, out, mxDOUBLE_CLASS, mxREAL);
  double *feat = (double *)mxGetPr(mxfeat);
  
  for (int i=0; i<blocks[1]; i++)
  {
      for (int j=0; j<blocks[0]; j++)
      {
            int idx = i*blocks[0] + j;
            color_sum[idx] = 0;
            color_sumsq[idx] = 0;
            color_weight[idx] = 0;
            
            idx += blocks[0]*blocks[1];
            color_sum[idx] = 0;
            color_sumsq[idx] = 0;
            
            idx += blocks[0]*blocks[1];
            color_sum[idx] = 0;
            color_sumsq[idx] = 0;      
      }
  }
  
  double *p_r = im, *p_g = im + dims[0]*dims[1], *p_b = im + 2*dims[0]*dims[1];
  double color[3];
  
  for (int x = 0; x < dims[1]; x++) {
      for (int y = 0; y < dims[0]; y++) {
          double color[3];
          color[0] = (*p_r)/255.0;
          color[1] = (*p_g)/255.0;
          color[2] = (*p_b)/255.0;
          
          p_r ++;
          p_g ++;
          p_b ++;
          
          // add to 4 histograms around pixel using linear interpolation between blocks
          double xp = ((double)x+0.5)/(double)sbin - 0.5;
          double yp = ((double)y+0.5)/(double)sbin - 0.5;
          int ixp = (int)floor(xp);
          int iyp = (int)floor(yp);
          double vx0 = xp-ixp;
          double vy0 = yp-iyp;
          double vx1 = 1.0-vx0;
          double vy1 = 1.0-vy0;
          
          if (ixp >= 0 && iyp >= 0) {
              int blk_idx_r = ixp*blocks[0] + iyp;
              int blk_idx_g = ixp*blocks[0] + iyp + blocks[0]*blocks[1];
              int blk_idx_b = ixp*blocks[0] + iyp + 2*blocks[0]*blocks[1];
              *(color_sum +  blk_idx_r) += vx1*vy1*color[0];
              *(color_sum +  blk_idx_g) += vx1*vy1*color[1];
              *(color_sum +  blk_idx_b) += vx1*vy1*color[2];
              *(color_sumsq +  blk_idx_r) += vx1*vy1*color[0]*color[0];
              *(color_sumsq +  blk_idx_g) += vx1*vy1*color[1]*color[1];
              *(color_sumsq +  blk_idx_b) += vx1*vy1*color[2]*color[2];
              *(color_weight +  blk_idx_r) += vx1*vy1;
          }
          
          if (ixp+1 < blocks[1] && iyp >= 0) {
              int blk_idx_r = (ixp+1)*blocks[0] + iyp;
              int blk_idx_g = (ixp+1)*blocks[0] + iyp + blocks[0]*blocks[1];
              int blk_idx_b = (ixp+1)*blocks[0] + iyp + 2*blocks[0]*blocks[1];
              *(color_sum +  blk_idx_r) += vx0*vy1*color[0];
              *(color_sum +  blk_idx_g) += vx0*vy1*color[1];
              *(color_sum +  blk_idx_b) += vx0*vy1*color[2];
              *(color_sumsq +  blk_idx_r) += vx0*vy1*color[0]*color[0];
              *(color_sumsq +  blk_idx_g) += vx0*vy1*color[1]*color[1];
              *(color_sumsq +  blk_idx_b) += vx0*vy1*color[2]*color[2];
              *(color_weight +  blk_idx_r) += vx0*vy1;
          }
          
          if (ixp >= 0 && iyp+1 < blocks[0]) {
              int blk_idx_r = ixp*blocks[0] + iyp+1;
              int blk_idx_g = ixp*blocks[0] + iyp+1 + blocks[0]*blocks[1];
              int blk_idx_b = ixp*blocks[0] + iyp+1 + 2*blocks[0]*blocks[1];
              *(color_sum +  blk_idx_r) += vx1*vy0*color[0];
              *(color_sum +  blk_idx_g) += vx1*vy0*color[1];
              *(color_sum +  blk_idx_b) += vx1*vy0*color[2];
              *(color_sumsq +  blk_idx_r) += vx1*vy0*color[0]*color[0];
              *(color_sumsq +  blk_idx_g) += vx1*vy0*color[1]*color[1];
              *(color_sumsq +  blk_idx_b) += vx1*vy0*color[2]*color[2];
              *(color_weight +  blk_idx_r) += vx1*vy0;
          }
          
          if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
              int blk_idx_r = (ixp+1)*blocks[0] + iyp+1;
              int blk_idx_g = (ixp+1)*blocks[0] + iyp+1 + blocks[0]*blocks[1];
              int blk_idx_b = (ixp+1)*blocks[0] + iyp+1 + 2*blocks[0]*blocks[1];
              *(color_sum +  blk_idx_r) += vx0*vy0*color[0];
              *(color_sum +  blk_idx_g) += vx0*vy0*color[1];
              *(color_sum +  blk_idx_b) += vx0*vy0*color[2];
              *(color_sumsq +  blk_idx_r) += vx0*vy0*color[0]*color[0];
              *(color_sumsq +  blk_idx_g) += vx0*vy0*color[1]*color[1];
              *(color_sumsq +  blk_idx_b) += vx0*vy0*color[2]*color[2];
              *(color_weight +  blk_idx_r) += vx0*vy0;
          }
      }
  }
  
  int idx = 0, idx2 = blocks[0]*blocks[1], idx3 = 2*blocks[0]*blocks[1];
  for (int i = 0; i < blocks[1]; i++) {
      for (int j = 0; j < blocks[0]; j++) {
            double cmean = color_sum[idx]/(color_weight[idx] + eps);
            double cmeansq = color_sumsq[idx]/(color_weight[idx] + eps);
            feat[idx] = cmean;
            feat[idx+3*blocks[0]*blocks[1]] = sqrt(cmeansq - cmean*cmean);
            
            cmean = color_sum[idx2]/(color_weight[idx] + eps);
            cmeansq = color_sumsq[idx2]/(color_weight[idx] + eps);
            feat[idx2] = cmean;
            feat[idx2+3*blocks[0]*blocks[1]] = sqrt(cmeansq - cmean*cmean);
            
            cmean = color_sum[idx3]/(color_weight[idx] + eps);
            cmeansq = color_sumsq[idx3]/(color_weight[idx] + eps);
            feat[idx3] = cmean;
            feat[idx3+3*blocks[0]*blocks[1]] = sqrt(cmeansq - cmean*cmean);  
            idx ++;
            idx2 ++;
            idx3 ++;
      }
  }
  
  mxFree(color_sum);
  mxFree(color_sumsq);
  mxFree(color_weight);
  return mxfeat;
}

// matlab entry point
// F = features(image, bin)
// image should be color with double values
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2)
        mexErrMsgTxt("Wrong number of inputs");
    if (nlhs != 1)
        mexErrMsgTxt("Wrong number of outputs");
    plhs[0] = process(prhs[0], prhs[1]);
}
