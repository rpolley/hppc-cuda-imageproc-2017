#include "book.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#define CAMERA_PIXEL_SCALE 0.0000617 //meters at 1cm distance
#define CAMERA_DEPTH_UNIT 4.5 //cm

#define swap(a,b) a^=b;b^=a;a^=b
#define xyzlt(a,b) ((a[0]<b[0])||((a[0]==b[0])&&(a[1]<b[1]))||((a[0]==b[0])&&(a[1]==b[1])&&(a[2]<b[2])))



/*
*** bitmap reading code courtesy of @BeholderOf from http://www.vbforums.com/showthread.php?261522-C-C-Loading-Bitmap-Files-%28Manually%29
*** with modifications by @ollo from https://stackoverflow.com/questions/14279242/read-bitmap-file-into-structure
*/
#pragma pack(push, 1)

typedef struct tagBITMAPFILEHEADER
{
    uint16_t bfType;  //specifies the file type
    uint32_t bfSize;  //specifies the size in bytes of the bitmap file
    uint16_t bfReserved1;  //reserved; must be 0
    uint16_t bfReserved2;  //reserved; must be 0
    uint32_t bfOffBits;  //species the offset in bytes from the bitmapfileheader to the bitmap bits
}BITMAPFILEHEADER;

#pragma pack(pop)
#pragma pack(push, 1)

typedef struct tagBITMAPINFOHEADER
{
    uint32_t biSize;  //specifies the number of bytes required by the struct
    int32_t biWidth;  //specifies width in pixels
    int32_t biHeight;  //species height in pixels
    uint16_t biPlanes; //specifies the number of color planes, must be 1
    uint16_t biBitCount; //specifies the number of bit per pixel
    uint32_t biCompression;//spcifies the type of compression
    uint32_t biSizeImage;  //size of image in bytes
    int32_t biXPelsPerMeter;  //number of pixels per meter in x axis
    int32_t biYPelsPerMeter;  //number of pixels per meter in y axis
    uint32_t biClrUsed;  //number of colors used by th ebitmap
    uint32_t biClrImportant;  //number of colors that are important
}BITMAPINFOHEADER;

#pragma pack(pop)
unsigned char *LoadBitmapFile(char *filename, BITMAPINFOHEADER *bitmapInfoHeader)
{
    FILE *filePtr; //our file pointer
    BITMAPFILEHEADER bitmapFileHeader; //our bitmap file header
    unsigned char *bitmapImage;  //store image data
    int imageIdx=0;  //image index counter
    unsigned char tempRGB;  //our swap variable

    //open filename in read binary mode
    filePtr = fopen(filename,"rb");
    if (filePtr == NULL)
        return NULL;

    //read the bitmap file header
    fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER),1,filePtr);

    //verify that this is a bmp file by check bitmap id
    if (bitmapFileHeader.bfType !=0x4D42)
    {
        fclose(filePtr);
        return NULL;
    }

    //read the bitmap info header
    fread(bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,filePtr); // small edit. forgot to add the closing bracket at sizeof

    //move file point to the begging of bitmap data
    fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);

    //allocate enough memory for the bitmap image data
    bitmapImage = (unsigned char*)malloc(bitmapInfoHeader->biSizeImage);

    //verify memory allocation
    if (!bitmapImage)
    {
        free(bitmapImage);
        fclose(filePtr);
        return NULL;
    }

    //read in the bitmap image data
    fread(bitmapImage,bitmapInfoHeader->biSizeImage,1,filePtr);

    //make sure bitmap image data was read
    if (bitmapImage == NULL)
    {
        fclose(filePtr);
        return NULL;
    }

    //swap the r and b values to get RGB (bitmap is BGR)
    for (imageIdx = 0;imageIdx < bitmapInfoHeader->biSizeImage;imageIdx+=3) // fixed semicolon
    {
        tempRGB = bitmapImage[imageIdx];
        bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
        bitmapImage[imageIdx + 2] = tempRGB;
    }

    //close file and return bitmap iamge data
    fclose(filePtr);
    return bitmapImage;
}
/*
*** end of external code
*/



struct rawImageData{
    int32_t width;
    int32_t height;
    unsigned char* image_data;
};

#define data(n,dim)  (*(&(n->x)+dim))
#define idx(p2,ix,dim) *((float*)(p2+ix+dim))
#define idxa(p2,ix,dim) ((float*)(p2+ix+dim))
#define idxn(n,ix,dim) *((float*)(((kdNode**) (&n+ix))+3)+dim)
#define idxf(f,ix1,ix2,w) *(f+ix1*w+ix2)
#define idxfa(f,ix1,ix2,w) (f+ix1*w+ix2)
struct point3D{
    float x;
    float y;
    float z;
};

struct kdNode{
    kdNode* parent;
    kdNode* left;
    kdNode* right;
    float x;
    float y;
    float z;
};

struct superArray{
    int length;
    int width;
    int height;
    float* data;
};

#define indexSuperArray(a,i, j,  k) *(a->data+i*a->width*a->height+j*a->height+k)

__host__ __device__ superArray* allocSuperArray(int length, int width, int height){
  superArray* res = (superArray*)malloc(sizeof(int)*3+sizeof(float*));
  res->length = length;
  res->width = width;
  res->height = height;
  res->data = (float*) malloc(sizeof(float)*length*width*height);
  return res;
}

size_t rawImageDataSize(int, int);
void  loadImage(char* fname, rawImageData** d){
    BITMAPINFOHEADER bitmapInfoHeader;
    unsigned char* temp;
    temp = LoadBitmapFile(fname,&bitmapInfoHeader);
    int width = bitmapInfoHeader.biWidth;
    int height = bitmapInfoHeader.biHeight;
    *d = (rawImageData*) malloc(sizeof(int)*2+sizeof(char*));
    (**d).width = width;
    (**d).height = height;
    (**d).image_data = (unsigned char*)malloc(width*height*sizeof(char));
    for(int i = 0; i<width*height; i++){
       (**d).image_data[i] = temp[2*i];
    }

}

size_t rawImageDataSize(int width, int height){
    size_t size = 2*sizeof(int)+width*height*sizeof(unsigned char);
    return size;
}


#define p3idx(i,dim) 3*i+dim
__global__ void get3DPoints(int* width, int* height, unsigned char* image_data, point3D* p){
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    int dims_prod = (*width)*(*height);
    int elems_per_thread = dims_prod/(blockDim.x*gridDim.x);
    int base = tid*elems_per_thread;
    int remainder = dims_prod%(blockDim.x*gridDim.x);
    int end = base+elems_per_thread;
    for(int i = base; i<end; i+=1){
        //use reverse raytracing to figure out the location of the point in 3D space
        unsigned char h = (image_data[i]);
	
        int x = i/(*width);
        int y = i-x*(*height);
        float x_meters = x * CAMERA_PIXEL_SCALE;
        float y_meters = y * CAMERA_PIXEL_SCALE;
        float h_camera = sqrt(0.0001+x_meters*x_meters+y_meters*y_meters);
        float h_real = h * CAMERA_DEPTH_UNIT;
        float scale = h_real/h_camera;
        idx(p,i,0) = x_meters * scale;
        idx(p,i,1) = y_meters * scale;
        idx(p,i,2) = 0.01 * scale;
    }
    
}

__global__ void doBatchTransformation(point3D* a, int length, float* m, point3D* b){
    int tid = threadIdx.x+blockIdx.x+blockDim.x;
    int elems_per_thread =length/(blockDim.x*gridDim.x);
    int base = tid*elems_per_thread;
    int remainder = length%(blockDim.x*gridDim.x);
    if(tid<remainder){
        elems_per_thread++;
        base+=tid;
    }
    else{
        base+=remainder;
    }
    int end = base+elems_per_thread;
    for(int i = base; i<end; i++){
        float x = idx(a,i,0);
        float y = idx(a,i,1);
        float z = idx(a,i,2);
	idx(b,i,0) =  x*idxf(m,0,0,4)+y*idxf(m,0,1,4)+z*idxf(m,0,2,4)+idxf(m,0,3,4);
        idx(b,i,1) =  x*idxf(m,1,0,4)+y*idxf(m,1,1,4)+z*idxf(m,1,2,4)+idxf(m,1,3,4);
        idx(b,i,2) =  x*idxf(m,2,0,4)+y*idxf(m,2,1,4)+z*idxf(m,2,2,4)+idxf(m,2,3,4);
        float w = x*idxf(m,3,0,4)+y*idxf(m,3,1,4)+z*idxf(m,3,2,4)+idxf(m,3,3,4);
        idx(b,i,0) /= w;
        idx(b,i,1) /= w;
        idx(b,i,2) /= w;
    }
}


__global__ void getMean(point3D* a, int* length, float* sum){
    int tid = threadIdx.x+blockIdx.x+blockDim.x;
    int elems_per_thread = (length[blockIdx.x])/(gridDim.x);
    if(elems_per_thread < 2){
      return;
    }
    int base = 0;
    for(int i = 0; i<blockIdx.x; i++){
      base+=length[blockIdx.x];
    }
    int end = base+elems_per_thread;
    float local_sum[3] = {0,0,0};
    for(int i = base; i<end; i++){
        for(int dim = 0; dim<3; dim++){
            local_sum[dim] += idx(a,i,dim);
	}
    }
    if(threadIdx.x==0){
      idx(sum,blockIdx.x,0) = 0;
      idx(sum,blockIdx.x,1) = 0;
      idx(sum,blockIdx.x,2) = 0;

        for(int dim = 0; dim<3; dim++){
	  idx(sum,blockIdx.x,dim) += local_sum[dim];
        }
	for(int dim = 0; dim<3; dim++){
	  idx(sum,blockIdx.x,dim) /= length[blockIdx.x];
        }
    }
    
  }

__global__ void doPartitionStep(point3D* a, int* length, int* dim, float* med, point3D* b, int *div){
    int tid = threadIdx.x+blockIdx.x+blockDim.x;
    int elems_per_thread = (int) (length[blockIdx.x])/(gridDim.x);
    if(elems_per_thread<2){
      div[blockIdx.x] = -1;
      length[2*blockIdx.x] = 0;
      length[2*blockIdx.x+1] =  0;
      return;
    }
    int base = 0;
    for(int i = 0; i<blockIdx.x; i++){
      base+=length[i];
    }
    int end = base+elems_per_thread;
    
    //__shared__ int* lsizes;
    //__shared__ int* rsizes;
    unsigned int curr;
    __shared__  unsigned int* r;
    __shared__  unsigned int* l;
    if(threadIdx.x==0){
        r = (unsigned int*)malloc(sizeof(int));
        l = (unsigned int*)malloc(sizeof(int));
        *r = end-1;
        *l = base;
    }
    for(int i = base; i<end; i++){
        if(idx(a,i,*dim)>*med){
	  curr = atomicAdd(l,1U);            
        }
        else{
	  curr = atomicSub(r,1U);

        }
        b[curr] = a[i];
    }
    if(threadIdx.x==0){
       div[blockIdx.x] = *r;
       length[blockIdx.x*2+1] = *l;
       length[blockIdx.x*2] = length[blockIdx.x]-(*l);
       free(r);
       free(l);
    }
    
}
void finishKDTree( int* lengths, int llength, kdNode* root, kdNode* tree){
  int loc = 0;
  for(int i = 0; i<llength; i++){
    if(lengths[i]<=1){
      loc+=lengths[i];
      continue;
    }
    for(int j = 0; j<lengths[i]; j++){
      kdNode* point = tree+loc+j;
      kdNode* curr = root;
      int dim = 0;
      while(true){
	if(data(curr,dim)>data(point,dim)){
	  if(curr->left!=NULL){
	    curr = curr->left;
	  }
	  else{
	    curr->right = point;
	    break;
	  }
	}
	else{
	  if(curr->right!=NULL){
	    curr = curr->right;
	  }
	  else{
	    curr->left = point;
	    break;
	  }
	}
	dim++;
	dim%=3;
      }
    }
    loc+=lengths[i];
  }
}

void buildKDTree(point3D* a, int length, kdNode* b, int nthreads, kdNode* head){
    int blocks = 1;
    int max_blocks = length/nthreads; //number of blocks to use before we have a thread for every block
    point3D* curr;
    point3D* next;
    cudaMalloc((void**)&curr,length*sizeof(float)*3);
    cudaMalloc((void**)&next,length*sizeof(float)*3);
    point3D* temp;
    int* tree_struct;
    cudaMalloc((void**)&tree_struct,length*sizeof(int));
    float* means;
    cudaMalloc((void**)&means,max_blocks*sizeof(float));
  
    int* dev_length;
    cudaMalloc((void**)&dev_length,sizeof(int)*max_blocks);
    cudaMemcpy(dev_length, &length, 1*sizeof(int), cudaMemcpyHostToDevice);
    int dim = 0;
    int tree_ptr;
    int* dev_dim;
    cudaMalloc((void**)&dev_dim, sizeof(int));
    for(;blocks<max_blocks;blocks++){
        cudaThreadSynchronize();
        getMean<<<blocks,nthreads>>>(curr,dev_length,means);
        cudaMemcpy(dev_dim,&dim,1*sizeof(int),cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
        doPartitionStep<<<blocks,nthreads>>>(curr,dev_length,dev_dim,means,next,(tree_struct));

        dim++;
        dim%=3;
        
        temp = curr;
        curr = next;
        next = temp;
       
    }
    
    kdNode* host_tree;
    host_tree = (kdNode*) malloc(length*sizeof(kdNode));
    //    cudaMemcpy(host_tree, b, length,cudaMemcpyDeviceToHost);
    int* host_tree_struct = (int*) malloc(length*sizeof(int));
    cudaMemcpy(host_tree_struct, tree_struct, length*sizeof(int), cudaMemcpyDeviceToHost);
    int c;
    int p;
    int r;
    int l;

    for(int i = 0; i<length; i++){
        c = host_tree_struct[i];
	if(c<0||c>length)continue;
        l = host_tree_struct[2*(i+1)];
        r = host_tree_struct[2*(i+1)+1];

        host_tree[c].right = host_tree+r;
        host_tree[c].left = host_tree+l;

        if(i==0){
            host_tree[c].parent = NULL;
	    *head = host_tree[c];
        }
        else{
            host_tree[c].parent = host_tree+p;
            
        }
        p = (i+1)/2-1;
    }
    int* lengths = (int*)malloc(max_blocks*sizeof(int));
    cudaMemcpy(lengths, dev_length, max_blocks*sizeof(int), cudaMemcpyDeviceToHost);
    finishKDTree(lengths,max_blocks,host_tree,head);
    
    cudaFree(curr);
    cudaFree(next);
    cudaFree(tree_struct);
    cudaFree(means);
    cudaFree(dev_length);
    cudaFree(dev_dim);
}
#define k 4
#define dist(f1, f2) ((f1.x-f2.x)*(f1.x-f2.x)+(f1.y-f2.y)*(f1.y-f2.y)+(f1.z-f2.z)*(f1.z-f2.z))
__global__ void doKNN(kdNode* kdt, kdNode* ktRoot, int* size, kdNode** ktg){
    int tid = threadIdx.x+blockIdx.x+blockDim.x;
    int elems_per_thread = (*size)/(blockDim.x*gridDim.x);
    int base = tid*elems_per_thread;
    int end = base+elems_per_thread;
    
    kdNode buffer[k];
    float dists[k];
    int head = 0;
    kdNode* curr;
    kdNode* point;
    int dim;
    for(int i = base; i<end; i++){
	curr = ktRoot;
        point = kdt+i;
        dim = 0;
        while(true){
               if(data(curr,dim)>data(point,dim)){
                   if(curr->left!=NULL){
                       curr = curr->left;
                   }
                   else{break;}
               }
               else{
                   if(curr->right!=NULL){
                       curr = curr->right;
                   }
                   else{break;}
               }
               dim++;
               dim%=3;
        }
        buffer[head] = *curr;
        for(int j = 1; j<k; j++){
            dists[j] = INFINITY;
        }
        int mi = 1;
        int dir = 0; //top right, top left, bottom left, bottom right
        while(true){
            
            if(dist((*curr),(*point))<dists[mi]){
               buffer[mi] = *curr;
               if(dir==2){
                   dir = 0;
                   curr = curr->left;
               }
               else if(dir==3){
                   dir = 1;
                   curr = curr->right;
               }
               else{
                       if(data(curr,dim)>data(point,dim)){
                           if(curr->left!=NULL){
                               dir = 0;
                       	       curr = curr->left;
                               dim++;
                               dim%=3;
                           }
                           else{
                               dir += 2;
                               curr = curr->parent;
                               dim--;
                               dim%=3;
                           }
                       }
                       else{
                           if(curr->right!=NULL){
                               dir = 1;
                               curr = curr->right;
                               dim++;
                               dim%=3;
                           }
                           else{
                               dir+=2;
                               curr = curr->parent;
                               dim--;
                               dim%=3;
                           }
                       }
                   }
            }
            else if( curr->parent != NULL){
               if(curr->parent->left==curr){
                  dir = 2;
               }
               else{
                  dir = 3;
               }
               curr = curr->parent;
               dim--;
               dim%=3;
            }
            else{break;}
            for(int j = 0; j<k; j++){
                if(dists[j]>dists[mi]){
                    mi = j;
                }
            }
        }
        for(int j = 0; j<k; j++){
            ktg[i][j] = buffer[j];
        }
        
    }
    
}

__global__ void getCenterOfMass(point3D* pts, int*  length, float* com){
    int tid = threadIdx.x+blockIdx.x+blockDim.x;
    int elems_per_thread = (*length)/(blockDim.x*gridDim.x);
    int base = tid*elems_per_thread;
    int end = base+elems_per_thread;
    float* local_sum; 
    local_sum = (float*) malloc(sizeof(point3D));
    idx(local_sum,threadIdx.x,0) = 0;
    idx(local_sum,threadIdx.x,1) = 0;
    idx(local_sum,threadIdx.x,2) = 0;
    __shared__ float* block_sum; 
    block_sum = (float*) malloc(sizeof(point3D));
    block_sum[0] = 0;
    block_sum[1] = 0;
    block_sum[2] = 0;
    for(int i = base; i<end; i++){
        idx(local_sum,threadIdx.x,0) += idx(pts,i,0);
        idx(local_sum,threadIdx.x,1) += idx(pts,i,1);
        idx(local_sum,threadIdx.x,2) += idx(pts,i,2);
    }
    atomicAdd(block_sum+0, local_sum[0]);
    atomicAdd(block_sum+1, local_sum[1]);
    atomicAdd(block_sum+2, local_sum[2]);
    if(blockIdx.x==0){
       atomicAdd(com+0, block_sum[0]);
       atomicAdd(com+1, block_sum[1]);
       atomicAdd(com+2, block_sum[2]);
    }
    if(tid==0){
       com[0] /= blockDim.x*gridDim.x;
       com[1] /= blockDim.x*gridDim.x;
       com[2] /= blockDim.x*gridDim.x;
    }
    free(local_sum);
    free(block_sum);
}
#define dot3(a,b) (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])
static inline void getQRDecomposition(float A[3][3], float Q[3][3]){
   float u[3][3];
   float sum;
   for(int i = 0; i<3; i++){
       for(int j = 0; j<3; j++){
           u[i][j] = A[i][j];
       }
   }
   float dpb;
   for(int i = 0; i<3; i++){
       sum = 0;
       for(int j = 0; j<3; j++){
          sum += u[i][j]*u[i][j];
       }
       for(int j = 0; j<3; j++){
          Q[i][j] = u[i][j]/sqrt(sum);
       }
       for(int j = 3; j>3-i; j--){
           dpb = dot3(A[i], Q[j]);
           for(int jj = 0; jj<3; jj++){
               u[i][jj] -= Q[j][jj]*dpb;
           }
       }
   }
}
static inline void  getRQDecomposition(float A[3][3], float Q[3][3]){
   float B[3][3];
   float S[3][3];
   for(int i = 0; i<3; i++){
       for(int j = 0; j<3; j++){
           B[3-j][3-i] = A[i][j];
       }
   }
   getQRDecomposition(B,S);
   for(int i = 0; i<3; i++){
       for(int j = 0; j<3; j++){
           Q[3-j][3-i] = S[i][j];
       }
   }
}

__global__ void consolodate(point3D* x, point3D* p, int* length, float* W){
    int tid = threadIdx.x+blockIdx.x+blockDim.x;
    int threads =(blockDim.x*gridDim.x);
    int elems_per_thread = (*length)/threads;
    int base = tid*elems_per_thread;
    int end  = base+elems_per_thread;

    __shared__ superArray* sum_t;
    //change this to proper alloc
    sum_t = allocSuperArray(gridDim.x,3,3);

    for(int i = base; i<end; i++){
       for(int m = 0; m<3; m++){
           for(int n = 0; n<3; n++){
               indexSuperArray(sum_t,threadIdx.x,m,n) = idx(x,i,m)*idx(p,i,n);
           }
       }
    }
    for(int d = 1; d<threads; d*=2){
       if(threadIdx.x%d==0&&threadIdx.x!=0){
           for(int i = 0; i<3; i++){
               for(int j = 0; j<3; j++){
                   indexSuperArray(sum_t,threadIdx.x-d,i,j)+=indexSuperArray(sum_t,threadIdx.x,i,j);
               }
           }
       }
    }
    for(int i = 0; i<3; i++){
       for(int j = 0; j<3; j++){
           atomicAdd(&idx(W,i,j),indexSuperArray(sum_t,threadIdx.x,i,j));
        }
    }
    free(sum_t);
}
__device__ point3D* p2blockbuffer;
__device__ point3D* f2blockbuffer;
__device__ point3D* distsblockbuffer;
#undef k
__global__ void selectSubset(kdNode* pt, kdNode* pthead, point3D* p, int* plength, int* flength, int* newlength, point3D* p2, point3D* f2){
    int tid = threadIdx.x+blockIdx.x+blockDim.x;
    int threads =(blockDim.x*gridDim.x);
    int elems_per_thread = (*plength)/threads;
    int base = tid*elems_per_thread;
    int end  = base+elems_per_thread;

    point3D* buffer;
    buffer = (point3D*) malloc(elems_per_thread*sizeof(point3D));
    point3D* buffer2;
    buffer2 = (point3D*) malloc(elems_per_thread*sizeof(point3D));
    float* dists;
    dists = (float*)malloc(elems_per_thread*sizeof(float));

    for(int i = base; i<end; i++){
       kdNode curr = *pthead;
       int dim = 0;
       //get closest point in pt
       int stop = false;
        while(!stop){
               if(data((&curr),dim)>idx(p,i,dim)){
                   if(curr.left!=NULL){
                       curr = *curr.left;
                   }
                   else{stop = true;}
               }
               else{
                   if(curr.right!=NULL){
                       curr = *curr.right;
                   }
                   else{stop = true;}
               }
               dim++;
               dim%=3;
        }
        
        
        memcpy((float*)(buffer+i-base),  &curr.x,3*sizeof(float));
        buffer2[i-base] = p[i];
        dists[i-base] = dist(curr,p[i]);
    }
    
    __shared__ point3D* p2buffer;
    __shared__ point3D* f2buffer;
    __shared__ point3D* distsbuffer;
    point3D* temp;
    point3D* temp2;
    temp = (point3D*)malloc( (*newlength)*sizeof(point3D));
    temp2 = (point3D*)malloc( (*newlength)*sizeof(point3D));
    if(threadIdx.x==0){
        p2buffer = (point3D*)malloc(sizeof(point3D*)*gridDim.x);
        f2buffer = (point3D*)malloc(sizeof(point3D*)*gridDim.x);
        distsbuffer = (point3D*) malloc(gridDim.x*elems_per_thread*sizeof(point3D));
    }
    memcpy(distsbuffer+threadIdx.x, dists, (*newlength)*sizeof(point3D));
    memcpy(temp, buffer, (*newlength)*sizeof(point3D));
    memcpy(temp+(*newlength), buffer2, (*newlength)*sizeof(point3D));
    memcpy(p2buffer+threadIdx.x ,temp, (*newlength)*2*sizeof(point3D));
    //reduce by threads


    for(int d = 1; d<threads; d*=2){
       if(threadIdx.x%d==0&&threadIdx.x!=0){
           int j = 0;
           int k = 0;
           for(int i = 0; i<(*newlength); i++){
               if(idx(distsbuffer,threadIdx.x,j)>idx(distsbuffer,threadIdx.x-d,k)){
		 memcpy(idxa(p2buffer,threadIdx.x,i), idxa(p2buffer,threadIdx.x,k), 3*sizeof(float));
               }
               else{
		 memcpy(idxa(p2buffer,threadIdx.x,i), idxa(p2buffer,threadIdx.x,k), 3*sizeof(float));
               }
           }
       }
    }

    free(buffer);



    if(threadIdx.x==0){
        p2blockbuffer = (point3D*)malloc(sizeof(point3D*)*blockDim.x);
        f2blockbuffer = (point3D*)malloc(sizeof(point3D*)*blockDim.x);
        distsblockbuffer = (point3D*)malloc(blockDim.x*(*newlength)*sizeof(point3D));
        memcpy(distsblockbuffer+blockIdx.x,distsbuffer,(*newlength)*sizeof(point3D));
	memcpy(p2blockbuffer+blockIdx.x,p2buffer,(*newlength)*sizeof(point3D));
        memcpy(f2blockbuffer+blockIdx.x,f2buffer,(*newlength)*sizeof(point3D));
 
    }
    for(int d = 1; d<threads; d*=2){
       if(blockIdx.x%d==0&&blockIdx.x!=0){
           int j = 0;
           int k = 0;
           for(int i = 0; i<(*newlength); i++){
               if(idx(distsblockbuffer,blockIdx.x,j)>idx(distsblockbuffer,blockIdx.x-d,k)){
		 memcpy(idxa(p2blockbuffer,blockIdx.x,i), idxa(p2blockbuffer,blockIdx.x,k), 3*sizeof(float));
		 memcpy(idxa(f2blockbuffer,blockIdx.x,i), idxa(f2blockbuffer,blockIdx.x,k), 3*sizeof(float));
               }
               else{
		 memcpy(idxa(p2blockbuffer,blockIdx.x,i), idxa(p2blockbuffer,blockIdx.x,k), 3*sizeof(float));
		 memcpy(idxa(f2blockbuffer,blockIdx.x,i), idxa(f2blockbuffer,blockIdx.x,k), 3*sizeof(float));
               }
           }
       }
    }
    if(blockIdx.x == 0){
      memcpy(p2, p2blockbuffer, (*newlength)*sizeof(point3D));
      memcpy(f2, f2blockbuffer, (*newlength)*sizeof(point3D));
    }
    free(temp);
    free(temp2);
    
    if(threadIdx.x==0){
        free(p2buffer);
        free(f2buffer);
        free(distsbuffer);       
    }
    
    if(blockIdx.x==0){
        free(p2blockbuffer);
        free(f2blockbuffer);
        free(distsblockbuffer);
    }

    free(buffer2);
    free(dists);
}


#define fl2size 3*sizeof(float*)+3*3*sizeof(float)
void doICP(point3D* f, point3D* p, kdNode* ft, kdNode* pt, kdNode* ftHead, int plength, int flength){
    float* WDev;
    float* TDev;
    cudaMalloc((void**)&WDev, 9*sizeof(float));
    cudaMalloc((void**)&TDev, 16*sizeof(float));

    point3D* p2;
    cudaMalloc((void**)&p2, plength*sizeof(point3D));
    
    point3D* f2;
    cudaMalloc((void**)&f2, flength*sizeof(point3D));

    float* com;
    float* com_host;
    cudaMalloc((void**)&com,sizeof(float)*3);
    com_host = (float*)malloc(sizeof(float)*3);

    float U[3][3];
    float V[3][3];
    float W[3][3];
    float T[4][4];
    float R[3][3];

    int* p2length;
    int p2length_host = flength/8;
    cudaMalloc((void**) &p2length, sizeof(int));
    cudaMemcpy(p2length,&p2length_host, 1*sizeof(int), cudaMemcpyHostToDevice);
    
    int* plength_dev;
    cudaMalloc((void**) &plength_dev, sizeof(int));
    cudaMemcpy(plength_dev,&plength, 1*sizeof(int), cudaMemcpyHostToDevice);
    int* flength_dev;
    cudaMalloc((void**) &flength_dev, sizeof(int));
    cudaMemcpy(plength_dev,&flength, 1*sizeof(int), cudaMemcpyHostToDevice);


    for(int n = 0; n<100; n++){
        cudaThreadSynchronize();
        selectSubset<<<128,128>>>(ft,ftHead,p,plength_dev,flength_dev,p2length,f2,p2);
	cudaThreadSynchronize();
        consolodate<<<128,128>>>(f2,p2,p2length,WDev);
	cudaThreadSynchronize();
        cudaMemcpy(W, WDev,fl2size, cudaMemcpyDeviceToHost);
        getQRDecomposition(W,U);
        getRQDecomposition(W,V);
        for(int i = 0; i<3; i++){
            for(int j = 0; j<3; j++){
                R[i][j] = 0;
             }
        }
        for(int i = 0; i<3; i++){
            for(int j = 0; j<3; j++){
                for(int k = 0; k<3; k++){
                   R[i][k] += U[i][j]*V[j][k];
                }
            }
        }
     

        cudaThreadSynchronize();
        getCenterOfMass<<<128,128>>>(f,flength_dev,com);
	cudaThreadSynchronize();
	cudaMemcpy(com_host, com, 3*sizeof(float),cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
         /*
         ** [R, R, R, c]
         ** [R, R, R, c]
         ** [R, R, R, c]
         ** [0, 0, 0, 1]
         */
         for(int i = 0; i<3; i++){
             for(int j = 0; j<3; j++){
                 T[i][j] = R[i][j];
             }
         }
         for(int i = 0; i<3; i++){
             T[i][3] = com_host[i];
         }
         for(int i = 0; i<3; i++){
             T[3][i] = 0;
         }
         T[3][3] = 1;
         cudaMemcpy(&T,TDev,fl2size, cudaMemcpyDeviceToHost);
	 cudaThreadSynchronize();
         doBatchTransformation<<<128,128>>>(f, flength, TDev, f);
     }
     cudaFree(WDev);
     cudaFree(TDev);
     cudaFree(p2);
     cudaFree(f2);
     cudaFree(com);
     cudaFree(p2length);
     cudaFree(plength_dev);
     cudaFree(flength_dev);
}

int main( void ) {
    char* filesToLoad[5] = {"snapshot0.bmp","snapshot1.bmp","snapshot2.bmp","snapshot3.bmp","snapshot4.bmp"};
    int numfiles = 5;
    rawImageData* img;
    int* width_dev;
    int* height_dev;
    unsigned char* data_dev;
    point3D* p1_dev;
    point3D* p2_dev;
    point3D* temp;
    
    
    kdNode* pt1;
    kdNode* pt2;

   
    loadImage(filesToLoad[0], &img);
 
 
    int length = img->width*img->height;
    int p1length = length;
    cudaMalloc((void**)&width_dev, sizeof(int));
    cudaMalloc((void**)&height_dev, sizeof(int));
    cudaMalloc((void**)&data_dev, length*sizeof(char));
    
    
    cudaMalloc((void**)&p1_dev, length*sizeof(point3D));
    cudaMalloc((void**)&p2_dev, length*sizeof(point3D));
    cudaMalloc((void**)&pt1, length*sizeof(kdNode));
    cudaMalloc((void**)&pt2, length*sizeof(kdNode));
    
    cudaMemcpy(data_dev,img->image_data,length*sizeof(char),cudaMemcpyHostToDevice);
    cudaMemcpy(width_dev,&img->width,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(height_dev,&img->height,sizeof(int),cudaMemcpyHostToDevice);
    cudaThreadSynchronize();
    get3DPoints<<<128,128>>>(width_dev,height_dev,data_dev, p1_dev);
    kdNode* head = (kdNode*)malloc(sizeof(kdNode));
    kdNode* trash = (kdNode*)malloc(sizeof(kdNode));
    buildKDTree(p1_dev, length, pt1, 300, head);
    for(int i  = 1; i<numfiles; i++){
        free(img);
        loadImage(filesToLoad[0], &img);
	cudaMemcpy(data_dev,img->image_data,length*sizeof(char),cudaMemcpyHostToDevice);
	cudaMemcpy(width_dev,&img->width,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(height_dev,&img->height,sizeof(int),cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
        get3DPoints<<<128,128>>>(width_dev,height_dev,data_dev, p2_dev);
        buildKDTree(p2_dev, length, pt2, 300, trash);
        doICP(p1_dev,p2_dev,pt1,pt2,head,p1length,length);
        cudaMalloc((void**)&temp, (p1length+length)*sizeof(point3D));
        cudaMemcpy(temp,p1_dev, p1length*sizeof(point3D),cudaMemcpyDeviceToDevice);
        cudaMemcpy(temp+p1length,p2_dev,length*sizeof(point3D), cudaMemcpyDeviceToDevice);
        p1length += length;
	cudaThreadSynchronize();
        cudaFree(p1_dev);
        p1_dev = temp;
    }

    //read the data back and write it to a file
    point3D* p1;
    p1 = (point3D*)malloc(p1length*sizeof(point3D));
    cudaMemcpy(p1,p1_dev,p1length*sizeof(point3D),cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    FILE* fp;
    fp = fopen("result.bin","w");
    fwrite(p1,sizeof(point3D),p1length,fp);
    printf("%d",p1length);
    return 0;
}
