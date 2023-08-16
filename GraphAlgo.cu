/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/

//this kernel counts the number of nodes in the 0th level by finding change in apr value from zero to non zero
__global__ void countKernel(int* d_apr,int* dnodes_cur,int len){
    int id=blockIdx.x*1024+threadIdx.x*32+threadIdx.y;
    if(id<len && id!=0 && d_apr[id]!=0 && d_apr[id-1]==0)
        *dnodes_cur=id-1;
}

// this kernel is basically launched for number of nodes in a particular level
__global__ void activecountKernel(int* d_aid,int* d_apr,int* dnodes_cur,int* d_activeVertex,int* d_offset,int* d_csrList,int l,int len,int start){
    int id=blockIdx.x*1024+threadIdx.x*32+threadIdx.y;
    if(id<len)
    {
        //check if its one of the corner nodes of the level, and if not check either one of the neighbour is activated
        if(id==0 || id==len-1 || ( *(d_aid+start+id-1) >= *(d_apr+start+id-1) || *(d_aid+start+id+1) >= *(d_apr+start+id+1) ))
        {
            //now check if the node "id" itself is activated or not
            if( *(d_aid+start+id) >= *(d_apr+start+id) )
            {
                //if it is activated then increase the d_aid value of all nodes pointed by this node
                for(int i = 0; i < d_offset[start+id+1] - d_offset[start+id]; i++)
                {
                    //offset stores the value of node pointed by the activated node "id"
                    int offset=*(d_csrList+*(d_offset+id+start)+i);
                    //atomically increment value because different nodes can try to increment at the same time
                    atomicAdd(d_aid+offset,1);
                }
                //count the node "id" as activated nodes of this level
                atomicAdd(d_activeVertex+l,1);
            }
        }
        //now for all the nodes pointed by the nodes in level "l" store the value of maximum node to count length of next level
        for(int i=0;i<d_offset[start+id+1]-d_offset[start+id];i++)
        {
            //val stores the index of nodes pointed by node "id"
            int val=*(d_csrList+*(d_offset+id+start)+i);
            atomicMax(dnodes_cur,val);
        }
    }
}

/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
	cudaMalloc(&d_activeVertex, L*sizeof(int));
    cudaMemcpy(d_activeVertex, h_activeVertex, L*sizeof(int), cudaMemcpyHostToDevice);


/***Important***/

    // Initialize d_aid array to zero for each vertex
    cudaMemset(d_aid, 0, V*sizeof(int));
    // Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/
// len variable to launch kernell with maximum 10000 because a level has atmax 10000 nodes
int len=(V>10000)?10001:V;
//hnodes_cur stores the index of max node in current level
int *dnodes_cur;
//dnodes_cur for GPU memory
int *hnodes_cur;
cudaMalloc(&dnodes_cur, sizeof(int));
hnodes_cur=(int*)malloc(sizeof(int));

dim3 gridDim(ceil(float(len)/1024),1,1);  
dim3 blockDim(32,32,1);	
//finds and stores the index of the last node of 0th level in dnodes_cur
countKernel<<<gridDim,blockDim>>>(d_apr,dnodes_cur,len);
cudaMemcpy(hnodes_cur,dnodes_cur,sizeof(int),cudaMemcpyDeviceToHost);

int start,l;
start=0;
l=0;
while(l<L){
    // len stores the number of nodes in level "l"
    len=*hnodes_cur-start+1;
    dim3 gridDim(ceil(float(len)/1024),1,1);  
    dim3 blockDim(32,32,1);
    activecountKernel<<<gridDim,blockDim>>>(d_aid,d_apr,dnodes_cur,d_activeVertex,d_offset,d_csrList,l,len,start);
    //update the value of start for next iteration before hnodes_cur gets updated
    start=*hnodes_cur+1;
    cudaMemcpy(hnodes_cur,dnodes_cur,sizeof(int),cudaMemcpyDeviceToHost);
    l++;
}
//stores back the value of d_activeVertex in h_activeVertex after all levels are done
cudaMemcpy(h_activeVertex,d_activeVertex,L*sizeof(int),cudaMemcpyDeviceToHost);

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
