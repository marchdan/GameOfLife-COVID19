#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
#include<math.h>

//Declare all needed extern variables and functions

// Result from last compute of world.
extern unsigned char *g_resultData;

// Current state of world. 
extern unsigned char *g_data;

// Current width of world.
extern size_t g_worldWidth;

/// Current height of world.
extern size_t g_worldHeight;

/// Current data length (product of width and height)
extern size_t g_dataLength;  // g_worldWidth * g_worldHeight

// Timing variables
extern size_t s_time;
extern size_t e_time;

//Global storage of total casses and deaths
extern unsigned int totalCases;
extern unsigned int totalDeaths;


extern "C" //Extern functions in C for understandability by mpi file
{
    void initMaster( int myrank, int numranks, unsigned int pattern, size_t worldSize, size_t caseSeed, size_t deathSeed );
    bool kernalLaunch(int myrank, int numranks, 
                    unsigned char** d_data, unsigned char** d_resultData,
                    size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount, size_t pattern,
                    unsigned int infectRate, unsigned int deathRate, unsigned char** recv);
    
    void recvData(size_t myrank, size_t numranks, size_t worldLength, unsigned char** *recv);
    void sendData(size_t myrank, size_t numranks, size_t worldLength, unsigned char* sData);
    void exportStats(unsigned char** data, int myrank, int numranks, int day);


    void freeData();    
    void finishCuda();
}


// Initialize each space in the data grids after initialization
static inline void gol_initData( unsigned char fill )
{
    for(int i = 0; i < g_dataLength; i++){
        g_data[i] = fill; //Fill the grid with the given fill number (0 or 1)
        g_resultData[i] = 0; //Fill result grid with default zero values
    }
}

static inline void gol_initEveryOther( size_t worldWidth, size_t worldHeight, size_t infected, size_t numranks, unsigned int* rands )
{
    size_t local = infected;
    size_t num_rows = local/(worldWidth/2); //Number of rows with infected people
    size_t start = (worldHeight/2) - (num_rows/2); //Row to start on
    size_t current = start*worldWidth; //Starting cell

    //Global variable init
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;
    
    //Allocate data for mian grid and result grid
    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char)));   

    gol_initData(0); //Initalize grid to zero for input of argv[1] = 0

    while(local > 0 && current < g_dataLength){ //Initialize every other spot in the middle of the grid
        g_data[current] = (rands[current]%20)+1;
        local-=1; current+=2;
    }
}

static inline void gol_initClustered( size_t worldWidth, size_t worldHeight, size_t infected, size_t numranks, unsigned int* rands )
{   
    size_t local = infected;
    size_t current = 0;
    size_t spacing = 0;
    size_t clusters = local/4;
    unsigned int i;

    // Set all global values for later use
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    //Allocate data for main grid and result grid
    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char)));

    gol_initData(0); //Initialze main grid to one for input of argv[1] = 1
    
    current = g_worldWidth+1;

    spacing = (g_dataLength-(2*g_worldWidth))/clusters;

    while(local>3){ //Initiate maximum spacing between groups of 4
        if(current&g_worldWidth == 0){ current++;}

        if(current%g_worldWidth < g_worldWidth-5){
            for(i = 0; i < 4; i++, local--, current++){
                g_data[current] = (rands[current]%20)+1;
            }
            current+= (spacing/2);
        }
        else{
            while(current%g_worldWidth > 0){
                current++;
            }
        }
    }

    while(local > 0){
        if(current >= g_dataLength){
            current = 1;
        }

        if(g_data[current-1] == 0 && g_data[current] == 0 && g_data[current+1] == 0){
            g_data[current] = (rands[current]%20)+1;
        }

        local--; current += 3;
    }
}

static inline void gol_initMiddle( size_t worldWidth, size_t worldHeight, size_t infected, size_t numranks, unsigned int* rands )
{
    size_t local = infected;
    size_t dim = sqrt(local);
    size_t start = (worldHeight/2) - (dim/2); //Row to start on
    size_t current = (start*worldWidth)+((worldWidth/2)-(dim/2)); //Starting cell

    int i, j, begin;

    // Set all global values for later use
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Allocate data for main grid and result grid
    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char)));

    gol_initData(0); //Initialize grids to zero

    // Create clustered square in the middle of the grid
    for( i = 0; i < dim; i++){
        begin = current + (i*worldWidth);
        for( j = begin; j < begin + dim; j++){
            g_data[j] = (rands[j]%20)+1;
        }
    } 
}

void gol_initDistancing( size_t worldWidth, size_t worldHeight, size_t infected, size_t numranks, unsigned int* rands )
{
    size_t local = infected;
    size_t current = worldWidth;

    // Set all global values for later use
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    //Allocate data for main grid and result grid
    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char)));

    gol_initData(0); //Initialize grids to zero

    size_t available_space = g_dataLength-(2* g_worldWidth);
    size_t distance = available_space/local;
    size_t front = distance/2;
    size_t back = distance-front;

    //Initialize with maximum distancing possible
    while(local > 0 && current < g_dataLength){
        current += front;
        g_data[current] = (rands[current]%20)+1; 
        local-=1; current+=back;
    }
}

void gol_initRandom( size_t worldWidth, size_t worldHeight, size_t infected, size_t numranks, unsigned int* rands)
{
    size_t local = infected;
    size_t current = worldWidth;
    
    // Set all global valeus for later use
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    //Allocate data for main grid and result grid
    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char)));

    gol_initData(0); //Initialize grids to zero

    while(local > 0){ //Randomly initialize spaces in the grid
        if(current == g_dataLength){ current = 0; }
        
        if(g_data[current] == 1){
            current++;
            continue;
        }
        else if(rands[current]%100 <= 5){ 
            g_data[current] = (rands[current]%20)+1; 
            local-=1;
        }
        current++;
    }
}

//Initiate the cuda world 
void initMaster( int myrank, int numranks, unsigned int pattern, size_t worldSize, size_t caseSeed, size_t deathSeed )
{
    int cudaDeviceCount = -1;
    cudaError_t cE = cudaSuccess;

    unsigned int *rands;
    curandGenerator_t gen;

    //Initialize cuda devices
    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n",
        cE, cudaDeviceCount );
        exit(-1);
    }
    if( (cE = cudaSetDevice( myrank % cudaDeviceCount )) != cudaSuccess )
    {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n",
        myrank, (myrank % cudaDeviceCount), cE);
        exit(-1);
    }

    totalCases = caseSeed;
    totalDeaths = deathSeed;

    //Alocate and create the host random number generator with curand
    cudaMallocManaged( &rands, (worldSize*worldSize * sizeof(unsigned int)));
    for(int i = 0; i < worldSize*worldSize; i++){
        rands[i] = 0;
    }

    curandStatus_t stat1, stat2, stat4;
    stat1 = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    
    stat2 = curandSetPseudoRandomGeneratorSeed(gen, 1234ULL*(unsigned long long) myrank);
    
    stat4 = curandGenerate( gen, rands, worldSize*worldSize);
    cudaDeviceSynchronize();
    if(stat1 != CURAND_STATUS_SUCCESS){
        printf("ERROR: Generator creation failed\n");
        exit(-1);
    } 
    else if(stat2 != CURAND_STATUS_SUCCESS){
        printf("ERROR: Generator seeding failed\n");
        exit(-1);
    }
    else if(stat4 != CURAND_STATUS_SUCCESS){
        printf("ERROR: RNG Production failed(%d)\n", stat4);
        exit(-1);
    }
    else if(rands[0] == 0){
        printf("ERROR: Rands did not generate!\n");
        exit(-1);
    }

    printf("Rank %d, of size %ldx%ld, is running GOL kernel on device %d with %ld infected\n", myrank, worldSize, worldSize, (myrank%cudaDeviceCount), caseSeed);
   
    //Initialize the world based on the given pattern
    switch(pattern)
    {
    case 0:
	gol_initEveryOther( worldSize, worldSize, caseSeed, numranks, rands);
	break;
	
    case 1:
	gol_initClustered( worldSize, worldSize, caseSeed, numranks, rands);
	break;
	
    case 2:
	gol_initMiddle( worldSize, worldSize, caseSeed, numranks, rands);
	break;
	
    case 3:
	gol_initDistancing( worldSize, worldSize, caseSeed, numranks, rands);
	break;

    case 4:
	gol_initRandom( worldSize, worldSize, caseSeed, numranks, rands);
	break;

    default:
	printf("Pattern %u has not been implemented \n", pattern);
	exit(-1);
    }

    cudaFree(rands);
    curandDestroyGenerator(gen);
}

//Swap the information in the given arrays
static inline void gol_swap( unsigned char **pA, unsigned char **pB)
{
    unsigned char *temp = *pA; //Save pA for use later
    *pA = *pB; //Swap pB into pA
    *pB = temp; //Swap pA into pB to complete the swap
}

//Cout the number of alive cells in the current world
__device__ 
static inline unsigned int gol_countAliveCells(unsigned char* data, 
					   size_t x0, size_t x1, size_t x2, 
					   size_t y0, size_t y1,size_t y2, size_t width) 
{
    // Compute the number of infected cells around the current cell
    // Infected cells have a value of 1 - 16, healthy cells have a value of 0
    // Adding up the appropiate cells produces a result containing the number of infected cells

    int BL, L, BR, R, B, UL, UR, U; // Initialize a variable for each direction

    L = data[x0+y1]; // Left
    R = data[x2+y1]; // Right

    BL = data[x0+y2]; //Bottom left
    B = data[x1+y2]; //Bottom
    BR = data[x2+y2]; //Bottom Right

    UL = data[x0+y0]; //Upper left
    U = data[x1+y0]; //Upper
    UR = data[x2+y0]; //Upper right

    int alive = 0; // Computer total number infected
    alive += ((BL < 17 && BL > 0) ? 1 : 0);
    alive += ((B < 17 && B > 0) ? 1 : 0);
    alive += ((BR < 17 && BR > 0) ? 1 : 0);
    alive += ((L < 17 && L > 0) ? 1 : 0);
    alive += ((R < 17 && R > 0) ? 1 : 0);
    alive += ((UL < 17 && UL > 0) ? 1 : 0);
    alive += ((U < 17 && U > 0) ? 1 : 0);
    alive += ((UR < 17 && UR > 0) ? 1 : 0);
    return alive;
}

__device__
unsigned int getDefaultStatus(unsigned int current, int alive, unsigned int *tCases, unsigned int *tDeaths){
    if(current > 0){ // Decide next cell state of a currently living cell.
        if(alive > 4){ // cell doesn't change state unless its healed
            return (current == 1 ? 0 : current);
        }
        else{ //Cell gets one day closer to being healthy
            return current-1;
        }
    }
    else{ //Currently healthy cells
        if(alive < 2){ // Co infection
            return 0;
        }
        else if(alive < 4){ // Minimal infection
            (*tCases)+=1;
            return 14;
        }
        else{ //Full infection
            (*tCases)+=1;
            return 21;
        }
    }
}

__device__
unsigned int getStatsStatus(unsigned int current, int alive, int rate, unsigned int *tCases, unsigned int *tDeaths, curandState_t* state){
    if(current > 0){
        return current-1;
    }
    else{
        if(alive == 0){ return 0; }
        while(alive > 0){
            int rand = fabsf(curand(state));
            if(rand%100 < rate){ //Spread rate
                (*tCases)++;
                return 21;
            }
            else{
                alive--;
            }
        }
        return 0;
    }
}

__device__
unsigned int getWorstStatus(unsigned int current, int alive, unsigned int *tCases, unsigned int *tDeaths){
    if(current > 0){
        if(alive > 5){
            return (current == 1 ? 0 : current);
        }
        else{
            return current-1;
        }
    }
    else{
        if(alive > 1){
            (*tCases)++;
            return 21;
        }
        else{
            return 0;
        }
    }
}

__device__
unsigned int getBestStatus(unsigned int current, int alive, unsigned int *tCases, unsigned int *tDeaths){
    if(current > 0){
        return current-1;
    }
    else{
        if(alive>6){
            (*tCases)++;
            return 14;
        }
        else{
            return 0;
        }
    }
}

/*
    CUDA kernal for running GOL calculations in parallel with the specificed number of threads/blocks
*/
__global__ void gol_kernal(unsigned int myrank, unsigned int numranks, 
                        unsigned char* d_data,
                        unsigned int worldWidth, unsigned int worldHeight,
                        unsigned char* d_resultData, unsigned int pattern,
                        unsigned int infectRate, unsigned int deathRate, unsigned int *tCases, unsigned int *tDeaths, size_t worldLength)
    {
        unsigned int index, x0, x2, y0, y1, y2, y, x; //Initialize all needed variables for the function

        index = blockIdx.x * blockDim.x + threadIdx.x; //The provided index calculation
        x = index%worldWidth; //The remainder of the index divided by the grid width produced the x component of the index.
        y = index/worldWidth; //The integer answer of index divided by world width provides the y component of the index.
        
        curandState_t state;
        
        while(index < worldLength){ //continue as long as the current index is valid in the scope of the provided grid.
            // Provided variable calculations
            curand_init(2020, index, 0, &state); //Device rng init
            y0 = ((y+worldHeight-1)%worldHeight)*worldWidth;
            y1 = y*worldWidth;
            y2 = ((y + 1) % worldHeight) * worldWidth;
            x0 = (x + worldWidth - 1) % worldWidth;
            x2 = (x + 1) % worldWidth;

            int alive = gol_countAliveCells(d_data, x0, x, x2, y0, y1, y2, worldWidth); // Retrive the number of current infected cells
            
            if(d_data[index] > 0){ //Chance of cell death
                unsigned int rand = fabsf(curand(&state));
                if(rand%100 < deathRate){
                    d_resultData[index] = 0;
                    (*tDeaths)++;
                    index += (blockDim.x * gridDim.x);
                    continue;
                }
            }

            switch(pattern){
                case(0):
                d_resultData[index] = getDefaultStatus(d_data[index], alive, tCases, tDeaths);
                break;

                case(1):
                d_resultData[index] = getStatsStatus(d_data[index], alive, infectRate, tCases, tDeaths, &state);
                break;

                case(2): 
                d_resultData[index] = getWorstStatus(d_data[index], alive, tCases, tDeaths);
                break;

                case(3): 
                d_resultData[index] = getBestStatus(d_data[index], alive, tCases, tDeaths);
                break;

                default:
                printf("Pattern %u has not been implemented \n", pattern);
                return;

            }
            index += (blockDim.x * gridDim.x); //Increase index by one block.
        }
    }

bool kernalLaunch(int myrank, int numranks, 
                    unsigned char** d_data, unsigned char** d_resultData,
                    size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount, size_t pattern,
                    unsigned int infectRate, unsigned int deathRate, unsigned char** recv)
    {
        int i;
        size_t N; //Initialize needed variables
        
        N = worldWidth * worldHeight; //N equals total grid size 
        
        for(i = 0; i < iterationsCount; i++){
            gol_kernal<<<N/threadsCount,threadsCount>>>(myrank, numranks, *d_data, worldWidth, worldHeight, *d_resultData, pattern, infectRate, deathRate, &totalCases, &totalDeaths, N); //Call the Parallel kernel and specify the number of blocks and threads per block.
            gol_swap(d_data, d_resultData);// Swap the current data with the result data
            
            //printf("\tRank %d, Day %d- Cases: %u | Deaths: %u\n", myrank, i, totalCases, totalDeaths);
            
            unsigned char* cpy = *d_data;
            if(numranks > 1){
                cudaDeviceSynchronize(); //Synchronize before full exchange
                recvData(myrank, numranks, N, &recv);
                sendData(myrank, numranks, N, cpy);
                cudaDeviceSynchronize();//Synchronize before export
            }
            recv[myrank] = cpy;
            exportStats(recv, myrank, numranks, i);
        }
        return true;
    }

void finishCuda(){
    cudaDeviceSynchronize(); // Function to synchronize cuda since this call must happen from the MPI code base
}

void freeData(){ //Function to free all CUDA memomory allocated
    cudaFree(g_data);
    cudaFree(g_resultData);
}
