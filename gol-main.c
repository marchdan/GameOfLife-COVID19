#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<mpi.h>
#include<string.h>
#include<math.h>
#include <ctype.h>

#define WORLD_POP 329588430 //Current US population

typedef unsigned long long ticks;

//Pre-declarations of needed functions
void initMaster( int myrank, int numranks, unsigned int pattern, size_t worldSize, size_t caseSeed, size_t deathSeed );
bool kernalLaunch(int myrank, int numranks, 
                    unsigned char** d_data, unsigned char** d_resultData,
                    size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount, size_t pattern,
                    unsigned int infectRate, unsigned int deathRate, unsigned char** recv);
void gol_printWorld(MPI_File outfile, unsigned char* data, size_t myrank);
void freeData();
void finishCuda();

//Declaration of global variables
unsigned char *g_resultData, *g_data; //Data arrays
size_t g_worldWidth, g_worldHeight, g_dataLength; //World sizes
double s_time, e_time; //Timing varaibles
unsigned int totalCases, totalDeaths;

unsigned int *cases, *deaths; //Storage of input file data

//Tick timing variables
unsigned long long start = 0;
unsigned long long finish = 0;


//Global files
MPI_Offset tick_offset = 0;
MPI_File tickFp;

MPI_Offset stats_offset = 0;
MPI_File statsFp;

MPI_Offset results_offset = 0;
MPI_File resultsFp;



MPI_Request* reqs; //MPI requests storage for waiting later

static __inline__ ticks getticks(void) //Provided ticks code
{
  unsigned int tbl, tbu0, tbu1;

  do {
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);

  return (((unsigned long long)tbu0) << 32) | tbl;
}


int fileRead(MPI_File fp, unsigned int* *cases, unsigned int* *deaths, double sFactor, size_t myrank, size_t numranks){
	int i = 0;
	MPI_Offset filesize;
	MPI_File_get_size(fp, &filesize);

	int start = 0;
	int end = filesize-1;

	if(myrank==0){
		printf("File size: %lld\n", filesize);
	}
	char* data=calloc((end-start+3), sizeof(char));
	if(!data){
		perror("ERROR: Malloc Failed");
		exit(-1);
	}
	MPI_Status temp;
	int ret = MPI_File_read(fp, data, filesize, MPI_CHAR, &temp);
	if(ret != MPI_SUCCESS){
		perror("ERROR: Invalid read");
		exit(-1);
	}

	data[strlen(data)] = '\n';

	//Read in data from the string which was read
	char* hold = data;
	while(*hold){
		int comma = 0;
		char* case_count;
		char* death_count;
		while(*hold != '\n'){
			if(*hold == ','){
				if(comma == 0){
					hold++;
					case_count = hold;
				}
				else if(comma == 1){
					*hold = '\0';
					hold++;
					death_count = hold;
				}
				comma++;
			}
			else{
				hold++;
			}
		}
		*hold = '\0';
		if(comma != 2 || !isdigit(*case_count)){
			hold++;
			continue;
		}
		else{
			(*cases)[i] = (atoi(case_count)*sFactor);
			(*deaths)[i] = (atoi(death_count)*sFactor);
			i++;
		}
		hold++;
	}
	free(data); //Free temp data
	return i; //Number of days in files
}

//MPI recv call set up
void recvData(size_t myrank, size_t numranks, size_t worldLength, unsigned char** *recv){
	int i;
	
	*recv = calloc(numranks, sizeof(unsigned char*));

	for(i = 0; i < numranks; i++){ //Recive from all ranks
		MPI_Request rReq; 
		if(i != myrank){
			(*recv)[i] = calloc(worldLength,sizeof(unsigned char));
			MPI_Irecv((*recv)[i], worldLength, MPI_UNSIGNED_CHAR, i, myrank, MPI_COMM_WORLD, &rReq);
		}
		reqs[i] = rReq;
	}
	reqs[myrank] = MPI_REQUEST_NULL;

}

//MPI send call init
void sendData(size_t myrank, size_t numranks, size_t worldLength, unsigned char* sData){
	int i;

	MPI_Barrier(MPI_COMM_WORLD);

	for(i = 0; i < numranks; i++){ //Send to all ranks
		MPI_Request sReq; 
		if(i != myrank){
			MPI_Isend(sData, worldLength, MPI_UNSIGNED_CHAR, i, i, MPI_COMM_WORLD, &sReq);
		}
		reqs[i+numranks] = sReq;
	}
	reqs[myrank+numranks] = MPI_REQUEST_NULL;

	MPI_Barrier(MPI_COMM_WORLD);

	if(MPI_Waitall(numranks*2, reqs, MPI_STATUSES_IGNORE)){ //Wait for all message passing to complete
		perror("ERROR: Waitall returned with error");
	}
}

int countCases(unsigned char* arr, unsigned int length){ //Count the number of cases in the given array
    int i, cases = 0;
    for(i = 0; i < length; i++){
        if(arr[i] > 0){
            cases++;
        }
    }
    return cases;
}

//Print the current world to the given file
void gol_printWorld(MPI_File outfile, unsigned char* data, size_t myrank)
{
    int i, j;
    
    MPI_Offset offset = g_dataLength*sizeof(unsigned char)*g_worldWidth*myrank;

    for( i = 0; i < g_worldHeight; i++)
    {
	
	for( j = 0; j < g_worldWidth; j++)
	{
		char temp[4];
		sprintf(temp, "%2d%c", data[(i*g_worldWidth) + j], ' '); 
		strcat(temp, "\0");

		MPI_File_write_at(outfile, offset, temp, strlen(temp), MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
		offset+=strlen(temp);
	}
	
	MPI_File_write_at(outfile, offset, "\n", 1, MPI_CHAR, MPI_STATUS_IGNORE);
	offset+=1;
    }
    MPI_File_write_at(outfile, offset, "\n\n", 2, MPI_CHAR, MPI_STATUS_IGNORE);
    offset+=2;
}

//Export current statistics to file for later data use
void exportStats(unsigned char** data, int myrank, int numranks, int day){
	int i, j,r;
	int err;
	char* filename = "data/temp";
	char file[256];
	sprintf(file, "%s%d%s", filename, myrank, ".txt");
	file[strlen(file)] = '\0';
	if(day > 0){
		MPI_File rfp;
		start = getticks();
		err = MPI_File_open(MPI_COMM_SELF, file, MPI_MODE_RDONLY, MPI_INFO_NULL, &rfp);
		//printf("Err: %d\n", err);
		if(err || !rfp){
			perror("ERROR: Unable to open imtermediate file");
			MPI_Finalize();
			exit(-1);
		}
		for(r = 0; r < numranks; r++){ //Read old data
			int local_cases = 0;
			MPI_Offset offset = g_dataLength*sizeof(unsigned char)*g_worldWidth*myrank;
			for( i = 0; i < g_worldHeight; i++)
		    {
			for( j = 0; j < g_worldWidth; j++)
			{
				char temp[4];

				MPI_File_read_at(rfp, offset, temp, 3, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
				if(atoi(temp) > 0){ local_cases++; }
				offset+=strlen(temp);
			}
			offset+=1;
		    }
		    offset+=2;

		    int new_cases = countCases(data[r], g_dataLength);
		    
		    char cur_string[512]; //New case sfile output
		    sprintf(cur_string, "%s%d%s%d%s%d%s", "Rank ", r, ", after day ", day, ": ", new_cases - local_cases, " new cases\n");
		    
		    MPI_File_write_at(statsFp, stats_offset, cur_string, strlen(cur_string), MPI_CHAR, MPI_STATUS_IGNORE);
			stats_offset+=strlen(cur_string);
		}

		MPI_File_close(&rfp);

		finish = getticks();
		char tick_string1[256]; //Ticks read data
		sprintf(tick_string1, "%s%d%s%d%s%llu%s", "Day ", day, ", Rank ", myrank, " MPI Read I/O tick time: ", (finish-start)/512000000, " seconds\n");
		MPI_File_write_at(tickFp, tick_offset, tick_string1, strlen(tick_string1), MPI_CHAR, MPI_STATUS_IGNORE);
		tick_offset+=strlen(tick_string1);

	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_File_delete(file, MPI_INFO_NULL);

	MPI_File wfp;
	start = getticks();
	err = MPI_File_open(MPI_COMM_SELF, file, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &wfp);
	if(err || !wfp){
		perror("ERROR: Unable to open imtermediate file");
		MPI_Finalize();
		exit(-1);
	}
	i = 0;
	unsigned char** temp = data;
	while(*temp){ //Export current world 
		gol_printWorld(wfp, *temp, i);
		temp++;
		i++;
	}

	MPI_File_close(&wfp);
	finish = getticks();

	char res_string[256]; //Data string
	sprintf(res_string, "%s%d%s%d%s%u%s%u%s", "Rank ", myrank, ", Day ", day, "- Cases: ", totalCases, " | Deaths: ", totalDeaths, "\n");
	
	MPI_File_write_at(resultsFp, results_offset, res_string, strlen(res_string), MPI_CHAR, MPI_STATUS_IGNORE);
	results_offset+=strlen(res_string);

	if(day>0){
		char tick_string2[256]; //Ticks write data
		sprintf(tick_string2, "%s%d%s%d%s%llu%s", "Day ", day, ", Rank ", myrank, " MPI Write I/O tick time: ", (finish-start)/512000000, " seconds\n");
		MPI_File_write_at(tickFp, tick_offset, tick_string2, strlen(tick_string2), MPI_CHAR, MPI_STATUS_IGNORE);
		tick_offset+=strlen(tick_string2);
	}
}


int main(int argc, char* argv[])
{	
	//Intitialization of all input variables to zero
	unsigned int initPattern = 0;
	unsigned int spreadPattern = 0;
	unsigned int infectionRate = 0;
	unsigned int deathRate = 0;
    unsigned int worldSize = 0;
    unsigned int worldDim = 0;
    unsigned int iterations = 0;
    unsigned int threads = 0;
    unsigned int output = 0;
    unsigned int subWorldSize = 0;

    
    unsigned int *valid_cases;
    unsigned int *valid_deaths;
    unsigned int days;

    int myrank = 0;
    int numranks = 0;
    unsigned int i = 0;

    if( argc != 9 ) //Check for correct input argumnet count
    {
		printf("GOL requires 9 arguments: ./gol pattern1 pattern2 infect-rate death-rate world-size iterations threads output, e.g. ./gol 0 0 3 3 300000 10 512 0\n");
		exit(-1);
    }
    int err;

    // Assign all inputs to the proper variables
    initPattern = atoi(argv[1]);
    spreadPattern = atoi(argv[2]);
    infectionRate = atoi(argv[3]);
    deathRate = atoi(argv[4]);
    worldSize = atoi(argv[5]); //Number of seperate grids
    if( worldSize > 2500000 ) //Check for correct input argumnet count
    {
		printf("worldSize argument (argv[5]) must be <= 2,500,000\n");
		exit(-1);
    }
    iterations = atoi(argv[6]); //Number of days to run for
    threads = atoi(argv[7]); 
    output = atoi(argv[8]);
    
    

	// MPI init stuff
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);

	char* stats_fileHead= "data/stats-rank";
	char stats_file[256];
	sprintf(stats_file, "%s%d%s", stats_fileHead, myrank, ".txt");

	//Open all global files
	MPI_File_delete(stats_file, MPI_INFO_NULL);
	err = MPI_File_open(MPI_COMM_SELF, stats_file, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &statsFp);
	if(err){
		perror("Couldn't open local stats file");
		exit(-1);
	}

	char* results_fileHead= "data/results-rank";
	char results_file[256];
	sprintf(results_file, "%s%d%s", results_fileHead, myrank, ".txt");

	MPI_File_delete(results_file, MPI_INFO_NULL);
	err = MPI_File_open(MPI_COMM_SELF, results_file, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &resultsFp);
	if(err){
		perror("Couldn't open local results file");
		exit(-1);
	}
	
	MPI_File_delete("data/tickResults.txt", MPI_INFO_NULL);
	err = MPI_File_open(MPI_COMM_WORLD, "data/tickResults.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &tickFp);
	if(err){
		perror("Couldn't open tick file");
		exit(-1);
	}

	tick_offset = iterations*2*128*myrank*sizeof(char);

	reqs = calloc(numranks*2, sizeof(MPI_Request)); //Allocate space for storage of MPI Requests
	
    worldDim = sqrt(worldSize);

    double scaling = (double) (worldDim*worldDim) / (double) WORLD_POP; //Scaling factor of worldSize/actual US pop
    printf("Scaling factor: %f\n", scaling);

    unsigned int *cases = calloc(365, sizeof(unsigned int));
    unsigned int *deaths = calloc(365, sizeof(unsigned int));

    unsigned char **recv = calloc(numranks, sizeof(unsigned char*));

    MPI_File fp; 
	//int err;
	err = MPI_File_open(MPI_COMM_WORLD, "/gpfs/u/home/PCP9/PCP9ckrm/scratch/project/us.csv", MPI_MODE_RDONLY, MPI_INFO_NULL, &fp); //Input data file
	if(err || !fp){
		fprintf(stderr, "ERROR: Unable to open file!\n");
		MPI_Finalize();
		exit(-1);
	}

    days = fileRead(fp, &cases, &deaths, scaling, myrank, numranks); //Read input file
	MPI_File_close(&fp);
	
	//Needed number of case and death data dates
	valid_cases = (cases+(days-iterations-1)); 
    valid_deaths = (deaths+(days-iterations-1));

	initMaster(myrank, numranks, initPattern, worldDim, valid_cases[0], valid_deaths[0]); // Initialize the world correclty using cudaMallocManaged
	
	if(myrank == 0){ s_time = MPI_Wtime(); } //Start time computation

	if(myrank == 0){printf("Initial- Cases: %u | Deaths: %u\n", totalCases, totalDeaths);}
	
	kernalLaunch(myrank, numranks, &g_data, &g_resultData, g_worldWidth, g_worldHeight, iterations, threads, spreadPattern, infectionRate, deathRate, recv); //Call to the kernal initializtion function to compute result for the current iteration

	MPI_Barrier(MPI_COMM_WORLD); //Barrier to all ranks finish information exchange	

	finishCuda(); //Cuda Device Synchronize
	
	if(myrank == 0) { //Stop clock and output total computation time
    	e_time = MPI_Wtime(); 
        printf("Elapsed time: %f\n", (e_time - s_time));
    }

    if(myrank == 0){ //Output expected results
		int e;
	    MPI_File expectFp;
	    MPI_Offset expect_offset = 0;
	    MPI_File_delete("data/expected.txt", MPI_INFO_NULL);
	    MPI_File_open(MPI_COMM_SELF, "data/expected.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &expectFp);
		for(e = 0; e < iterations; e++){
			char expect_string[256];
			sprintf(expect_string, "%s%d%s%u%s%u%s", "Day ", e, ": Expected cases- ", valid_cases[e], " | Expected deaths- ", valid_deaths[e], "\n");
			MPI_File_write_at(expectFp, expect_offset, expect_string, strlen(expect_string), MPI_CHAR, MPI_STATUS_IGNORE);
			expect_offset+=strlen(expect_string);
		}
		MPI_File_close(&expectFp);
	}

	//Close global files
    MPI_File_close(&statsFp);
    MPI_File_close(&tickFp);
    MPI_File_close(&resultsFp);

	
	if(output) { //Export result to file if specified
		char outfile[256];
		char cwd[512];
		
		getcwd(cwd, sizeof(cwd));

		sprintf(outfile, "%s%d%c%d%s", "/outputs/output-", worldSize, '-', numranks, "_ranks.txt"); //Name the outputfile appropriately, requires the folder "outputs" to exist
		
		strcat(cwd, outfile);

		//Declare and open the output file
		MPI_File ofp; 
		int err;

		MPI_File_delete(cwd, MPI_INFO_NULL);
		err = MPI_File_open(MPI_COMM_WORLD, cwd, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &ofp);
		
		if(err){
			if(ofp){
				MPI_File_close(&ofp);
			}
			perror("ERROR: Unable to open file!");
			fprintf(stderr, "%s\n", cwd);
			MPI_Finalize();
			exit(-1);
		}
		
		gol_printWorld(ofp, g_data, myrank); // Print out final result
		MPI_File_close(&ofp);
    }

    MPI_Finalize();
    
    freeData();

    // Free all delcared variables
	free(cases);
	free(deaths);

	free(recv);
	
    free(reqs);

	return EXIT_SUCCESS;
}
