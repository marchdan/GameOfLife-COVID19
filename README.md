# GameOfLife-COVID19
---
## COVID-19 Modeling with the Game of Life
	title: "COVID-19 Modeling with the Game of Life"
	date: 2020-05-05
	author: "Daniel Ackermans"
---

### Testing enviroment setup:
- pull us.csv from NYT github files
- "outputs" folder created
- "data" folder created
- set WORLD_POP to current number from: 
	* https://www.census.gov/popclock/
- compile with "Makefile"
- run with slurm script:
	* Example slurm script: slurpSpectrum.sh
	* Maximum worldSize = 2,000,0000

### Minimun needed directory structure:
project
- data/
- outputs/
- gol-main.c
- gol-with-cuda.cu
- Makefile
- slurmSpectrum.sh
- us.csv
- README.md

### Compile line arguments 
> slurm*.sh:
``
2 1 3 3 300000 7 128 0
^ ^ ^ ^ ^      ^ ^   ^
| | | | |      | |   |
| | | | |      | |   output(on/off) (0 or 1)
| | | | | 	   | number of threads (1-128)
| | | | |      iterations (> 0)
| | | | sim. world size ( < 2,000,000)
| | | death rate (0-100)
| | infection rate (0-100)
| spread pattern (0-3)
initialization pattern (0-4)
``
