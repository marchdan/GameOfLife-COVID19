# GameOfLife-COVID19
---
## COVID-19 Modeling with the Game of Life
	title: "COVID-19 Modeling with the Game of Life"
	date: 2020-05-05
	author: "Daniel Ackermans"
---

## (Report)[https://github.com/marchdan/GameOfLife-COVID19/blob/main/Ackermans-Daniel_Covid19-Modeling-with-the-Game-of-Life_2020.pdf]

### Testing enviroment setup:
- pull us.csv from NYT github files (https://github.com/nytimes/covid-19-data)
- "outputs" folder created
- "data" folder created
- set WORLD_POP to current number from: 
	* https://www.census.gov/popclock/
- compile with "Makefile"
- run with slurm script:
	* Example slurm script: slurpSpectrum.sh
	* Maximum worldSize = 2,000,0000

### Minimun needed directory structure:
- project
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
> 
> 2 1 3 3 300000 7 128 0
>
> Line items:
> 1. initialization pattern (0-4)
> 2. spread pattern (0-3)
> 3. infection rate (0-100)
> 4. death rate (0-100)
> 5. sim. world size ( < 2,000,000)
> 6. iterations (> 0)
> 7. number of threads (1-128)
> 8. output(on/off) (0 or 1)
