# Object files to either reference or create                          
OBJ = inputData.cpp Random.cpp main.cpp MPsoAmp.cpp
#OBJ = $(wildcard *.cpp)

# The executable file that will be created at the end                 
EXEC = run

# The flags to use for compilation     
FLAGS = -Wall -fopenmp -g

# The code compiler to use for compilation                            
CC = g++ -std=c++11

# Includes
INC = 											\
	#-IC:/Apps/NLopt.v2.6.1/include/ 			\

# Libraries
LIBS = 											\
	#-LC:/Apps/NLopt.v2.6.1/bin -llibnlopt		\
                                                   
# Perform action on all object files (May or may not exist)           
all: info                                                     
	$(CC) $(INC) $(FLAGS) $(OBJ) $(LIBS) -o $(EXEC)
	@echo "Executavel construido com sucesso"
	
info:
	@echo "Compilando codigo. Aguarde ..."