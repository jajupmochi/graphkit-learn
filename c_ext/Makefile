# You must specify your env variable LSAPE_DIR
#LSAPE_DIR=/home/bgauzere/Téléchargements/lsape/include/

liblsap.so:lsap.cpp
	g++ -fPIC -I/home/bgauzere/Téléchargements/lsape/include/ -shared lsap.cpp -o liblsap.so -O3 -I$(LSAPE_DIR)
