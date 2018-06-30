libmetrics.so : metrics.o
	gcc -shared -Wl,-soname,libmetrics.so -o libmetrics.so metrics.o

metrics.o : metrics.c
	gcc -c -fPIC metrics.c -o metrics.o
