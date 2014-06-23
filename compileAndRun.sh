
#!/bin/sh
make clean
qmake Project.pro
make -j 8

./dist/RunActiveContoursQt
