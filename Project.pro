#OPENCL="/opt/AMDAPP/"
OPENCL="/usr/local/cuda"
GLM="/home/olmozavala/Dropbox/OzOpenCL/"
OZLIB="/home/olmozavala/Dropbox/OzOpenCL/OZlib"
#This is used because I was getting the following erro (April 2014)
# error running a compiled C++ file (uses OpenGL). Error: “Inconsistency detected by ld.so: dl-version.c: 224”

LIBS += -L$${OZLIB} # Adds lib folder (for ozlib)

LIBS += -lGL -lGLU -lglut -lGLEW -lX11 -lm -lFileManager -lOpenCL 
LIBS +=  -lGLManager -lCLManager -lImageManager -lGordonTimers -lfreeimage -lniftiio

#INCLUDEPATH += $${OZLIB}/.."/khronos"

INCLUDEPATH += $${OZLIB}
INCLUDEPATH += $${GLM}
INCLUDEPATH += $${OPENCL}"/include"
INCLUDEPATH += "../SignedDistanceFunction/src/headers"
INCLUDEPATH += "/usr/include/GL" # For glew.h
INCLUDEPATH += "/usr/include/nifti" # For nifti library
INCLUDEPATH += "./src/headers/"  # All headers
INCLUDEPATH += "./src/forms/headers/"  # All headers

HEADERS += src/headers/*.h
HEADERS += src/forms/headers/*.h
HEADERS += ../SignedDistanceFunction/src/headers/SignedDistFunc.h

SOURCES += src/*.cpp
SOURCES += src/forms/src/*.cpp
SOURCES += ../SignedDistanceFunction/src/SignedDistFunc.cpp

FORMS += src/forms/ui/*.ui
MOD_DIR = build/moc
UI_SOURCES_DIR = src/forms/src
UI_HEADERS_DIR = src/forms/headers

TARGET = RunActiveContoursQt
OBJECTS_DIR = build
DESTDIR = dist
MOC_DIR = build/moc

DEFINES += DEBUG

#CONFIG += qt debug
CONFIG += qt debug

QT +=core gui opengl
QMAKE_CXXFLAGS += -w
