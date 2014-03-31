CUDA="/usr/local/cuda-5.5"
OZLIB="/home/olmozavala/Dropbox/OzOpenCL/OZlib/"
GLM="/home/olmozavala/Dropbox/OzOpenCL/"
#
LIBS += -L$${OZLIB}
LIBS += -lOpenCL -lGL -lGLU -lglut -lGLEW -lX11 -lm -lFileManager
LIBS +=  -lGLManager -lCLManager -lImageManager -lGordonTimers -lfreeimage

INCLUDEPATH += $${OZLIB}
INCLUDEPATH += $${GLM}
INCLUDEPATH += $${OZLIB}/.."/khronos"
INCLUDEPATH += "../SignedDistanceFunction"
INCLUDEPATH += $${CUDA}"/include"
INCLUDEPATH += "/usr/include/GL"
INCLUDEPATH += "./src/headers/"

HEADERS += src/headers/*.h
HEADERS += src/forms/headers/*.h
HEADERS += ../SignedDistanceFunction/SignedDistFunc.h

SOURCES += src/*.cpp
SOURCES += src/forms/src/*.cpp
SOURCES += ../SignedDistanceFunction/SignedDistFunc.cpp


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
