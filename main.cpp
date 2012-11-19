// This is very crazy but glew.h HAS TO BE included
// before QGLWidget, gl.h and glext.h so it should be the first include to
// avoir problems.
#include <GL/glew.h>
#include <QtGui/QApplication>
#include <QtOpenGL/QGLWidget>
#include "MainWindow.h"

int main(int argc, char *argv[]) {

    QApplication app(argc, argv);

    //GLWidget window;
    
    MainWindow window;

    window.resize(800,600);
    window.show();

    return app.exec();
}
