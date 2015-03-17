// This is very crazy but glew.h HAS TO BE included
// before QGLWidget, gl.h and glext.h so it should be the first include to
// avoir problems.
#include <GL/glew.h>
#include <QtGui/QApplication>
#include <QtOpenGL/QGLWidget>
#include "ui_MainWindow.h"
#include "GLWidget.h"
#include "MainWindowOZ.h"

/**
 * This is the main function. It simply initializes the Qt main window.
 * @param argc
 * @param argv
 * @return 
 */
int main(int argc, char *argv[]) {

    QApplication app(argc, argv);
    
	MainWindowOZ *mainWin = new MainWindowOZ();
    mainWin->resize(1200,900);
	mainWin->show();
	
    return app.exec();
}
