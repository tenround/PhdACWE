/*
 * File:   MainWindow.cpp
 * Author: olmozavala
 *
 * Created on September 17, 2012, 8:32 AM
 */

#include "MainWindow.h"
#include <iostream>
#include <QKeyEvent>
#include <QDebug>

using namespace std;
MainWindow::MainWindow() {
	glWidget = new GLWidget();

	setCentralWidget(glWidget);
//    widget.setupUi(this);

}

MainWindow::~MainWindow() {
}

bool MainWindow::eventFilter(QObject *o, QEvent *event){
    cout << "aqui ta" << endl;
    if(event->type() == QEvent::KeyPress){
        qWarning() << "Fuck: " << o;
    }
}

void MainWindow::keyPressEvent(QKeyEvent* event) {
    
    switch (event->key()) {
        case 27:
            close();
            break;
        case Qt::Key_Escape:
            close();
            break;
        default:
            event->ignore();
            break;
    }
}
